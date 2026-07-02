"""
Replay Psi0 SIMPLE teleop data and write an enriched LeRobot-style dataset.

The source dataset is left untouched.  Videos are copied verbatim; parquet files
are rewritten with additional low-dimensional state measured during MuJoCo/WBC
replay, such as floating-base pose and FK end-effector poses.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import transforms3d as t3d
import typer
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from typing_extensions import Annotated

import simple.envs as _  # noqa: F401
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic


DEFAULT_SOURCE = "/pfs/pfs-ilWc5D/yzh/Psi0/data/simple/G1WholebodyXMovePickTeleop-v0"
DEFAULT_OUTPUT = "/pfs/pfs-ilWc5D/yzh/Psi0/data/simple/G1WholebodyXMovePickTeleop-v0-fullstate"

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _load_sonic_config() -> dict:
    import tyro

    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _as_array(value: Any, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    return arr.reshape(-1)


def _quat_wxyz_from_matrix(mat: np.ndarray) -> np.ndarray:
    return np.asarray(t3d.quaternions.mat2quat(mat), dtype=np.float32)


def _pose7_from_placement(placement: Any) -> np.ndarray:
    pos = np.asarray(placement.translation, dtype=np.float32)
    quat = _quat_wxyz_from_matrix(np.asarray(placement.rotation, dtype=np.float32))
    return np.concatenate([pos, quat]).astype(np.float32)


def _world_from_base_pose(base_pose: np.ndarray, local_pose7: np.ndarray) -> np.ndarray:
    base_pos = np.asarray(base_pose[:3], dtype=np.float64)
    base_quat = np.asarray(base_pose[3:7], dtype=np.float64)
    local_pos = np.asarray(local_pose7[:3], dtype=np.float64)
    local_quat = np.asarray(local_pose7[3:7], dtype=np.float64)

    base_rot = t3d.quaternions.quat2mat(base_quat)
    world_pos = base_pos + base_rot @ local_pos
    world_quat = t3d.quaternions.qmult(base_quat, local_quat)
    return np.concatenate([world_pos, world_quat]).astype(np.float32)


class Psi0ReplayWbcAgent(SonicDecoupledWbcAgent):
    """Maps Psi0 36D teleop-style actions to decoupled WBC actions."""

    def __init__(self, robot: G1Sonic, sonic_config: dict):
        super().__init__(robot, sonic_config)
        self._wbc_policy.lower_body_policy.use_policy_action = True
        self._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)
        self._joint_to_full_idx = {name: i for i, name in enumerate(self._dwbc_robot_model.joint_names)}

    def reset_policy(self) -> None:
        self._wbc_policy.reset(init_time=time.monotonic())
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._wbc_policy.lower_body_policy.use_policy_action = True
        self._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)

    def _goal_from_psi0_action(self, action36: np.ndarray, proprio: dict) -> tuple[dict, np.ndarray, np.ndarray]:
        from decoupled_wbc.control.main.constants import DEFAULT_NAV_CMD

        action36 = np.asarray(action36, dtype=np.float32).reshape(-1)
        if action36.size < 36:
            raise ValueError(f"Expected >=36 action dims, got {action36.size}")

        left_hand_q = action36[0:7].copy()
        right_hand_q = action36[7:14].copy()
        arm_q = action36[14:28].copy()
        torso_rpy = action36[28:31].copy()
        base_height = np.asarray([action36[31]], dtype=np.float32)
        nav = np.asarray(DEFAULT_NAV_CMD, dtype=np.float32).copy()
        nav[:4] = action36[32:36]

        wbc_obs = self._build_wbc_observation(proprio)
        target_full_q = wbc_obs["q"].copy()

        for joint_name, value in zip(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS, arm_q):
            idx = self._joint_to_full_idx.get(joint_name)
            if idx is not None:
                target_full_q[idx] = value

        # Source stores torso_rpy as roll, pitch, yaw; G1 waist joints are yaw, roll, pitch.
        waist_values = {
            "waist_roll_joint": torso_rpy[0],
            "waist_pitch_joint": torso_rpy[1],
            "waist_yaw_joint": torso_rpy[2],
        }
        for joint_name, value in waist_values.items():
            idx = self._joint_to_full_idx.get(joint_name)
            if idx is not None:
                target_full_q[idx] = value

        t_now = time.monotonic()
        goal = {
            "target_upper_body_pose": target_full_q[self._upper_body_indices].astype(np.float32),
            "navigate_cmd": nav,
            "base_height_command": base_height,
            "target_time": t_now + 1.0 / self._control_frequency,
            "interpolation_garbage_collection_time": t_now - 2.0 / self._control_frequency,
            "timestamp": t_now,
        }
        return goal, left_hand_q, right_hand_q

    def get_action_from_psi0(self, action36: np.ndarray, proprio: dict) -> ActionCmd:
        wbc_obs = self._build_wbc_observation(proprio)
        goal, left_hand_q, right_hand_q = self._goal_from_psi0_action(action36, proprio)
        self._wbc_policy.set_observation(wbc_obs)
        self._wbc_policy.set_goal(goal)
        wbc_action = self._wbc_policy.get_action(time=time.monotonic())
        target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        return ActionCmd(
            "decoupled_wbc",
            target_q=np.asarray(target_q, dtype=np.float32),
            left_hand_q=np.asarray(left_hand_q, dtype=np.float32),
            right_hand_q=np.asarray(right_hand_q, dtype=np.float32),
        )


def _collect_fullstate(agent: Psi0ReplayWbcAgent, info: dict) -> dict[str, np.ndarray]:
    proprio = info["proprio"]
    base_pose = _as_array(proprio["floating_base_pose"])
    base_vel = _as_array(proprio["floating_base_vel"])
    base_acc = _as_array(proprio.get("floating_base_acc", np.zeros(6, dtype=np.float32)))

    body_q = _as_array(proprio["body_q"])
    left_hand_q = _as_array(proprio.get("left_hand_q", np.zeros(7, dtype=np.float32)))
    right_hand_q = _as_array(proprio.get("right_hand_q", np.zeros(7, dtype=np.float32)))

    rm = agent._dwbc_robot_model
    q_full = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=body_q,
        left_hand_actuated_joint_values=left_hand_q,
        right_hand_actuated_joint_values=right_hand_q,
    )
    rm.cache_forward_kinematics(q_full)

    left_frame = rm.supplemental_info.hand_frame_names["left"]
    right_frame = rm.supplemental_info.hand_frame_names["right"]
    left_base = _pose7_from_placement(rm.frame_placement(left_frame))
    right_base = _pose7_from_placement(rm.frame_placement(right_frame))
    left_world = _world_from_base_pose(base_pose, left_base)
    right_world = _world_from_base_pose(base_pose, right_base)

    object_poses = []
    object_names = []
    for name, value in info.items():
        if name == "proprio":
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 7:
            object_names.append(str(name))
            object_poses.append(arr)

    return {
        "observation.base_pose": base_pose.astype(np.float32),
        "observation.base_vel": base_vel.astype(np.float32),
        "observation.base_acc": base_acc.astype(np.float32),
        "observation.left_eef_pose_base": left_base.astype(np.float32),
        "observation.right_eef_pose_base": right_base.astype(np.float32),
        "observation.left_eef_pose_world": left_world.astype(np.float32),
        "observation.right_eef_pose_world": right_world.astype(np.float32),
        "observation.object_poses": (
            np.concatenate(object_poses).astype(np.float32)
            if object_poses
            else np.zeros(0, dtype=np.float32)
        ),
        "_object_names": np.asarray(object_names, dtype=object),
    }


def _feature(dtype: str, shape: list[int], names: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"dtype": dtype, "shape": shape}
    if names is not None:
        out["names"] = names
    return out


def _update_info_json(
    dst_root: Path,
    object_names: list[str],
    total_episodes: int,
    total_frames: int,
) -> None:
    info_path = dst_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_videos"] = int(total_episodes)
    features = info.setdefault("features", {})
    features["observation.base_pose"] = _feature(
        "float32", [7], ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]
    )
    features["observation.base_vel"] = _feature(
        "float32", [6], ["lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"]
    )
    features["observation.base_acc"] = _feature(
        "float32", [6], ["lin_acc_x", "lin_acc_y", "lin_acc_z", "ang_acc_x", "ang_acc_y", "ang_acc_z"]
    )
    pose_names = ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]
    for key in [
        "observation.left_eef_pose_base",
        "observation.right_eef_pose_base",
        "observation.left_eef_pose_world",
        "observation.right_eef_pose_world",
    ]:
        features[key] = _feature("float32", [7], pose_names)

    object_feature_names: list[str] = []
    for obj_name in object_names:
        object_feature_names.extend([f"{obj_name}.{suffix}" for suffix in pose_names])
    features["observation.object_poses"] = _feature(
        "float32", [len(object_feature_names)], object_feature_names
    )
    info_path.write_text(json.dumps(info, indent=4))


def _rewrite_selected_meta(dst_root: Path, selected_meta: list[dict[str, Any]], lengths: dict[int, int]) -> int:
    cursor = 0
    selected_indices = {int(row["episode_index"]) for row in selected_meta}
    rewritten = []
    for row in selected_meta:
        ep_idx = int(row["episode_index"])
        length = int(lengths[ep_idx])
        out = dict(row)
        out["length"] = length
        out["dataset_from_index"] = cursor
        out["dataset_to_index"] = cursor + length - 1
        rewritten.append(out)
        cursor += length
    _write_jsonl(dst_root / "meta" / "episodes.jsonl", rewritten)

    stats_path = dst_root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        stats_rows = [
            row
            for row in _read_jsonl(stats_path)
            if int(row.get("episode_index", -1)) in selected_indices
        ]
        _write_jsonl(stats_path, stats_rows)
    return cursor


def _copy_static_tree(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for name in ("meta", "videos"):
        src = src_root / name
        if src.exists():
            shutil.copytree(src, dst_root / name)
    (dst_root / "data").mkdir(parents=True, exist_ok=True)


def main(
    source_dir: Annotated[str, typer.Option("--source-dir")] = DEFAULT_SOURCE,
    output_dir: Annotated[str, typer.Option("--output-dir")] = DEFAULT_OUTPUT,
    env_id: Annotated[str, typer.Option("--env-id")] = "simple/G1WholebodyXMovePickTeleop-v0",
    sim_mode: Annotated[str, typer.Option("--sim-mode")] = "mujoco",
    headless: Annotated[bool, typer.Option("--headless/--no-headless")] = True,
    num_episodes: Annotated[int, typer.Option("--num-episodes")] = -1,
    episode_start: Annotated[int, typer.Option("--episode-start")] = 0,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    copy_videos: Annotated[bool, typer.Option("--copy-videos/--no-copy-videos")] = True,
    render_videos: Annotated[bool, typer.Option("--render-videos/--no-render-videos")] = False,
    real_time: Annotated[bool, typer.Option("--real-time/--no-real-time")] = False,
) -> None:
    src_root = Path(source_dir).expanduser().resolve()
    dst_root = Path(output_dir).expanduser().resolve()
    if not (src_root / "meta" / "episodes.jsonl").exists():
        raise FileNotFoundError(f"Missing source meta/episodes.jsonl under {src_root}")

    _copy_static_tree(src_root, dst_root, overwrite=overwrite)
    if (render_videos or not copy_videos) and (dst_root / "videos").exists():
        shutil.rmtree(dst_root / "videos")
    if render_videos:
        (dst_root / "videos").mkdir(parents=True, exist_ok=True)

    episodes_meta = _read_jsonl(src_root / "meta" / "episodes.jsonl")
    selected = episodes_meta[episode_start:]
    if num_episodes >= 0:
        selected = selected[:num_episodes]

    sonic_config = _load_sonic_config()
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt

    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=50,
        physics_dt=sim_dt,
        headless=headless,
        max_episode_steps=30000,
        sonic_config=sonic_config,
    )
    robot = env.unwrapped.task.robot  # type: ignore[attr-defined]
    if not isinstance(robot, G1Sonic):
        raise TypeError(f"Expected G1Sonic robot, got {type(robot).__name__}")
    agent = Psi0ReplayWbcAgent(robot, sonic_config)

    object_names_for_meta: list[str] = []
    processed_meta: list[dict[str, Any]] = []
    processed_lengths: dict[int, int] = {}
    try:
        for ep_meta in tqdm(selected, desc="Replaying episodes", unit="ep"):
            ep_idx = int(ep_meta["episode_index"])
            chunk_idx = ep_idx // int(json.loads((src_root / "meta" / "info.json").read_text()).get("chunks_size", 1000))
            src_parquet = src_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            dst_parquet = dst_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_parquet(src_parquet)
            env_conf = json.loads(ep_meta["environment_config"])
            # Older Psi0 datasets were saved before the teleop task uid was
            # renamed.  The DR/layout payload is still compatible with the
            # current env, but Task.reset checks the uid string exactly.
            env_conf["uid"] = env.unwrapped.task.uid  # type: ignore[attr-defined]
            ep_meta_out = dict(ep_meta)
            ep_meta_out["environment_config"] = json.dumps(env_conf)
            observation, info = env.reset(options={"state_dict": env_conf})
            agent.reset_policy()

            enriched_rows: list[dict[str, Any]] = []
            rendered_frames: list[np.ndarray] = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  ep {ep_idx:06d}", leave=False):
                fullstate = _collect_fullstate(agent, info)
                object_names = [str(x) for x in fullstate.pop("_object_names").tolist()]
                if object_names and not object_names_for_meta:
                    object_names_for_meta = object_names

                row_dict = row.to_dict()
                row_dict.update(fullstate)
                enriched_rows.append(row_dict)
                if render_videos:
                    rendered_frames.append(np.asarray(observation["head_stereo_left"]))

                action = agent.get_action_from_psi0(np.asarray(row["action"], dtype=np.float32), info["proprio"])
                step_start = time.monotonic()
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    # Continue writing the remaining rows from the terminal state, so
                    # the enriched parquet stays frame-aligned with the source video.
                    pass
                if real_time:
                    sleep_time = control_dt - (time.monotonic() - step_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            pd.DataFrame(enriched_rows).to_parquet(dst_parquet, index=False)
            if render_videos:
                video_path = (
                    dst_root
                    / "videos"
                    / f"chunk-{chunk_idx:03d}"
                    / "egocentric"
                    / f"episode_{ep_idx:06d}.mp4"
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(video_path, rendered_frames, fps=50, codec="libx264")
            processed_meta.append(ep_meta_out)
            processed_lengths[ep_idx] = len(enriched_rows)
    finally:
        env.close()
        close = getattr(agent, "close", None)
        if callable(close):
            close()

    total_frames = _rewrite_selected_meta(dst_root, processed_meta, processed_lengths)
    _update_info_json(dst_root, object_names_for_meta, len(processed_meta), total_frames)
    print(f"Wrote enriched dataset to {dst_root}")


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
