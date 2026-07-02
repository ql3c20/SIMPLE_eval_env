from __future__ import annotations

import importlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import typer
import transforms3d as t3d
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm
from typing_extensions import Annotated

import simple.envs as _  # noqa: F401
from simple.cli.eval_decoupled_wbc import _fall_stop_triggered, _make_sonic_config
from simple.datasets.lerobot import get_episode_lerobot
from simple.utils import NumpyArrayEncoder
from simple.utils import snake_to_pascal


def _init_exporter(
    save_dir: str,
    task_prompt: str,
    robot_model,
    obj_names: list[str],
    joint_names: list[str],
    policy_action_dim: int | None = None,
):
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    features = get_dataset_features(robot_model)
    features["observation.state"]["names"] = joint_names
    features["observation.base_pose"] = {
        "dtype": "float64",
        "shape": (9,),
        "names": [
            "base_pose.x",
            "base_pose.y",
            "base_pose.z",
            "base_pose.r00",
            "base_pose.r01",
            "base_pose.r10",
            "base_pose.r11",
            "base_pose.r20",
            "base_pose.r21",
        ],
    }
    features["observation.body_joint_vel"] = {
        "dtype": "float64",
        "shape": (29,),
        "names": [f"body_joint_vel.{name}" for name in joint_names[:29]],
    }
    for key in (
        "observation.left_wrist_yaw_link_pose",
        "observation.right_wrist_yaw_link_pose",
        "observation.left_ankle_roll_link_pose",
        "observation.right_ankle_roll_link_pose",
    ):
        features[key] = {
            "dtype": "float64",
            "shape": (9,),
            "names": [
                f"{key}.x",
                f"{key}.y",
                f"{key}.z",
                f"{key}.r00",
                f"{key}.r01",
                f"{key}.r10",
                f"{key}.r11",
                f"{key}.r20",
                f"{key}.r21",
            ],
        }

    modality_config = get_modality_config(robot_model)
    if obj_names:
        obj_names_flat = []
        for name in obj_names:
            for suffix in ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]:
                obj_names_flat.append(f"{name}.{suffix}")
        features["observation.object_poses"] = {
            "dtype": "float64",
            "shape": (len(obj_names) * 7,),
            "names": obj_names_flat,
        }

    if policy_action_dim is not None and policy_action_dim > 0:
        features["action.policy_action"] = {
            "dtype": "float32",
            "shape": (policy_action_dim,),
            "names": [f"policy_action.{i}" for i in range(policy_action_dim)],
        }

    return Gr00tDataExporter.create(
        save_root=save_dir,
        fps=50,
        features=features,
        modality_config=modality_config,
        task=task_prompt,
    )


def _save_episode_env_config(exporter, env_config: dict, episode_index: int) -> None:
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    lines[episode_index]["environment_config"] = json.dumps(env_config, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def _as_array(value: Any, shape: int | None = None, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=dtype).reshape(-1)
    if shape is not None and arr.size != shape:
        out = np.zeros(shape, dtype=dtype)
        out[: min(shape, arr.size)] = arr[: min(shape, arr.size)]
        return out
    return arr


def _collect_object_poses(sonic_env: Any) -> np.ndarray:
    poses: list[np.ndarray] = []
    for mj_obj in sonic_env.mujoco.mj_objects.values():
        poses.append(
            np.concatenate(
                [
                    np.asarray(mj_obj.xpos, dtype=np.float64).reshape(3),
                    np.asarray(mj_obj.xquat, dtype=np.float64).reshape(4),
                ]
            )
        )
    return np.concatenate(poses) if poses else np.zeros(0, dtype=np.float64)


def _resolve_body_name(sonic_env: Any, candidates: tuple[str, ...]) -> str:
    for body_name in candidates:
        try:
            sonic_env.mujoco.mjModel.body(body_name).id
            return body_name
        except KeyError:
            continue
    raise KeyError(f"None of the candidate body names exist: {candidates}")


def _pose_world(sonic_env: Any, body_name: str) -> np.ndarray:
    body_id = sonic_env.mujoco.mjModel.body(body_name).id
    return np.concatenate(
        [
            np.asarray(sonic_env.mujoco.mjData.xpos[body_id], dtype=np.float64).reshape(3),
            np.asarray(sonic_env.mujoco.mjData.xquat[body_id], dtype=np.float64).reshape(4),
        ]
    )


def _pose_in_base(base_pose: np.ndarray, world_pose: np.ndarray) -> np.ndarray:
    base_pos = _as_array(base_pose[:3], 3)
    base_quat = _as_array(base_pose[3:7], 4)
    world_pos = _as_array(world_pose[:3], 3)
    world_quat = _as_array(world_pose[3:7], 4)

    base_rot = t3d.quaternions.quat2mat(base_quat)
    local_pos = base_rot.T @ (world_pos - base_pos)
    local_quat = t3d.quaternions.qmult(t3d.quaternions.qinverse(base_quat), world_quat)
    return np.concatenate([local_pos, local_quat]).astype(np.float64)


def _pose_xyz_rot6d(pose: np.ndarray) -> np.ndarray:
    pos = _as_array(pose[:3], 3)
    quat = _as_array(pose[3:7], 4)
    mat = t3d.quaternions.quat2mat(quat)
    rot6d = mat[:, :2].reshape(-1)
    return np.concatenate([pos, rot6d]).astype(np.float64)


def _base_pose_quat_from_sim(sonic_env: Any) -> np.ndarray:
    isaac = getattr(sonic_env, "isaac", None)
    isaac_robot = getattr(isaac, "robot", None)
    if isaac_robot is not None:
        try:
            pos, quat = isaac_robot.get_world_pose()
            return np.concatenate([_as_array(pos, 3), _as_array(quat, 4)]).astype(np.float64)
        except Exception:
            pass

    mj_data = sonic_env.mujoco.mjData
    return _as_array(mj_data.qpos[:7], 7)


def _root_stats(frames: list[dict[str, np.ndarray]]) -> dict[str, list[float]] | None:
    if not frames:
        return None
    roots = np.stack([_as_array(frame["observation.base_pose"], 9) for frame in frames])
    return {
        "base_pose_min": roots.min(axis=0).astype(float).tolist(),
        "base_pose_max": roots.max(axis=0).astype(float).tolist(),
    }


def _action43(action: Any) -> np.ndarray:
    if getattr(action, "type", None) != "decoupled_wbc":
        return np.zeros(43, dtype=np.float64)
    target_q = _as_array(action["target_q"], 29)
    left_hand_q = _as_array(action["left_hand_q"], 7)
    right_hand_q = _as_array(action["right_hand_q"], 7)
    return np.concatenate([target_q, left_hand_q, right_hand_q]).astype(np.float64)


def _sim_joint_qpos43(observation: dict[str, Any], sonic_env: Any) -> np.ndarray:
    try:
        qpos = np.asarray(list(sonic_env.mujoco.get_robot_qpos().values()), dtype=np.float64).reshape(-1)
    except Exception:
        qpos = _as_array(observation.get("joint_qpos"), 43)
    return _as_array(qpos, 43)


def _frame_from_step(
    observation: dict[str, Any],
    sonic_env: Any,
    action: Any,
    *,
    agent: Any | None = None,
    export_policy_action: bool = False,
) -> dict[str, np.ndarray]:
    mj_data = sonic_env.mujoco.mjData
    base_pose_arr = _base_pose_quat_from_sim(sonic_env)
    base_vel = _as_array(mj_data.qvel[:6], 6)
    proprio = sonic_env.task.robot.prepare_obs()
    left_wrist_pose = _pose_world(
        sonic_env,
        _resolve_body_name(sonic_env, ("left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link")),
    )
    right_wrist_pose = _pose_world(
        sonic_env,
        _resolve_body_name(sonic_env, ("right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link")),
    )
    left_ankle_pose = _pose_world(
        sonic_env,
        _resolve_body_name(sonic_env, ("left_ankle_roll_link", "left_ankle_pitch_link")),
    )
    right_ankle_pose = _pose_world(
        sonic_env,
        _resolve_body_name(sonic_env, ("right_ankle_roll_link", "right_ankle_pitch_link")),
    )
    frame = {
        "observation.images.ego_view": np.asarray(observation["head_stereo_left"]),
        "observation.state": _sim_joint_qpos43(observation, sonic_env),
        "observation.body_joint_vel": _as_array(proprio.get("body_dq", np.zeros(29)), 29),
        "observation.base_pose": _pose_xyz_rot6d(base_pose_arr),
        "observation.base_vel": base_vel,
        "observation.left_wrist_yaw_link_pose": _pose_xyz_rot6d(left_wrist_pose),
        "observation.right_wrist_yaw_link_pose": _pose_xyz_rot6d(right_wrist_pose),
        "observation.left_ankle_roll_link_pose": _pose_xyz_rot6d(left_ankle_pose),
        "observation.right_ankle_roll_link_pose": _pose_xyz_rot6d(right_ankle_pose),
        "observation.eef_state": np.zeros(14, dtype=np.float64),
        "observation.img_state_delta": np.zeros(1, dtype=np.float32),
        "teleop.navigate_command": np.zeros(4, dtype=np.float64),
        "teleop.base_height_command": np.zeros(1, dtype=np.float64),
        "action": _action43(action),
        "action.eef": np.zeros(14, dtype=np.float64),
        "observation.object_poses": _collect_object_poses(sonic_env),
    }
    if export_policy_action:
        raw_policy = getattr(agent, "_last_policy_action_raw", None) if agent is not None else None
        if raw_policy is not None:
            frame["action.policy_action"] = np.asarray(raw_policy, dtype=np.float32).reshape(-1)
    return frame


def _save_attempt_video(
    frames: list[dict[str, np.ndarray]],
    save_dir: Path,
    attempt: int,
    source_episode_index: int,
    success: bool,
) -> None:
    if not frames:
        return
    video_dir = save_dir / "attempt_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    status = "success" if success else "failed"
    path = video_dir / f"attempt_{attempt:04d}_source_{source_episode_index:06d}_{status}.mp4"
    images = np.stack([frame["observation.images.ego_view"] for frame in frames])
    iio.imwrite(path, images, fps=50, codec="libx264")
    print(f"[AttemptVideo] saved {path}")


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyXMovePickTeleop-v0",
    policy: Annotated[str, typer.Argument()] = "psi0_decoupled_wbc",
    data_dir: Annotated[str, typer.Option()] = "/pfs/pfs-ilWc5D/yzh/Psi0/data/simple/G1WholebodyXMovePickTeleop-v0",
    save_dir: Annotated[str, typer.Option()] = "/pfs/pfs-ilWc5D/yzh/Psi0/data/simple/G1WholebodyXMovePickTeleop-v0-eval-fullstate",
    split: Annotated[str, typer.Option()] = "train",
    host: Annotated[str, typer.Option()] = "172.17.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,
    target_successes: Annotated[int, typer.Option("--target-successes")] = 1,
    max_attempts: Annotated[int, typer.Option("--max-attempts")] = 20,
    episode_start: Annotated[int, typer.Option("--episode-start")] = 0,
    max_episode_steps: Annotated[int | None, typer.Option("--max-episode-steps")] = None,
    success_criteria: Annotated[float, typer.Option("--success-criteria")] = 0.7,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    skip_stabilize: Annotated[bool, typer.Option("--skip-stabilize/--no-skip-stabilize")] = False,
    save_attempt_videos: Annotated[bool, typer.Option("--save-attempt-videos/--no-save-attempt-videos")] = True,
    reseed_dr_level_after_first_pass: Annotated[int | None, typer.Option("--reseed-dr-level-after-first-pass")] = None,
    force_dr_level: Annotated[int | None, typer.Option("--force-dr-level")] = None,
    export_policy_action: Annotated[bool, typer.Option("--export-policy-action/--no-export-policy-action")] = False,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sonic_config = _make_sonic_config()
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt

    dataset = LeRobotDataset(repo_id=env_id, root=data_dir)
    dataset_size = dataset.num_episodes
    if dataset_size <= 0:
        raise ValueError(f"No source episodes found in {data_dir}")

    level_name = Path(data_dir).name
    run_save_dir = Path(save_dir) / env_id.split("/", 1)[1] / level_name
    if run_save_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_save_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(run_save_dir)

    raw_env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=50,
        headless=headless,
        success_criteria=success_criteria,
        sonic_config=sonic_config,
    )
    sonic_env = raw_env.unwrapped  # type: ignore[attr-defined]
    task = sonic_env.task
    if max_episode_steps is None:
        max_episode_steps = task.metadata.get("max_episode_steps")
    env = TimeLimit(raw_env, max_episode_steps=max_episode_steps) if max_episode_steps else raw_env

    policy_module = importlib.import_module(f"simple.baselines.{policy}")
    agent_clazz = getattr(policy_module, f"{snake_to_pascal(policy)}Agent")
    agent = agent_clazz(task.robot, host, port, sonic_config=sonic_config)

    exporter = None
    saved = 0
    attempts = 0
    debug_dir = run_save_dir.parent / f"{run_save_dir.name}_debug"
    if overwrite and debug_dir.exists():
        shutil.rmtree(debug_dir)
    stats_path = debug_dir / "eval_to_lerobot_stats.jsonl"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pbar = tqdm(total=target_successes, desc="Successful eval episodes", unit="ep")
        while saved < target_successes and attempts < max_attempts:
            src_ep_idx = (episode_start + attempts) % dataset_size
            attempts += 1
            print(f"[Attempt] {attempts}/{max_attempts}: source episode {src_ep_idx}")
            env_conf, episode = get_episode_lerobot(dataset, src_ep_idx)
            # Older Psi0 exports predate the teleop task uid rename. The saved
            # DR/layout payload is compatible, but Task.reset checks uid exactly.
            if isinstance(env_conf, dict):
                env_conf["uid"] = task.uid

            reset_options: dict[str, Any] = {"state_dict": env_conf}
            if force_dr_level is not None:
                reset_options["dr_level"] = force_dr_level
            elif (
                reseed_dr_level_after_first_pass is not None
                and attempts > dataset_size
            ):
                reset_options["dr_level"] = reseed_dr_level_after_first_pass

            observation, info = env.reset(options=reset_options)
            agent._wbc_policy.lower_body_policy.use_policy_action = True

            sim_cnt = 0
            if not skip_stabilize:
                while not task.robot.stabilized and sim_cnt < 300:
                    step_start = time.monotonic()
                    stabilize_action = agent.get_stabilize_action(observation)
                    observation, *_, info = env.step(stabilize_action)
                    sonic_env.update_viewer()
                    sonic_env.update_reward()
                    sleep_time = control_dt - (time.monotonic() - step_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    sim_cnt += 1

            agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)
            agent.reset()

            frames: list[dict[str, np.ndarray]] = []
            frame_idx = 0
            episode_over = False
            fall_stop_reason = ""
            last_action: Any = None
            instruction = task.instruction

            while not episode_over:
                try:
                    action = agent.get_action(observation, info=info, instruction=instruction)
                    frames.append(
                        _frame_from_step(
                            observation,
                            sonic_env,
                            action,
                            agent=agent,
                            export_policy_action=export_policy_action,
                        )
                    )
                    observation, reward, terminated, truncated, info = env.step(action)
                    last_action = action
                    frame_idx += 1
                    episode_over = terminated or truncated
                    fall_stop, fall_stop_reason = _fall_stop_triggered(task.robot)
                    if fall_stop:
                        episode_over = True
                except StopIteration:
                    episode_over = True

            is_success = bool(raw_env.unwrapped._success) and not fall_stop_reason  # type: ignore[attr-defined]
            if save_attempt_videos:
                _save_attempt_video(frames, debug_dir, attempts, src_ep_idx, is_success)
            with open(stats_path, "a", buffering=1) as f:
                root_stats = _root_stats(frames) or {}
                f.write(
                    json.dumps(
                        {
                            "attempt": attempts,
                            "source_episode_index": src_ep_idx,
                            "saved_episode_index": saved if is_success else None,
                            "success": is_success,
                            "steps": frame_idx,
                            "fall_stop": fall_stop_reason,
                            **root_stats,
                        }
                    )
                    + "\n"
                )

            if not is_success:
                print(f"[Skip] source episode {src_ep_idx} failed after {frame_idx} steps")
                continue

            if exporter is None:
                obj_names = list(sonic_env.mujoco.mj_objects.keys())
                policy_action_dim = None
                if export_policy_action and frames:
                    raw = frames[0].get("action.policy_action")
                    if raw is not None:
                        policy_action_dim = int(np.asarray(raw).reshape(-1).shape[0])
                exporter = _init_exporter(
                    str(run_save_dir),
                    task.instruction,
                    agent._dwbc_robot_model,
                    obj_names,
                    task.robot.joint_names,
                    policy_action_dim=policy_action_dim,
                )
                print(f"[Record] Exporter initialized at {run_save_dir}")

            for frame in frames:
                exporter.add_frame(frame)
            exporter.save_episode()
            _save_episode_env_config(exporter, task.state_dict(), saved)
            print(f"[Record] saved episode {saved} from source episode {src_ep_idx}")
            saved += 1
            pbar.update(1)

        pbar.close()
        print(f"Saved {saved}/{target_successes} successful episodes after {attempts} attempts")
        if saved < target_successes:
            raise SystemExit(2)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if exporter is not None:
            exporter.stop_video_writers()
        env.close()


if __name__ == "__main__":
    typer.run(main)
