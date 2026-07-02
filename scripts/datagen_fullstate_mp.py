#!/usr/bin/env python3
"""Run SIMPLE MP datagen while recording full MuJoCo state.

This script intentionally does not modify ``simple.cli.datagen`` or the
default ``LerobotRecorder``. It is a side path for collecting raw data with
root/body/EE ground truth so downstream Psi0/TextOp formats can be derived
without integrating root x/y from velocity.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import transforms3d as t3d
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

import simple.envs as _  # noqa: F401, register envs
from simple.agents.mp import MotionPlannerAgent
from simple.envs.lerobot import _from_gym_action_space
from simple.mp.curobo import CuRoboPlanner
from simple.robots.protocols import DualArm, Humanoid, Wholebody
from simple.utils import NumpyArrayEncoder


BODY_JOINT_COUNT = 29
HAND_JOINT_COUNT = 14

EE_BODY_CANDIDATES = (
    ("left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"),
    ("right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"),
    ("left_ankle_roll_link", "left_ankle_pitch_link"),
    ("right_ankle_roll_link", "right_ankle_pitch_link"),
)


def _feature(dtype: str, shape: tuple[int, ...], name: str) -> dict:
    return {"dtype": dtype, "shape": shape, "names": [name]}


def _robot_mj(robot):
    mjdata = getattr(robot, "mjData", None)
    if mjdata is None:
        mjdata = getattr(robot, "mjdata", None)
    mjmodel = getattr(robot, "mjModel", None)
    if mjmodel is None:
        mjmodel = getattr(robot, "mjmodel", None)
    if mjdata is None or mjmodel is None:
        raise AttributeError("Robot does not expose MuJoCo mjData/mjModel")
    return mjdata, mjmodel


def _body_pose_6d(robot, candidates: tuple[str, ...]) -> np.ndarray:
    mjdata, mjmodel = _robot_mj(robot)
    for name in candidates:
        body_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            xyz = np.asarray(mjdata.xpos[body_id], dtype=np.float32)
            quat = np.asarray(mjdata.xquat[body_id], dtype=np.float32)
            rpy = np.asarray(t3d.euler.quat2euler(quat, axes="sxyz"), dtype=np.float32)
            return np.concatenate([xyz, rpy]).astype(np.float32)
    raise ValueError(f"Could not find MuJoCo body from candidates: {candidates}")


def _joint_qvel(joint) -> float:
    qvel = getattr(joint, "qvel", None)
    if qvel is None:
        return 0.0
    return float(qvel[0] if hasattr(qvel, "__len__") else qvel)


class FullStateLerobotRecorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        root_dir: str,
        agent: MotionPlannerAgent,
        shard_size: int = 100,
        debug: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(self, root_dir=root_dir, shard_size=shard_size)
        gym.Wrapper.__init__(self, env)

        task_name = env.unwrapped.spec.id
        dr_level = env.unwrapped.task.dr.level
        dataset_root_dir = f"{os.path.abspath(root_dir)}/{task_name}/level-{dr_level}"
        if debug and os.path.exists(dataset_root_dir):
            shutil.rmtree(dataset_root_dir)

        self.task = env.unwrapped.task
        self.robot = env.unwrapped.task.robot
        self.agent = agent
        self.root_dir = dataset_root_dir
        self.episode_index = 0

        features = {"action": _from_gym_action_space(self.task.action_space)}
        for key, space in self.task.observation_space.items():
            if len(space.shape) == 3 and space.shape[2] == 3:
                features[f"observation.rgb_{key}"] = {
                    "dtype": "video",
                    "shape": space.shape,
                    "names": ["height", "width", "channels"],
                }
            elif len(space.shape) == 1:
                features[f"observation.{key}"] = _feature(space.dtype.name, space.shape, key)

        features.update(
            {
                "observation.root_qpos7": _feature("float32", (7,), "root_xyz_quat_wxyz"),
                "observation.root_pose6": _feature("float32", (6,), "root_xyz_rpy"),
                "observation.root_qvel6": _feature("float32", (6,), "root_lin_ang_vel"),
                "observation.body_qpos29": _feature("float32", (29,), "body_qpos29"),
                "observation.body_qvel29": _feature("float32", (29,), "body_qvel29"),
                "observation.hand_qpos14": _feature("float32", (14,), "hand_qpos14"),
                "observation.hand_qvel14": _feature("float32", (14,), "hand_qvel14"),
                "observation.ee_pose24": _feature("float32", (24,), "four_ee_xyz_rpy"),
                "observation.qpos36": _feature("float32", (36,), "root7_body29"),
                "observation.full_qpos50": _feature("float32", (50,), "root7_body29_hand14"),
            }
        )

        if isinstance(self.task.robot, Wholebody) and hasattr(self.task.robot, "amo_policy"):
            features["observation.amo_policy_obs_prop"] = _feature(
                "float32", (3 + 2 + 2 + 23 * 3 + 2 + 15,), "amo_policy_obs_prop"
            )
            features["observation.amo_policy_output_torque"] = _feature("float32", (15,), "amo_policy_output_torque")
            features["observation.amo_policy_command"] = _feature("float32", (9,), "amo_policy_command")
            features["observation.amo_policy_rpy"] = _feature("float32", (3,), "amo_policy_rpy")
            features["observation.amo_policy_turning_flag"] = _feature("float32", (1,), "amo_policy_turning_flag")
            features["observation.amo_policy_target_yaw"] = _feature("float32", (1,), "amo_policy_target_yaw")

        self.dataset = LeRobotDataset.create(
            repo_id=task_name,
            root=dataset_root_dir,
            fps=self.task.metadata["render_hz"],
            features=features,
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.dataset.clear_episode_buffer()
        self.agent.reset()
        self.env_conf = self.task.state_dict()
        return obs, info

    def clear_episode_buffer(self):
        if self.dataset.episode_buffer is None:
            return
        for cam_key in self.dataset.meta.camera_keys:
            img_dir = self.dataset._get_image_file_path(
                episode_index=self.dataset.episode_buffer["episode_index"],
                image_key=cam_key,
                frame_index=0,
            ).parent
            if img_dir.is_dir():
                shutil.rmtree(img_dir)
        self.dataset.clear_episode_buffer()

    def _full_state_frame(self) -> dict[str, np.ndarray]:
        mjdata, _ = _robot_mj(self.robot)
        joint_names = list(getattr(self.robot, "joint_names", []))
        if len(joint_names) < BODY_JOINT_COUNT + HAND_JOINT_COUNT:
            raise ValueError(f"Expected at least 43 robot joints, got {len(joint_names)}")

        body_names = joint_names[:BODY_JOINT_COUNT]
        hand_names = joint_names[BODY_JOINT_COUNT : BODY_JOINT_COUNT + HAND_JOINT_COUNT]
        body_q = np.asarray([self.robot.joints[n].qpos[0] for n in body_names], dtype=np.float32)
        body_dq = np.asarray([_joint_qvel(self.robot.joints[n]) for n in body_names], dtype=np.float32)
        hand_q = np.asarray([self.robot.joints[n].qpos[0] for n in hand_names], dtype=np.float32)
        hand_dq = np.asarray([_joint_qvel(self.robot.joints[n]) for n in hand_names], dtype=np.float32)

        root_qpos7 = np.asarray(mjdata.qpos[:7], dtype=np.float32)
        root_qvel6 = np.asarray(mjdata.qvel[:6], dtype=np.float32)
        root_rpy = np.asarray(t3d.euler.quat2euler(root_qpos7[3:7], axes="sxyz"), dtype=np.float32)
        ee_pose24 = np.concatenate([_body_pose_6d(self.robot, c) for c in EE_BODY_CANDIDATES]).astype(np.float32)

        return {
            "observation.root_qpos7": root_qpos7,
            "observation.root_pose6": np.concatenate([root_qpos7[:3], root_rpy]).astype(np.float32),
            "observation.root_qvel6": root_qvel6,
            "observation.body_qpos29": body_q,
            "observation.body_qvel29": body_dq,
            "observation.hand_qpos14": hand_q,
            "observation.hand_qvel14": hand_dq,
            "observation.ee_pose24": ee_pose24,
            "observation.qpos36": np.concatenate([root_qpos7, body_q]).astype(np.float32),
            "observation.full_qpos50": np.concatenate([root_qpos7, body_q, hand_q]).astype(np.float32),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        frame = {}
        for key, value in obs.items():
            if value.ndim == 3:
                frame[f"observation.rgb_{key}"] = value
            elif value.ndim == 1:
                frame[f"observation.{key}"] = value
        frame.update(self._full_state_frame())

        action_1d = []
        if isinstance(self.robot, Humanoid):
            for jname in self.robot.joint_names:
                action_1d.append(float(self.robot.actuators[jname].ctrl.item()))
            if isinstance(self.robot, Wholebody) and hasattr(self.robot, "pd_target"):
                action_1d[:15] = self.robot.pd_target.copy()
                if hasattr(self.robot, "hand_target_qpos"):
                    action_1d[-14:] = list(self.robot.hand_target_qpos().values())
        else:
            seen_grippers = set()
            for jname, jvalue in action.parameters["target_qpos"].items():
                if "finger" in jname:
                    gripper_side = jname.split("_")[0]
                    if gripper_side in seen_grippers:
                        continue
                    seen_grippers.add(gripper_side)
                action_1d.append(jvalue)
        frame["action"] = np.asarray(action_1d, dtype=np.float32)

        if isinstance(self.robot, Wholebody) and hasattr(self.robot, "amo_policy"):
            frame["observation.amo_policy_obs_prop"] = self.robot.amo_policy.obs_prop.astype(np.float32)
            frame["observation.amo_policy_output_torque"] = _robot_mj(self.robot)[0].ctrl[:15].astype(np.float32)
            frame["observation.amo_policy_command"] = self.robot.amo_policy.obs_command.astype(np.float32)
            frame["observation.amo_policy_rpy"] = self.robot.amo_policy.obs_rpy.astype(np.float32)
            frame["observation.amo_policy_turning_flag"] = self.robot.amo_policy.obs_turning_flag.astype(np.float32)
            frame["observation.amo_policy_target_yaw"] = self.robot.amo_policy.obs_target_yaw.astype(np.float32)

        self.dataset.add_frame(frame, task=self.unwrapped.task.instruction)

        if terminated or truncated:
            if float(reward) > 0.9:
                self.dataset.save_episode()
                self.write_env_config(self.env_conf, self.episode_index)
                self.episode_index += 1
                self.dataset.clear_episode_buffer()
            else:
                self.clear_episode_buffer()

        return obs, reward, terminated, truncated, info

    def write_env_config(self, env_conf, episode_index):
        meta_file = Path(self.root_dir) / "meta" / "episodes.jsonl"
        if not meta_file.exists():
            return
        with meta_file.open("r") as f:
            lines = [json.loads(line) for line in f]
        lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
        with meta_file.open("w") as f:
            for entry in lines:
                f.write(json.dumps(entry) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", nargs="?", default="simple/G1WholebodyXMoveAndPickMP-v0")
    parser.add_argument("--scene-uid", default="hssd:scene1")
    parser.add_argument("--target-object", default=None)
    parser.add_argument("--sim-mode", default="mujoco_isaac")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--webrtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-episode-steps", type=int, default=30000)
    parser.add_argument("--render-hz", type=int, default=50)
    parser.add_argument("--save-dir", default="data/datagen_fullstate")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--dr-level", type=int, default=0)
    parser.add_argument("--plan-batch-size", type=int, default=1)
    parser.add_argument("--ignore-target-collision", action="store_true")
    parser.add_argument("--easy-motion-gen", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = gym.make(
        args.env_id,
        scene_uid=args.scene_uid,
        target_object=args.target_object,
        sim_mode=args.sim_mode,
        headless=args.headless,
        webrtc=args.webrtc,
        max_episode_steps=args.max_episode_steps,
        render_hz=args.render_hz,
        dr_level=args.dr_level,
    )
    task = env.unwrapped.task
    planner = CuRoboPlanner(
        robot=task.robot,
        plan_dt=0.01,
        plan_batch_size=1,
        easy_motion_gen=args.easy_motion_gen,
        ignore_target_collisions=args.ignore_target_collision,
    )
    mp_agent = MotionPlannerAgent(task, planner, debug=args.debug, plan_batch_size=args.plan_batch_size)
    env = FullStateLerobotRecorder(
        env=env,
        root_dir=args.save_dir,
        agent=mp_agent,
        shard_size=args.shard_size,
        debug=args.debug,
    )

    observation = None
    info = None
    success_count = 0

    def execute_action_sequence(stage: str):
        nonlocal observation, info, success_count
        while True:
            try:
                action = mp_agent.get_action(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if terminated:
                        success_count += 1
                    return "episode_end"
            except StopIteration:
                if stage == "phase":
                    return "phase_done"
                print("Motion plan exhausted before episode end.")
                env.clear_episode_buffer()
                return "episode_end"
            except Exception as exc:
                print(f"Error during action execution: {exc}", file=sys.stderr)
                env.clear_episode_buffer()
                traceback.print_exc()
                return "episode_end"

    pbar = tqdm(total=args.num_episodes, desc="Full-state MP datagen")
    while success_count < args.num_episodes:
        before = success_count
        observation, info = env.reset(options={"state_dict": None})
        phase = 1
        while True:
            try:
                state = mp_agent.synthesize()
                if state is False:
                    break
                if state == "phase_break":
                    result = execute_action_sequence("phase")
                    if result == "phase_done":
                        phase += 1
                        continue
                    break
                execute_action_sequence("final")
                break
            except Exception as exc:
                print(f"Error during episode synthesis: {exc}", file=sys.stderr)
                traceback.print_exc()
                break
        mp_agent.reset()
        if success_count > before:
            pbar.update(success_count - before)
    pbar.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
