#!/usr/bin/env python3
"""
Experimental MP -> decoupled-WBC full-state datagen.

This is intentionally separate from simple.cli.datagen and
scripts/datagen_fullstate_mp.py.  It reuses the MP task decomposition/CuRobo
planner, but executes queued primitive targets through G1Sonic's decoupled WBC
path instead of the g1_wholebody AMO+PD execution path.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from simple.agents.mp import MotionPlannerAgent
from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd
from simple.core.types import Pose
from simple.envs.sonic_loco_manip import SonicLocoManipEnv
from simple.mp.curobo import CuRoboPlanner
from simple.tasks.g1_wholebody_x_move_and_pick_mp import G1WholebodyXMoveAndPickMP

from datagen_fullstate_mp import FullStateLerobotRecorder


class G1WholebodyXMoveAndPickMPSonic(G1WholebodyXMoveAndPickMP):
    """MP task variant that resets g1_sonic with the spawn pose it requires."""

    def reset(self, seed=None, options=None) -> None:
        super(G1WholebodyXMoveAndPickMP, self).reset(seed, options)
        split = self.metadata.get("split", "train")
        self._target = self.layout.actors.get("target")
        self._container = self.layout.actors.get("container")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name)
        self._init_target_height = None
        self._contact_started = False
        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)


def _make_sonic_config() -> dict:
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    import tyro

    config = tyro.cli(
        SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]
    )
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


class MpToWbcAgent(SonicDecoupledWbcAgent):
    """Convert MotionPlannerAgent primitive actions into decoupled WBC goals."""

    def __init__(self, robot, sonic_config: dict, stabilize_steps: int = 100):
        super().__init__(robot, sonic_config)
        self._mp_queue = deque()
        self._current_mp_action = None
        self._current_action_steps = 0
        self._stabilize_steps = stabilize_steps
        self._step_idx = 0
        self._last_body_q = np.zeros(29, dtype=np.float32)
        self._last_left_hand_q = np.zeros(7, dtype=np.float32)
        self._last_right_hand_q = np.zeros(7, dtype=np.float32)
        self._default_base_height = None

        left_eef = getattr(getattr(robot, "controller_cfg", None), "left_eef", None)
        right_eef = getattr(getattr(robot, "controller_cfg", None), "right_eef", None)
        self._open_left_hand_q = np.zeros(7, dtype=np.float32)
        self._open_right_hand_q = np.zeros(7, dtype=np.float32)
        self._close_left_hand_q = np.asarray(
            getattr(left_eef, "close_qpos", self._open_left_hand_q),
            dtype=np.float32,
        )
        self._close_right_hand_q = np.asarray(
            getattr(right_eef, "close_qpos", self._open_right_hand_q),
            dtype=np.float32,
        )
        self.loco_xy_tol = 0.08
        self.loco_yaw_tol = 0.15
        self.move_q_tol = 0.08
        self.hand_q_tol = 0.12
        self.loco_timeout_steps = 500
        self.move_timeout_steps = 160
        self.hand_timeout_steps = 120

    def load_mp_queue(self, mp_agent: MotionPlannerAgent, move_repeat: int = 5, hand_repeat: int = 15):
        self._mp_queue = deque(mp_agent._action_queue)
        self.move_timeout_steps = max(self.move_timeout_steps, int(move_repeat))
        self.hand_timeout_steps = max(self.hand_timeout_steps, int(hand_repeat))
        self._current_mp_action = None
        self._current_action_steps = 0
        # Mirror PrimitiveAgent.get_action() consumption so multi-phase MP
        # synthesis can continue from a clean queue after PhaseBreakSpec.
        mp_agent._action_queue.clear()

    @staticmethod
    def _yaw_from_wxyz(quat_wxyz: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in quat_wxyz]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _sync_last_targets_from_proprio(self, proprio: dict) -> None:
        body_q = np.asarray(proprio.get("body_q", self._last_body_q), dtype=np.float32).reshape(-1)
        if body_q.size == 29:
            self._last_body_q = body_q.copy()
        left_hand_q = np.asarray(proprio.get("left_hand_q", self._last_left_hand_q), dtype=np.float32).reshape(-1)
        if left_hand_q.size == 7:
            self._last_left_hand_q = left_hand_q.copy()
        right_hand_q = np.asarray(proprio.get("right_hand_q", self._last_right_hand_q), dtype=np.float32).reshape(-1)
        if right_hand_q.size == 7:
            self._last_right_hand_q = right_hand_q.copy()
        root_pose = np.asarray(proprio.get("floating_base_pose", []), dtype=np.float32).reshape(-1)
        if self._default_base_height is None and root_pose.size >= 3:
            self._default_base_height = float(root_pose[2])

    def _target_arrays_from_qpos(self, target_qpos: dict | None, update: bool = True):
        if not target_qpos:
            return self._last_body_q, self._last_left_hand_q, self._last_right_hand_q

        body_names = self.robot.joint_names[:29]
        hand_names = self.robot.joint_names[29:43]
        body_q = np.asarray(
            [target_qpos.get(n, self._last_body_q[i]) for i, n in enumerate(body_names)],
            dtype=np.float32,
        )
        hand_q = np.asarray(
            [target_qpos.get(n, 0.0) for n in hand_names],
            dtype=np.float32,
        )
        left_hand_q = hand_q[:7]
        right_hand_q = hand_q[7:]
        if update:
            self._last_body_q = body_q
            self._last_left_hand_q = left_hand_q
            self._last_right_hand_q = right_hand_q
        return body_q, left_hand_q, right_hand_q

    def _apply_hand_primitive(
        self,
        action_type: str,
        hand_uid: str,
        left_hand_q: np.ndarray,
        right_hand_q: np.ndarray,
        update: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if action_type not in ("open_eef", "close_eef"):
            return left_hand_q, right_hand_q

        if action_type == "open_eef":
            left_target = self._open_left_hand_q
            right_target = self._open_right_hand_q
        else:
            left_target = self._close_left_hand_q
            right_target = self._close_right_hand_q

        hand_uid = hand_uid or ""
        if "left" in hand_uid:
            left_hand_q = left_target.copy()
        elif "right" in hand_uid or "dex3" in hand_uid:
            right_hand_q = right_target.copy()
        else:
            left_hand_q = left_target.copy()
            right_hand_q = right_target.copy()

        if update:
            self._last_left_hand_q = left_hand_q.copy()
            self._last_right_hand_q = right_hand_q.copy()
        return left_hand_q, right_hand_q

    def _current_action_timeout(self, action: ActionCmd) -> int:
        if action.type == "loco_command":
            if action.parameters.get("desired_robot_pose", None) is None:
                return 1
            return self.loco_timeout_steps
        if action.type in ("open_eef", "close_eef"):
            return self.hand_timeout_steps
        return self.move_timeout_steps

    def _loco_done(self, action: ActionCmd, proprio: dict) -> bool:
        desired_pose = action.parameters.get("desired_robot_pose", None)
        if desired_pose is None:
            return False
        root_pose = np.asarray(proprio.get("floating_base_pose", []), dtype=np.float32).reshape(-1)
        desired_pose = np.asarray(desired_pose, dtype=np.float32).reshape(-1)
        if root_pose.size < 7 or desired_pose.size < 7:
            return False
        xy_err = float(np.linalg.norm(root_pose[:2] - desired_pose[:2]))
        yaw = self._yaw_from_wxyz(root_pose[3:7])
        desired_yaw = self._yaw_from_wxyz(desired_pose[3:7])
        yaw_err = abs(self._wrap_angle(yaw - desired_yaw))
        return xy_err < self.loco_xy_tol and yaw_err < self.loco_yaw_tol

    def _qpos_done(self, action: ActionCmd, proprio: dict) -> bool:
        target_qpos = action.parameters.get("target_qpos", None)
        if not target_qpos:
            return False
        target_body, target_left_hand, target_right_hand = self._target_arrays_from_qpos(target_qpos, update=False)
        body_q = np.asarray(proprio.get("body_q", []), dtype=np.float32).reshape(-1)
        left_q = np.asarray(proprio.get("left_hand_q", []), dtype=np.float32).reshape(-1)
        right_q = np.asarray(proprio.get("right_hand_q", []), dtype=np.float32).reshape(-1)
        if body_q.size != target_body.size:
            return False
        try:
            current_full_q = self._dwbc_robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
                left_hand_actuated_joint_values=left_q if left_q.size == 7 else self._last_left_hand_q,
                right_hand_actuated_joint_values=right_q if right_q.size == 7 else self._last_right_hand_q,
            )
            target_full_q = self._dwbc_robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=target_body,
                left_hand_actuated_joint_values=target_left_hand,
                right_hand_actuated_joint_values=target_right_hand,
            )
            body_err = float(
                np.max(
                    np.abs(
                        current_full_q[self._upper_body_indices]
                        - target_full_q[self._upper_body_indices]
                    )
                )
            )
        except Exception:
            body_err = float(np.max(np.abs(body_q - target_body)))
        hand_err = 0.0
        if left_q.size == target_left_hand.size:
            hand_err = max(hand_err, float(np.max(np.abs(left_q - target_left_hand))))
        if right_q.size == target_right_hand.size:
            hand_err = max(hand_err, float(np.max(np.abs(right_q - target_right_hand))))
        return body_err < self.move_q_tol and hand_err < self.hand_q_tol

    def _hand_done(self, action: ActionCmd, proprio: dict) -> bool:
        left_target, right_target = self._apply_hand_primitive(
            action.type,
            action.parameters.get("hand_uid", "") or "",
            self._last_left_hand_q.copy(),
            self._last_right_hand_q.copy(),
            update=False,
        )
        left_q = np.asarray(proprio.get("left_hand_q", []), dtype=np.float32).reshape(-1)
        right_q = np.asarray(proprio.get("right_hand_q", []), dtype=np.float32).reshape(-1)
        errs = []
        hand_uid = action.parameters.get("hand_uid", "") or ""
        if ("left" in hand_uid or not hand_uid) and left_q.size == left_target.size:
            errs.append(float(np.max(np.abs(left_q - left_target))))
        if ("right" in hand_uid or "dex3" in hand_uid or not hand_uid) and right_q.size == right_target.size:
            errs.append(float(np.max(np.abs(right_q - right_target))))
        return bool(errs) and max(errs) < self.hand_q_tol

    def _action_done(self, action: ActionCmd, proprio: dict) -> bool:
        if action.type == "loco_command":
            return self._loco_done(action, proprio)
        if action.type in ("open_eef", "close_eef"):
            return self._hand_done(action, proprio)
        if "move_qpos" in action.type:
            return self._qpos_done(action, proprio)
        return True

    def _goal_from_mp_action(self, mp_action: ActionCmd, proprio: dict) -> tuple[dict, np.ndarray, np.ndarray]:
        from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT, DEFAULT_NAV_CMD

        t_now = time.monotonic()
        wbc_obs = self._build_wbc_observation(proprio)

        nav = np.asarray(DEFAULT_NAV_CMD, dtype=np.float32).copy()
        base_height = float(DEFAULT_BASE_HEIGHT)
        if self._default_base_height is not None:
            base_height = self._default_base_height
        target_qpos = mp_action.parameters.get("target_qpos", None)

        if mp_action.type == "loco_command":
            cmd = np.asarray(mp_action.parameters.get("command", DEFAULT_NAV_CMD), dtype=np.float32)
            nav[0] = cmd[0] if len(cmd) > 0 else 0.0
            nav[1] = cmd[2] if len(cmd) > 2 else 0.0
            nav[2] = cmd[7] if len(cmd) > 7 else 0.0
            nav[3] = cmd[1] if len(cmd) > 1 else 0.0
            if len(cmd) > 3:
                base_height += float(cmd[3])
            if "desired_height" in mp_action.parameters:
                base_height = float(mp_action.parameters["desired_height"])
            target_upper = wbc_obs["q"][self._upper_body_indices].copy()
            left_hand_q = self._last_left_hand_q
            right_hand_q = self._last_right_hand_q
        else:
            body_q, left_hand_q, right_hand_q = self._target_arrays_from_qpos(target_qpos)
            full_q = self._dwbc_robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
                left_hand_actuated_joint_values=left_hand_q,
                right_hand_actuated_joint_values=right_hand_q,
            )
            target_upper = full_q[self._upper_body_indices]

            left_hand_q, right_hand_q = self._apply_hand_primitive(
                mp_action.type,
                mp_action.parameters.get("hand_uid", "") or "",
                left_hand_q,
                right_hand_q,
            )

        goal = {
            "target_upper_body_pose": np.asarray(target_upper, dtype=np.float32),
            "navigate_cmd": nav,
            "base_height_command": np.atleast_1d(np.asarray(base_height, dtype=np.float32)),
            "target_time": t_now + (1.0 / self._control_frequency),
            "interpolation_garbage_collection_time": t_now - 2.0 / self._control_frequency,
            "timestamp": t_now,
        }
        return goal, left_hand_q, right_hand_q

    def get_action(self, observation, instruction=None, **kwargs):
        proprio = kwargs.get("privileged_info", {}).get("proprio")
        if proprio is None:
            proprio = self.robot.prepare_obs()
        self._sync_last_targets_from_proprio(proprio)

        if self._step_idx < self._stabilize_steps:
            self._step_idx += 1
            return self.get_stabilize_action(proprio)

        while True:
            if self._current_mp_action is not None:
                timeout = self._current_action_timeout(self._current_mp_action)
                if self._action_done(self._current_mp_action, proprio) or self._current_action_steps >= timeout:
                    self._current_mp_action = None
                    self._current_action_steps = 0
                else:
                    break

            try:
                self._current_mp_action = self._mp_queue.popleft()
                self._current_action_steps = 0
            except IndexError:
                raise StopIteration("No more MP actions queued for WBC execution.")

        mp_action = self._current_mp_action
        self._current_action_steps += 1

        t_now = time.monotonic()
        wbc_obs = self._build_wbc_observation(proprio)
        goal, left_hand_q, right_hand_q = self._goal_from_mp_action(mp_action, proprio)
        self._wbc_policy.set_observation(wbc_obs)
        self._wbc_policy.set_goal(goal)
        wbc_action = self._wbc_policy.get_action(time=t_now)

        target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        return ActionCmd(
            "decoupled_wbc",
            target_q=target_q,
            left_hand_q=np.asarray(left_hand_q, dtype=np.float32),
            right_hand_q=np.asarray(right_hand_q, dtype=np.float32),
        )

    def reset(self, **kwargs):
        self._wbc_policy.reset(init_time=time.monotonic())
        super().reset(**kwargs)
        self._mp_queue = deque()
        self._current_mp_action = None
        self._current_action_steps = 0
        self._step_idx = 0
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", nargs="?", default="simple/G1WholebodyXMoveAndPickMP-v0")
    parser.add_argument("--sim-mode", default="mujoco")
    parser.add_argument("--render-hz", type=int, default=50)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--save-dir", default="data/datagen_fullstate_xmove_pick_mp_wbc_smoke")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--plan-batch-size", type=int, default=1)
    parser.add_argument("--stabilize-steps", type=int, default=100)
    parser.add_argument("--max-episode-actions", type=int, default=3000)
    parser.add_argument("--move-repeat", type=int, default=5)
    parser.add_argument("--hand-repeat", type=int, default=15)
    parser.add_argument("--easy-motion-gen", action="store_true", default=True)
    parser.add_argument("--ignore-target-collision", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.env_id != "simple/G1WholebodyXMoveAndPickMP-v0":
        raise ValueError("This smoke script currently supports only G1WholebodyXMoveAndPickMP-v0")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    sonic_config = _make_sonic_config()
    # The MP task normally instantiates g1_wholebody with no extra kwargs.
    # For this isolated smoke path, inject sonic_config so RobotRegistry can
    # construct g1_sonic without changing the task implementation globally.
    G1WholebodyXMoveAndPickMPSonic.robot_cfg = {
        **G1WholebodyXMoveAndPickMPSonic.robot_cfg,
        "uid": "g1_sonic",
        "sonic_config": sonic_config,
    }
    task = G1WholebodyXMoveAndPickMPSonic(
        robot_uid="g1_sonic",
        render_hz=args.render_hz,
    )
    env = SonicLocoManipEnv(
        task=task,
        sonic_config=sonic_config,
        sim_mode=args.sim_mode,
        headless=args.headless,
    )
    env.spec = SimpleNamespace(id=args.env_id)

    planner = CuRoboPlanner(
        task.robot,
        plan_batch_size=args.plan_batch_size,
        easy_motion_gen=args.easy_motion_gen,
        ignore_target_collisions=args.ignore_target_collision,
    )
    mp_agent = MotionPlannerAgent(task, planner, debug=args.debug, plan_batch_size=args.plan_batch_size)
    wbc_agent = MpToWbcAgent(task.robot, sonic_config, stabilize_steps=args.stabilize_steps)
    wbc_agent._wbc_policy.lower_body_policy.use_policy_action = True
    wbc_agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)
    env = FullStateLerobotRecorder(
        env=env,
        root_dir=args.save_dir,
        agent=wbc_agent,
        shard_size=args.shard_size,
        debug=args.debug,
    )

    pbar = tqdm(total=args.num_episodes, desc="Full-state MP-WBC datagen")
    success_count = 0

    def sync_layout_robot_pose(info: dict) -> None:
        proprio = info.get("proprio", {}) if isinstance(info, dict) else {}
        base_pose = np.asarray(proprio.get("floating_base_pose", []), dtype=np.float32).reshape(-1)
        if base_pose.size >= 7:
            task.layout.actors["robot"].pose = Pose(
                position=base_pose[:3].tolist(),
                quaternion=base_pose[3:7].tolist(),
            )

    while success_count < args.num_episodes:
        observation, info = env.reset(options={"state_dict": None})
        wbc_agent._wbc_policy.lower_body_policy.use_policy_action = True
        wbc_agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)
        episode_failed = False
        try:
            phase = 1
            while True:
                state = mp_agent.synthesize()
                if state is False:
                    print(
                        "MP synthesis failed "
                        f"(phase={phase}, subtask_index={mp_agent._subtask_index}, "
                        f"queued_actions={len(mp_agent._action_queue)}).",
                        file=sys.stderr,
                    )
                    episode_failed = True
                    break

                is_phase_break = state == "phase_break"
                wbc_agent.load_mp_queue(
                    mp_agent,
                    move_repeat=args.move_repeat,
                    hand_repeat=args.hand_repeat,
                )

                phase_result = "queue_done"
                for _ in range(args.max_episode_actions):
                    try:
                        action = wbc_agent.get_action(observation, privileged_info=info)
                    except StopIteration:
                        phase_result = "queue_done"
                        break

                    observation, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        success_count += int(terminated)
                        phase_result = "episode_done"
                        break
                else:
                    print("MP-WBC action execution hit max-episode-actions.", file=sys.stderr)
                    episode_failed = True
                    break

                if phase_result == "episode_done":
                    break

                if is_phase_break:
                    sync_layout_robot_pose(info)
                    if args.debug:
                        proprio = info.get("proprio", {}) if isinstance(info, dict) else {}
                        base_pose = np.asarray(
                            proprio.get("floating_base_pose", task.robot.get_robot_pose()),
                            dtype=np.float32,
                        ).reshape(-1)
                        target_actor = task.layout.actors.get("target")
                        if target_actor is not None and base_pose.size >= 2:
                            target_xy = np.asarray(target_actor.pose.position[:2], dtype=np.float32)
                            base_xy = base_pose[:2]
                            print(
                                "[MP-WBC] Phase "
                                f"{phase} base_xy={base_xy.tolist()} "
                                f"target_xy={target_xy.tolist()} "
                                f"xy_dist={float(np.linalg.norm(base_xy - target_xy)):.3f}"
                            )
                    print(f"[MP-WBC] Phase {phase} completed; continuing synthesis.")
                    phase += 1
                    continue

                print("MP-WBC queue exhausted before success.", file=sys.stderr)
                episode_failed = True
                break

            if episode_failed:
                env.clear_episode_buffer()
        except StopIteration:
            print("MP-WBC queue exhausted before success.", file=sys.stderr)
            env.clear_episode_buffer()
        except Exception as exc:
            print(f"Error during MP-WBC episode: {exc}", file=sys.stderr)
            traceback.print_exc()
            env.clear_episode_buffer()
        finally:
            mp_agent.reset()
            wbc_agent.reset()

        pbar.n = success_count
        pbar.refresh()

    pbar.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
