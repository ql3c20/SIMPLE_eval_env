"""
Psi0 + Kimodo adapter agent for SIMPLE decoupled WBC evaluation.

This agent talks to a Kimodo-finetuned Psi0 server whose policy action is:

    hand14 + root6 + four ee poses * 6 = 44D

The output must be converted through Kimodo into the original SIMPLE 36D action
format before being handed to the existing decoupled WBC logic.
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from simple.baselines.client import HttpActionClient
from simple.baselines.kimodo_adapter import KimodoPolicyAdapter
from simple.baselines.psi0_decoupled_wbc import from_psi0_upper_joints
from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd


HAND_STATE_SLICES = [
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
]


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return R.from_quat(quat_wxyz, scalar_first=True).as_euler("xyz").astype(np.float32)


class Psi0KimodoDecoupledWbcAgent(SonicDecoupledWbcAgent):
    """SIMPLE-side agent for a Kimodo-action Psi0 checkpoint.

    Responsibilities:
      1. Build the 49D policy state used during Kimodo Psi0 fine-tuning.
      2. Query the Psi0 server and receive 30x44D policy actions.
      3. Convert 44D policy actions to the original SIMPLE 36D action format.
      4. Reuse the existing decoupled WBC tracking logic.

    Step 3 is intentionally isolated in ``_policy44_to_simple36``. The current
    file wires the eval agent into SIMPLE; the Kimodo call/conversion should be
    implemented there next.
    """

    def __init__(self, robot, host: str, port: int, upsample_factor=1, **kwargs):
        super().__init__(robot, **kwargs)

        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor
        self.client = HttpActionClient(self.server_ip, self.server_port)

        self._global_step_idx = 0
        self._reset_history = True
        self._last_target_yaw = 0.0
        self._last_kimodo_qpos_mujoco = None
        self._last_simple_actions = None
        self._last_simple_actions = None
        self._policy_action_queue = []
        self._last_policy_action_raw = None
        self._chunk_blend_frames = int(os.environ.get("KIMODO_BLEND_FRAMES", "4"))
        self._kimodo_adapter = KimodoPolicyAdapter()
        self._rot6d59_convention = os.environ.get("ROT6D59_CONVENTION", "cols_rowmajor")
        self._printed_rot6d59_convention = False

        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [
            name for name, idx in self._dwbc_robot_model.joint_to_dof_index.items() if idx in indices
        ]

    @staticmethod
    def _rot6d_to_rpy(rot6d: np.ndarray, *, convention: str = "cols_rowmajor") -> np.ndarray:
        rot6d = np.asarray(rot6d, dtype=np.float64)
        if rot6d.shape[-1] != 6:
            raise ValueError(f"Expected rot6d last dim 6, got {rot6d.shape}")

        if convention == "cols_rowmajor":
            a1 = rot6d[..., [0, 2, 4]]
            a2 = rot6d[..., [1, 3, 5]]
            build_matrix_from_basis = "columns"
        elif convention == "cols":
            a1 = rot6d[..., :3]
            a2 = rot6d[..., 3:6]
            build_matrix_from_basis = "columns"
        elif convention == "rows":
            a1 = rot6d[..., :3]
            a2 = rot6d[..., 3:6]
            build_matrix_from_basis = "rows"
        else:
            raise ValueError(
                f"Unsupported ROT6D59_CONVENTION={convention!r}; "
                "use cols_rowmajor, cols, or rows."
            )

        b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
        a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = a2_orth / np.maximum(np.linalg.norm(a2_orth, axis=-1, keepdims=True), 1e-8)
        b3 = np.cross(b1, b2)

        mats = np.empty(rot6d.shape[:-1] + (3, 3), dtype=np.float64)
        if build_matrix_from_basis == "columns":
            mats[..., :, 0] = b1
            mats[..., :, 1] = b2
            mats[..., :, 2] = b3
        else:
            mats[..., 0, :] = b1
            mats[..., 1, :] = b2
            mats[..., 2, :] = b3
        return R.from_matrix(mats.reshape(-1, 3, 3)).as_euler("xyz").reshape(rot6d.shape[:-1] + (3,))

    def _policy59_to_policy44(self, policy_action: np.ndarray) -> np.ndarray:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2 or policy_action.shape[1] != 59:
            raise ValueError(f"Expected rot6d59 policy action shape (T, 59), got {policy_action.shape}")
        if not self._printed_rot6d59_convention:
            print(f"[Rot6D59Adapter/WBC] ROT6D59_CONVENTION={self._rot6d59_convention}")
            self._printed_rot6d59_convention = True

        hand14 = policy_action[:, :14]
        root9 = policy_action[:, 14:23]
        ee36 = policy_action[:, 23:59].reshape(policy_action.shape[0], 4, 9)

        root6 = np.concatenate(
            [
                root9[:, :3],
                self._rot6d_to_rpy(root9[:, 3:9], convention=self._rot6d59_convention).astype(np.float32),
            ],
            axis=1,
        )
        ee_chunks = []
        for i in range(4):
            pose9 = ee36[:, i]
            ee_chunks.append(
                np.concatenate(
                    [
                        pose9[:, :3],
                        self._rot6d_to_rpy(pose9[:, 3:9], convention=self._rot6d59_convention).astype(np.float32),
                    ],
                    axis=1,
                )
            )
        return np.concatenate([hand14, root6.astype(np.float32), *ee_chunks], axis=1).astype(np.float32)

    def _normalize_policy_action(self, policy_action: np.ndarray) -> np.ndarray:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2:
            raise ValueError(f"Expected policy action shape (T, D), got {policy_action.shape}")
        if policy_action.shape[1] == 44:
            return policy_action
        if policy_action.shape[1] == 59:
            converted = self._policy59_to_policy44(policy_action)
            print("[Rot6D59Adapter/WBC] converted policy action 59D -> 44D for Kimodo/WBC")
            return converted
        raise ValueError(f"Expected policy action dim 44 or 59, got {policy_action.shape}")

    def _build_policy_state49(self, observation) -> np.ndarray:
        """Build ``hand14 + body29 + root6`` as shape ``(1, 49)``."""
        joint_qpos = np.asarray(observation["joint_qpos"], dtype=np.float32)[None]
        hand14 = np.concatenate([joint_qpos[:, s:e] for _, s, e in HAND_STATE_SLICES], axis=1)

        proprio = self.robot.prepare_obs()
        body29 = np.asarray(proprio["body_q"], dtype=np.float32).reshape(1, -1)
        if body29.shape[1] != 29:
            raise ValueError(f"Expected body_q to be 29D, got {body29.shape}")

        root_pose = np.asarray(proprio.get("floating_base_pose", np.zeros(7)), dtype=np.float32).reshape(-1)
        if root_pose.size >= 7:
            root_xyz = root_pose[:3].reshape(1, 3)
            root_rpy = _quat_wxyz_to_rpy(root_pose[3:7]).reshape(1, 3)
        else:
            root_xyz = np.asarray([[0.0, 0.0, 0.74]], dtype=np.float32)
            root_rpy = np.zeros((1, 3), dtype=np.float32)
        root6 = np.concatenate([root_xyz, root_rpy], axis=1).astype(np.float32)

        state49 = np.concatenate([hand14, body29, root6], axis=1).astype(np.float32)
        if state49.shape != (1, 49):
            raise ValueError(f"Expected policy state shape (1, 49), got {state49.shape}")
        return state49

    def _policy44_to_simple36(self, policy_action: np.ndarray, instruction: str | None = None) -> np.ndarray:
        """Convert ``30x44`` Kimodo-policy actions to original SIMPLE ``Tx36`` actions.

        Expected input layout:
            hand14 + root6(xyz,rpy) + four ee poses * 6

        Expected output layout:
            hand14 + arm14 + torso_rpy3 + height1 + vx + vy + vyaw + target_yaw

        The adapter:
          - converts root6 + EE poses into 28D constraints,
          - converts the 28D constraints from MuJoCo z-up to Kimodo y-up,
          - run Kimodo to obtain qpos ``root_xyz3 + root_quat4 + joint29``,
          - resample qpos to 50Hz if needed,
          - extract arm14/root rpy/height and finite-difference vx, vy, vyaw,
          - combine them with hand14 and root yaw from ``policy_action``.
        """
        policy_action = self._normalize_policy_action(policy_action)
        current_qpos_mujoco = self._current_mujoco_qpos36()
        simple36, qpos50 = self._kimodo_adapter.policy44_to_simple36(
            policy_action,
            prev_qpos_mujoco=current_qpos_mujoco,
            current_constraints28_mujoco=self._current_constraints28_mujoco(),
            instruction=instruction,
        )
        self._last_kimodo_qpos_mujoco = qpos50[-1].copy()
        if self._last_simple_actions is not None and self._chunk_blend_frames > 0:
            simple36 = self._blend_chunk_start(self._last_simple_actions, simple36, self._chunk_blend_frames)
        self._last_simple_actions = simple36.copy()
        return simple36

    @staticmethod
    def _blend_chunk_start(prev_actions: np.ndarray, next_actions: np.ndarray, blend_frames: int) -> np.ndarray:
        blend_frames = min(int(blend_frames), len(prev_actions), len(next_actions))
        if blend_frames <= 0:
            return next_actions
        blended = next_actions.copy()
        prev_tail = prev_actions[-blend_frames:]
        for idx in range(blend_frames):
            alpha = float(idx + 1) / float(blend_frames + 1)
            blended[idx, :35] = (1.0 - alpha) * prev_tail[idx, :35] + alpha * next_actions[idx, :35]
            yaw_delta = np.arctan2(
                np.sin(next_actions[idx, 35] - prev_tail[idx, 35]),
                np.cos(next_actions[idx, 35] - prev_tail[idx, 35]),
            )
            blended[idx, 35] = prev_tail[idx, 35] + alpha * yaw_delta
        return blended.astype(np.float32)

    def _current_mujoco_qpos36(self) -> np.ndarray:
        proprio = self.robot.prepare_obs()
        root_pose = np.asarray(proprio.get("floating_base_pose", np.zeros(7)), dtype=np.float32).reshape(-1)
        if root_pose.size < 7:
            root_pose = np.asarray([0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        body_q = np.asarray(proprio["body_q"], dtype=np.float32).reshape(-1)
        if body_q.size != 29:
            raise ValueError(f"Expected body_q to be 29D, got {body_q.shape}")
        return np.concatenate([root_pose[:7], body_q], axis=0).astype(np.float32)

    def _body_pose_6d(self, candidates: tuple[str, ...]) -> np.ndarray:
        import mujoco

        for name in candidates:
            body_id = mujoco.mj_name2id(self.robot.mjModel, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                xyz = np.asarray(self.robot.mjData.xpos[body_id], dtype=np.float32)
                quat = np.asarray(self.robot.mjData.xquat[body_id], dtype=np.float32)
                rpy = R.from_quat(quat, scalar_first=True).as_euler("xyz").astype(np.float32)
                return np.concatenate([xyz, rpy], axis=0).astype(np.float32)
        raise ValueError(f"Could not find any MuJoCo body from candidates: {candidates}")

    def _current_constraints28_mujoco(self) -> np.ndarray:
        qpos = self._current_mujoco_qpos36()
        root_yaw = R.from_quat(qpos[3:7], scalar_first=True).as_euler("xyz")[2]
        root_xyzyaw = np.asarray([qpos[0], qpos[1], qpos[2], root_yaw], dtype=np.float32)

        left_hand = self._body_pose_6d(("left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"))
        right_hand = self._body_pose_6d(("right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"))
        left_foot = self._body_pose_6d(("left_ankle_roll_link", "left_ankle_pitch_link"))
        right_foot = self._body_pose_6d(("right_ankle_roll_link", "right_ankle_pitch_link"))
        return np.concatenate([root_xyzyaw, left_hand, right_hand, left_foot, right_foot], axis=0).astype(np.float32)

    def _queue_simple36_actions(self, simple_actions: np.ndarray, *, policy_actions_raw: np.ndarray | None = None) -> None:
        simple_actions = np.asarray(simple_actions, dtype=np.float32)
        if simple_actions.ndim != 2 or simple_actions.shape[1] != 36:
            raise ValueError(f"Expected SIMPLE actions shape (T, 36), got {simple_actions.shape}")
        if policy_actions_raw is not None:
            policy_actions_raw = np.asarray(policy_actions_raw, dtype=np.float32)
            if policy_actions_raw.ndim != 2 or policy_actions_raw.shape[0] != simple_actions.shape[0]:
                raise ValueError(
                    f"Expected raw policy chunk shape (T, D) aligned with simple_actions, "
                    f"got {policy_actions_raw.shape} vs {simple_actions.shape}"
                )

        for idx, action in enumerate(simple_actions):
            self._last_target_yaw = float(action[35])
            for _ in range(self.upsample_factor):
                if policy_actions_raw is not None:
                    self._policy_action_queue.append(policy_actions_raw[idx].copy())
                target_qpos = dict(
                    zip(
                        self.robot.joint_names[15:],
                        from_psi0_upper_joints(action[:28]),
                    )
                )
                target_waist_qpos = {
                    "waist_yaw_joint": action[30],
                    "waist_roll_joint": action[28],
                    "waist_pitch_joint": action[29],
                }
                self.queue_action(
                    ActionCmd(
                        "vla_cmd",
                        target_upper_body_pose={**target_qpos, **target_waist_qpos},
                        navigate_cmd=action[32:36],
                        base_height_command=action[31:32],
                    )
                )

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if len(self._action_queue) == 0:
            observations = {
                "rgb_head_stereo_left": observation["head_stereo_left"],
            }
            state_dict = {"states": self._build_policy_state49(observation)}

            if self._reset_history:
                history = {"reset": True}
                self._reset_history = False
            else:
                history = {}

            policy_action, *_ = self.client.query_action(
                observations,
                instruction or "bend to pick up the object",
                state_dict,
                {},
                history=history,
                dataset="simple",
            )
            print(f"Received {policy_action.shape[0]} Kimodo-policy actions from server.")
            simple_actions = self._policy44_to_simple36(policy_action, instruction=instruction)
            self._queue_simple36_actions(simple_actions, policy_actions_raw=policy_action)

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type != "vla_cmd":
            raise ValueError(f"Unexpected action type {action_cmd.type} from queue.")
        if self._policy_action_queue:
            self._last_policy_action_raw = self._policy_action_queue.pop(0)
        else:
            self._last_policy_action_raw = None

        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)

        t_now = time.monotonic()
        control_freq = self._control_frequency
        target_time = t_now + 1 / control_freq
        target_upper_body_pose = np.array(
            [action_cmd["target_upper_body_pose"][jName] for jName in self.sonic_upper_joint_names],
            dtype=np.float32,
        )
        goal = {
            "target_upper_body_pose": target_upper_body_pose,
            "navigate_cmd": action_cmd["navigate_cmd"],
            "base_height_command": action_cmd["base_height_command"],
            "target_time": target_time,
            "interpolation_garbage_collection_time": t_now - 2 / control_freq,
            "timestamp": t_now,
        }
        self._wbc_policy.set_goal(goal)
        wbc_action = self._wbc_policy.get_action(time=t_now)
        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

        self._last_pred_action = ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )
        self._global_step_idx += 1
        return self._last_pred_action

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._kimodo_adapter.begin_episode()
        self._global_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_target_yaw = 0.0
        self._last_kimodo_qpos_mujoco = None
        self._policy_action_queue = []
        self._last_policy_action_raw = None
