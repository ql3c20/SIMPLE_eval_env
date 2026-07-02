from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.baselines.client import HttpActionClient
from simple.baselines.psi0_kimodo_decoupled_wbc import HAND_STATE_SLICES, _quat_wxyz_to_rpy
from simple.baselines.textop_tracker_adapter import TextOpTrackerAdapter
from simple.core.action import ActionCmd


class Psi0Direct49TextOpTrackerAgent(SonicDecoupledWbcAgent):
    """Psi0 direct49 action -> qpos36 reference -> TextOp tracker + direct hand14.

    Policy action layout:
        hand14 + body29(leg15+arm14 in MuJoCo order) + root6(xyz,rpy)
    """

    def __init__(self, robot, host: str, port: int, upsample_factor=1, **kwargs):
        super().__init__(robot, **kwargs)
        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor
        self.client = HttpActionClient(self.server_ip, self.server_port)
        self._textop_adapter: TextOpTrackerAdapter | None = None
        self._direct_queue: list[tuple[dict[str, np.ndarray], int, np.ndarray]] = []
        self._global_step_idx = 0
        self._reset_history = True
        self._debug = os.environ.get("TEXTOP_DEBUG", "0") == "1"
        self._debug_dir = os.environ.get("TEXTOP_DEBUG_DIR", "/pfs/pfs-ilWc5D/yzh/Psi0/outputs/direct49_textop_debug")
        self._debug_trace = []
        self._episode_idx = -1
        self._dump_qpos_dir = os.environ.get("DIRECT49_QPOS_DUMP_DIR", "")
        self._dumped_qpos_chunks: list[np.ndarray] = []

    def _build_policy_state49(self, observation) -> np.ndarray:
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

        state49 = np.concatenate([hand14, body29, root_xyz, root_rpy], axis=1).astype(np.float32)
        if state49.shape != (1, 49):
            raise ValueError(f"Expected policy state shape (1, 49), got {state49.shape}")
        return state49

    @staticmethod
    def _policy49_to_qpos36(policy_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2 or policy_action.shape[1] != 49:
            raise ValueError(f"Expected direct49 policy action shape (T, 49), got {policy_action.shape}")
        hand14 = policy_action[:, :14].astype(np.float32)
        body29 = policy_action[:, 14:43].astype(np.float32)
        root6 = policy_action[:, 43:49].astype(np.float32)
        quat = R.from_euler("xyz", root6[:, 3:6]).as_quat(scalar_first=True).astype(np.float32)
        qpos36 = np.concatenate([root6[:, :3], quat, body29], axis=1).astype(np.float32)
        return hand14, qpos36

    def _policy49_to_textop_actions(self, policy_action: np.ndarray) -> list[tuple[dict[str, np.ndarray], int, np.ndarray]]:
        if self._textop_adapter is None:
            self._textop_adapter = TextOpTrackerAdapter(self.robot.mjModel)
        hand14, qpos36 = self._policy49_to_qpos36(policy_action)
        self._append_qpos36_dump(qpos36)
        ref = self._textop_adapter.prepare_reference(qpos36)
        if self._debug:
            self._dump_ref_debug(qpos36)
        return [(ref, t, np.asarray(hand_q, dtype=np.float32)) for t, hand_q in enumerate(hand14)]

    def _make_textop_action(self, item) -> ActionCmd:
        ref, t, hand_q = item
        assert self._textop_adapter is not None
        body_q = self._textop_adapter.target_from_reference(ref, self.robot.mjData, t)
        return ActionCmd(
            "textop_tracker",
            target_q=np.asarray(body_q, dtype=np.float32),
            left_hand_q=np.asarray(hand_q[:7], dtype=np.float32),
            right_hand_q=np.asarray(hand_q[7:14], dtype=np.float32),
        )

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if not self._direct_queue:
            observations = {"rgb_head_stereo_left": observation["head_stereo_left"]}
            state_dict = {"states": self._build_policy_state49(observation)}
            history = {"reset": True} if self._reset_history else {}
            self._reset_history = False

            policy_action, *_ = self.client.query_action(
                observations,
                instruction or "bend to pick up the object",
                state_dict,
                {},
                history=history,
                dataset="simple",
            )
            print(f"Received {policy_action.shape[0]} direct49 actions for TextOp tracker.")
            self._direct_queue.extend(self._policy49_to_textop_actions(policy_action))

        self._last_pred_action = self._make_textop_action(self._direct_queue.pop(0))
        self._record_debug_trace()
        self._global_step_idx += 1
        return self._last_pred_action

    def _dump_ref_debug(self, qpos36: np.ndarray) -> None:
        os.makedirs(self._debug_dir, exist_ok=True)
        np.savetxt(os.path.join(self._debug_dir, "latest_direct49_qpos36.csv"), qpos36, delimiter=",", fmt="%.8f")
        np.savez(os.path.join(self._debug_dir, "latest_direct49_qpos36.qpos.npz"), qpos=qpos36)

    def _append_qpos36_dump(self, qpos36: np.ndarray) -> None:
        if not self._dump_qpos_dir:
            return
        qpos36 = np.asarray(qpos36, dtype=np.float32)
        self._dumped_qpos_chunks.append(qpos36.copy())
        out = np.concatenate(self._dumped_qpos_chunks, axis=0)
        ep_dir = Path(self._dump_qpos_dir) / f"ep{max(self._episode_idx, 0):04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(ep_dir / "direct49_qpos36.csv", out, delimiter=",", fmt="%.8f")
        np.savez(ep_dir / "direct49_qpos36.qpos.npz", qpos=out)

    def _record_debug_trace(self) -> None:
        if not self._debug:
            return
        root = np.asarray(self.robot.mjData.qpos[:7], dtype=np.float32)
        rpy = R.from_quat(root[3:7], scalar_first=True).as_euler("xyz").astype(np.float32)
        self._debug_trace.append([self._global_step_idx, root[2], rpy[0], rpy[1], rpy[2]])
        os.makedirs(self._debug_dir, exist_ok=True)
        np.save(os.path.join(self._debug_dir, "root_trace.npy"), np.asarray(self._debug_trace, dtype=np.float32))

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._episode_idx += 1
        self._global_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._debug_trace = []
        self._dumped_qpos_chunks = []
        self._direct_queue.clear()
        if self._textop_adapter is not None:
            self._textop_adapter.reset()


Psi0Direct49TextopTrackerAgent = Psi0Direct49TextOpTrackerAgent
