from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from simple.baselines.psi0_direct49_textop_tracker import Psi0Direct49TextOpTrackerAgent
from simple.core.action import ActionCmd


class Psi0Direct73TextOpTrackerAgent(Psi0Direct49TextOpTrackerAgent):
    """Psi0 direct73 action -> TextOp tracker.

    Policy action layout:
        hand14 + body29 + root6(xyz,rpy) + ee24

    TextOp VAE features are built from root/body qpos36. Future anchor/EE
    observations are taken directly from predicted root6/ee24.
    """

    def __init__(self, robot, host: str, port: int, upsample_factor=1, **kwargs):
        super().__init__(robot, host, port, upsample_factor=upsample_factor, **kwargs)
        self._debug_dir = os.environ.get("TEXTOP_DEBUG_DIR", "/pfs/pfs-ilWc5D/yzh/Psi0/outputs/direct73_textop_debug")
        self._dump_qpos_dir = os.environ.get("DIRECT73_QPOS_DUMP_DIR", "")
        self._dumped_qpos_chunks: list[np.ndarray] = []

    @staticmethod
    def _policy73_to_reference_parts(policy_action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2 or policy_action.shape[1] != 73:
            raise ValueError(f"Expected direct73 policy action shape (T, 73), got {policy_action.shape}")
        hand14 = policy_action[:, :14].astype(np.float32)
        body29 = policy_action[:, 14:43].astype(np.float32)
        root6 = policy_action[:, 43:49].astype(np.float32)
        ee24 = policy_action[:, 49:73].astype(np.float32)
        quat = R.from_euler("xyz", root6[:, 3:6]).as_quat(scalar_first=True).astype(np.float32)
        qpos36 = np.concatenate([root6[:, :3], quat, body29], axis=1).astype(np.float32)
        return hand14, qpos36, root6, ee24

    def _policy73_to_textop_actions(self, policy_action: np.ndarray) -> list[tuple[dict[str, np.ndarray], int, np.ndarray]]:
        if self._textop_adapter is None:
            from simple.baselines.textop_tracker_adapter import TextOpTrackerAdapter

            self._textop_adapter = TextOpTrackerAdapter(self.robot.mjModel)
        hand14, qpos36, root6, ee24 = self._policy73_to_reference_parts(policy_action)
        self._append_qpos36_dump(qpos36)
        ref = self._textop_adapter.prepare_reference_external(qpos36, root6=root6, ee24=ee24)
        if self._debug:
            self._dump_ref_debug(qpos36, root6, ee24)
        return [(ref, t, np.asarray(hand_q, dtype=np.float32)) for t, hand_q in enumerate(hand14)]

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
            print(f"Received {policy_action.shape[0]} direct73 actions for TextOp tracker.")
            self._direct_queue.extend(self._policy73_to_textop_actions(policy_action))

        self._last_pred_action = self._make_textop_action(self._direct_queue.pop(0))
        self._record_debug_trace()
        self._global_step_idx += 1
        return self._last_pred_action

    def _dump_ref_debug(self, qpos36: np.ndarray, root6: np.ndarray, ee24: np.ndarray) -> None:
        os.makedirs(self._debug_dir, exist_ok=True)
        np.savetxt(os.path.join(self._debug_dir, "latest_direct73_qpos36.csv"), qpos36, delimiter=",", fmt="%.8f")
        np.savez(
            os.path.join(self._debug_dir, "latest_direct73_reference.npz"),
            qpos=qpos36,
            root6=np.asarray(root6, dtype=np.float32),
            ee24=np.asarray(ee24, dtype=np.float32),
        )

    def _append_qpos36_dump(self, qpos36: np.ndarray) -> None:
        if not self._dump_qpos_dir:
            return
        qpos36 = np.asarray(qpos36, dtype=np.float32)
        self._dumped_qpos_chunks.append(qpos36.copy())
        out = np.concatenate(self._dumped_qpos_chunks, axis=0)
        ep_dir = Path(self._dump_qpos_dir) / f"ep{max(self._episode_idx, 0):04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(ep_dir / "direct73_qpos36.csv", out, delimiter=",", fmt="%.8f")
        np.savez(ep_dir / "direct73_qpos36.qpos.npz", qpos=out)


Psi0Direct73TextopTrackerAgent = Psi0Direct73TextOpTrackerAgent
