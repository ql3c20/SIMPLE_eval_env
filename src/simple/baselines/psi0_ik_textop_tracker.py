from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.baselines.fullbody_ik_textop_adapter import FullBodyIkTextOpAdapter
from simple.baselines.psi0_kimodo_decoupled_wbc import Psi0KimodoDecoupledWbcAgent
from simple.baselines.textop_tracker_adapter import TextOpTrackerAdapter
from simple.core.action import ActionCmd


class Psi0IkTextOpTrackerAgent(Psi0KimodoDecoupledWbcAgent):
    """Psi0 44D -> full-body IK qpos36 -> TextOp tracker + direct hand14."""

    def __init__(self, robot, host: str, port: int, upsample_factor=1, **kwargs):
        super().__init__(robot, host, port, upsample_factor=upsample_factor, **kwargs)
        self._ik_textop_adapter: FullBodyIkTextOpAdapter | None = None
        self._textop_adapter: TextOpTrackerAdapter | None = None
        self._direct_queue: list[tuple[dict[str, np.ndarray], int, np.ndarray]] = []
        self._debug = os.environ.get("TEXTOP_DEBUG", "0") == "1"
        self._debug_dir = os.environ.get("TEXTOP_DEBUG_DIR", "/pfs/pfs-ilWc5D/yzh/Psi0/outputs/ik_textop_debug")
        self._dump_qpos_dir = os.environ.get("IK_TEXTOP_QPOS_DUMP_DIR", "")
        self._episode_idx = -1
        self._dumped_qpos_chunks: list[np.ndarray] = []
        self._debug_trace = []

    def _policy44_to_textop_actions(self, policy_action: np.ndarray) -> list[tuple[dict[str, np.ndarray], int, np.ndarray]]:
        if self._ik_textop_adapter is None:
            self._ik_textop_adapter = FullBodyIkTextOpAdapter(self.robot.mjModel)
        if self._textop_adapter is None:
            self._textop_adapter = TextOpTrackerAdapter(self.robot.mjModel)

        hand14, qpos36 = self._ik_textop_adapter.policy44_to_qpos36(
            policy_action,
            current_qpos36=self._current_mujoco_qpos36(),
        )
        self._append_qpos36_dump(qpos36)
        if self._debug:
            self._dump_ref_debug(qpos36)
        ref = self._textop_adapter.prepare_reference(qpos36)
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
            print(f"Received {policy_action.shape[0]} 44D actions for IK+TextOp tracker.")
            self._direct_queue.extend(self._policy44_to_textop_actions(policy_action))

        self._last_pred_action = self._make_textop_action(self._direct_queue.pop(0))
        self._record_debug_trace()
        self._global_step_idx += 1
        return self._last_pred_action

    def _append_qpos36_dump(self, qpos36: np.ndarray) -> None:
        if not self._dump_qpos_dir:
            return
        qpos36 = np.asarray(qpos36, dtype=np.float32)
        self._dumped_qpos_chunks.append(qpos36.copy())
        out = np.concatenate(self._dumped_qpos_chunks, axis=0)
        ep_dir = Path(self._dump_qpos_dir) / f"ep{max(self._episode_idx, 0):04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(ep_dir / "ik_textop_qpos36.csv", out, delimiter=",", fmt="%.8f")
        np.savez(ep_dir / "ik_textop_qpos36.qpos.npz", qpos=out)

    def _dump_ref_debug(self, qpos36: np.ndarray) -> None:
        os.makedirs(self._debug_dir, exist_ok=True)
        np.savetxt(os.path.join(self._debug_dir, "latest_ik_textop_qpos36.csv"), qpos36, delimiter=",", fmt="%.8f")
        np.savez(os.path.join(self._debug_dir, "latest_ik_textop_qpos36.qpos.npz"), qpos=qpos36)

    def _record_debug_trace(self) -> None:
        if not self._debug:
            return
        root = np.asarray(self.robot.mjData.qpos[:7], dtype=np.float32)
        rpy = R.from_quat(root[3:7], scalar_first=True).as_euler("xyz").astype(np.float32)
        self._debug_trace.append([self._global_step_idx, root[2], rpy[0], rpy[1], rpy[2]])
        os.makedirs(self._debug_dir, exist_ok=True)
        np.save(os.path.join(self._debug_dir, "root_trace.npy"), np.asarray(self._debug_trace, dtype=np.float32))

    def reset(self, **kwargs):
        SonicDecoupledWbcAgent.reset(self, **kwargs)
        self._global_step_idx = 0
        self._episode_idx += 1
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_target_yaw = 0.0
        self._last_simple_actions = None
        self._direct_queue.clear()
        self._dumped_qpos_chunks = []
        self._debug_trace = []
        if self._ik_textop_adapter is not None:
            self._ik_textop_adapter.reset()
        if self._textop_adapter is not None:
            self._textop_adapter.reset()


Psi0IkTextopTrackerAgent = Psi0IkTextOpTrackerAgent
