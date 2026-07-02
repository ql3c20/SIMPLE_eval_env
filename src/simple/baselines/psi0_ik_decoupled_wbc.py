"""Psi0 44D policy debug agent that bypasses Kimodo with wrist MuJoCo IK.

This agent keeps the same Psi0 checkpoint/interface as
``psi0_kimodo_decoupled_wbc``:

    hand14 + root6 + four ee poses * 6 = 44D

Instead of sending EE constraints to Kimodo, it solves left/right wrist IK for
the 14 arm joints and directly builds the original SIMPLE 36D WBC action:

    hand14 + arm14 + torso_rpy + base_height + vx + vy + vyaw + target_yaw
"""

from __future__ import annotations

import numpy as np

from simple.baselines.ik_policy_adapter import MujocoWristIkPolicyAdapter
from simple.baselines.psi0_kimodo_decoupled_wbc import Psi0KimodoDecoupledWbcAgent


class Psi0IkDecoupledWbcAgent(Psi0KimodoDecoupledWbcAgent):
    """Debug path for checking whether Kimodo is the source of grasp error."""

    def __init__(self, robot, host: str, port: int, upsample_factor=1, **kwargs):
        super().__init__(robot, host, port, upsample_factor=upsample_factor, **kwargs)
        self._ik_adapter = None

    def _policy44_to_simple36(self, policy_action: np.ndarray, instruction: str | None = None) -> np.ndarray:
        del instruction
        policy_action = self._normalize_policy_action(policy_action)
        if self._ik_adapter is None:
            if not hasattr(self.robot, "mjModel"):
                raise AttributeError("robot.mjModel is not initialized yet; cannot construct MuJoCo IK adapter")
            self._ik_adapter = MujocoWristIkPolicyAdapter(self.robot.mjModel)
        simple36 = self._ik_adapter.policy44_to_simple36(
            policy_action,
            current_qpos36=self._current_mujoco_qpos36(),
        )
        if self._last_simple_actions is not None and self._chunk_blend_frames > 0:
            simple36 = self._blend_chunk_start(self._last_simple_actions, simple36, self._chunk_blend_frames)
        self._last_simple_actions = simple36.copy()
        return simple36

    def reset(self, **kwargs):
        super().reset(**kwargs)
        if self._ik_adapter is not None:
            self._ik_adapter.reset()
        self._last_simple_actions = None
