from __future__ import annotations

import numpy as np

from simple.baselines.psi0_kimodo_textop_tracker import (
    Psi0KimodoTextOpTrackerAgent,
)


class _Adapter:
    def target_from_reference(self, ref, mjdata, t):
        del ref, mjdata, t
        return np.arange(29, dtype=np.float32)


class _ReferenceAdapter:
    def __init__(self):
        self.qpos36 = None

    def prepare_reference(self, qpos36):
        self.qpos36 = np.asarray(qpos36)
        return {"full_reference": True}


class _KimodoAdapter:
    def __init__(self):
        self.policy_action = None

    def policy44_to_simple36(self, policy_action, **kwargs):
        del kwargs
        self.policy_action = np.asarray(policy_action)
        frame_count = self.policy_action.shape[0]
        qpos50 = np.arange(frame_count * 36, dtype=np.float32).reshape(frame_count, 36)
        return np.zeros((frame_count, 36), dtype=np.float32), qpos50


class _PolicyClient:
    def __init__(self):
        self.calls = 0

    def query_action(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return np.zeros((40, 44), dtype=np.float32), None


def _agent_for_action_test():
    agent = Psi0KimodoTextOpTrackerAgent.__new__(Psi0KimodoTextOpTrackerAgent)
    agent._textop_adapter = _Adapter()
    agent._rate_limit_body_target = lambda value: value
    agent._debug = False
    agent.robot = type("Robot", (), {"mjData": object()})()
    return agent


def test_textop_action_carries_current_kimodo_reference_for_ghost():
    agent = _agent_for_action_test()
    qpos36 = np.arange(36, dtype=np.float32)
    hand14 = np.arange(14, dtype=np.float32)

    action = agent._make_textop_action(({}, 3, hand14, qpos36))

    np.testing.assert_array_equal(action["debug_reference_qpos36"], qpos36)
    np.testing.assert_array_equal(action["left_hand_q"], hand14[:7])
    np.testing.assert_array_equal(action["right_hand_q"], hand14[7:])


def test_legacy_textop_queue_item_remains_supported():
    agent = _agent_for_action_test()
    hand14 = np.zeros(14, dtype=np.float32)

    action = agent._make_textop_action(({}, 0, hand14))

    assert action["debug_reference_qpos36"] is None


def test_policy_execution_horizon_preserves_full_40_reference_but_queues_30():
    agent = Psi0KimodoTextOpTrackerAgent.__new__(Psi0KimodoTextOpTrackerAgent)
    agent._textop_adapter = _ReferenceAdapter()
    agent._kimodo_adapter = _KimodoAdapter()
    agent._normalize_policy_action = lambda value: np.asarray(value, dtype=np.float32)
    agent._current_mujoco_qpos36 = lambda: np.zeros(36, dtype=np.float32)
    agent._current_constraints28_mujoco = lambda: np.zeros(28, dtype=np.float32)
    agent._init_to_ref = False
    agent._init_hold_action = False
    agent._did_init_to_ref = False
    agent._use_policy_root_ee = False
    agent._last_kimodo_qpos_mujoco = None

    policy_action = np.zeros((40, 44), dtype=np.float32)
    actions = agent._policy44_to_textop_actions(
        policy_action,
        execution_horizon=30,
    )

    assert agent._kimodo_adapter.policy_action.shape == (40, 44)
    assert agent._textop_adapter.qpos36.shape == (40, 36)
    assert len(actions) == 30
    assert actions[-1][1] == 29
    np.testing.assert_array_equal(
        agent._last_kimodo_qpos_mujoco,
        agent._textop_adapter.qpos36[29],
    )


def test_policy_execution_horizon_replans_at_configured_prefix():
    agent = Psi0KimodoTextOpTrackerAgent.__new__(Psi0KimodoTextOpTrackerAgent)
    agent._policy_execution_horizon = 34
    assert agent._execution_horizon_for_chunk(40) == 34

    agent._policy_execution_horizon = None
    assert agent._execution_horizon_for_chunk(40) == 40


def test_policy_chunk_contract_is_40_execute_30(monkeypatch):
    monkeypatch.setenv("POLICY_ACTION_HORIZON", "40")
    monkeypatch.setenv("POLICY_EXECUTION_HORIZON", "30")
    agent = Psi0KimodoTextOpTrackerAgent.__new__(Psi0KimodoTextOpTrackerAgent)
    agent._policy_action_horizon = 40
    agent._policy_execution_horizon = 30

    assert agent._execution_horizon_for_chunk(40) == 30
    assert 40 - agent._execution_horizon_for_chunk(40) == 10


def test_get_action_replans_after_34_of_40_policy_frames():
    agent = Psi0KimodoTextOpTrackerAgent.__new__(Psi0KimodoTextOpTrackerAgent)
    agent.client = _PolicyClient()
    agent._policy_execution_horizon = 34
    agent._direct_queue = []
    agent._reset_history = True
    agent._global_step_idx = 0
    agent._build_policy_state49 = lambda observation: np.zeros((1, 49), dtype=np.float32)
    agent._policy44_to_textop_actions = (
        lambda policy_action, instruction=None, execution_horizon=None: list(
            range(execution_horizon)
        )
    )
    agent._make_textop_action = lambda item: item
    agent._record_debug_trace = lambda: None
    observation = {
        "joint_qpos": np.zeros(50, dtype=np.float32),
        "head_stereo_left": np.zeros((8, 8, 3), dtype=np.uint8),
    }

    for expected in range(34):
        assert agent.get_action(observation) == expected
    assert agent.client.calls == 1

    assert agent.get_action(observation) == 0
    assert agent.client.calls == 2
