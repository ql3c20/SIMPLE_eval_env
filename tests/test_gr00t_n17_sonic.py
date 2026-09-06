from __future__ import annotations

from collections import deque

import numpy as np

from simple.baselines.gr00t_n17_sonic import Gr00tN17SonicAgent


class _Client:
    def __init__(self):
        self.histories = []

    def query_action(
        self,
        image,
        instruction,
        state,
        condition,
        *,
        history,
        dataset,
    ):
        del image, instruction, state, condition, dataset
        self.histories.append(history)
        action = np.zeros((40, 78), dtype=np.float32)
        action[:, 0] = np.arange(40, dtype=np.float32)
        return action, None


class _Stabilizer:
    _cached_target_q = None
    _cached_left_hand_q = None
    _cached_right_hand_q = None


def _agent_for_queue_test() -> Gr00tN17SonicAgent:
    agent = Gr00tN17SonicAgent.__new__(Gr00tN17SonicAgent)
    agent.client = _Client()
    agent.execution_horizon = 34
    agent.initial_pose_steps = 0
    agent._initial_pose_step = 0
    agent._pending = deque()
    agent._history = deque(maxlen=10)
    agent._last_action_isaac = np.zeros(29, dtype=np.float32)
    agent._stabilize_target = None
    agent._decode_count = 0
    agent._initial_pose_start_q = None
    agent._reset_history = True
    agent.debug_dir = None
    agent._debug_query_count = 0
    agent._last_pred_action = None
    agent._stabilizer = _Stabilizer()
    agent._append_history = lambda: None
    agent._sonic_state46 = lambda observation: np.zeros(46, dtype=np.float32)
    agent._decode = lambda action: np.asarray(action, dtype=np.float32)
    return agent


def test_prefix_rtc_executes_34_of_40_before_replanning():
    agent = _agent_for_queue_test()
    observation = {"head_stereo_left": np.zeros((8, 8, 3), dtype=np.uint8)}

    for expected in range(34):
        action = agent.get_action(observation)
        assert action[0] == expected
    assert len(agent.client.histories) == 1

    action = agent.get_action(observation)
    assert action[0] == 0
    assert len(agent.client.histories) == 2


def test_first_query_of_each_episode_resets_server_rtc_history():
    agent = _agent_for_queue_test()
    observation = {"head_stereo_left": np.zeros((8, 8, 3), dtype=np.uint8)}

    agent.get_action(observation)
    agent._pending.clear()
    agent.get_action(observation)
    assert agent.client.histories == [{"reset": True}, {}]

    agent.reset()
    agent.get_action(observation)
    assert agent.client.histories[-1] == {"reset": True}


def test_sonic_hand_action_is_reordered_to_mujoco_joint_order():
    hand = np.arange(7, dtype=np.float32)
    np.testing.assert_array_equal(
        Gr00tN17SonicAgent._hand_to_mujoco(hand),
        hand[[0, 1, 2, 5, 6, 3, 4]],
    )
