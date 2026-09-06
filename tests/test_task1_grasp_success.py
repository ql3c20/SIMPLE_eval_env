from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1


class _Named:
    def __init__(self, *, name: str = "", bodyid: int = -1):
        self.name = name
        self.bodyid = bodyid


class _Model:
    def __init__(self, body_names: list[str]):
        self._bodies = [_Named(name=name) for name in body_names]
        self._geoms = [_Named(bodyid=index) for index in range(len(body_names))]

    def body(self, index: int) -> _Named:
        return self._bodies[index]

    def geom(self, index: int) -> _Named:
        return self._geoms[index]


def _env(hand_bodies: list[str]):
    names = ["green_grasp_cylinder_object", *hand_bodies]
    model = _Model(names)
    contacts = [SimpleNamespace(geom1=0, geom2=index) for index in range(1, len(names))]
    data = SimpleNamespace(ncon=len(contacts), contact=contacts)
    return SimpleNamespace(mjModel=model, mjData=data)


def _task() -> G1Fullstate20260615Task1:
    task = G1Fullstate20260615Task1.__new__(G1Fullstate20260615Task1)
    task._init_target_height = 0.8
    task._init_target_xy = np.array([1.0, 0.0], dtype=np.float64)
    task._task1_lift_steps = 0
    task._task1_grasp_steps = 0
    task._task1_grasp_anchor_position = None
    task._task1_last_grasp_sides = ()
    task._task1_last_horizontal_move = 0.0
    task._task1_last_grasp_lift = 0.0
    task._task1_last_grasp_horizontal_move = 0.0
    task._task1_success = False
    task.reward = 0.0
    return task


def _info(x: float, y: float = 0.0, z: float = 0.8):
    return {"target": np.array([x, y, z, 1.0, 0.0, 0.0, 0.0])}


def test_single_finger_push_does_not_count_as_grasp_or_success():
    task = _task()
    env = _env(["right_hand_index_1_link"])

    task.compute_reward(_info(1.04), mujoco_env=env)

    assert not task._task1_success
    assert task._task1_last_grasp_sides == ()
    assert task._task1_grasp_anchor_position is None


def test_continuous_thumb_finger_grasp_then_three_cm_motion_succeeds():
    task = _task()
    env = _env(["right_hand_thumb_2_link", "right_hand_index_1_link"])

    task.compute_reward(_info(1.0), mujoco_env=env)
    assert not task._task1_success
    task.compute_reward(_info(1.031), mujoco_env=env)

    assert task._task1_success
    assert task._task1_last_grasp_sides == ("right",)
    assert task._task1_last_grasp_horizontal_move > 0.03


def test_continuous_left_grasp_then_three_cm_lift_succeeds():
    task = _task()
    env = _env(["left_hand_thumb_2_link", "left_wrist_yaw_link"])

    task.compute_reward(_info(1.0), mujoco_env=env)
    task.compute_reward(_info(1.0, z=0.831), mujoco_env=env)

    assert task._task1_success
    assert task._task1_last_grasp_sides == ("left",)
    assert task._task1_last_grasp_lift > 0.03


def test_motion_before_grasp_cannot_be_reused_for_success():
    task = _task()
    touch_env = _env(["right_hand_index_1_link"])
    grasp_env = _env(["right_hand_thumb_2_link", "right_hand_middle_1_link"])

    task.compute_reward(_info(1.04), mujoco_env=touch_env)
    task.compute_reward(_info(1.04), mujoco_env=grasp_env)

    assert not task._task1_success
    assert task._task1_last_grasp_horizontal_move == 0.0


def test_losing_grasp_resets_carried_displacement_anchor():
    task = _task()
    grasp_env = _env(["right_hand_thumb_2_link", "right_hand_middle_1_link"])
    no_grasp_env = _env([])

    task.compute_reward(_info(1.0), mujoco_env=grasp_env)
    task.compute_reward(_info(1.02), mujoco_env=grasp_env)
    task.compute_reward(_info(1.02), mujoco_env=no_grasp_env)
    task.compute_reward(_info(1.04), mujoco_env=grasp_env)

    assert not task._task1_success
    assert task._task1_last_grasp_horizontal_move == 0.0
    assert task._task1_grasp_steps == 1
