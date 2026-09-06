from __future__ import annotations

import numpy as np
import pytest

from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1
from simple.tasks.g1_fullstate_20260804_task3 import G1Fullstate20260804Task3
from simple.tasks.g1_fullstate_20260805_task2 import G1Fullstate20260805Task2
from simple.tasks.g1_fullstate_20260828_task6 import G1Fullstate20260828Task6
from simple.tasks.g1_fullstate_20260729_task4 import G1Fullstate20260729Task4


def test_task1_requires_ten_consecutive_lift_steps():
    task = object.__new__(G1Fullstate20260615Task1)
    task._init_target_height = 0.8
    task._task1_lift_steps = 0
    task._task1_success = False
    task.reward = 0.0
    for _ in range(9):
        task.compute_reward({"target": np.asarray([0.0, 0.0, 0.90])})
    assert not task.check_success({})
    task.compute_reward({"target": np.asarray([0.0, 0.0, 0.90])})
    assert task.check_success({})


def test_task1_hold_resets_below_threshold():
    task = object.__new__(G1Fullstate20260615Task1)
    task._init_target_height = 0.8
    task._task1_lift_steps = 7
    task._task1_success = False
    task.reward = 0.0
    task.compute_reward({"target": np.asarray([0.0, 0.0, 0.88])})
    assert task._task1_lift_steps == 0


def test_task4_local_box_uses_xy_center_and_bottle_bottom():
    upright_inside = G1Fullstate20260729Task4._placement_inside_box(
        np.asarray([0.0, 0.0, 0.02]),
        np.asarray([0.0, 0.0, 0.13]),
        np.asarray([0.0, 0.0, 1.0]),
    )
    outside_xy = G1Fullstate20260729Task4._placement_inside_box(
        np.asarray([0.4, 0.0, 0.02]),
        np.asarray([0.4, 0.0, 0.13]),
        np.asarray([0.0, 0.0, 1.0]),
    )
    hovering_above = G1Fullstate20260729Task4._placement_inside_box(
        np.asarray([0.0, 0.0, 0.20]),
        np.asarray([0.0, 0.0, 0.30]),
        np.asarray([0.0, 0.0, 1.0]),
    )
    assert upright_inside
    assert not outside_xy
    assert not hovering_above


def test_task4_succeeds_immediately_when_inside_even_if_moving():
    task = object.__new__(G1Fullstate20260729Task4)
    task._bottle_start_z = 0.8
    task._was_lifted = False
    task._task4_inside_steps = 0
    task._task4_success = False
    task.reward = 0.0
    task._read_task_state = lambda _env: (np.zeros(3), True, False)

    assert task.compute_reward(
        {"target": np.asarray([0.0, 0.0, 0.8])}, mujoco_env=object()
    ) == 1.0
    assert task.check_success({})
    assert task._task4_inside_steps == 1


def test_task2_local_bin_interior_uses_bottle_center():
    bounds = (-0.15, 0.15, 0.03, 0.48, -0.13, 0.13)
    assert G1Fullstate20260805Task2._inside_interior(
        np.asarray([0.0, 0.24, 0.0]), np.asarray([0.03, 0.12, 0.03]), bounds
    )
    assert G1Fullstate20260805Task2._inside_interior(
        np.asarray([0.14, 0.24, 0.0]), np.asarray([0.03, 0.12, 0.03]), bounds
    )
    for center in (
        np.asarray([0.151, 0.24, 0.0]),
        np.asarray([0.0, 0.029, 0.0]),
        np.asarray([0.0, 0.481, 0.0]),
        np.asarray([0.0, 0.24, 0.131]),
    ):
        assert not G1Fullstate20260805Task2._inside_interior(
            center, np.asarray([0.03, 0.12, 0.03]), bounds
        )


def test_task2_succeeds_immediately_when_center_is_inside_even_if_moving():
    task = object.__new__(G1Fullstate20260805Task2)
    task._task2_inside_steps = 0
    task._task2_success = False
    task.reward = 0.0
    task._read_task_state = lambda _env: (
        True,
        False,
        {"bottle_linear_speed_mps": 1.0, "bottle_angular_speed_radps": 5.0},
    )

    assert task.compute_reward({}, mujoco_env=object()) == 1.0
    assert task.check_success({})
    assert task._task2_inside_steps == 1


def test_task2_trash_translate_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("TASK2_TRASH_TRANSLATE", raising=False)
    np.testing.assert_allclose(
        G1Fullstate20260805Task2._trash_translate_from_env(), np.zeros(3)
    )


def test_task2_trash_translate_reads_xyz(monkeypatch):
    monkeypatch.setenv("TASK2_TRASH_TRANSLATE", "-0.05 0.05 0.0")
    np.testing.assert_allclose(
        G1Fullstate20260805Task2._trash_translate_from_env(),
        np.asarray([-0.05, 0.05, 0.0]),
    )


def test_task2_bottle_translate_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("TASK2_BOTTLE_TRANSLATE", raising=False)
    np.testing.assert_allclose(
        G1Fullstate20260805Task2._bottle_translate_from_env(), np.zeros(3)
    )


def test_task2_bottle_translate_reads_xyz(monkeypatch):
    monkeypatch.setenv("TASK2_BOTTLE_TRANSLATE", "0.02 0.0 0.0")
    np.testing.assert_allclose(
        G1Fullstate20260805Task2._bottle_translate_from_env(),
        np.asarray([0.02, 0.0, 0.0]),
    )


def test_task6_fixed_can_xy_preserves_recorded_height_and_orientation(monkeypatch):
    monkeypatch.setenv("TASK6_CAN_RANDOMIZE_LEFT", "1")
    monkeypatch.setenv("TASK6_CAN_FIXED_X", "1.220")
    monkeypatch.setenv("TASK6_CAN_FIXED_Y", "0.020")
    monkeypatch.setenv("TASK6_CAN_LEFT_JITTER_X", "0")
    monkeypatch.setenv("TASK6_CAN_LEFT_JITTER_Y", "0")
    selected = np.asarray([1.30, -0.08, 1.039, 1.0, 0.0, 0.0, 0.0])

    pose, source = G1Fullstate20260828Task6._sample_left_can_from_recordings(
        selected, [], episode_index=0
    )

    np.testing.assert_allclose(pose[:2], np.asarray([1.220, 0.020]))
    np.testing.assert_allclose(pose[2:], selected[2:])
    assert source == "fixed_xy"


def test_task6_face_can_replaces_only_yaw():
    roll, pitch, yaw = np.radians([3.0, -2.0, 15.0])
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    original = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )

    aligned = G1Fullstate20260828Task6._quaternion_with_yaw(original, 0.0)

    assert np.degrees(G1Fullstate20260828Task6._quaternion_yaw(aligned)) == (
        pytest.approx(0.0, abs=1e-8)
    )
    expected = G1Fullstate20260828Task6._quaternion_with_yaw(original, yaw)
    np.testing.assert_allclose(expected, original, atol=1e-12)


class _Joint:
    def __init__(self, angle):
        self.qpos = np.asarray([angle])


class _Data:
    def __init__(self, angle):
        self.angle = angle

    def joint(self, name):
        return _Joint(self.angle)


class _Mujoco:
    def __init__(self, angle):
        self.mjData = _Data(angle)


def test_task3_requires_ten_consecutive_open_steps():
    task = object.__new__(G1Fullstate20260804Task3)
    task._task3_lid_steps = 0
    task._task3_success = False
    task._task3_lid_angle = 0.0
    task.reward = 0.0
    for _ in range(9):
        task.compute_reward({}, mujoco_env=_Mujoco(0.8))
    assert not task.check_success({})
    task.compute_reward({}, mujoco_env=_Mujoco(0.8))
    assert task.check_success({})


def test_task3_hold_resets_below_relaxed_threshold():
    task = object.__new__(G1Fullstate20260804Task3)
    task._task3_lid_steps = 7
    task._task3_success = False
    task._task3_lid_angle = 0.8
    task.reward = 0.0

    task.compute_reward({}, mujoco_env=_Mujoco(0.79))

    assert task._task3_lid_steps == 0
    assert not task.check_success({})
