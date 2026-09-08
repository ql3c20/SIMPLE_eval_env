from __future__ import annotations

import numpy as np
import pytest

from simple.evals.task234_reproduction import (
    EGO_CAMERA,
    TASK_SPECS,
    ego_camera_quaternion_wxyz,
    get_task_spec,
    validate_policy_rgb,
    validate_state_packet,
)

def test_frozen_task_dimensions_and_assets() -> None:
    assert TASK_SPECS[2].expected_dimensions() == (59, 57, 43)
    assert TASK_SPECS[3].expected_dimensions() == (52, 51, 43)
    assert TASK_SPECS[4].expected_dimensions() == (57, 55, 43)
    assert TASK_SPECS[3].task_translate == (0.6, -0.5, 0.0)
    assert TASK_SPECS[3].hssd.usd_relative_path.endswith("102344250.usd")


def test_ego_quaternion_is_normalized() -> None:
    quat = ego_camera_quaternion_wxyz()
    assert quat.shape == (4,)
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    assert EGO_CAMERA.width == 640
    assert EGO_CAMERA.height == 480


def test_state_packet_is_strict_and_normalizes_root_quaternion() -> None:
    spec = get_task_spec(3)
    qpos = np.zeros(spec.expected_nq)
    qpos[3:7] = [2.0, 0.0, 0.0, 0.0]
    packet = {
        "nq": spec.expected_nq,
        "nv": spec.expected_nv,
        "qpos": qpos,
        "qvel": np.zeros(spec.expected_nv),
        "sim_time": 1.25,
        "running": 1,
    }
    normalized, _, sim_time, running = validate_state_packet(packet, spec)
    np.testing.assert_allclose(normalized[3:7], [1.0, 0.0, 0.0, 0.0])
    assert sim_time == 1.25
    assert running

    packet["nq"] -= 1
    with pytest.raises(ValueError, match="expected nq/nv"):
        validate_state_packet(packet, spec)


def test_policy_frame_contract() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    validate_policy_rgb(
        image,
        image_sim_time=1.0,
        mujoco_sim_time=1.2,
        frame_id=9,
        previous_frame_id=8,
    )
    with pytest.raises(RuntimeError, match="did not advance"):
        validate_policy_rgb(
            image,
            image_sim_time=1.0,
            mujoco_sim_time=1.2,
            frame_id=8,
            previous_frame_id=8,
        )
    with pytest.raises(ValueError, match="640x480"):
        validate_policy_rgb(
            image[:360], image_sim_time=1.0, mujoco_sim_time=1.0, frame_id=1
        )
