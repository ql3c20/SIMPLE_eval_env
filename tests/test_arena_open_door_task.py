from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from simple.sensors import CameraCfg, StereoCameraCfg
from simple.tasks.g1_fullstate_arena_open_door import G1FullstateArenaOpenDoor


def _uninitialized_task() -> G1FullstateArenaOpenDoor:
    task = object.__new__(G1FullstateArenaOpenDoor)
    task._arena_open_door_latch_unlocked = False
    task._arena_open_door_model_id = None
    return task


def _compiled_door():
    extra = G1FullstateArenaOpenDoor.mujoco_extra_mjcf[0]
    scene = mujoco.MjSpec()
    child = mujoco.MjSpec.from_file(str(extra["path"]))
    frame = scene.worldbody.add_frame(pos=extra["pos"], quat=extra["quat"])
    scene.attach(child, prefix=extra["prefix"], frame=frame)
    model = scene.compile()
    return model, mujoco.MjData(model)


def test_open_door_uses_arena_front_camera() -> None:
    assert tuple(G1FullstateArenaOpenDoor.sensor_cfgs) == ("front_camera",)
    camera = G1FullstateArenaOpenDoor.sensor_cfgs["front_camera"]
    assert isinstance(camera, CameraCfg)
    assert not isinstance(camera, StereoCameraCfg)
    assert camera.mount == "eye_in_head"
    assert camera.resolution == (640, 480)
    assert camera.focal_length == pytest.approx(7.6)
    assert np.rad2deg(camera.fov) == pytest.approx(105.53033203685067)
    assert camera.quaternion == pytest.approx([0.5, 0.5, -0.5, -0.5])
    assert G1FullstateArenaOpenDoor.isaac_camera_axes == "usd"
    assert G1FullstateArenaOpenDoor.metadata["video_camera_keys"] == (
        "front_camera",
    )


def test_door_mjcf_contains_collidable_leaf_handle_and_frame() -> None:
    model, data = _compiled_door()
    for body_name in (
        G1FullstateArenaOpenDoor._door_root_body,
        G1FullstateArenaOpenDoor._leaf_body,
        G1FullstateArenaOpenDoor._handle_body,
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0
    for geom_name in (
        "arena_open_door_leaf_collision",
        "arena_open_door_handle_hub",
        "arena_open_door_handle_lever",
        "arena_open_door_hinge_post",
        "arena_open_door_latch_post",
    ):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert geom_id >= 0
        assert model.geom_contype[geom_id] != 0
        assert model.geom_conaffinity[geom_id] != 0

    leaf_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, G1FullstateArenaOpenDoor._leaf_joint
    )
    handle_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, G1FullstateArenaOpenDoor._handle_joint
    )
    np.testing.assert_allclose(model.jnt_axis[leaf_id], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(model.jnt_range[leaf_id], [0.0, np.pi / 2])
    np.testing.assert_allclose(
        model.jnt_range[handle_id],
        np.deg2rad([-36.741695404052734, 16.29798698425293]),
        atol=1e-8,
    )

    # The visible USD lever extends inboard from the handle pivot.  The handle
    # body's authored 180-degree Y rotation makes local +X point in that
    # direction, so the MuJoCo capsule and COM must both lie left of the hub.
    mujoco.mj_forward(model, data)
    handle_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, G1FullstateArenaOpenDoor._handle_body
    )
    handle_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "arena_open_door_handle_lever"
    )
    leaf_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "arena_open_door_leaf_collision"
    )
    hub_x = float(data.xpos[handle_body_id, 0])
    lever_center_x = float(data.geom_xpos[handle_geom_id, 0])
    lever_half_length = float(model.geom_size[handle_geom_id, 1])
    lever_min_x = lever_center_x - lever_half_length
    lever_max_x = lever_center_x + lever_half_length
    leaf_max_x = float(
        data.geom_xpos[leaf_geom_id, 0] + model.geom_size[leaf_geom_id, 0]
    )

    assert lever_min_x < hub_x
    assert lever_max_x == pytest.approx(hub_x)
    assert lever_max_x <= leaf_max_x
    assert model.body_ipos[handle_body_id, 0] > 0.0

    # HumanoidArena disables self-collision for the complete door
    # articulation.  The authored closed pose has tiny frame/leaf overlaps,
    # so allowing those internal contacts jams an otherwise free hinge.
    assert data.ncon == 0


def test_latch_releases_only_after_handle_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPEN_DOOR_HANDLE_UNLOCK_ANGLE_DEG", "-20")
    monkeypatch.delenv("OPEN_DOOR_LEAF_UNLOCK_STIFFNESS", raising=False)
    monkeypatch.delenv("OPEN_DOOR_LEAF_UNLOCK_DAMPING", raising=False)
    monkeypatch.delenv("OPEN_DOOR_LEAF_UNLOCK_MAX_FORCE", raising=False)
    model, data = _compiled_door()
    env = type("Env", (), {"mjModel": model, "mjData": data})()
    task = _uninitialized_task()
    task._ensure_door_indices(env)

    data.qpos[task._leaf_qpos_adr] = 0.3
    data.qpos[task._handle_qpos_adr] = np.deg2rad(-19.9)
    task.before_mujoco_step(env)
    assert task._arena_open_door_latch_unlocked is False
    assert data.qpos[task._leaf_qpos_adr] == pytest.approx(0.0)

    data.qpos[task._handle_qpos_adr] = np.deg2rad(-20.1)
    task.before_mujoco_step(env)
    assert task._arena_open_door_latch_unlocked is True

    data.qpos[task._handle_qpos_adr] = 0.0
    task.after_mujoco_step(env)
    assert task._arena_open_door_latch_unlocked is True

    # Unlocking is sticky, and the leaf is no longer hard-clamped.  Match the
    # native Arena behavior by restoring a free, undriven hinge.
    data.qpos[task._leaf_qpos_adr] = 0.1
    data.qvel[task._leaf_dof_adr] = 0.0
    task.before_mujoco_step(env)
    assert data.qpos[task._leaf_qpos_adr] == pytest.approx(0.1)
    assert data.qfrc_applied[task._leaf_dof_adr] == pytest.approx(0.0)


def test_inboard_contact_pushes_handle_down_and_unlocks_latch() -> None:
    extra = G1FullstateArenaOpenDoor.mujoco_extra_mjcf[0]
    scene = mujoco.MjSpec()
    scene.option.timestep = 0.001
    child = mujoco.MjSpec.from_file(str(extra["path"]))
    frame = scene.worldbody.add_frame(
        pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    scene.attach(child, prefix=extra["prefix"], frame=frame)

    # Press downward on the inboard side occupied by the rendered lever.  The
    # old, reversed collision capsule did not exist at this location and could
    # not transmit a moment to the handle hinge.
    probe = scene.worldbody.add_body(
        name="handle_press_probe", pos=[0.25, -0.102, 1.02]
    )
    probe.add_freejoint(name="handle_press_probe_free")
    probe.add_geom(
        name="handle_press_probe_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.03, 0.0, 0.0],
        mass=2.0,
        friction=[0.8, 0.02, 0.002],
    )

    model = scene.compile()
    data = mujoco.MjData(model)
    env = type("Env", (), {"mjModel": model, "mjData": data})()
    task = _uninitialized_task()
    task._ensure_door_indices(env)
    probe_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "handle_press_probe_free"
    )
    probe_dof_adr = int(model.jnt_dofadr[probe_joint_id])
    data.qvel[probe_dof_adr + 2] = -1.0

    for _ in range(1000):
        task.before_mujoco_step(env)
        mujoco.mj_step(model, data)
        task.after_mujoco_step(env)
        if task._arena_open_door_latch_unlocked:
            break

    assert np.rad2deg(data.qpos[task._handle_qpos_adr]) <= -20.0
    assert task._arena_open_door_latch_unlocked is True


def test_unlocked_leaf_is_a_free_hinge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPEN_DOOR_LEAF_UNLOCK_STIFFNESS", raising=False)
    monkeypatch.delenv("OPEN_DOOR_LEAF_UNLOCK_DAMPING", raising=False)
    monkeypatch.delenv("OPEN_DOOR_LEAF_UNLOCK_MAX_FORCE", raising=False)
    model, data = _compiled_door()
    env = type("Env", (), {"mjModel": model, "mjData": data})()
    task = _uninitialized_task()
    task._ensure_door_indices(env)

    data.qpos[task._handle_qpos_adr] = np.deg2rad(-20.1)
    task.before_mujoco_step(env)
    assert task._arena_open_door_latch_unlocked is True

    for _ in range(1000):
        task.before_mujoco_step(env)
        data.qfrc_applied[task._leaf_dof_adr] += 1.0
        mujoco.mj_step(model, data)
        task.after_mujoco_step(env)

    assert data.qpos[task._leaf_qpos_adr] > np.deg2rad(1.0)


def test_leaf_drive_torque_is_clipped() -> None:
    assert G1FullstateArenaOpenDoor._clipped_drive_torque(
        position=0.1, velocity=0.0, stiffness=5.0, damping=20.0, max_force=5.0
    ) == pytest.approx(-0.5)
    assert G1FullstateArenaOpenDoor._clipped_drive_torque(
        position=1.0, velocity=1.0, stiffness=5.0, damping=20.0, max_force=5.0
    ) == pytest.approx(-5.0)


def test_usd_visuals_are_read_only_and_track_all_door_bodies() -> None:
    task = _uninitialized_task()
    door = next(
        item
        for item in task.isaac_extra_usd_references()
        if item["name"] == "arena_open_door_door_usd"
    )
    assert door["disable_collision"] is True
    assert door["sync_body"] == task._door_root_body
    assert door["sync_subprims"] == {
        "E_leaf_2": task._leaf_body,
        "E_handle_4": task._handle_body,
    }
    assert task.metadata["isaac_read_only_mirror"] is True


def test_recording_loader_reads_episode_initial_door_pose(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "episode.npz",
        robot_root_position=np.asarray([[-1.6, 0.2, 0.8]], dtype=np.float32),
        robot_root_orientation=np.asarray(
            [[0.70711, 0.0, 0.0, 0.70711]], dtype=np.float32
        ),
        robot_qpos_before_decimation=np.zeros((1, 29), dtype=np.float32),
        episode_init_env_obj_door_position=np.asarray(
            [-1.2, 1.8, 0.002], dtype=np.float32
        ),
        episode_init_env_obj_door_orientation=np.asarray(
            [0.99749684, 0.0, 0.0, 0.0], dtype=np.float32
        ),
    )
    recordings = G1FullstateArenaOpenDoor._load_recording_initializations(tmp_path)
    assert len(recordings) == 1
    assert recordings[0][1].shape == (50,)
    np.testing.assert_allclose(recordings[0][2], [-1.2, 1.8, 0.002], atol=1e-6)
    np.testing.assert_allclose(recordings[0][3], [1.0, 0.0, 0.0, 0.0])


def test_recording_selection_is_seeded_without_replacement() -> None:
    selected = [
        G1FullstateArenaOpenDoor._select_recording_index(100, i, 0, None)
        for i in range(100)
    ]
    assert selected[:5] == [82, 36, 20, 5, 93]
    assert sorted(selected) == list(range(100))


def test_standing_check_uses_root_height_and_up_axis() -> None:
    assert G1FullstateArenaOpenDoor._root_up_axis_z([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert G1FullstateArenaOpenDoor._root_up_axis_z(
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    ) == pytest.approx(0.0)
