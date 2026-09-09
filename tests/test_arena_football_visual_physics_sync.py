from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from simple.engines.mujoco import MujocoSimulator
from simple.sensors import CameraCfg, StereoCameraCfg
from simple.tasks.arena_eval_camera import configure_arena_eval_cameras
from simple.tasks.g1_fullstate_arena_football import G1FullstateArenaFootball


def _uninitialized_task() -> G1FullstateArenaFootball:
    return object.__new__(G1FullstateArenaFootball)


def test_offscreen_framebuffer_fits_policy_and_wide_video_cameras() -> None:
    model = SimpleNamespace(
        vis=SimpleNamespace(
            global_=SimpleNamespace(offwidth=640, offheight=480)
        )
    )
    cameras = {
        "front_camera": SimpleNamespace(resolution=(640, 480)),
        "video_camera": SimpleNamespace(resolution=(1280, 720)),
    }

    MujocoSimulator._fit_offscreen_framebuffer(model, cameras)

    assert model.vis.global_.offwidth == 1280
    assert model.vis.global_.offheight == 720


def _write_robot_recording(
    path: Path,
    body_qpos29: np.ndarray | None = None,
    **object_fields: np.ndarray,
) -> None:
    np.savez(
        path,
        robot_root_position=np.asarray([[0.0, 0.0, 0.8]], dtype=np.float32),
        robot_root_orientation=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        robot_qpos_before_decimation=np.asarray(
            np.zeros((1, 29), dtype=np.float32)
            if body_qpos29 is None
            else np.asarray(body_qpos29, dtype=np.float32).reshape(1, 29)
        ),
        **object_fields,
    )


def test_recording_loader_reads_sonic_episode_initial_ball_pose(tmp_path: Path) -> None:
    _write_robot_recording(
        tmp_path / "sonic.npz",
        episode_init_env_obj_football_position=np.asarray(
            [-0.3, -1.4, 0.11], dtype=np.float32
        ),
        episode_init_env_obj_football_orientation=np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
    )

    recordings = G1FullstateArenaFootball._load_recording_initializations(tmp_path)

    assert len(recordings) == 1
    np.testing.assert_allclose(recordings[0][2], [-0.3, -1.4, 0.11])
    np.testing.assert_allclose(recordings[0][3], [1.0, 0.0, 0.0, 0.0])


def test_recording_loader_keeps_legacy_per_frame_ball_pose(tmp_path: Path) -> None:
    _write_robot_recording(
        tmp_path / "legacy.npz",
        env_obj_football_position=np.asarray([[0.2, -0.7, 0.11]], dtype=np.float32),
        env_obj_football_orientation=np.asarray(
            [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        ),
    )

    recordings = G1FullstateArenaFootball._load_recording_initializations(tmp_path)

    assert len(recordings) == 1
    np.testing.assert_allclose(recordings[0][2], [0.2, -0.7, 0.11])


def test_recording_loader_reorders_sonic_body29_for_mujoco(tmp_path: Path) -> None:
    grouped = G1FullstateArenaFootball._default_body_qpos29
    interleaved = np.empty(29, dtype=np.float32)
    interleaved[G1FullstateArenaFootball._sonic_to_mujoco_body29] = grouped
    _write_robot_recording(
        tmp_path / "sonic.npz",
        body_qpos29=interleaved,
        episode_init_env_obj_football_position=np.asarray(
            [0.0, 0.0, 0.11], dtype=np.float32
        ),
        episode_init_env_obj_football_orientation=np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
    )

    robot_qpos = G1FullstateArenaFootball._load_recording_initializations(tmp_path)[0][1]

    np.testing.assert_allclose(robot_qpos[7:36], grouped)


def test_football_uses_one_arena_aligned_policy_camera() -> None:
    assert tuple(G1FullstateArenaFootball.sensor_cfgs) == ("front_camera",)
    camera = G1FullstateArenaFootball.sensor_cfgs["front_camera"]

    assert isinstance(camera, CameraCfg)
    assert not isinstance(camera, StereoCameraCfg)
    assert camera.mount == "eye_in_head"
    assert camera.resolution == (640, 480)
    assert camera.focal_length == pytest.approx(7.6)
    assert np.rad2deg(camera.fov) == pytest.approx(105.53033203685067)
    assert camera.near == pytest.approx(0.1)
    assert camera.far == pytest.approx(1.0e5)
    assert camera.position == pytest.approx([0.0, 0.0, 0.0])
    assert camera.quaternion == pytest.approx([0.5, 0.5, -0.5, -0.5])
    assert G1FullstateArenaFootball.isaac_camera_axes == "usd"
    assert G1FullstateArenaFootball.metadata["video_camera_keys"] == (
        "front_camera",
    )


def test_football_wide_video_camera_preserves_policy_camera_and_vertical_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARENA_AUX_VIDEO_ENABLED", "1")
    task = _uninitialized_task()
    configure_arena_eval_cameras(task)

    policy_camera = task.sensor_cfgs["front_camera"]
    video_camera = task.sensor_cfgs["video_camera"]
    assert policy_camera.resolution == (640, 480)
    assert np.rad2deg(policy_camera.fov) == pytest.approx(105.53033203685067)
    assert video_camera.resolution == (1280, 720)
    assert np.rad2deg(video_camera.fov) == pytest.approx(120.63371964175342)
    assert video_camera.pose == policy_camera.pose
    assert task.metadata["video_camera_keys"] == ("video_camera",)

    policy_vertical_aperture = 20.0 * 480.0 / 640.0
    video_horizontal_aperture = (
        2.0 * video_camera.focal_length * np.tan(video_camera.fov / 2.0)
    )
    video_vertical_aperture = video_horizontal_aperture * 720.0 / 1280.0
    assert video_vertical_aperture == pytest.approx(policy_vertical_aperture)


def test_grass_texture_scale_matches_arena() -> None:
    task = _uninitialized_task()
    ground = next(
        prim
        for prim in task.isaac_extra_primitives()
        if prim["name"] == "arena_football_ground"
    )

    assert ground["uv_scale"] == pytest.approx([15.0, 15.0])


def test_usd_ball_tracks_published_mujoco_body_and_radius() -> None:
    task = _uninitialized_task()
    ball_spec = next(
        spec
        for spec in task.isaac_extra_usd_references()
        if spec["name"] == "arena_football_ball_usd"
    )

    assert ball_spec["sync_body"] == task.mujoco_object_body_names["target"]
    assert ball_spec["disable_collision"] is True
    assert task.metadata["isaac_read_only_mirror"] is True
    assert ball_spec["scale"] == pytest.approx([task._ball_usd_scale] * 3)
    assert task._ball_usd_authored_radius * ball_spec["scale"][0] == pytest.approx(
        task._ball_collision_radius
    )


def test_fallback_ball_tracks_the_same_mujoco_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_FOOTBALL_USE_ARENA_USD_VISUALS", "0")
    task = _uninitialized_task()
    ball_spec = next(
        spec
        for spec in task.isaac_extra_primitives()
        if spec["name"] == "arena_football_ball_fallback"
    )

    assert ball_spec["sync_body"] == task.mujoco_object_body_names["target"]
    assert ball_spec["radius"] == pytest.approx(task._ball_collision_radius)


def test_mujoco_ball_xml_uses_declared_collision_radius() -> None:
    xml_path = (
        Path(__file__).resolve().parents[1]
        / "data/assets/humanoidarena/local_mjcf/arena_football/soccer_ball.xml"
    )
    root = ET.parse(xml_path).getroot()
    collision = root.find(".//geom[@name='ball_collision']")

    assert collision is not None
    assert float(collision.attrib["size"]) == pytest.approx(
        G1FullstateArenaFootball._ball_collision_radius
    )


def test_mujoco_fragment_publishes_the_visual_sync_body() -> None:
    extra = G1FullstateArenaFootball.mujoco_extra_mjcf[0]
    scene = mujoco.MjSpec()
    football = mujoco.MjSpec.from_file(str(extra["path"]))
    body = next(
        item
        for item in football.worldbody.find_all("body")
        if item.name == extra["freejoint_body"]
    )
    body.add_freejoint(name=extra["freejoint_name"])
    frame = scene.worldbody.add_frame(pos=extra["pos"], quat=extra["quat"])
    scene.attach(football, prefix=extra["prefix"], frame=frame)
    model = scene.compile()

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        G1FullstateArenaFootball._ball_body_name,
    )
    collision_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "arena_football_ball_ball_collision",
    )

    assert body_id >= 0
    assert collision_id >= 0
    assert model.geom_contype[collision_id] != 0
    assert model.geom_conaffinity[collision_id] != 0
    assert model.geom_size[collision_id, 0] == pytest.approx(
        G1FullstateArenaFootball._ball_collision_radius
    )


def test_visual_sync_name_is_present_in_mujoco_render_state() -> None:
    simulator = object.__new__(MujocoSimulator)
    simulator.render_step = 1
    simulator.mj_objects = {
        "target": SimpleNamespace(
            xpos=np.asarray([0.2, 0.3, 0.11]),
            xquat=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
    }
    simulator.obj_names = [G1FullstateArenaFootball._ball_body_name]
    simulator.task = SimpleNamespace(
        robot=SimpleNamespace(get_robot_qpos=lambda: {})
    )
    simulator.mjData = SimpleNamespace(qpos=np.zeros(7))
    simulator.articulated_object_joints = None

    _, _, published_names, positions, _, _ = simulator.get_states()
    task = _uninitialized_task()
    ball_spec = next(
        spec
        for spec in task.isaac_extra_usd_references()
        if spec["name"] == "arena_football_ball_usd"
    )

    assert ball_spec["sync_body"] in published_names
    ball_position = positions[published_names.index(ball_spec["sync_body"])]
    np.testing.assert_allclose(ball_position, [0.2, 0.3, 0.11])
