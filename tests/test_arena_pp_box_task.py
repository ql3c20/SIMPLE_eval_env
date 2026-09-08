from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from simple.tasks.g1_fullstate_arena_pp_box import G1FullstateArenaPPBox


def _uninitialized_task() -> G1FullstateArenaPPBox:
    return object.__new__(G1FullstateArenaPPBox)


def _write_recording(root: Path, name: str, *, with_video: bool) -> None:
    videos = root / "videos"
    videos.mkdir(exist_ok=True)
    video_rel = Path("videos") / f"{name}_front_rgb.mp4"
    np.savez(
        root / f"{name}.npz",
        vision_rgb_video_path=np.asarray(str(video_rel)),
        robot_root_position=np.asarray([[1.0, 2.0, 0.8]], dtype=np.float32),
        robot_root_orientation=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        robot_qpos_before_decimation=np.zeros((1, 29), dtype=np.float32),
        env_obj_box_position=np.asarray([[0.0, 0.0, 0.75]], dtype=np.float32),
        env_obj_box_orientation=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        env_obj_shelf_position=np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32),
        env_obj_shelf_orientation=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    if with_video:
        (root / video_rel).touch()


def test_recording_loader_uses_only_complete_front_video(tmp_path: Path) -> None:
    _write_recording(tmp_path, "episode_000", with_video=True)
    _write_recording(tmp_path, "episode_001", with_video=False)
    _write_recording(tmp_path, "episode_002", with_video=True)

    recordings = G1FullstateArenaPPBox._load_recording_initializations(tmp_path)

    assert [item[0] for item in recordings] == ["episode_000.npz", "episode_002.npz"]
    assert G1FullstateArenaPPBox._last_missing_video_recordings == ("episode_001.npz",)
    assert recordings[0][1].shape == (50,)
    np.testing.assert_allclose(recordings[0][1][36:], 0.0)


def test_recording_selection_is_seeded_and_cycles_without_replacement() -> None:
    selected = [
        G1FullstateArenaPPBox._select_recording_index(100, index, seed=0)
        for index in range(100)
    ]

    assert selected[:10] == [82, 36, 20, 5, 93, 16, 94, 52, 72, 90]
    assert sorted(selected) == list(range(100))
    assert G1FullstateArenaPPBox._select_recording_index(100, 100, seed=0) == 48


def test_fixed_recording_selection_validates_complete_pool_index() -> None:
    assert G1FullstateArenaPPBox._select_recording_index(
        100, sequence_index=9, seed=0, fixed_index=7
    ) == 7
    with pytest.raises(ValueError, match="complete-video range 0..99"):
        G1FullstateArenaPPBox._select_recording_index(
            100, sequence_index=0, seed=0, fixed_index=100
        )


def test_usd_box_tracks_the_published_mujoco_body() -> None:
    task = _uninitialized_task()
    box_spec = next(
        spec
        for spec in task.isaac_extra_usd_references()
        if spec["name"] == "arena_pp_box_box_usd"
    )
    shelf_spec = next(
        spec
        for spec in task.isaac_extra_usd_references()
        if spec["name"] == "arena_pp_box_shelf_usd"
    )

    assert box_spec["sync_body"] == task.mujoco_object_body_names["target"]
    assert shelf_spec["sync_body"] == task.mujoco_object_body_names["shelf"]
    assert box_spec["disable_collision"] is True
    assert shelf_spec["disable_collision"] is True
    assert box_spec["scale"] == pytest.approx([task._box_usd_scale] * 3)
    assert task.metadata["isaac_read_only_mirror"] is True


def test_fallback_box_tracks_the_same_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_PP_BOX_USE_ARENA_USD_VISUALS", "0")
    task = _uninitialized_task()
    box_spec = next(
        spec
        for spec in task.isaac_extra_primitives()
        if spec["name"] == "arena_pp_box_box_fallback"
    )

    assert box_spec["sync_body"] == task._box_body_name
    assert box_spec["scale"] == pytest.approx([0.21, 0.21, 0.21])


def test_mujoco_fragments_publish_box_and_shelf_collision_bodies() -> None:
    scene = mujoco.MjSpec()
    for extra in G1FullstateArenaPPBox.mujoco_extra_mjcf:
        child = mujoco.MjSpec.from_file(str(extra["path"]))
        if extra.get("freejoint_body"):
            body = next(
                item
                for item in child.worldbody.find_all("body")
                if item.name == extra["freejoint_body"]
            )
            body.add_freejoint(name=extra["freejoint_name"])
        frame = scene.worldbody.add_frame(pos=extra["pos"], quat=extra["quat"])
        scene.attach(child, prefix=extra["prefix"], frame=frame)
    model = scene.compile()

    box_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, G1FullstateArenaPPBox._box_body_name
    )
    shelf_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, G1FullstateArenaPPBox._shelf_body_name
    )
    box_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "arena_pp_box_box_box_collision"
    )
    support_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "arena_pp_box_shelf_active_support_collision",
    )

    assert box_body_id >= 0
    assert shelf_body_id >= 0
    assert box_geom_id >= 0
    assert support_geom_id >= 0
    np.testing.assert_allclose(model.geom_size[box_geom_id], [0.105, 0.105, 0.105])
    assert model.body_mass[box_body_id] == pytest.approx(0.05)
    assert model.geom_contype[box_geom_id] != 0
    support_top = model.geom_pos[support_geom_id, 2] + model.geom_size[support_geom_id, 2]
    assert support_top == pytest.approx(G1FullstateArenaPPBox._shelf_active_support_top_z)


def test_box_collision_contacts_the_tabletop() -> None:
    scene = mujoco.MjSpec.from_file(
        str(G1FullstateArenaPPBox.mujoco_scene_mjcf_path)
    )
    box_spec = G1FullstateArenaPPBox.mujoco_extra_mjcf[0]
    box = mujoco.MjSpec.from_file(str(box_spec["path"]))
    body = next(
        item
        for item in box.worldbody.find_all("body")
        if item.name == box_spec["freejoint_body"]
    )
    body.add_freejoint(name=box_spec["freejoint_name"])
    # Add 3 mm overlap so contact generation is deterministic at mj_forward().
    position = np.asarray(box_spec["pos"], dtype=np.float64)
    position[2] -= 0.003
    frame = scene.worldbody.add_frame(pos=position, quat=box_spec["quat"])
    scene.attach(box, prefix=box_spec["prefix"], frame=frame)
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    contact_pairs = {
        frozenset((model.geom(contact.geom1).name, model.geom(contact.geom2).name))
        for contact in data.contact
    }
    assert frozenset(("top_collision", "arena_pp_box_box_box_collision")) in (
        contact_pairs
    )


def test_placement_geometry_requires_full_support_and_correct_height() -> None:
    identity = np.eye(3)
    shelf_center = np.zeros(3)
    valid_center = np.asarray(
        [0.0, 0.0, G1FullstateArenaPPBox._shelf_active_support_top_z + 0.105]
    )

    valid = G1FullstateArenaPPBox._placement_geometry(
        valid_center, identity, shelf_center, identity
    )
    outside = G1FullstateArenaPPBox._placement_geometry(
        valid_center + np.asarray([0.40, 0.0, 0.0]), identity, shelf_center, identity
    )
    too_high = G1FullstateArenaPPBox._placement_geometry(
        valid_center + np.asarray([0.0, 0.0, 0.20]), identity, shelf_center, identity
    )

    assert valid["inside_xy"] is True
    assert valid["aligned_z"] is True
    assert outside["inside_xy"] is False
    assert too_high["aligned_z"] is False


def test_placement_geometry_is_evaluated_in_shelf_frame() -> None:
    yaw = np.deg2rad(35.0)
    rotation = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shelf_center = np.asarray([-2.0, -3.0, 0.0])
    local_box_center = np.asarray(
        [0.1, 0.0, G1FullstateArenaPPBox._shelf_active_support_top_z + 0.105]
    )
    world_box_center = shelf_center + rotation @ local_box_center

    result = G1FullstateArenaPPBox._placement_geometry(
        world_box_center, rotation, shelf_center, rotation
    )

    assert result["inside_xy"] is True
    assert result["aligned_z"] is True
    np.testing.assert_allclose(result["relative_center"], local_box_center, atol=1e-12)
