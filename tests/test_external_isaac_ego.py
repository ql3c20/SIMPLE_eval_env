from __future__ import annotations

import io
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest
import gymnasium as gym
from PIL import Image

import simple.envs  # noqa: F401 - register the Task1 Gym environment
from simple.cli.eval_decoupled_wbc import _make_sonic_config
from simple.engines.external_isaac_ego import (
    FRAME_MAGIC,
    ExternalIsaacEgoClient,
    _xml_joint_layout,
)


class _Task:
    recording_env_prefix = "TASKX"
    uid = "task_x"
    selected_recording_index = 17
    selected_recording_name = "recording_17"
    selected_recording_seed = 42


def _write_frame(
    path: Path,
    width: int,
    height: int,
    sequence: int = 7,
    **extra_metadata,
):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 1] = 127
    jpeg = io.BytesIO()
    Image.fromarray(rgb).save(jpeg, format="JPEG", quality=95)
    metadata = json.dumps(
        {
            "sequence": sequence,
            "sim_time": 1.25,
            "published_monotonic_s": time.monotonic(),
            "width": width,
            "height": height,
            **extra_metadata,
        }
    ).encode()
    path.write_bytes(FRAME_MAGIC + struct.pack("<I", len(metadata)) + metadata + jpeg.getvalue())


def test_hvego_frame_protocol_and_size(tmp_path, monkeypatch):
    path = tmp_path / "ego.frame"
    monkeypatch.setenv("EXTERNAL_ISAAC_EGO_FRAME_PATH", str(path))
    client = ExternalIsaacEgoClient(_Task())
    _write_frame(path, 640, 480)
    metadata, rgb = client._read_frame()
    assert metadata["sequence"] == 7
    assert rgb.shape == (480, 640, 3)
    client.close()


def test_hvego_wrong_size_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "ego.frame"
    monkeypatch.setenv("EXTERNAL_ISAAC_EGO_FRAME_PATH", str(path))
    client = ExternalIsaacEgoClient(_Task())
    _write_frame(path, 320, 240)
    with pytest.raises(ValueError, match="shape"):
        client._read_frame()
    client.close()


def test_udp_state_carries_recording_identity(tmp_path, monkeypatch):
    path = tmp_path / "ego.frame"
    monkeypatch.setenv("EXTERNAL_ISAAC_EGO_FRAME_PATH", str(path))
    client = ExternalIsaacEgoClient(_Task())
    client._source_qpos = lambda simulator: np.arange(7, dtype=np.float64)

    class _Data:
        time = 1.25

    class _Simulator:
        mjData = _Data()

    sent = []

    class _Socket:
        def sendto(self, payload, address):
            sent.append((json.loads(payload.decode("utf-8")), address))

        def close(self):
            pass

    client.socket.close()
    client.socket = _Socket()
    client._read_frame = lambda: (
        {
            "sequence": 1,
            "sim_time": 1.25,
            "published_monotonic_s": time.monotonic(),
            "width": 640,
            "height": 480,
            "session_id": client.session_id,
            "request_id": 0,
            "state_sequence": 0,
        },
        np.zeros((480, 640, 3), dtype=np.uint8),
    )

    client.sync_and_render(_Simulator())

    assert sent[0][0]["recording_index"] == 17
    assert sent[0][0]["recording_name"] == "recording_17"
    assert sent[0][0]["recording_seed"] == 42
    assert sent[0][0]["schema_version"] == 2
    assert sent[0][0]["session_id"] == client.session_id
    assert sent[0][0]["request_id"] == 0
    assert sent[0][0]["state_sequence"] == 0
    assert sent[0][1][1] == 23331
    client.close()


def test_sync_waits_for_exact_request_id(tmp_path, monkeypatch):
    path = tmp_path / "ego.frame"
    monkeypatch.setenv("EXTERNAL_ISAAC_EGO_FRAME_PATH", str(path))
    client = ExternalIsaacEgoClient(_Task())
    client._source_qpos = lambda simulator: np.arange(7, dtype=np.float64)

    class _Data:
        time = 1.25

    class _Simulator:
        mjData = _Data()

    class _Socket:
        def sendto(self, payload, address):
            pass

        def close(self):
            pass

    client.socket.close()
    client.socket = _Socket()
    frames = iter(
        [
            {
                "sequence": 99,
                "sim_time": 1.25,
                "published_monotonic_s": time.monotonic(),
                "width": 640,
                "height": 480,
                "session_id": client.session_id,
                "request_id": 123,
                "state_sequence": 123,
            },
            {
                "sequence": 1,
                "sim_time": 1.25,
                "published_monotonic_s": time.monotonic(),
                "width": 640,
                "height": 480,
                "session_id": client.session_id,
                "request_id": 0,
                "state_sequence": 0,
            },
        ]
    )
    reads = []

    def _read_frame():
        metadata = next(frames)
        reads.append(metadata)
        return metadata, np.zeros((480, 640, 3), dtype=np.uint8)

    client._read_frame = _read_frame
    client.sync_and_render(_Simulator())

    assert len(reads) == 2
    assert client.last_sequence == 1
    client.close()


def test_task1_external_reset_renders_csv_row0_before_mj_step(monkeypatch):
    recordings = Path("/home/ubuntu/yzh/mujoco_recordings/20260615_task1_lqb")
    if not recordings.is_dir():
        pytest.skip(f"Task1 LQB recordings unavailable: {recordings}")
    monkeypatch.setenv("TASK1_RECORDINGS_DIR", str(recordings))
    monkeypatch.setenv("TASK1_RECORDING_INDEX", "0")
    rendered = []

    def _capture_reset_frame(self, simulator):
        rendered.append((float(simulator.mjData.time), simulator.mjData.qpos.copy()))
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        return {"head_stereo_left": rgb, "head_stereo_right": rgb}

    monkeypatch.setattr(ExternalIsaacEgoClient, "sync_and_render", _capture_reset_frame)
    env = gym.make(
        "simple/G1Fullstate20260615Task1-v0",
        sim_mode="mujoco_external_isaac",
        render_hz=50,
        headless=True,
        success_criteria=None,
        sonic_config=_make_sonic_config(),
    )
    try:
        env.reset(options={"episode_index": 0})
        raw = env.unwrapped
        expected = raw.task._recording_initializations[0].qpos
        assert rendered[0][0] == 0.0
        assert raw.mujoco.render_step == 0
        np.testing.assert_array_equal(rendered[0][1][: expected.size], expected)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scene", "expected_nq"),
    [
        ("/home/ubuntu/yzh/mujoco_recordings/20260615_task1_new/20260615_141610_g1_sim/model_snapshot/mujoco/model/g1/scene_43dof.xml", 57),
        ("/home/ubuntu/yzh/mujoco_recordings/20260805_task2_new/20260805_152442_g1_sim/model_snapshot/mujoco/model/g1/scene_43dof.xml", 59),
        ("/home/ubuntu/yzh/mujoco_recordings/20260804_task3_new/20260804_170954_g1_sim/model_snapshot/mujoco/model/g1/scene_43dof.xml", 52),
        ("/home/ubuntu/yzh/mujoco_recordings/20260729_task4/20260729_151117_g1_sim/model_snapshot/mujoco/model/g1/scene_43dof.xml", 57),
    ],
)
def test_snapshot_xml_qpos_layout_does_not_need_meshes(scene, expected_nq):
    layout, nq = _xml_joint_layout(Path(scene))
    assert nq == expected_nq
    assert layout["floating_base_joint"] == (0, 7)
