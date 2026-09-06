"""Strict localhost bridge from authoritative MuJoCo state to Isaac Ego RGB."""

from __future__ import annotations

import io
import json
import os
import socket
import struct
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image


FRAME_MAGIC = b"HVEGO01\0"


def _joint_layout(model: mujoco.MjModel) -> dict[str, tuple[int, int]]:
    layout: dict[str, tuple[int, int]] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise ValueError(f"MuJoCo joint {joint_id} has no name")
        start = int(model.jnt_qposadr[joint_id])
        stop = int(model.jnt_qposadr[joint_id + 1]) if joint_id + 1 < model.njnt else int(model.nq)
        if name in layout:
            raise ValueError(f"Duplicate MuJoCo joint name: {name}")
        layout[name] = (start, stop)
    return layout


def _xml_joint_layout(mjcf_path: Path) -> tuple[dict[str, tuple[int, int]], int]:
    """Resolve qpos addresses without loading meshes from a recording snapshot."""
    layout: dict[str, tuple[int, int]] = {}
    qpos_address = 0
    include_stack: list[Path] = []

    def visit(element: ET.Element, source_dir: Path, in_worldbody: bool) -> None:
        nonlocal qpos_address
        if element.tag == "include":
            include_path = (source_dir / element.attrib["file"]).resolve()
            if include_path in include_stack:
                raise ValueError(f"Recursive MJCF include: {include_path}")
            include_stack.append(include_path)
            included = ET.parse(include_path).getroot()
            for child in included:
                visit(child, include_path.parent, in_worldbody)
            include_stack.pop()
            return
        if element.tag == "worldbody":
            for child in element:
                visit(child, source_dir, True)
            return
        if element.tag in {"body", "frame"} and in_worldbody:
            for child in element:
                visit(child, source_dir, True)
            return
        if not in_worldbody or element.tag not in {"joint", "freejoint"}:
            return
        name = element.get("name")
        if not name:
            raise ValueError(f"Unnamed {element.tag} in {mjcf_path}")
        joint_type = "free" if element.tag == "freejoint" else element.get("type", "hinge")
        width = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}.get(joint_type)
        if width is None:
            raise ValueError(f"Unsupported joint type {joint_type}: {name}")
        if name in layout:
            raise ValueError(f"Duplicate source joint name: {name}")
        layout[name] = (qpos_address, qpos_address + width)
        qpos_address += width

    resolved = mjcf_path.expanduser().resolve()
    include_stack.append(resolved)
    root = ET.parse(resolved).getroot()
    for child in root:
        visit(child, resolved.parent, False)
    include_stack.pop()
    return layout, qpos_address


class ExternalIsaacEgoClient:
    """Send one latest-only state and wait for its matching HVEGO frame."""

    def __init__(self, task: Any):
        self.task = task
        self.host = os.environ.get("EXTERNAL_ISAAC_UDP_HOST", "127.0.0.1")
        self.port = int(os.environ.get("EXTERNAL_ISAAC_UDP_PORT", "23331"))
        self.frame_path = Path(
            os.environ.get(
                "EXTERNAL_ISAAC_EGO_FRAME_PATH",
                f"/dev/shm/simple_{task.recording_env_prefix.lower()}_isaac_ego.frame",
            )
        )
        self.width = int(os.environ.get("EXTERNAL_ISAAC_EGO_WIDTH", "640"))
        self.height = int(os.environ.get("EXTERNAL_ISAAC_EGO_HEIGHT", "480"))
        self.timeout_s = float(os.environ.get("EXTERNAL_ISAAC_EGO_TIMEOUT_S", "5.0"))
        self.max_age_s = float(os.environ.get("EXTERNAL_ISAAC_EGO_MAX_AGE_S", "1.0"))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_sequence = -1
        self.session_id = uuid.uuid4().hex
        self.next_request_id = 0
        self.last_frame: np.ndarray | None = None
        self._source_nq = 0
        self._source_scene_hash: str | None = None
        self._mapping: list[tuple[slice, slice]] = []

    def _ensure_mapping(self, simulator: Any) -> None:
        source_path = Path(self.task.external_isaac_mjcf).expanduser().resolve()
        scene_hash = str(self.task.selected_recording_scene_sha256)
        if self._source_scene_hash == scene_hash and self._mapping:
            return
        if not source_path.is_file():
            raise FileNotFoundError(f"External Isaac source MJCF is missing: {source_path}")
        source_layout, source_nq = _xml_joint_layout(source_path)
        target_layout = _joint_layout(simulator.mjModel)
        aliases = getattr(self.task, "external_isaac_joint_aliases", {})
        mapping: list[tuple[slice, slice]] = []
        missing: list[str] = []
        for source_name, (source_start, source_stop) in source_layout.items():
            target_name = aliases.get(source_name, source_name)
            target = target_layout.get(target_name)
            if target is None:
                missing.append(f"{source_name}->{target_name}")
                continue
            target_start, target_stop = target
            if source_stop - source_start != target_stop - target_start:
                raise ValueError(
                    f"qpos width mismatch for {source_name}->{target_name}: "
                    f"source={source_stop - source_start}, target={target_stop - target_start}"
                )
            mapping.append((slice(source_start, source_stop), slice(target_start, target_stop)))
        if missing:
            raise ValueError(
                "SIMPLE MuJoCo model cannot reconstruct snapshot qpos by name; "
                f"missing joints={missing}"
            )
        if sum(source.stop - source.start for source, _ in mapping) != source_nq:
            raise ValueError("External Isaac qpos mapping does not cover source nq")
        self._source_nq = source_nq
        self._source_scene_hash = scene_hash
        self._mapping = mapping
        print(
            f"[ExternalIsaacEgo] mapped source={source_path} nq={source_nq} "
            f"joints={len(mapping)} udp={self.host}:{self.port} frame={self.frame_path}",
            flush=True,
        )

    def _source_qpos(self, simulator: Any) -> np.ndarray:
        self._ensure_mapping(simulator)
        result = np.empty(self._source_nq, dtype=np.float64)
        for source_slice, target_slice in self._mapping:
            result[source_slice] = simulator.mjData.qpos[target_slice]
        return result

    def _read_frame(self) -> tuple[dict[str, Any], np.ndarray]:
        try:
            payload = self.frame_path.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Isaac Ego frame is missing: {self.frame_path}; start the task's isaac terminal first"
            ) from exc
        if len(payload) < len(FRAME_MAGIC) + 4 or not payload.startswith(FRAME_MAGIC):
            raise ValueError(f"Invalid HVEGO01 frame header: {self.frame_path}")
        metadata_size = struct.unpack_from("<I", payload, len(FRAME_MAGIC))[0]
        metadata_start = len(FRAME_MAGIC) + 4
        metadata_stop = metadata_start + metadata_size
        if metadata_stop >= len(payload):
            raise ValueError(f"Truncated HVEGO01 frame: {self.frame_path}")
        metadata = json.loads(payload[metadata_start:metadata_stop].decode("utf-8"))
        with Image.open(io.BytesIO(payload[metadata_stop:])) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Isaac Ego shape is {rgb.shape}, expected ({self.height}, {self.width}, 3)"
            )
        if int(metadata.get("width", -1)) != self.width or int(metadata.get("height", -1)) != self.height:
            raise ValueError(f"Isaac Ego metadata size mismatch: {metadata}")
        return metadata, rgb

    def sync_and_render(self, simulator: Any) -> dict[str, np.ndarray]:
        qpos = self._source_qpos(simulator)
        sim_time = float(simulator.mjData.time)
        request_id = self.next_request_id
        self.next_request_id += 1
        payload = json.dumps(
            {
                "schema_version": 2,
                "session_id": self.session_id,
                "request_id": request_id,
                "state_sequence": request_id,
                "model_id": self.task.uid,
                "nq": int(qpos.size),
                "sim_time": sim_time,
                "running": 1,
                "recording_index": getattr(
                    self.task, "selected_recording_index", None
                ),
                "recording_name": getattr(
                    self.task, "selected_recording_name", None
                ),
                "recording_seed": getattr(
                    self.task, "selected_recording_seed", None
                ),
                "qpos": qpos.tolist(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > 65507:
            raise ValueError(f"External Isaac UDP payload is too large: {len(payload)}")
        self.socket.sendto(payload, (self.host, self.port))

        deadline = time.monotonic() + self.timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                metadata, rgb = self._read_frame()
                sequence = int(metadata["sequence"])
                published = float(metadata["published_monotonic_s"])
                frame_sim_time = float(metadata["sim_time"])
                age = time.monotonic() - published
                frame_session_id = str(metadata.get("session_id", ""))
                frame_request_id = int(metadata.get("request_id", -1))
                frame_state_sequence = int(metadata.get("state_sequence", -1))
                if (
                    frame_session_id != self.session_id
                    or frame_request_id != request_id
                    or frame_state_sequence != request_id
                ):
                    raise RuntimeError(
                        "Isaac Ego request mismatch: "
                        f"expected=({self.session_id},{request_id},{request_id}), "
                        f"got=({frame_session_id},{frame_request_id},{frame_state_sequence})"
                    )
                if age < -0.1 or age > self.max_age_s:
                    raise RuntimeError(
                        f"Isaac Ego frame is stale: age={age:.3f}s, limit={self.max_age_s:.3f}s"
                    )
                if frame_sim_time != sim_time:
                    raise RuntimeError(
                        f"Isaac Ego sim_time mismatch: frame={frame_sim_time!r}, "
                        f"MuJoCo={sim_time!r}"
                    )
                self.last_sequence = sequence
                self.last_frame = rgb
                return {
                    "head_stereo_left": rgb,
                    "head_stereo_right": rgb,
                }
            except (FileNotFoundError, RuntimeError) as exc:
                last_error = exc
                time.sleep(0.002)
        raise TimeoutError(
            f"Timed out after {self.timeout_s:.1f}s waiting for synchronized Isaac "
            f"Ego frame at {self.frame_path}: {last_error}"
        )

    def cached_frame(self) -> dict[str, np.ndarray]:
        if self.last_frame is None:
            raise RuntimeError("No cached external Isaac Ego frame is available")
        return {
            "head_stereo_left": self.last_frame,
            "head_stereo_right": self.last_frame,
        }

    def close(self) -> None:
        self.socket.close()
