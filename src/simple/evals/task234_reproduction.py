"""Frozen Task2/3/4 MuJoCo-physics + Isaac-render reproduction contract.

This module deliberately has no Isaac Sim or DDS imports.  It is shared by the
SIMPLE tasks, the MuJoCo/Isaac engines, preflight tooling, and lightweight unit
tests.  The renderer must treat MuJoCo state as authoritative; these structures
only describe validation and visual mirroring.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class EgoCameraSpec:
    parent_prim: str
    eye: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]
    pitch_deg: float
    vertical_fov_deg: float
    near_clip: float
    width: int
    height: int
    max_state_lag_seconds: float = 0.25

    def manifest(self) -> dict[str, Any]:
        return {
            "parent_prim": self.parent_prim,
            "eye": list(self.eye),
            "forward": list(self.forward),
            "up": list(self.up),
            "pitch_deg": self.pitch_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
            "near_clip": self.near_clip,
            "width": self.width,
            "height": self.height,
            "max_state_lag_seconds": self.max_state_lag_seconds,
        }


@dataclass(frozen=True, slots=True)
class HssdSpec:
    scene_uid: str
    usd_relative_path: str
    scale: float
    insertion_surface: str
    hidden_prims: tuple[str, ...] = ()
    target_surface_center: tuple[float, float, float] = (0.0, 0.25, 0.0)
    rotation_xyz_deg: tuple[float, float, float] = (90.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class TaskReproductionSpec:
    task_number: int
    gym_id: str
    task_uid: str
    recording_root_default: str
    reference_instance: str
    expected_nq: int
    expected_nv: int
    expected_nu: int
    task_translate: tuple[float, float, float]
    robot_visual_z_offset: float
    trash_visual_z_offset: float | None
    free_object_visual_z_offset: float
    hssd: HssdSpec
    world_camera_eye: tuple[float, float, float]
    world_camera_target: tuple[float, float, float]
    world_camera_vfov_deg: float = 69.24

    def expected_dimensions(self) -> tuple[int, int, int]:
        return self.expected_nq, self.expected_nv, self.expected_nu

    def recording_mjcf(self, instance: str | None = None) -> Path:
        selected = instance or self.reference_instance
        return (
            Path(self.recording_root_default)
            / selected
            / "model_snapshot/mujoco/model/g1/scene_43dof.xml"
        )


EGO_CAMERA = EgoCameraSpec(
    parent_prim=(
        "/World/Task3/Geometry/pelvis/waist_yaw_link/"
        "waist_roll_link/torso_link"
    ),
    eye=(0.06, 0.06, 0.45),
    forward=(0.71735609, 0.0, -0.69670671),
    up=(0.69670649, 0.00079633, 0.71735586),
    pitch_deg=15.0,
    vertical_fov_deg=70.0,
    near_clip=0.2,
    width=640,
    height=480,
)


SCENE3 = HssdSpec(
    scene_uid="scene3",
    usd_relative_path="scenes/hssd/102344280/102344280.usd",
    scale=0.01,
    insertion_surface="furniture/d68aaf2484eec4c754d3b6e07adc09293d8b36de",
)

SCENE13 = HssdSpec(
    scene_uid="scene13",
    # The official SIMPLE archive contains 102344250.usd.  Conflicting room
    # furniture is hidden dynamically by the Isaac renderer, so no private
    # machine-local ``_local`` copy is required for inference.
    usd_relative_path="scenes/hssd/102344250/102344250.usd",
    scale=1.0,
    insertion_surface="furniture/node_e7c674ea7231612862a5bb66960f0cfef5d8f0e",
    hidden_prims=(
        "furniture/node_a9c2d2f765a2399442f845b7da0ddaa61e82071",
        "furniture/ccf3f0ee76dd2263f77ad90a7cec59da84e047c8",
        "furniture/f71c22e2956fcc6e97ace5d6bc9334ddcd1842c1",
    ),
)


TASK_SPECS: dict[int, TaskReproductionSpec] = {
    2: TaskReproductionSpec(
        task_number=2,
        gym_id="simple/G1Fullstate20260805Task2IsaacEval-v0",
        task_uid="g1_fullstate_20260805_task2_isaac_eval",
        recording_root_default="/home/ubuntu/yzh/mujoco_recordings/20260805_task2_new",
        reference_instance="20260805_152442_g1_sim",
        expected_nq=59,
        expected_nv=57,
        expected_nu=43,
        task_translate=(0.0, 0.0, 0.0),
        robot_visual_z_offset=0.012,
        trash_visual_z_offset=0.006,
        free_object_visual_z_offset=0.0,
        hssd=SCENE3,
        world_camera_eye=(-1.35, -1.05, 1.45),
        world_camera_target=(0.50, -0.18, 0.78),
    ),
    3: TaskReproductionSpec(
        task_number=3,
        gym_id="simple/G1Fullstate20260804Task3IsaacEval-v0",
        task_uid="g1_fullstate_20260804_task3_isaac_eval",
        recording_root_default="/home/ubuntu/yzh/mujoco_recordings/20260804_task3_new",
        reference_instance="20260804_171003_g1_sim",
        expected_nq=52,
        expected_nv=51,
        expected_nu=43,
        task_translate=(0.6, -0.5, 0.0),
        robot_visual_z_offset=0.012,
        trash_visual_z_offset=0.006,
        free_object_visual_z_offset=0.0,
        hssd=SCENE13,
        world_camera_eye=(-2.50, -1.65, 1.80),
        world_camera_target=(0.55, -0.05, 0.72),
    ),
    4: TaskReproductionSpec(
        task_number=4,
        gym_id="simple/G1Fullstate20260729Task4IsaacEval-v0",
        task_uid="g1_fullstate_20260729_task4_isaac_eval",
        recording_root_default="/home/ubuntu/yzh/mujoco_recordings/20260729_task4",
        reference_instance="20260729_151117_g1_sim",
        expected_nq=57,
        expected_nv=55,
        expected_nu=43,
        task_translate=(0.0, 0.0, 0.0),
        robot_visual_z_offset=0.012,
        trash_visual_z_offset=None,
        free_object_visual_z_offset=0.0,
        hssd=SCENE3,
        world_camera_eye=(-1.35, -1.05, 1.45),
        world_camera_target=(0.50, -0.18, 0.78),
    ),
}

TASK_ASSET_RELATIVE_PATHS: dict[int, tuple[str, ...]] = {
    2: (
        "local_mjcf/primitive_round_table_green_cylinder/round_table.xml",
        "local_mjcf/step_trash_can/trash_can.xml",
        "local_mjcf/obj_001_bottle_13_8e13a37c387158c8/bottle.xml",
    ),
    3: ("local_mjcf/step_trash_can/trash_can.xml",),
    4: (
        "local_mjcf/primitive_round_table_green_cylinder/round_table.xml",
        "local_mjcf/task4_green_box/green_box.xml",
        "local_mjcf/task4_bottle/bottle.xml",
    ),
}


def inference_asset_paths(
    spec: TaskReproductionSpec,
    *,
    simple_data_root: str | Path,
    task_assets_root: str | Path,
) -> tuple[Path, ...]:
    """Return all top-level assets needed for checkpoint-only SIMPLE eval."""

    data_root = Path(simple_data_root)
    assets_root = Path(task_assets_root)
    return (
        data_root / spec.hssd.usd_relative_path,
        *(assets_root / path for path in TASK_ASSET_RELATIVE_PATHS[spec.task_number]),
    )


def missing_inference_assets(
    spec: TaskReproductionSpec,
    *,
    simple_data_root: str | Path,
    task_assets_root: str | Path,
) -> list[Path]:
    return [
        path
        for path in inference_asset_paths(
            spec,
            simple_data_root=simple_data_root,
            task_assets_root=task_assets_root,
        )
        if not path.is_file()
    ]


def get_task_spec(task_number: int) -> TaskReproductionSpec:
    try:
        return TASK_SPECS[int(task_number)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported reproduction task: {task_number!r}") from exc


def ego_camera_quaternion_wxyz(camera: EgoCameraSpec = EGO_CAMERA) -> np.ndarray:
    """Return the USD camera local rotation (local -Z forward, +Y up)."""

    forward = np.asarray(camera.forward, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up -= forward * float(np.dot(forward, up))
    up /= np.linalg.norm(up)
    pitch = math.radians(camera.pitch_deg)
    pitched_forward = math.cos(pitch) * forward - math.sin(pitch) * up
    pitched_up = math.sin(pitch) * forward + math.cos(pitch) * up
    right = np.cross(pitched_forward, pitched_up)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, pitched_up, -pitched_forward))

    # Stable matrix -> wxyz conversion without scipy/Isaac dependencies.
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= np.linalg.norm(quat)
    return quat


def validate_model_dimensions(model: Any, spec: TaskReproductionSpec) -> None:
    actual = (int(model.nq), int(model.nv), int(model.nu))
    expected = spec.expected_dimensions()
    if actual != expected:
        raise ValueError(
            f"Task{spec.task_number} MJCF dimension mismatch: expected "
            f"nq/nv/nu={expected}, got {actual}. Refusing to truncate or pad state."
        )


def validate_state_packet(
    packet: Mapping[str, Any], spec: TaskReproductionSpec
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    nq = int(packet.get("nq", -1))
    nv = int(packet.get("nv", -1))
    if (nq, nv) != (spec.expected_nq, spec.expected_nv):
        raise ValueError(
            f"Task{spec.task_number} state packet expected nq/nv="
            f"({spec.expected_nq}, {spec.expected_nv}), got ({nq}, {nv})"
        )
    qpos = np.asarray(packet.get("qpos", ()), dtype=np.float64)
    qvel = np.asarray(packet.get("qvel", ()), dtype=np.float64)
    if qpos.shape != (nq,) or qvel.shape != (nv,):
        raise ValueError(
            f"State vector shape mismatch: qpos={qpos.shape}, qvel={qvel.shape}, "
            f"declared nq/nv=({nq}, {nv})"
        )
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("State packet contains NaN or Inf")
    qpos = qpos.copy()
    root_norm = float(np.linalg.norm(qpos[3:7]))
    if not math.isfinite(root_norm) or root_norm < 1e-8:
        raise ValueError("Floating-base quaternion is invalid")
    qpos[3:7] /= root_norm
    return qpos, qvel.copy(), float(packet["sim_time"]), bool(packet["running"])


def validate_task_joint_layout(model: Any, spec: TaskReproductionSpec) -> dict[str, int]:
    """Validate task qpos semantics by joint name and return role -> qpos address."""

    joints: list[tuple[str, int, int]] = []
    for joint_id in range(int(model.njnt)):
        joint = model.joint(joint_id)
        name = str(joint.name or "")
        address = int(model.jnt_qposadr[joint_id])
        next_address = (
            int(model.jnt_qposadr[joint_id + 1])
            if joint_id + 1 < int(model.njnt)
            else int(model.nq)
        )
        joints.append((name, address, next_address - address))

    task_joints = [(name, address, width) for name, address, width in joints if address >= 50]
    result: dict[str, int] = {}

    if spec.task_number == 2:
        scalar = [(name, address) for name, address, width in task_joints if width == 1]
        free = [(name, address) for name, address, width in task_joints if width == 7]
        if len(scalar) != 2 or [address for _, address in scalar] != [50, 51]:
            raise ValueError(f"Task2 expected two scalar task hinges at qpos 50/51, got {task_joints}")
        if len(free) != 1 or free[0][1] != 52 or "bottle" not in free[0][0].lower():
            raise ValueError(f"Task2 expected a bottle free joint at qpos 52:59, got {task_joints}")
        for name, address in scalar:
            lower = name.lower()
            if "lid" in lower:
                result["trash_lid"] = address
            elif "pedal" in lower:
                result["trash_pedal"] = address
        if set(result) != {"trash_lid", "trash_pedal"}:
            raise ValueError(f"Task2 hinge names must identify lid and pedal, got {scalar}")
        result["bottle_free"] = 52
    elif spec.task_number == 3:
        scalar = [(name, address) for name, address, width in task_joints if width == 1]
        if len(scalar) != 2 or sorted(address for _, address in scalar) != [50, 51]:
            raise ValueError(
                "Task3 expected two trash-can hinges at qpos 50/51; "
                f"got {task_joints}"
            )
        for name, address in scalar:
            lower = name.lower()
            if "lid" in lower:
                result["trash_lid"] = address
            elif "pedal" in lower:
                result["trash_pedal"] = address
        if set(result) != {"trash_lid", "trash_pedal"}:
            raise ValueError(
                f"Task3 hinge names must identify lid and pedal, got {scalar}"
            )
    else:
        free = [(name, address) for name, address, width in task_joints if width == 7]
        if len(free) != 1 or free[0][1] != 50 or "bottle" not in free[0][0].lower():
            raise ValueError(f"Task4 expected a bottle free joint at qpos 50:57, got {task_joints}")
        result["bottle_free"] = 50
    return result


def validate_policy_rgb(
    image: np.ndarray,
    *,
    image_sim_time: float,
    mujoco_sim_time: float,
    frame_id: int,
    previous_frame_id: int | None = None,
    camera: EgoCameraSpec = EGO_CAMERA,
) -> None:
    if image.shape != (camera.height, camera.width, 3):
        raise ValueError(
            f"Policy RGB must be {camera.width}x{camera.height} RGB; got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"Policy RGB must use uint8 pixels; got {image.dtype}")
    lag = float(mujoco_sim_time) - float(image_sim_time)
    if lag < -1e-6 or lag > camera.max_state_lag_seconds:
        raise RuntimeError(
            f"Isaac Ego frame is not synchronized: image_sim_time={image_sim_time:.6f}, "
            f"mujoco_sim_time={mujoco_sim_time:.6f}, lag={lag:.6f}s"
        )
    if previous_frame_id is not None and int(frame_id) <= int(previous_frame_id):
        raise RuntimeError(
            f"Isaac Ego frame did not advance: previous={previous_frame_id}, current={frame_id}"
        )


def mjcf_bundle_hash(mjcf_path: str | Path) -> str:
    """Hash the MJCF plus files referenced by local ``file=`` attributes."""

    import xml.etree.ElementTree as ET

    root_path = Path(mjcf_path).resolve()
    paths = {root_path}
    pending = [root_path]
    while pending:
        current = pending.pop()
        try:
            xml_root = ET.parse(current).getroot()
        except ET.ParseError:
            continue
        for element in xml_root.iter():
            raw = element.attrib.get("file")
            if not raw:
                continue
            candidate = (current.parent / raw).resolve()
            if candidate.is_file() and candidate not in paths:
                paths.add(candidate)
                if candidate.suffix.lower() == ".xml":
                    pending.append(candidate)

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.relative_to(root_path.parent)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_recording_mjcf_bundle(
    scene_path: str | Path,
    spec: TaskReproductionSpec,
    *,
    task_assets_root: str | Path | None = None,
    canonical_model_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """Create a lightweight runtime view for assets omitted by snapshots."""

    scene_path = Path(scene_path).resolve()
    task_assets_root = Path(
        task_assets_root
        or os.environ.get(
            "SIMPLE_TASK_ASSETS_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets",
        )
    ).resolve()
    canonical_model_root = Path(
        canonical_model_root
        or os.environ.get(
            "SIMPLE_MUJOCO_MODEL_ROOT",
            (
                "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/"
                "HumanoidVLA_MJ_1/mujoco/model"
            ),
        )
    ).resolve()
    if not task_assets_root.is_dir():
        raise FileNotFoundError(
            f"Task asset root does not exist: {task_assets_root}. "
            "Set SIMPLE_TASK_ASSETS_ROOT."
        )
    if not (canonical_model_root / "g1/meshes").is_dir():
        raise FileNotFoundError(
            f"Canonical G1 model assets are missing: {canonical_model_root}"
        )

    digest = hashlib.sha256()
    digest.update(scene_path.read_bytes())
    digest.update(os.fspath(task_assets_root).encode("utf-8"))
    digest.update(os.fspath(canonical_model_root).encode("utf-8"))
    cache_root = Path(
        cache_root
        or os.environ.get(
            "SIMPLE_RECORDING_BUNDLE_CACHE",
            "/tmp/simple_task234_recording_bundles",
        )
    )
    bundle_root = cache_root / digest.hexdigest()
    bundle_g1 = bundle_root / "model/g1"
    bundle_g1.mkdir(parents=True, exist_ok=True)

    def ensure_link(destination: Path, source: Path) -> None:
        if destination.exists() or destination.is_symlink():
            return
        if not source.exists():
            raise FileNotFoundError(
                f"Task{spec.task_number} runtime bundle source is missing: {source}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=source.is_dir())

    for source in scene_path.parent.iterdir():
        if source.is_file():
            ensure_link(bundle_g1 / source.name, source)
    for name in ("meshes", "objects", "objects_coacd", "textures"):
        snapshot_source = scene_path.parent / name
        source = (
            snapshot_source
            if snapshot_source.exists()
            else canonical_model_root / "g1" / name
        )
        ensure_link(bundle_g1 / name, source)

    ensure_link(bundle_root / "task_assets", task_assets_root)
    ensure_link(bundle_root / "model/task_assets", task_assets_root)
    for name in ("robotwin_assets", "adam_pro", "adam_pick"):
        source = canonical_model_root / name
        if source.exists():
            ensure_link(bundle_root / name, source)
            ensure_link(bundle_root / "model" / name, source)
    return bundle_g1 / scene_path.name


def missing_assets(
    spec: TaskReproductionSpec,
    *,
    simple_data_root: str | Path,
    recording_root: str | Path | None = None,
    instance: str | None = None,
) -> list[Path]:
    root = Path(recording_root) if recording_root is not None else Path(spec.recording_root_default)
    instance_root = root / (instance or spec.reference_instance)
    mjcf = instance_root / "model_snapshot/mujoco/model/g1/scene_43dof.xml"
    initial_state = instance_root / "data.csv"
    hssd = Path(simple_data_root) / spec.hssd.usd_relative_path
    return [path for path in (mjcf, initial_state, hssd) if not path.is_file()]


def write_camera_manifest(path: str | Path, task_spec: TaskReproductionSpec) -> None:
    payload = {
        "task": task_spec.task_number,
        "gym_id": task_spec.gym_id,
        "ego_camera": EGO_CAMERA.manifest(),
        "task_translate": list(task_spec.task_translate),
        "world_camera": {
            "eye": list(task_spec.world_camera_eye),
            "target": list(task_spec.world_camera_target),
            "vertical_fov_deg": task_spec.world_camera_vfov_deg,
        },
        "policy_image_source": "isaac_ego_render_product",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
