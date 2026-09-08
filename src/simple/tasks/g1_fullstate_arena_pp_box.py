"""MuJoCo evaluation task aligned with HumanoidArena twist2 box demos."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simple.core.layout import Layout
from simple.core.types import Pose
from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_fullstate_arena_pp_box")
class G1FullstateArenaPPBox(G1Fullstate20260615Task1):
    """Pick up the recorded cardboard box and place it on the shelf."""

    uid = "g1_fullstate_arena_pp_box"
    label = "G1 Fullstate Arena Pick-and-Place Box"
    description = "Pick up the box and place it on the shelf."

    metadata = {
        **G1Fullstate20260615Task1.metadata,
        # Match the twist2 evaluation configuration: dt=0.001, decimation=20.
        "physics_dt": 0.001,
        "control_hz": 1000,
        "render_hz": 50,
        "max_episode_steps": 1450,
        # MuJoCo owns all contacts; Isaac is a read-only visual mirror.
        "isaac_read_only_mirror": True,
    }

    _simple_root = Path(__file__).resolve().parents[3]
    _asset_root = _simple_root / "data/assets/humanoidarena/local_mjcf/arena_pp_box"
    _arena_repo_root = Path(
        os.environ.get(
            "HUMANOID_ARENA_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena/isaaclab_twist2_g1",
        )
    )
    _raw_data_root = Path(
        os.environ.get(
            "ARENA_PP_BOX_RAW_DIR",
            (
                "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/"
                "HumanoidArena_pp_box/twist2/yb"
            ),
        )
    )

    # HumanoidArena's g1_front_camera is shared by football and P&P Box.
    sensor_cfgs = copy.deepcopy(G1Fullstate20260615Task1.sensor_cfgs)
    sensor_cfgs["head_stereo"].width = 640
    sensor_cfgs["head_stereo"].height = 480
    sensor_cfgs["head_stereo"].focal_length = 7.6
    sensor_cfgs["head_stereo"].fov = np.deg2rad(
        float(os.environ.get("ARENA_PP_BOX_CAMERA_HFOV_DEG", "105.53033203685067"))
    )
    sensor_cfgs["head_stereo"].near = 0.1
    sensor_cfgs["head_stereo"].far = 1.0e5
    sensor_cfgs["head_stereo"].baseline = 0.06
    sensor_cfgs["head_stereo"].pose = {
        "position": [0.0, 0.0, 0.0],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    mujoco_native_head_camera = True

    _default_root_pos = np.asarray([-1.6, -5.0, 0.8], dtype=np.float64)
    _default_root_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _default_body_qpos29 = np.asarray(
        [
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )
    _default_hand_qpos14 = np.zeros(14, dtype=np.float64)
    _default_box_pos = np.asarray([-0.42994285, -4.72239637, 0.75], dtype=np.float64)
    _default_box_quat = [0.95721936, 0.0, 0.0, 0.28936329]
    _default_shelf_pos = np.asarray([-2.49047208, -3.10416484, 0.0], dtype=np.float64)
    _default_shelf_quat = [0.99955928, 0.0, 0.0, -0.02968508]

    _box_body_name = "arena_pp_box_box_object"
    _box_freejoint_name = "arena_pp_box_box_free"
    _shelf_body_name = "arena_pp_box_shelf_shelf"
    _box_half_extents = np.asarray([0.105, 0.105, 0.105], dtype=np.float64)
    _box_mass = 0.05
    _box_usd_scale = 0.01
    _shelf_support_half_extents_xy = np.asarray([0.45, 0.18], dtype=np.float64)
    _shelf_active_support_top_z = 0.6477
    _placement_xy_tolerance = 0.01
    _placement_z_tolerance = 0.06
    _stable_linear_speed = 0.10
    _stable_angular_speed = 2.0
    _stable_steps_required = 8
    _lift_threshold = 0.05

    mujoco_scene_mjcf_path = _asset_root / "table.xml"
    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "box.xml",
            "prefix": "arena_pp_box_box_",
            "pos": _default_box_pos.tolist(),
            "quat": _default_box_quat,
            "freejoint_body": "object",
            "freejoint_name": "free",
        },
        {
            "path": _asset_root / "shelf.xml",
            "prefix": "arena_pp_box_shelf_",
            "pos": _default_shelf_pos.tolist(),
            "quat": _default_shelf_quat,
        },
    ]
    mujoco_object_body_names = {
        "target": _box_body_name,
        "shelf": _shelf_body_name,
    }
    mujoco_initial_robot_qpos = np.concatenate(
        [
            _default_root_pos,
            _default_root_quat,
            _default_body_qpos29,
            _default_hand_qpos14,
        ]
    ).tolist()
    mujoco_ground_friction = [1.0, 0.005, 0.0001]
    mujoco_ground_visual = {
        "builtin": "mjBUILTIN_FLAT",
        "rgb1": [0.58, 0.58, 0.56],
        "rgb2": [0.58, 0.58, 0.56],
        "mark": "mjMARK_EDGE",
        "markrgb": [0.58, 0.58, 0.56],
        "width": 64,
        "height": 64,
        "texrepeat": [8.0, 8.0],
        "reflectance": 0.0,
    }

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _normalize_quat(quat: np.ndarray, *, source: str) -> np.ndarray:
        value = np.asarray(quat, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(value).all() or norm <= 1.0e-12:
            raise ValueError(f"Invalid quaternion from {source}: {value}")
        return value / norm

    @staticmethod
    def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = G1FullstateArenaPPBox._normalize_quat(
            quat_wxyz, source="quaternion-to-matrix"
        )
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _arena_isaac_asset_root(cls) -> Path:
        raw = os.environ.get("HUMANOID_ARENA_ISAAC_ASSET_ROOT") or os.environ.get(
            "ARENA_PP_BOX_ISAAC_ASSET_ROOT"
        )
        if raw:
            return Path(raw)
        for candidate in (cls._arena_repo_root / "assets1", cls._arena_repo_root / "assets"):
            if candidate.exists():
                return candidate
        return cls._arena_repo_root / "assets1"

    @classmethod
    def _arena_isaac_asset_paths(cls) -> dict[str, Path]:
        root = cls._arena_isaac_asset_root()
        scene = root / "objects/small_warehouse/small_warehouse_box"
        interaction = scene / "intreaction_obj"
        return {
            "room_usd": scene / "small_warehouse_box.usd",
            "box_usd": (
                interaction
                / "CubeBox_A03_21cm_PR_NVD_01/"
                "CubeBox_A03_21cm_PR_NVD_01_physics_rigid.usd"
            ),
            "shelf_usd": interaction / "HeavyDutySteelShelving_A06_PR_NVD_01_physics.usd",
        }

    def isaac_extra_usd_references(self) -> list[dict[str, Any]]:
        if not self._env_flag("ARENA_PP_BOX_USE_ARENA_USD_VISUALS", True):
            return []
        paths = self._arena_isaac_asset_paths()
        required = self._env_flag("ARENA_PP_BOX_REQUIRE_ARENA_USD_VISUALS", False)
        box_pos = getattr(self, "_arena_pp_box_start_box_pos", self._default_box_pos)
        box_quat = getattr(self, "_arena_pp_box_start_box_quat", self._default_box_quat)
        shelf_pos = getattr(self, "_arena_pp_box_start_shelf_pos", self._default_shelf_pos)
        shelf_quat = getattr(
            self, "_arena_pp_box_start_shelf_quat", self._default_shelf_quat
        )
        return [
            {
                "name": "arena_pp_box_room_usd",
                "usd_path": str(paths["room_usd"]),
                "prim_path": "/World/workspace/ArenaPPBoxRoom",
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "scale": [100.0, 100.0, 100.0],
                "disable_collision": True,
                "required": required,
            },
            {
                "name": "arena_pp_box_box_usd",
                "usd_path": str(paths["box_usd"]),
                "prim_path": "/World/workspace/ArenaPPBox",
                "pos": np.asarray(box_pos, dtype=np.float64).tolist(),
                "quat": list(box_quat),
                "scale": [self._box_usd_scale] * 3,
                "sync_body": self._box_body_name,
                "disable_collision": True,
                "required": required,
            },
            {
                "name": "arena_pp_box_shelf_usd",
                "usd_path": str(paths["shelf_usd"]),
                "prim_path": "/World/workspace/ArenaPPBoxShelf",
                "pos": np.asarray(shelf_pos, dtype=np.float64).tolist(),
                "quat": list(shelf_quat),
                "scale": [1.0, 1.0, 1.0],
                "sync_body": self._shelf_body_name,
                "disable_collision": True,
                "required": required,
            },
        ]

    @classmethod
    def _world_box_primitive(
        cls,
        *,
        name: str,
        local_pos: list[float],
        scale: list[float],
        shelf_pos: np.ndarray,
        shelf_quat: np.ndarray,
        color: list[float],
    ) -> dict[str, Any]:
        rotation = cls._quat_to_matrix(shelf_quat)
        position = np.asarray(shelf_pos, dtype=np.float64) + rotation @ np.asarray(
            local_pos, dtype=np.float64
        )
        return {
            "name": name,
            "type": "cube",
            "pos": position.tolist(),
            "quat": np.asarray(shelf_quat, dtype=np.float64).tolist(),
            "scale": scale,
            "color": color,
        }

    def _fallback_shelf_primitives(self) -> list[dict[str, Any]]:
        shelf_pos = np.asarray(
            getattr(self, "_arena_pp_box_start_shelf_pos", self._default_shelf_pos),
            dtype=np.float64,
        )
        shelf_quat = np.asarray(
            getattr(self, "_arena_pp_box_start_shelf_quat", self._default_shelf_quat),
            dtype=np.float64,
        )
        dark = [0.05, 0.06, 0.07]
        blue = [0.03, 0.22, 0.55]
        specs = []
        for name, z in (("base", 0.0965), ("active", 0.6352), ("upper", 1.1739)):
            specs.append(
                self._world_box_primitive(
                    name=f"arena_pp_box_shelf_{name}_fallback",
                    local_pos=[0.0, 0.0, z],
                    scale=[0.9, 0.36, 0.025],
                    shelf_pos=shelf_pos,
                    shelf_quat=shelf_quat,
                    color=dark,
                )
            )
        for x in (-0.425, 0.425):
            for y in (-0.155, 0.155):
                specs.append(
                    self._world_box_primitive(
                        name=f"arena_pp_box_shelf_post_{x:+.3f}_{y:+.3f}_fallback",
                        local_pos=[x, y, 0.64],
                        scale=[0.05, 0.05, 1.28],
                        shelf_pos=shelf_pos,
                        shelf_quat=shelf_quat,
                        color=blue,
                    )
                )
        return specs

    def isaac_extra_primitives(self) -> list[dict[str, Any]]:
        paths = self._arena_isaac_asset_paths()
        use_usd = self._env_flag("ARENA_PP_BOX_USE_ARENA_USD_VISUALS", True)
        box_pos = getattr(self, "_arena_pp_box_start_box_pos", self._default_box_pos)
        prims: list[dict[str, Any]] = [
            {
                "name": "arena_pp_box_dome_light",
                "type": "dome_light",
                "pos": [0.0, 0.0, 0.0],
                "color": [0.75, 0.75, 0.75],
                "intensity": 3000.0,
            }
        ]
        if not use_usd or not paths["room_usd"].exists():
            prims.append(
                {
                    "name": "arena_pp_box_table_fallback",
                    "type": "cube",
                    "pos": [-0.48, -4.775, 0.61],
                    "scale": [1.10, 2.30, 0.07],
                    "color": [0.72, 0.72, 0.70],
                }
            )
        if not use_usd or not paths["box_usd"].exists():
            prims.append(
                {
                    "name": "arena_pp_box_box_fallback",
                    "type": "cube",
                    "pos": np.asarray(box_pos, dtype=np.float64).tolist(),
                    "scale": [0.21, 0.21, 0.21],
                    "color": [0.76, 0.58, 0.32],
                    "sync_body": self._box_body_name,
                }
            )
        if not use_usd or not paths["shelf_usd"].exists():
            prims.extend(self._fallback_shelf_primitives())
        return prims

    def preload_objects(self) -> list[Any]:
        return []

    @classmethod
    def _robot_qpos_from_recording(
        cls,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        body_qpos29: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(root_pos, dtype=np.float64).reshape(3),
                cls._normalize_quat(root_quat, source="robot root"),
                np.asarray(body_qpos29, dtype=np.float64).reshape(29),
                cls._default_hand_qpos14,
            ]
        )

    @classmethod
    def _load_recording_initializations(
        cls,
        recordings_dir: Path,
    ) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Load only the original twist2 recordings whose front MP4 exists."""

        initializations = []
        missing_video = []
        for data_path in sorted(recordings_dir.glob("*.npz")):
            with np.load(data_path, allow_pickle=True) as data:
                if "vision_rgb_video_path" not in data.files:
                    missing_video.append(data_path.name)
                    continue
                video_rel = Path(str(np.asarray(data["vision_rgb_video_path"]).item()))
                if not (data_path.parent / video_rel).is_file():
                    missing_video.append(data_path.name)
                    continue

                robot_qpos = cls._robot_qpos_from_recording(
                    data["robot_root_position"][0],
                    data["robot_root_orientation"][0],
                    data["robot_qpos_before_decimation"][0],
                )
                box_pos = np.asarray(data["env_obj_box_position"][0], dtype=np.float64)
                box_quat = cls._normalize_quat(
                    data["env_obj_box_orientation"][0], source=f"{data_path}:box"
                )
                shelf_pos = np.asarray(data["env_obj_shelf_position"][0], dtype=np.float64)
                shelf_quat = cls._normalize_quat(
                    data["env_obj_shelf_orientation"][0], source=f"{data_path}:shelf"
                )

            arrays = (robot_qpos, box_pos, box_quat, shelf_pos, shelf_quat)
            if robot_qpos.size != 50 or any(not np.isfinite(value).all() for value in arrays):
                raise ValueError(f"Invalid Arena P&P Box initialization in {data_path}")
            initializations.append((data_path.name, *arrays))

        if not initializations:
            raise FileNotFoundError(
                f"No Arena P&P Box recordings with complete front video in {recordings_dir}"
            )
        cls._last_missing_video_recordings = tuple(missing_video)
        return initializations

    @staticmethod
    def _select_recording_index(
        pool_size: int,
        sequence_index: int,
        seed: int,
        fixed_index: int | None = None,
    ) -> int:
        """Select a deterministic recording while cycling through the full pool."""

        if pool_size <= 0:
            raise ValueError(f"Recording pool must be non-empty, got {pool_size}")
        if fixed_index is not None:
            if not 0 <= fixed_index < pool_size:
                raise ValueError(
                    f"ARENA_PP_BOX_RECORDING_INDEX={fixed_index} is outside "
                    f"the complete-video range 0..{pool_size - 1}"
                )
            return fixed_index
        cycle_index, cycle_offset = divmod(sequence_index, pool_size)
        order = np.random.default_rng(seed + cycle_index).permutation(pool_size)
        return int(order[cycle_offset])

    def reset(
        self,
        seed: int | None = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        self._layout = Layout()
        self._layout.add_robot(self.robot)
        self._layout.robot.pose = Pose(
            position=[0.0, 0.0, 0.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        )

        reset_index = int(getattr(self, "_arena_pp_box_reset_index", 0))
        self._arena_pp_box_reset_index = reset_index + 1
        self.mujoco_initial_robot_qpos = list(type(self).mujoco_initial_robot_qpos)
        box_pos = self._default_box_pos.copy()
        box_quat = list(self._default_box_quat)
        shelf_pos = self._default_shelf_pos.copy()
        shelf_quat = list(self._default_shelf_quat)
        source = "default_first_complete_recording"
        selected_index: int | None = None
        recording_name: str | None = None
        recording_seed: int | None = None

        if self._env_flag("ARENA_PP_BOX_INIT_FROM_DATA", False):
            recordings_dir = Path(os.environ.get("ARENA_PP_BOX_RAW_DIR", str(self._raw_data_root)))
            if not hasattr(self, "_arena_pp_box_initializations"):
                self._arena_pp_box_initializations = self._load_recording_initializations(
                    recordings_dir
                )
                expected_count = os.environ.get(
                    "ARENA_PP_BOX_EXPECTED_COMPLETE_RECORDINGS"
                )
                if (
                    expected_count is not None
                    and len(self._arena_pp_box_initializations) != int(expected_count)
                ):
                    raise ValueError(
                        "Arena P&P Box complete-video recording count mismatch: "
                        f"expected {expected_count}, found "
                        f"{len(self._arena_pp_box_initializations)} in {recordings_dir}"
                    )
                self._arena_pp_box_recording_seed = int(
                    os.environ.get("ARENA_PP_BOX_RECORDING_SEED", "0")
                )
                print(
                    "[ArenaPPBoxData] "
                    f"complete_recordings={len(self._arena_pp_box_initializations)} "
                    f"missing_video={len(type(self)._last_missing_video_recordings)}"
                )

            pool_size = len(self._arena_pp_box_initializations)
            recording_seed = self._arena_pp_box_recording_seed
            fixed_index_raw = os.environ.get("ARENA_PP_BOX_RECORDING_INDEX")
            fixed_index = int(fixed_index_raw) if fixed_index_raw is not None else None
            selected_index = self._select_recording_index(
                pool_size,
                reset_index,
                recording_seed,
                fixed_index,
            )

            recording_name, robot_qpos, box_pos, box_quat_arr, shelf_pos, shelf_quat_arr = (
                self._arena_pp_box_initializations[selected_index]
            )
            self.mujoco_initial_robot_qpos = robot_qpos.tolist()
            box_pos = box_pos.copy()
            box_quat = box_quat_arr.tolist()
            shelf_pos = shelf_pos.copy()
            shelf_quat = shelf_quat_arr.tolist()
            source = f"recording:{recording_name}"

        if options is not None:
            override_keys = []
            if "box_pos" in options:
                box_pos = np.asarray(options["box_pos"], dtype=np.float64).reshape(3)
                override_keys.append("box_pos")
            if "box_quat" in options:
                box_quat = self._normalize_quat(
                    options["box_quat"], source="reset options box"
                ).tolist()
                override_keys.append("box_quat")
            if "shelf_pos" in options:
                shelf_pos = np.asarray(options["shelf_pos"], dtype=np.float64).reshape(3)
                override_keys.append("shelf_pos")
            if "shelf_quat" in options:
                shelf_quat = self._normalize_quat(
                    options["shelf_quat"], source="reset options shelf"
                ).tolist()
                override_keys.append("shelf_quat")
            if override_keys:
                source = f"reset_options:{','.join(override_keys)}"

        box_pos[0] += float(os.environ.get("ARENA_PP_BOX_BOX_X_OFFSET", "0"))
        box_pos[1] += float(os.environ.get("ARENA_PP_BOX_BOX_Y_OFFSET", "0"))
        shelf_pos[0] += float(os.environ.get("ARENA_PP_BOX_SHELF_X_OFFSET", "0"))
        shelf_pos[1] += float(os.environ.get("ARENA_PP_BOX_SHELF_Y_OFFSET", "0"))

        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[0]["pos"] = box_pos.tolist()
        self.mujoco_extra_mjcf[0]["quat"] = list(box_quat)
        self.mujoco_extra_mjcf[1]["pos"] = shelf_pos.tolist()
        self.mujoco_extra_mjcf[1]["quat"] = list(shelf_quat)

        self._arena_pp_box_start_box_pos = box_pos.copy()
        self._arena_pp_box_start_box_quat = list(box_quat)
        self._arena_pp_box_start_shelf_pos = shelf_pos.copy()
        self._arena_pp_box_start_shelf_quat = list(shelf_quat)

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)

        instruction = "Pick up the box and place it on the shelf."
        if self._env_flag("ARENA_PP_BOX_ALLOW_INSTRUCTION_OVERRIDE", False):
            instruction = os.environ.get("ARENA_PP_BOX_INSTRUCTION", instruction)
        self._instruction = instruction
        self._target = None
        self.reward = 0.0
        self._box_start_z = float(box_pos[2])
        self._was_lifted = False
        self._arena_pp_box_stable_steps = 0
        self._arena_pp_box_success = False

        if self._env_flag("ARENA_PP_BOX_LOG_RESETS", True):
            recording_details = ""
            if recording_name is not None and selected_index is not None:
                recording_details = (
                    f" recording_pool_index={selected_index}"
                    f" recording_sequence_index={reset_index}"
                    f" recording_seed={recording_seed}"
                )
            print(
                "[ArenaPPBoxInit] "
                f"episode_reset={reset_index} source={source} "
                f"box_xyz=({box_pos[0]:.4f}, {box_pos[1]:.4f}, {box_pos[2]:.4f}) "
                f"shelf_xyz=({shelf_pos[0]:.4f}, {shelf_pos[1]:.4f}, {shelf_pos[2]:.4f})"
                f"{recording_details}"
            )
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    @classmethod
    def _placement_geometry(
        cls,
        box_center_w: np.ndarray,
        box_rotation_w: np.ndarray,
        shelf_center_w: np.ndarray,
        shelf_rotation_w: np.ndarray,
    ) -> dict[str, Any]:
        box_center = np.asarray(box_center_w, dtype=np.float64).reshape(3)
        box_rotation = np.asarray(box_rotation_w, dtype=np.float64).reshape(3, 3)
        shelf_center = np.asarray(shelf_center_w, dtype=np.float64).reshape(3)
        shelf_rotation = np.asarray(shelf_rotation_w, dtype=np.float64).reshape(3, 3)
        relative_center = shelf_rotation.T @ (box_center - shelf_center)
        relative_rotation = shelf_rotation.T @ box_rotation
        projected_half_extents = np.abs(relative_rotation) @ cls._box_half_extents
        inner_xy = cls._shelf_support_half_extents_xy - projected_half_extents[:2]
        inside_xy = bool(
            np.all(np.abs(relative_center[:2]) <= inner_xy + cls._placement_xy_tolerance)
        )
        box_bottom_z = float(relative_center[2] - projected_half_extents[2])
        z_gap = box_bottom_z - cls._shelf_active_support_top_z
        aligned_z = abs(z_gap) <= cls._placement_z_tolerance
        return {
            "relative_center": relative_center,
            "projected_half_extents": projected_half_extents,
            "inside_xy": inside_xy,
            "aligned_z": bool(aligned_z),
            "box_bottom_z": box_bottom_z,
            "z_gap": float(z_gap),
        }

    def _read_placement_state(self, mujoco_env: Any) -> dict[str, Any]:
        data = mujoco_env.mjData
        box_body = data.body(self._box_body_name)
        shelf_body = data.body(self._shelf_body_name)
        geometry = self._placement_geometry(
            np.asarray(box_body.xpos, dtype=np.float64),
            np.asarray(box_body.xmat, dtype=np.float64).reshape(3, 3),
            np.asarray(shelf_body.xpos, dtype=np.float64),
            np.asarray(shelf_body.xmat, dtype=np.float64).reshape(3, 3),
        )
        box_qvel = np.asarray(data.joint(self._box_freejoint_name).qvel, dtype=np.float64)
        geometry["linear_speed"] = float(np.linalg.norm(box_qvel[:3]))
        geometry["angular_speed"] = float(np.linalg.norm(box_qvel[3:]))
        geometry["stable"] = bool(
            geometry["linear_speed"] <= self._stable_linear_speed
            and geometry["angular_speed"] <= self._stable_angular_speed
        )
        return geometry

    def compute_reward(
        self,
        info: dict[str, Any],
        *args: Any,
        mujoco_env: Any = None,
        **kwargs: Any,
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Arena P&P Box reward requires the MuJoCo environment")

        placement = self._read_placement_state(mujoco_env)
        box_z = float(np.asarray(info["target"], dtype=np.float64)[2])
        self._was_lifted |= box_z >= self._box_start_z + self._lift_threshold
        placed_and_stable = bool(
            placement["inside_xy"] and placement["aligned_z"] and placement["stable"]
        )
        self._arena_pp_box_stable_steps = (
            self._arena_pp_box_stable_steps + 1 if placed_and_stable else 0
        )
        if self._arena_pp_box_stable_steps >= self._stable_steps_required:
            self._arena_pp_box_success = True

        stable_progress = min(
            self._arena_pp_box_stable_steps / self._stable_steps_required, 1.0
        )
        self.reward = (
            1.0
            if self._arena_pp_box_success
            else 0.10 * float(self._was_lifted)
            + 0.25 * float(placement["inside_xy"])
            + 0.25 * float(placement["aligned_z"])
            + 0.30 * stable_progress
        )
        self._arena_pp_box_debug = {
            **placement,
            "was_lifted": self._was_lifted,
            "stable_steps": self._arena_pp_box_stable_steps,
            "success": self._arena_pp_box_success,
            "reward": float(self.reward),
        }
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._arena_pp_box_success)
