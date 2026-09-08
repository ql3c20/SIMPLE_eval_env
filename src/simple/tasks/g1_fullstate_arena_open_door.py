"""MuJoCo evaluation task aligned with HumanoidArena open-door demos."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simple.core.layout import Layout
from simple.core.types import Pose
from simple.sensors import CameraCfg
from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_fullstate_arena_open_door")
class G1FullstateArenaOpenDoor(G1Fullstate20260615Task1):
    """Depress the handle, unlatch the door, and open the leaf."""

    uid = "g1_fullstate_arena_open_door"
    label = "G1 Fullstate Arena Open Door"
    description = "Press the door handle down and open the door."

    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "physics_dt": 0.005,
        "render_hz": 50,
        "max_episode_steps": 1800,
        "isaac_read_only_mirror": True,
        "video_output_layout": "classified_flat",
        "video_camera_keys": ("front_camera",),
    }

    _simple_root = Path(__file__).resolve().parents[3]
    _asset_root = _simple_root / "data/assets/humanoidarena/local_mjcf/arena_open_door"
    _arena_repo_root = Path(
        os.environ.get(
            "HUMANOID_ARENA_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena/isaaclab_twist2_g1",
        )
    )
    _raw_data_root = Path(
        os.environ.get(
            "ARENA_OPEN_DOOR_RAW_DIR",
            (
                "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/"
                "HumanoidArena_open_door/HSI_open_door"
            ),
        )
    )

    sensor_cfgs = {
        "front_camera": CameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=640,
            height=480,
            focal_length=7.6,
            fov=np.deg2rad(
                float(
                    os.environ.get(
                        "ARENA_OPEN_DOOR_CAMERA_HFOV_DEG", "105.53033203685067"
                    )
                )
            ),
            near=0.1,
            far=1.0e5,
            pose={
                "position": [0.0, 0.0, 0.0],
                "quaternion": [0.5, 0.5, -0.5, -0.5],
            },
        )
    }
    # The quaternion above is already the authored USD/OpenGL transform used
    # by the corrected Arena football migration.  Without this opt-out Isaac
    # interprets it as world-camera axes and converts it a second time.
    isaac_camera_axes = "usd"
    mujoco_native_head_camera = True

    _default_root_pos = np.asarray([-1.6, 0.2, 0.8], dtype=np.float64)
    _default_root_quat = np.asarray([0.70711, 0.0, 0.0, 0.70711], dtype=np.float64)
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
    _default_door_pos = np.asarray([-1.614, 2.314, 0.002], dtype=np.float64)
    _default_door_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    _door_root_body = "arena_open_door_door_root"
    _leaf_body = "arena_open_door_leaf"
    _handle_body = "arena_open_door_handle"
    _leaf_joint = "arena_open_door_leaf_hinge"
    _handle_joint = "arena_open_door_handle_hinge"

    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "door.xml",
            "prefix": "arena_open_door_",
            "pos": _default_door_pos.tolist(),
            "quat": _default_door_quat.tolist(),
        }
    ]
    mujoco_object_body_names = {
        "target": _leaf_body,
        "door_root": _door_root_body,
        "door_handle": _handle_body,
    }
    mujoco_initial_robot_qpos = np.concatenate(
        [_default_root_pos, _default_root_quat, _default_body_qpos29, _default_hand_qpos14]
    ).tolist()
    mujoco_ground_friction = [1.0, 0.005, 0.0001]
    mujoco_ground_visual = {
        "builtin": "mjBUILTIN_FLAT",
        "rgb1": [0.52, 0.52, 0.50],
        "rgb2": [0.52, 0.52, 0.50],
        "mark": "mjMARK_EDGE",
        "markrgb": [0.52, 0.52, 0.50],
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
    def _normalize_quat(value: Any, *, source: str) -> np.ndarray:
        quat = np.asarray(value, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(quat).all() or norm <= 1.0e-8:
            raise ValueError(f"Invalid quaternion from {source}: {quat}")
        return quat / norm

    @staticmethod
    def _parse_range(name: str, default: str) -> np.ndarray:
        values = np.fromstring(os.environ.get(name, default), sep=",", dtype=np.float64)
        if values.size != 2 or not np.isfinite(values).all() or values[0] > values[1]:
            raise ValueError(f"{name} must be a comma-separated min,max pair")
        return values

    @classmethod
    def _robot_qpos_from_arena(
        cls, root_pos: Any, root_quat: Any, body_qpos29: Any
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(root_pos, dtype=np.float64).reshape(3),
                cls._normalize_quat(root_quat, source="Arena robot root"),
                np.asarray(body_qpos29, dtype=np.float64).reshape(29),
                cls._default_hand_qpos14,
            ]
        )

    @classmethod
    def _recordings_dir(cls) -> Path:
        root = Path(os.environ.get("ARENA_OPEN_DOOR_RAW_DIR", str(cls._raw_data_root)))
        branch = os.environ.get("ARENA_OPEN_DOOR_RAW_BRANCH", "twist2/zz").strip()
        candidate = root / branch if branch else root
        return candidate if candidate.exists() else root

    @classmethod
    def _load_recording_initializations(
        cls, recordings_dir: Path
    ) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        initializations = []
        for data_path in sorted(recordings_dir.rglob("*.npz")):
            with np.load(data_path, allow_pickle=False) as data:
                robot_qpos = cls._robot_qpos_from_arena(
                    data["robot_root_position"][0],
                    data["robot_root_orientation"][0],
                    data["robot_qpos_before_decimation"][0],
                )
                position_key = (
                    "episode_init_env_obj_door_position"
                    if "episode_init_env_obj_door_position" in data
                    else "env_obj_door_position"
                )
                orientation_key = (
                    "episode_init_env_obj_door_orientation"
                    if "episode_init_env_obj_door_orientation" in data
                    else "env_obj_door_orientation"
                )
                door_pos = np.asarray(data[position_key], dtype=np.float64)
                door_quat = np.asarray(data[orientation_key], dtype=np.float64)
                if door_pos.ndim > 1:
                    door_pos = door_pos[0]
                if door_quat.ndim > 1:
                    door_quat = door_quat[0]
                door_quat = cls._normalize_quat(door_quat, source=str(data_path))
            if robot_qpos.size != 50 or not np.isfinite(robot_qpos).all():
                raise ValueError(f"Invalid robot initialization in {data_path}")
            initializations.append(
                (data_path.name, robot_qpos, door_pos.reshape(3), door_quat)
            )
        if not initializations:
            raise FileNotFoundError(
                f"No Arena open-door *.npz recordings found in {recordings_dir}"
            )
        return initializations

    @staticmethod
    def _select_recording_index(
        pool_size: int, reset_index: int, seed: int, fixed_index: int | None
    ) -> int:
        if fixed_index is not None:
            if not 0 <= fixed_index < pool_size:
                raise ValueError(
                    f"ARENA_OPEN_DOOR_RECORDING_INDEX={fixed_index} is outside 0..{pool_size - 1}"
                )
            return fixed_index
        cycle, offset = divmod(reset_index, pool_size)
        return int(np.random.default_rng(seed + cycle).permutation(pool_size)[offset])

    @classmethod
    def _arena_asset_paths(cls) -> dict[str, Path]:
        override = os.environ.get("HUMANOID_ARENA_ISAAC_ASSET_ROOT")
        candidates = [Path(override)] if override else []
        candidates.extend([cls._arena_repo_root / "assets", cls._arena_repo_root / "assets1"])
        relative_scene = Path("objects/small_warehouse/small_warehouse_opendoor")
        for root in candidates:
            scene = root / relative_scene
            if scene.exists():
                return {
                    "room": scene / "small_warehouse_digital_twin_opendoor.usd",
                    "door": scene / "interaction_obj/door001/model_door001_vali.usd",
                }
        scene = candidates[0] / relative_scene
        return {
            "room": scene / "small_warehouse_digital_twin_opendoor.usd",
            "door": scene / "interaction_obj/door001/model_door001_vali.usd",
        }

    def isaac_extra_usd_references(self) -> list[dict[str, Any]]:
        if not self._env_flag("ARENA_OPEN_DOOR_USE_ARENA_USD_VISUALS", True):
            return []
        paths = self._arena_asset_paths()
        required = self._env_flag("ARENA_OPEN_DOOR_REQUIRE_ARENA_USD_VISUALS", False)
        door_pos = getattr(self, "_arena_open_door_start_pos", self._default_door_pos)
        door_quat = getattr(self, "_arena_open_door_start_quat", self._default_door_quat)
        return [
            {
                "name": "arena_open_door_room_usd",
                "usd_path": str(paths["room"]),
                "prim_path": "/World/workspace/ArenaOpenDoorRoom",
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "disable_collision": True,
                "required": required,
            },
            {
                "name": "arena_open_door_door_usd",
                "usd_path": str(paths["door"]),
                "prim_path": "/World/workspace/ArenaOpenDoor",
                "pos": np.asarray(door_pos, dtype=np.float64).tolist(),
                "quat": np.asarray(door_quat, dtype=np.float64).tolist(),
                "sync_body": self._door_root_body,
                "sync_subprims": {
                    "E_leaf_2": self._leaf_body,
                    "E_handle_4": self._handle_body,
                },
                "disable_collision": True,
                "required": required,
            },
        ]

    def preload_objects(self) -> list[Any]:
        return []

    def reset(
        self, seed: int | None = None, options: Optional[dict[str, Any]] = None
    ) -> None:
        self._layout = Layout()
        self._layout.add_robot(self.robot)
        self._layout.robot.pose = Pose(
            position=[0.0, 0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0]
        )

        reset_index = int(getattr(self, "_arena_open_door_reset_index", 0))
        self._arena_open_door_reset_index = reset_index + 1
        self.mujoco_initial_robot_qpos = list(type(self).mujoco_initial_robot_qpos)
        door_pos = self._default_door_pos.copy()
        door_quat = self._default_door_quat.copy()
        source = "default"

        profile = os.environ.get(
            "ARENA_OPEN_DOOR_EVAL_PROFILE", "recording"
        ).strip().lower()
        use_data = self._env_flag("ARENA_OPEN_DOOR_INIT_FROM_DATA", profile == "recording")
        if use_data:
            if not hasattr(self, "_arena_open_door_initializations"):
                recordings_dir = self._recordings_dir()
                self._arena_open_door_initializations = self._load_recording_initializations(
                    recordings_dir
                )
                self._arena_open_door_recording_seed = int(
                    os.environ.get("ARENA_OPEN_DOOR_RECORDING_SEED", "0")
                )
                print(
                    "[ArenaOpenDoorData] "
                    f"recordings={len(self._arena_open_door_initializations)} "
                    f"dir={recordings_dir}"
                )
            fixed_raw = os.environ.get("ARENA_OPEN_DOOR_RECORDING_INDEX")
            fixed = int(fixed_raw) if fixed_raw is not None else None
            selected = self._select_recording_index(
                len(self._arena_open_door_initializations),
                reset_index,
                self._arena_open_door_recording_seed,
                fixed,
            )
            name, robot_qpos, door_pos, door_quat = self._arena_open_door_initializations[
                selected
            ]
            self.mujoco_initial_robot_qpos = robot_qpos.tolist()
            door_pos = door_pos.copy()
            door_quat = door_quat.copy()
            source = f"recording:{name}"
        elif profile in {"random", "benchmark_random"}:
            object_seed = int(
                os.environ.get("ARENA_OPEN_DOOR_OBJECT_SEED", str(seed or 0))
            )
            rng = np.random.default_rng(object_seed + reset_index)
            x_range = self._parse_range("ARENA_OPEN_DOOR_X_RANGE", "-2.368,-1.064")
            y_range = self._parse_range("ARENA_OPEN_DOOR_Y_RANGE", "1.514,2.414")
            door_pos[0] = rng.uniform(*x_range)
            door_pos[1] = rng.uniform(*y_range)
            source = f"benchmark_random:seed{object_seed}"

        if options:
            if "door_pos" in options:
                door_pos = np.asarray(options["door_pos"], dtype=np.float64).reshape(3)
                source = "reset_options"
            if "door_quat" in options:
                door_quat = self._normalize_quat(
                    options["door_quat"], source="reset options door"
                )

        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[0]["pos"] = door_pos.tolist()
        self.mujoco_extra_mjcf[0]["quat"] = door_quat.tolist()
        self._arena_open_door_start_pos = door_pos.copy()
        self._arena_open_door_start_quat = door_quat.copy()

        self._layout.add_camera(
            "front_camera", copy.deepcopy(self.sensor_cfgs["front_camera"])
        )
        instruction = "Press the door handle down and open the door."
        if self._env_flag("ARENA_OPEN_DOOR_ALLOW_INSTRUCTION_OVERRIDE", False):
            instruction = os.environ.get("ARENA_OPEN_DOOR_INSTRUCTION", instruction)
        self._instruction = instruction
        self._target = None
        self.reward = 0.0
        self._arena_open_door_latch_unlocked = False
        self._arena_open_door_success = False
        self._arena_open_door_model_id = None

        if self._env_flag("ARENA_OPEN_DOOR_LOG_RESETS", True):
            print(
                "[ArenaOpenDoorInit] "
                f"episode_reset={reset_index} source={source} "
                f"door_xyz=({door_pos[0]:.4f}, {door_pos[1]:.4f}, {door_pos[2]:.4f})"
            )
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    @staticmethod
    def _clipped_drive_torque(
        position: float,
        velocity: float,
        stiffness: float,
        damping: float,
        max_force: float,
    ) -> float:
        return float(np.clip(-stiffness * position - damping * velocity, -max_force, max_force))

    def _ensure_door_indices(self, mujoco_env: Any) -> None:
        model = mujoco_env.mjModel
        if self._arena_open_door_model_id == id(model):
            return
        import mujoco

        leaf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._leaf_joint)
        handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._handle_joint)
        if leaf_id < 0 or handle_id < 0:
            raise RuntimeError("Arena open-door MuJoCo joints were not compiled")
        self._leaf_qpos_adr = int(model.jnt_qposadr[leaf_id])
        self._leaf_dof_adr = int(model.jnt_dofadr[leaf_id])
        self._handle_qpos_adr = int(model.jnt_qposadr[handle_id])
        self._handle_dof_adr = int(model.jnt_dofadr[handle_id])
        self._arena_open_door_model_id = id(model)

    def _update_latch(self, handle_angle: float) -> bool:
        threshold = np.deg2rad(
            float(os.environ.get("OPEN_DOOR_HANDLE_UNLOCK_ANGLE_DEG", "-20"))
        )
        if not self._arena_open_door_latch_unlocked and handle_angle <= threshold:
            self._arena_open_door_latch_unlocked = True
            if self._env_flag("OPEN_DOOR_LATCH_DEBUG", False):
                print(
                    "[ArenaOpenDoorLatch] unlocked "
                    f"handle_deg={np.rad2deg(handle_angle):.3f}"
                )
        return self._arena_open_door_latch_unlocked

    def before_mujoco_step(self, mujoco_env: Any) -> None:
        self._ensure_door_indices(mujoco_env)
        data = mujoco_env.mjData
        handle_q = float(data.qpos[self._handle_qpos_adr])
        self._update_latch(handle_q)

        data.qfrc_applied[self._handle_dof_adr] = self._clipped_drive_torque(
            handle_q,
            float(data.qvel[self._handle_dof_adr]),
            float(os.environ.get("OPEN_DOOR_HANDLE_STIFFNESS", "5")),
            float(os.environ.get("OPEN_DOOR_HANDLE_DAMPING", "0.1")),
            float(os.environ.get("OPEN_DOOR_HANDLE_MAX_FORCE", "0.001")),
        )
        if not self._arena_open_door_latch_unlocked:
            data.qpos[self._leaf_qpos_adr] = 0.0
            data.qvel[self._leaf_dof_adr] = 0.0
            data.qfrc_applied[self._leaf_dof_adr] = 0.0
        else:
            # Match HumanoidArena's runtime latch behavior.  The native task
            # temporarily applies a very stiff lock drive, then restores the
            # leaf to a frictionless, undriven hinge after the handle crosses
            # its unlock threshold.
            data.qfrc_applied[self._leaf_dof_adr] = self._clipped_drive_torque(
                float(data.qpos[self._leaf_qpos_adr]),
                float(data.qvel[self._leaf_dof_adr]),
                float(os.environ.get("OPEN_DOOR_LEAF_UNLOCK_STIFFNESS", "0")),
                float(os.environ.get("OPEN_DOOR_LEAF_UNLOCK_DAMPING", "0")),
                float(os.environ.get("OPEN_DOOR_LEAF_UNLOCK_MAX_FORCE", "0")),
            )

    def after_mujoco_step(self, mujoco_env: Any) -> None:
        self._ensure_door_indices(mujoco_env)
        data = mujoco_env.mjData
        self._update_latch(float(data.qpos[self._handle_qpos_adr]))
        if not self._arena_open_door_latch_unlocked:
            data.qpos[self._leaf_qpos_adr] = 0.0
            data.qvel[self._leaf_dof_adr] = 0.0

    @staticmethod
    def _root_up_axis_z(root_quat_wxyz: np.ndarray) -> float:
        quat = G1FullstateArenaOpenDoor._normalize_quat(
            root_quat_wxyz, source="MuJoCo robot root"
        )
        _, x, y, _ = quat
        return float(1.0 - 2.0 * (x * x + y * y))

    def _read_success_state(self, mujoco_env: Any) -> dict[str, Any]:
        self._ensure_door_indices(mujoco_env)
        data = mujoco_env.mjData
        leaf_angle = float(data.qpos[self._leaf_qpos_adr])
        standing = bool(
            float(data.qpos[2])
            >= float(os.environ.get("OPEN_DOOR_MIN_ROOT_HEIGHT", "0.45"))
            and self._root_up_axis_z(np.asarray(data.qpos[3:7], dtype=np.float64))
            >= float(os.environ.get("OPEN_DOOR_MIN_UP_AXIS_Z", "0.60"))
        )
        threshold = np.deg2rad(
            abs(float(os.environ.get("OPEN_DOOR_SUCCESS_LEAF_ANGLE_DEG", "60")))
        )
        return {
            "leaf_angle": leaf_angle,
            "handle_angle": float(data.qpos[self._handle_qpos_adr]),
            "standing": standing,
            "latch_unlocked": bool(self._arena_open_door_latch_unlocked),
            "leaf_open": abs(leaf_angle) >= threshold,
        }

    def compute_reward(
        self, info: dict[str, Any], *args: Any, mujoco_env: Any = None, **kwargs: Any
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Arena open-door reward requires the MuJoCo environment")
        state = self._read_success_state(mujoco_env)
        success = bool(
            state["standing"] and state["latch_unlocked"] and state["leaf_open"]
        )
        self._arena_open_door_success |= success
        leaf_progress = min(abs(state["leaf_angle"]) / np.deg2rad(60.0), 1.0)
        self.reward = (
            1.0
            if self._arena_open_door_success
            else 0.2 * float(state["latch_unlocked"]) + 0.6 * leaf_progress
        )
        self._arena_open_door_debug = {
            **state,
            "success": self._arena_open_door_success,
        }
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._arena_open_door_success)
