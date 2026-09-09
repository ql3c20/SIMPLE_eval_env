"""MuJoCo football evaluation task aligned with HumanoidArena football demos."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simple.core.layout import Layout
from simple.core.types import Pose
from simple.sensors import CameraCfg
from simple.tasks.arena_eval_camera import (
    add_arena_eval_cameras,
    configure_arena_eval_cameras,
)
from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1
from simple.tasks.registry import TaskRegistry


def _env_float_vector(name: str, default: list[float], size: int) -> list[float]:
    """Read a fixed-size, finite comma-separated vector from the environment."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    try:
        values = [float(item.strip()) for item in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated numbers; got {raw!r}") from exc
    if len(values) != size or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain {size} finite values; got {raw!r}")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} values must be non-negative; got {raw!r}")
    return values


def _contact_solref_from_env(scope: str) -> list[float] | None:
    """Convert a requested coefficient of restitution to MuJoCo solref.

    MuJoCo does not expose a geom-level restitution scalar.  For its standard
    positive solref format, the second value is a damping ratio.  The mapping
    below is the usual underdamped-contact approximation; restitution=0 uses
    critical damping. Direct SOLREF overrides remain available for exact
    MuJoCo ablations.
    """
    prefix = f"ARENA_FOOTBALL_{scope}_"
    solref_name = f"{prefix}SOLREF"
    restitution_name = f"{prefix}RESTITUTION"
    time_constant_name = f"{prefix}CONTACT_TIME_CONSTANT"
    direct = os.environ.get(solref_name)
    restitution = os.environ.get(restitution_name)
    if direct and restitution:
        raise ValueError(
            f"Set only one of {solref_name} and {restitution_name}"
        )
    if direct:
        return _env_float_vector(solref_name, [0.02, 1.0], 2)
    if restitution is None or not restitution.strip():
        return None
    try:
        coefficient = float(restitution)
        time_constant = float(os.environ.get(time_constant_name, "0.02"))
    except ValueError as exc:
        raise ValueError(
            f"{restitution_name} and {time_constant_name} must be numbers"
        ) from exc
    if not math.isfinite(coefficient) or not 0.0 <= coefficient < 1.0:
        raise ValueError(
            f"{restitution_name} must be in [0, 1); "
            f"got {restitution!r}"
        )
    if not math.isfinite(time_constant) or time_constant <= 0.0:
        raise ValueError(f"{time_constant_name} must be positive")
    if coefficient == 0.0:
        damping_ratio = 1.0
    else:
        log_e = math.log(coefficient)
        damping_ratio = -log_e / math.sqrt(math.pi**2 + log_e**2)
    return [time_constant, damping_ratio]


@TaskRegistry.register("g1_fullstate_arena_football")
class G1FullstateArenaFootball(G1Fullstate20260615Task1):
    """Kick the football across the Arena single-goal scoring line."""

    uid = "g1_fullstate_arena_football"
    label = "G1 Fullstate Arena Football"
    description = "Kick the football into the goal."

    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 1000,
        # MuJoCo is the sole physics owner.  Isaac only renders transforms
        # copied from the current MuJoCo step and must not advance a second
        # physics clock after synchronization.
        "isaac_read_only_mirror": True,
        # Keep football recordings compact and directly comparable to Arena:
        # one policy camera, classified into success/failure directories.
        "video_output_layout": "classified_flat",
        "video_camera_keys": ("front_camera",),
    }

    _simple_root = Path(__file__).resolve().parents[3]
    _asset_root = _simple_root / "data/assets/humanoidarena/local_mjcf/arena_football"
    _arena_repo_root = Path(
        os.environ.get(
            "HUMANOID_ARENA_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena/isaaclab_twist2_g1",
        )
    )
    _raw_data_root = Path(
        os.environ.get(
            "ARENA_FOOTBALL_RAW_DIR",
            (
                "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/"
                "HumanoidArena_football/HOI_football_v2"
            ),
        )
    )

    # Match HumanoidArena's IsaacLab g1_front_camera:
    # 640x480, focal_length=7.6, horizontal_aperture=20.0, mounted on d435_link
    # with ROS camera-frame offset (0.5, -0.5, 0.5, -0.5).
    sensor_cfgs = {
        "front_camera": CameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=640,
            height=480,
            focal_length=7.6,
            # Equivalent to Arena's horizontal_aperture=20.0 at focal=7.6.
            fov=np.deg2rad(
                float(
                    os.environ.get(
                        "ARENA_FOOTBALL_CAMERA_HFOV_DEG", "105.53033203685067"
                    )
                )
            ),
            near=0.1,
            far=1.0e5,
            pose={
                "position": [0.0, 0.0, 0.0],
                # Arena authors q=(0.5,-0.5,0.5,-0.5) in ROS camera axes.
                # IsaacLab converts it to this USD/OpenGL wxyz quaternion.
                "quaternion": [0.5, 0.5, -0.5, -0.5],
            },
        )
    }
    # The quaternion above is the authored USD/OpenGL transform obtained by
    # converting HumanoidArena's ROS camera offset.  Prevent Isaac Sim's
    # Camera API from interpreting and converting it as world-camera axes.
    isaac_camera_axes = "usd"
    mujoco_native_head_camera = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        configure_arena_eval_cameras(self)

    # HumanoidArena single-goal football geometry:
    # robot root [0, -2, 0.8], yaw +90 deg; goal origin [-2.5, 3.0].
    _default_root_pos = np.asarray([0.0, -2.0, 0.8], dtype=np.float64)
    _default_root_quat = np.asarray([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)
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
    # SONIC records robot_qpos_before_decimation in the interleaved Isaac Lab
    # controller order.  SIMPLE's MuJoCo model uses grouped left-leg,
    # right-leg, waist, left-arm, right-arm order.
    _sonic_to_mujoco_body29 = np.asarray(
        [
            0, 3, 6, 9, 13, 17,
            1, 4, 7, 10, 14, 18,
            2, 5, 8,
            11, 15, 19, 21, 23, 25, 27,
            12, 16, 20, 22, 24, 26, 28,
        ],
        dtype=np.int64,
    )
    _default_ball_pos = np.asarray(
        [0.31992957, 0.78888839, 0.11],
        dtype=np.float64,
    )
    _default_ball_quat = [1.0, 0.0, 0.0, 0.0]

    # MuJoCo owns football physics in mujoco_isaac mode.  The Arena USD is a
    # read-only render proxy whose authored radius is 0.11145 m, while the
    # MuJoCo collision sphere is 0.11 m.  Scale the proxy to the collision
    # sphere and synchronize it by the MJCF body name exported by
    # MujocoSimulator.get_states().
    _ball_body_name = "arena_football_ball_object"
    _ball_collision_radius = 0.11
    _ball_usd_authored_radius = 0.11145
    _ball_usd_scale = _ball_collision_radius / _ball_usd_authored_radius

    # Copied from HumanoidArena football reward geometry.
    _goal_origin_x = -2.5
    _goal_origin_y = 3.0
    _goal_z = 0.65
    _goal_mouth_x_min_local = 0.0
    _goal_mouth_x_max_local = 5.1
    _goal_line_width_m = 0.12 * 0.5
    _goal_height_z_min = 0.0
    _goal_height_z_max = 1.85
    _field_line_color = [1.0, 1.0, 1.0]
    _arena_backdrop_color = [0.35, 0.35, 0.35]

    mujoco_scene_mjcf_path = _asset_root / "football_field.xml"
    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "soccer_ball.xml",
            "prefix": "arena_football_ball_",
            "pos": _default_ball_pos.tolist(),
            "quat": _default_ball_quat,
            "freejoint_body": "object",
            "freejoint_name": "football_free",
        },
        {
            "path": _asset_root / "football_goal.xml",
            "prefix": "arena_football_goal_",
            "pos": [_goal_origin_x, _goal_origin_y, 0.0],
        },
    ]
    mujoco_object_body_names = {
        "target": _ball_body_name,
        "goal": "arena_football_goal_goal",
    }
    mujoco_hidden_sites = ("com_marker",)
    # Per-process physics overrides. MuJoCo friction is ordered as sliding,
    # torsional, rolling. Defaults preserve all existing evaluation behavior.
    mujoco_ground_friction = _env_float_vector(
        "ARENA_FOOTBALL_GROUND_FRICTION", [0.7, 0.005, 0.0001], 3
    )
    mujoco_ground_solref = _contact_solref_from_env("GROUND")
    # The G1 MJCF assigns friction=1.0 to every collision geom. MuJoCo uses
    # the larger coefficient from an equal-priority contact pair, so lowering
    # only the ground cannot lower foot-ground friction. Override precisely
    # the eight sole contact spheres under the two ankle-roll bodies.
    mujoco_foot_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    mujoco_expected_foot_collision_geoms = 8
    mujoco_foot_friction = (
        _env_float_vector(
            "ARENA_FOOTBALL_FOOT_FRICTION", [1.0, 0.005, 0.0001], 3
        )
        if os.environ.get("ARENA_FOOTBALL_FOOT_FRICTION")
        else None
    )
    mujoco_foot_solref = _contact_solref_from_env("FOOT")
    mujoco_ground_visual = {
        "builtin": "mjBUILTIN_FLAT",
        "rgb1": [0.23, 0.47, 0.25],
        "rgb2": [0.23, 0.47, 0.25],
        "mark": "mjMARK_EDGE",
        "markrgb": [0.23, 0.47, 0.25],
        "width": 128,
        "height": 128,
        "texrepeat": [1.0, 1.0],
        "reflectance": 0.0,
    }

    mujoco_initial_robot_qpos = np.concatenate(
        [
            _default_root_pos,
            _default_root_quat,
            _default_body_qpos29,
            _default_hand_qpos14,
        ]
    ).tolist()

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _yaw_quat(yaw: float) -> list[float]:
        return [float(np.cos(yaw * 0.5)), 0.0, 0.0, float(np.sin(yaw * 0.5))]

    @classmethod
    def _arena_isaac_asset_root(cls) -> Path:
        raw = os.environ.get("HUMANOID_ARENA_ISAAC_ASSET_ROOT") or os.environ.get(
            "ARENA_FOOTBALL_ISAAC_ASSET_ROOT"
        )
        if raw:
            return Path(raw)

        return cls._arena_repo_root / "assets"

    @classmethod
    def _arena_isaac_asset_paths(cls) -> dict[str, Path]:
        asset_root = cls._arena_isaac_asset_root()
        return {
            "asset_root": asset_root,
            "ball_usd": (
                asset_root
                / "objects/small_warehouse/football_scene/interaction_obj/"
                "soccer_ball/soccer_ball_physics.usd"
            ),
            "goal_usd": (
                asset_root
                / "objects/small_warehouse/football_scene/interaction_obj/"
                "football_goal/football_goal_physics.usd"
            ),
            "grass_textures": asset_root / "objects/materials/grass_turf",
            "grass_tool": cls._arena_repo_root / "tools/grass_ground_material.py",
        }

    @classmethod
    def _line_primitive(
        cls,
        name: str,
        start: tuple[float, float],
        end: tuple[float, float],
        z: float,
        width: float,
        height: float,
        color: list[float],
    ) -> dict[str, Any]:
        start_xy = np.asarray(start, dtype=np.float64)
        end_xy = np.asarray(end, dtype=np.float64)
        center = 0.5 * (start_xy + end_xy)
        delta = end_xy - start_xy
        length = float(np.linalg.norm(delta))
        yaw = float(np.arctan2(delta[1], delta[0]))
        return {
            "name": name,
            "type": "cube",
            "pos": [float(center[0]), float(center[1]), z],
            "quat": cls._yaw_quat(yaw),
            "scale": [length, width, height],
            "color": color,
        }

    @classmethod
    def _fallback_goal_primitives(cls) -> list[dict[str, Any]]:
        x0 = cls._goal_origin_x
        y0 = cls._goal_origin_y
        white = [0.92, 0.92, 0.88]
        net = [0.72, 0.78, 0.84, 0.30]
        return [
            {
                "name": "arena_football_goal_left_post_fallback",
                "type": "cube",
                "pos": [x0, y0, 0.925],
                "scale": [0.09, 0.09, 1.85],
                "color": white,
            },
            {
                "name": "arena_football_goal_right_post_fallback",
                "type": "cube",
                "pos": [x0 + 5.1, y0, 0.925],
                "scale": [0.09, 0.09, 1.85],
                "color": white,
            },
            {
                "name": "arena_football_goal_crossbar_fallback",
                "type": "cube",
                "pos": [x0 + 2.55, y0, 1.85],
                "scale": [5.1, 0.09, 0.09],
                "color": white,
            },
            {
                "name": "arena_football_goal_back_bar_fallback",
                "type": "cube",
                "pos": [x0 + 2.55, y0 + 0.65, 1.60],
                "scale": [5.1, 0.064, 0.064],
                "color": white,
            },
            {
                "name": "arena_football_goal_net_back_fallback",
                "type": "cube",
                "pos": [x0 + 2.55, y0 + 0.66, 0.8],
                "scale": [5.1, 0.02, 1.6],
                "color": net,
            },
            {
                "name": "arena_football_goal_net_left_fallback",
                "type": "cube",
                "pos": [x0, y0 + 0.325, 0.8],
                "scale": [0.02, 0.65, 1.6],
                "color": net,
            },
            {
                "name": "arena_football_goal_net_right_fallback",
                "type": "cube",
                "pos": [x0 + 5.1, y0 + 0.325, 0.8],
                "scale": [0.02, 0.65, 1.6],
                "color": net,
            },
        ]

    def isaac_extra_usd_references(self) -> list[dict[str, Any]]:
        if not self._env_flag("ARENA_FOOTBALL_USE_ARENA_USD_VISUALS", True):
            return []

        paths = self._arena_isaac_asset_paths()
        required = self._env_flag("ARENA_FOOTBALL_REQUIRE_ARENA_USD_VISUALS", False)
        ball_pos = getattr(self, "_arena_football_start_ball_pos", self._default_ball_pos)
        return [
            {
                "name": "arena_football_ball_usd",
                "usd_path": str(paths["ball_usd"]),
                "prim_path": "/World/workspace/ArenaFootballBall",
                "pos": np.asarray(ball_pos, dtype=np.float64).tolist(),
                "quat": list(self._default_ball_quat),
                "scale": [self._ball_usd_scale] * 3,
                "sync_body": self._ball_body_name,
                "disable_collision": True,
                "required": required,
            },
            {
                "name": "arena_football_goal_usd",
                "usd_path": str(paths["goal_usd"]),
                "prim_path": "/World/workspace/ArenaFootballGoal",
                "pos": [self._goal_origin_x, self._goal_origin_y, self._goal_z],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "disable_collision": True,
                "required": required,
            },
        ]

    def isaac_extra_primitives(self) -> list[dict[str, Any]]:
        if self._env_flag("ARENA_FOOTBALL_DISABLE_ISAAC_PRIMITIVES", False):
            return []

        paths = self._arena_isaac_asset_paths()
        use_usd = self._env_flag("ARENA_FOOTBALL_USE_ARENA_USD_VISUALS", True)
        use_grass_pbr = self._env_flag("ARENA_FOOTBALL_USE_GRASS_PBR", True)
        prims: list[dict[str, Any]] = [
            {
                "name": "arena_football_dome_light",
                "type": "dome_light",
                "pos": [0.0, 0.0, 0.0],
                "color": [0.75, 0.75, 0.75],
                "intensity": 3000.0,
            },
            {
                "name": "arena_football_ground",
                "type": "cube",
                "pos": [0.0, 0.0, -0.005],
                "scale": [14.0, 14.0, 0.01],
                "color": [0.52, 0.78, 0.20],
            },
            {
                "name": "arena_football_backdrop_front",
                "type": "cube",
                "pos": [0.0, 6.0, 1.3],
                "scale": [13.68, 0.08, 2.6],
                "color": list(self._arena_backdrop_color),
            },
            {
                "name": "arena_football_backdrop_back",
                "type": "cube",
                "pos": [0.0, -6.0, 1.3],
                "scale": [13.68, 0.08, 2.6],
                "color": list(self._arena_backdrop_color),
            },
            {
                "name": "arena_football_backdrop_left",
                "type": "cube",
                "pos": [-6.8, 0.0, 1.3],
                "scale": [0.08, 12.0, 2.6],
                "color": list(self._arena_backdrop_color),
            },
            {
                "name": "arena_football_backdrop_right",
                "type": "cube",
                "pos": [6.8, 0.0, 1.3],
                "scale": [0.08, 12.0, 2.6],
                "color": list(self._arena_backdrop_color),
            },
        ]

        if (
            use_grass_pbr
            and paths["grass_textures"].exists()
            and paths["grass_tool"].exists()
        ):
            for prim in prims:
                if prim["name"] == "arena_football_ground":
                    prim["grass_textures_dir"] = str(paths["grass_textures"])
                    prim["grass_material_tool"] = str(paths["grass_tool"])
                    prim["uv_scale"] = [15.0, 15.0]
                    break

        radius = 1.5
        angles = np.linspace(0.0, 2.0 * np.pi, 65)
        for idx in range(64):
            start = (float(radius * np.cos(angles[idx])), float(radius * np.sin(angles[idx])))
            end = (
                float(radius * np.cos(angles[idx + 1])),
                float(radius * np.sin(angles[idx + 1])),
            )
            prims.append(
                self._line_primitive(
                    f"arena_football_center_circle_{idx:02d}",
                    start,
                    end,
                    z=0.006,
                    width=0.12,
                    height=0.005,
                    color=list(self._field_line_color),
                )
            )

        if not use_usd or not paths["ball_usd"].exists():
            ball_pos = getattr(self, "_arena_football_start_ball_pos", self._default_ball_pos)
            prims.append(
                {
                    "name": "arena_football_ball_fallback",
                    "type": "sphere",
                    "pos": np.asarray(ball_pos, dtype=np.float64).tolist(),
                    "radius": self._ball_collision_radius,
                    "color": [0.92, 0.92, 0.86],
                    "sync_body": self._ball_body_name,
                }
            )

        if not use_usd or not paths["goal_usd"].exists():
            prims.extend(self._fallback_goal_primitives())

        return prims

    def preload_objects(self) -> list[Any]:
        return []

    @staticmethod
    def _parse_range(name: str, default: str) -> np.ndarray:
        values = np.fromstring(os.environ.get(name, default), sep=",", dtype=np.float64)
        if values.size != 2 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a comma-separated min,max pair")
        return values

    @classmethod
    def _robot_qpos_from_arena(
        cls,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        body_qpos29: np.ndarray,
    ) -> np.ndarray:
        root_quat = np.asarray(root_quat, dtype=np.float64).reshape(4)
        quat_norm = np.linalg.norm(root_quat)
        if quat_norm == 0.0:
            raise ValueError("Arena football root quaternion has zero norm")
        body_qpos29 = np.asarray(body_qpos29, dtype=np.float64).reshape(29)
        body_qpos29 = body_qpos29[cls._sonic_to_mujoco_body29]
        return np.concatenate(
            [
                np.asarray(root_pos, dtype=np.float64).reshape(3),
                root_quat / quat_norm,
                body_qpos29,
                cls._default_hand_qpos14,
            ]
        )

    @classmethod
    def _load_recording_initializations(
        cls,
        recordings_dir: Path,
    ) -> list[tuple[str, np.ndarray, np.ndarray, list[float]]]:
        initializations = []
        for data_path in sorted(recordings_dir.rglob("*.npz")):
            with np.load(data_path, allow_pickle=False) as data:
                robot_qpos = cls._robot_qpos_from_arena(
                    data["robot_root_position"][0],
                    data["robot_root_orientation"][0],
                    data["robot_qpos_before_decimation"][0],
                )
                if "episode_init_env_obj_football_position" in data:
                    ball_pos = np.asarray(
                        data["episode_init_env_obj_football_position"], dtype=np.float64
                    ).reshape(3)
                    ball_quat = np.asarray(
                        data["episode_init_env_obj_football_orientation"], dtype=np.float64
                    ).reshape(4)
                else:
                    ball_pos = np.asarray(
                        data["env_obj_football_position"][0], dtype=np.float64
                    )
                    ball_quat = np.asarray(
                        data["env_obj_football_orientation"][0], dtype=np.float64
                    )
            if robot_qpos.size != 50:
                raise ValueError(f"Expected 50D SIMPLE robot qpos from {data_path}")
            if not np.isfinite(robot_qpos).all() or not np.isfinite(ball_pos).all():
                raise ValueError(f"NaN/Inf Arena football initialization in {data_path}")
            initializations.append((data_path.name, robot_qpos, ball_pos, ball_quat.tolist()))

        if not initializations:
            raise FileNotFoundError(f"No Arena football *.npz recordings found in {recordings_dir}")
        return initializations

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

        reset_index = int(getattr(self, "_arena_football_reset_index", 0))
        self._arena_football_reset_index = reset_index + 1
        self.mujoco_initial_robot_qpos = list(type(self).mujoco_initial_robot_qpos)
        ball_pos = self._default_ball_pos.copy()
        ball_quat = list(self._default_ball_quat)
        source = "default_first_recording"

        if os.environ.get("ARENA_FOOTBALL_INIT_FROM_DATA", "0") == "1":
            recordings_dir = Path(os.environ.get("ARENA_FOOTBALL_RAW_DIR", str(self._raw_data_root)))
            if not hasattr(self, "_arena_football_initializations"):
                self._arena_football_initializations = self._load_recording_initializations(recordings_dir)
                self._arena_football_recording_seed = int(
                    os.environ.get("ARENA_FOOTBALL_RECORDING_SEED", "0")
                )

            pool_size = len(self._arena_football_initializations)
            recording_index = os.environ.get("ARENA_FOOTBALL_RECORDING_INDEX")
            if recording_index is not None:
                selected_index = int(recording_index)
                if not 0 <= selected_index < pool_size:
                    raise ValueError(
                        f"ARENA_FOOTBALL_RECORDING_INDEX={selected_index} is outside "
                        f"the available range 0..{pool_size - 1}"
                    )
            else:
                cycle_index, cycle_offset = divmod(reset_index, pool_size)
                recording_order = np.random.default_rng(
                    self._arena_football_recording_seed + cycle_index
                ).permutation(pool_size)
                selected_index = int(recording_order[cycle_offset])

            recording_name, robot_qpos, ball_pos, ball_quat = (
                self._arena_football_initializations[selected_index]
            )
            self.mujoco_initial_robot_qpos = robot_qpos.tolist()
            source = f"recording:{recording_name}"

        elif os.environ.get("ARENA_FOOTBALL_RANDOMIZE_BALL", "0") == "1":
            rng = np.random.default_rng(
                int(os.environ.get("ARENA_FOOTBALL_OBJECT_SEED", "0")) + reset_index
            )
            x_range = self._parse_range("ARENA_FOOTBALL_BALL_X_RANGE", "-1.5,1.5")
            y_range = self._parse_range("ARENA_FOOTBALL_BALL_Y_RANGE", "-1.4,1.5")
            ball_pos[0] = rng.uniform(float(x_range[0]), float(x_range[1]))
            ball_pos[1] = rng.uniform(float(y_range[0]), float(y_range[1]))
            source = "range_random"

        if options is not None and "ball_pos" in options:
            ball_pos = np.asarray(options["ball_pos"], dtype=np.float64).reshape(3)
            source = "reset_options"

        ball_pos[0] += float(os.environ.get("ARENA_FOOTBALL_BALL_X_OFFSET", "0"))
        ball_pos[1] += float(os.environ.get("ARENA_FOOTBALL_BALL_Y_OFFSET", "0"))

        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[0]["pos"] = ball_pos.tolist()
        self.mujoco_extra_mjcf[0]["quat"] = ball_quat

        add_arena_eval_cameras(self)

        instruction = "Kick the football into the goal."
        if os.environ.get("ARENA_FOOTBALL_ALLOW_INSTRUCTION_OVERRIDE", "0") == "1":
            instruction = os.environ.get("ARENA_FOOTBALL_INSTRUCTION", instruction)
        self._instruction = instruction
        self._target = None
        self.reward = -1.0
        self._arena_football_start_ball_pos = ball_pos.copy()

        if os.environ.get("ARENA_FOOTBALL_LOG_RESETS", "1") == "1":
            print(
                "[ArenaFootballInit] "
                f"episode_reset={reset_index} source={source} "
                f"ball_xyz=({ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f})"
            )
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    @classmethod
    def _scored(cls, ball_pos: np.ndarray) -> bool:
        ball = np.asarray(ball_pos, dtype=np.float64).reshape(3)
        front_x_min = cls._goal_origin_x + cls._goal_mouth_x_min_local
        front_x_max = cls._goal_origin_x + cls._goal_mouth_x_max_local
        front_goal_inner_y = cls._goal_origin_y + cls._goal_line_width_m * 0.5
        return bool(
            front_x_min < ball[0] < front_x_max
            and ball[1] > front_goal_inner_y
            and cls._goal_height_z_min < ball[2] < cls._goal_height_z_max
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        ball_pos = np.asarray(info["target"], dtype=np.float64)[:3]
        self.reward = 1.0 if self._scored(ball_pos) else -1.0
        return self.reward

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self._scored(np.asarray(info["target"], dtype=np.float64)[:3])
