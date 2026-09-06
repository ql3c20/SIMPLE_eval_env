"""Bottle-into-pedal-bin evaluation task from the 2026-08-05 recordings."""

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


@TaskRegistry.register("g1_fullstate_20260805_task2")
class G1Fullstate20260805Task2(G1Fullstate20260615Task1):
    uid = "g1_fullstate_20260805_task2"
    label = "G1 Fullstate 20260805 Task 2"
    description = "Pick up the bottle on the table in front of me and throw it into the trash can."
    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 900,
        "render_hz": 50,
    }
    recording_env_prefix = "TASK2"
    recording_default_dir = Path("/home/ubuntu/yzh/mujoco_recordings/20260805_task2_new")
    recording_nq = 59
    recording_semantic_fields = {
        0: "pelvis.floating_base_joint.x",
        49: "right_hand_index_1_joint.angle",
        50: "custom_step_trash_can_lid_hinge_joint.angle",
        51: "custom_step_trash_can_pedal_hinge_joint.angle",
        52: "task_obj_0_001_bottle_13_free.x",
        58: "task_obj_0_001_bottle_13_free.qz",
    }
    sensor_cfgs = copy.deepcopy(G1Fullstate20260615Task1.sensor_cfgs)
    sensor_cfgs["head_stereo"].pose["position"] = [0.06, 0.06, 0.45]

    _humanoid_vla_root = Path(
        os.environ.get(
            "HUMANOID_VLA_MJ_ROOT",
            "/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ",
        )
    )
    _asset_root = _humanoid_vla_root / "mujoco/model/task_assets"
    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "2real_asset/round_table/round_table_visual_05pct.xml",
            "prefix": "round_table_2real_05pct_",
            "pos": [0.95, 0.0, 0.0],
        },
        {
            "path": _asset_root / "2real_asset/trash/step_trash_can.xml",
            "prefix": "custom_step_trash_can_",
            "pos": [0.6, -0.45, 0.0],
            "quat": [0.377629, 0.0, 0.0, -0.925957],
            "scale": 0.395257,
        },
        {
            "path": _asset_root / "local_mjcf/task4_bottle/bottle.xml",
            "prefix": "task_obj_0_001_bottle_13_",
            "pos": [0.78, -0.08, 0.755],
            "quat": [0.707107, 0.707107, 0.0, 0.0],
            "freejoint_body": "object",
            "freejoint_name": "free",
        },
    ]
    mujoco_object_body_names = {
        "target": "task_obj_0_001_bottle_13_object",
        "container": "custom_step_trash_can_bin",
    }
    _stable_linear_speed = 0.10
    _stable_angular_speed = 2.0

    @staticmethod
    def _trash_translate_from_env() -> np.ndarray:
        raw_value = os.environ.get("TASK2_TRASH_TRANSLATE", "0.0 0.0 0.0")
        fields = raw_value.split()
        if len(fields) != 3:
            raise ValueError(
                "TASK2_TRASH_TRANSLATE must contain exactly three floats "
                f"(x y z), got {raw_value!r}"
            )
        try:
            translate = np.asarray([float(field) for field in fields], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(
                "TASK2_TRASH_TRANSLATE must contain exactly three floats "
                f"(x y z), got {raw_value!r}"
            ) from exc
        if not np.isfinite(translate).all():
            raise ValueError(
                f"TASK2_TRASH_TRANSLATE must be finite, got {raw_value!r}"
            )
        return translate

    @staticmethod
    def _bottle_translate_from_env() -> np.ndarray:
        raw_value = os.environ.get("TASK2_BOTTLE_TRANSLATE", "0.0 0.0 0.0")
        fields = raw_value.split()
        if len(fields) != 3:
            raise ValueError(
                "TASK2_BOTTLE_TRANSLATE must contain exactly three floats "
                f"(x y z), got {raw_value!r}"
            )
        try:
            translate = np.asarray([float(field) for field in fields], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(
                "TASK2_BOTTLE_TRANSLATE must contain exactly three floats "
                f"(x y z), got {raw_value!r}"
            ) from exc
        if not np.isfinite(translate).all():
            raise ValueError(
                f"TASK2_BOTTLE_TRANSLATE must be finite, got {raw_value!r}"
            )
        return translate

    def reset(
        self,
        seed: int | None = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        selected = self._select_recording(options)
        self._layout = Layout()
        self._layout.add_robot(self.robot)
        self._layout.robot.pose = Pose(
            position=[0.0, 0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0]
        )
        self.mujoco_initial_robot_qpos = selected.qpos[:50].tolist()
        self.mujoco_initial_joint_qpos = {
            "custom_step_trash_can_lid_hinge_joint": [selected.qpos[50]],
            "custom_step_trash_can_pedal_hinge_joint": [selected.qpos[51]],
        }
        bottle_pose = np.asarray(selected.qpos[52:59], dtype=np.float64).copy()
        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        trash_translate = self._trash_translate_from_env()
        bottle_translate = self._bottle_translate_from_env()
        trash_position = np.asarray(
            self.mujoco_extra_mjcf[1]["pos"], dtype=np.float64
        ) + trash_translate
        bottle_pose[:3] += bottle_translate
        self.mujoco_extra_mjcf[1]["pos"] = trash_position.tolist()
        self.mujoco_extra_mjcf[2]["pos"] = bottle_pose[:3].tolist()
        self.mujoco_extra_mjcf[2]["quat"] = bottle_pose[3:].tolist()
        self.task2_trash_translate = trash_translate.tolist()
        self.task2_bottle_translate = bottle_translate.tolist()
        print(
            "[Task2Trash] "
            f"translate={tuple(self.task2_trash_translate)} "
            f"world_position={tuple(self.mujoco_extra_mjcf[1]['pos'])}",
            flush=True,
        )
        print(
            "[Task2Bottle] "
            f"translate={tuple(self.task2_bottle_translate)} "
            f"world_position={tuple(self.mujoco_extra_mjcf[2]['pos'])}",
            flush=True,
        )
        self._layout.add_camera("head_stereo", copy.deepcopy(self.sensor_cfgs["head_stereo"]))
        self._instruction = os.environ.get("TASK2_INSTRUCTION", type(self).description)
        self._target = None
        self.reward = 0.0
        self._task2_inside_steps = 0
        self._task2_success = False
        self._task2_last_inside = False
        self._task2_last_stable = False
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    @staticmethod
    def _geom_aabb_in_body(data: Any, model: Any, geom_name: str, body_name: str):
        geom = data.geom(geom_name)
        body = data.body(body_name)
        body_rot = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
        geom_rot = np.asarray(geom.xmat, dtype=np.float64).reshape(3, 3)
        center = body_rot.T @ (np.asarray(geom.xpos) - np.asarray(body.xpos))
        half = np.abs(body_rot.T @ geom_rot) @ np.asarray(model.geom(geom_name).size[:3])
        return center, half

    def _read_task_state(self, mujoco_env: Any) -> tuple[bool, bool, dict[str, float]]:
        data, model = mujoco_env.mjData, mujoco_env.mjModel
        bin_name = "custom_step_trash_can_bin"
        bounds = {}
        for label in ("left_wall", "right_wall", "front_wall", "back_wall", "bottom"):
            name = f"custom_step_trash_can_bin_{label}_collision"
            bounds[label] = self._geom_aabb_in_body(data, model, name, bin_name)
        x_min = bounds["left_wall"][0][0] + bounds["left_wall"][1][0]
        x_max = bounds["right_wall"][0][0] - bounds["right_wall"][1][0]
        z_min = bounds["front_wall"][0][2] + bounds["front_wall"][1][2]
        z_max = bounds["back_wall"][0][2] - bounds["back_wall"][1][2]
        y_min = bounds["bottom"][0][1] + bounds["bottom"][1][1]
        y_rim = max(
            center[1] + half[1]
            for label, (center, half) in bounds.items()
            if label != "bottom"
        )

        bottle_geom = data.geom("task_obj_0_001_bottle_13_collision")
        bin_body = data.body(bin_name)
        bin_rot = np.asarray(bin_body.xmat, dtype=np.float64).reshape(3, 3)
        bottle_rot = np.asarray(bottle_geom.xmat, dtype=np.float64).reshape(3, 3)
        center = bin_rot.T @ (np.asarray(bottle_geom.xpos) - np.asarray(bin_body.xpos))
        axis = bin_rot.T @ bottle_rot[:, 2]
        radius, half_length = 0.0322801, 0.121928
        extent = np.abs(axis) * half_length + radius * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
        inside = self._inside_interior(
            center,
            extent,
            (x_min, x_max, y_min, y_rim, z_min, z_max),
        )
        qvel = np.asarray(data.joint("task_obj_0_001_bottle_13_free").qvel)
        linear_speed = float(np.linalg.norm(qvel[:3]))
        angular_speed = float(np.linalg.norm(qvel[3:]))
        stable = bool(
            linear_speed <= self._stable_linear_speed
            and angular_speed <= self._stable_angular_speed
        )
        metrics = {
            "bottle_linear_speed_mps": linear_speed,
            "bottle_angular_speed_radps": angular_speed,
            "bottle_top_below_rim_m": float(y_rim - (center[1] + extent[1])),
        }
        return inside, stable, metrics

    @staticmethod
    def _inside_interior(
        center: np.ndarray,
        _extent: np.ndarray,
        bounds: tuple[float, float, float, float, float, float],
    ) -> bool:
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,):
            return False
        if not np.isfinite(center).all():
            return False
        x_min, x_max, y_min, y_rim, z_min, z_max = bounds
        return bool(
            x_min <= center[0] <= x_max
            and z_min <= center[2] <= z_max
            and y_min <= center[1] <= y_rim
        )

    def compute_reward(
        self, info: dict[str, Any], *args: Any, mujoco_env: Any = None, **kwargs: Any
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Task2 reward requires the MuJoCo environment")
        inside, stable, metrics = self._read_task_state(mujoco_env)
        self._task2_last_inside = inside
        self._task2_last_stable = stable
        self._task2_last_metrics = metrics
        self._task2_inside_steps = 1 if inside else 0
        if inside:
            self._task2_success = True
        self.reward = 1.0 if self._task2_success else 0.0
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._task2_success)

    def evaluation_metrics(self) -> dict[str, Any]:
        return {
            "inside": bool(self._task2_last_inside),
            "stable": bool(self._task2_last_stable),
            "inside_hold_steps": int(self._task2_inside_steps),
            **getattr(self, "_task2_last_metrics", {}),
        }
