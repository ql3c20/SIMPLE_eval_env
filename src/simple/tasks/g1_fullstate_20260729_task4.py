"""MuJoCo evaluation task for the 2026-07-29 bottle-to-green-box demos."""

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


@TaskRegistry.register("g1_fullstate_20260729_task4")
class G1Fullstate20260729Task4(G1Fullstate20260615Task1):
    uid = "g1_fullstate_20260729_task4"
    label = "G1 Fullstate 20260729 Task 4"
    description = "Walk forward and put the bottle into the box."
    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 900,
        "render_hz": 50,
    }

    recording_env_prefix = "TASK4"
    recording_default_dir = Path("/home/ubuntu/yzh/mujoco_recordings/20260729_task4")
    recording_nq = 57
    recording_semantic_fields = {
        0: "pelvis.floating_base_joint.x",
        49: "right_hand_index_1_joint.angle",
        50: "task_obj_0_001_bottle_13_free.x",
        56: "task_obj_0_001_bottle_13_free.qz",
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
    _table_pos = [0.95, 0.0, 0.0]
    _box_pos = [0.98, 0.1, 0.75]
    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "2real_asset/round_table/round_table_visual_05pct.xml",
            "prefix": "round_table_2real_",
            "pos": _table_pos,
        },
        {
            "path": _asset_root / "local_mjcf/task4_green_box/green_box.xml",
            "prefix": "green_box_2real_",
            "pos": _box_pos,
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
        "container": "green_box_2real_object",
    }

    _bottle_radius = 0.0322801
    _bottle_half_length = 0.121928
    _box_inner_half_x = 0.1224493994
    _box_inner_half_y = 0.167
    _box_floor_top = 0.008
    _box_rim = 0.12
    _root_boundary_tolerance = 0.005
    _floor_penetration_tolerance = 0.012
    _stable_linear_speed = 0.10
    _stable_angular_speed = 2.0
    _lift_threshold = 0.05

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
        bottle_pose = selected.qpos[50:57].copy()
        bottle_pose[0] += float(os.environ.get("TASK4_BOTTLE_X_OFFSET", "0"))
        bottle_pose[1] += float(os.environ.get("TASK4_BOTTLE_Y_OFFSET", "0"))
        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[2]["pos"] = bottle_pose[:3].tolist()
        self.mujoco_extra_mjcf[2]["quat"] = bottle_pose[3:].tolist()

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)
        self._instruction = "Walk forward and put the bottle into the box."
        self._target = None
        self.reward = 0.0
        self._bottle_start_z = float(bottle_pose[2])
        self._was_lifted = False
        self._task4_inside_steps = 0
        self._task4_success = False
        self._task4_last_inside = False
        self._task4_last_stable = False
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    @classmethod
    def _placement_inside_box(
        cls,
        _bottle_root_local: np.ndarray,
        bottle_center_local: np.ndarray,
        bottle_axis_local: np.ndarray,
    ) -> bool:
        center = np.asarray(bottle_center_local, dtype=np.float64)
        axis = np.asarray(bottle_axis_local, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if any(value.shape != (3,) for value in (center, axis)):
            return False
        if not all(np.all(np.isfinite(value)) for value in (center, axis)) or axis_norm == 0:
            return False
        axis = axis / axis_norm
        vertical_half_extent = (
            abs(float(axis[2])) * cls._bottle_half_length
            + cls._bottle_radius * np.sqrt(max(0.0, 1.0 - float(axis[2]) ** 2))
        )
        bottle_bottom = float(center[2]) - vertical_half_extent
        tolerance = cls._root_boundary_tolerance
        return bool(
            abs(float(center[0])) <= cls._box_inner_half_x
            and abs(float(center[1])) <= cls._box_inner_half_y
            and cls._box_floor_top - cls._floor_penetration_tolerance
            <= bottle_bottom
            <= cls._box_rim + tolerance
        )

    def _read_task_state(self, mujoco_env: Any) -> tuple[np.ndarray, bool, bool]:
        data = mujoco_env.mjData
        bottle_body = data.body("task_obj_0_001_bottle_13_object")
        bottle_geom = data.geom("task_obj_0_001_bottle_13_collision")
        box_body = data.body("green_box_2real_object")
        box_rotation = np.asarray(box_body.xmat, dtype=np.float64).reshape(3, 3)
        bottle_root_local = box_rotation.T @ (
            np.asarray(bottle_body.xpos) - np.asarray(box_body.xpos)
        )
        bottle_center_local = box_rotation.T @ (
            np.asarray(bottle_geom.xpos) - np.asarray(box_body.xpos)
        )
        bottle_rotation = np.asarray(bottle_geom.xmat, dtype=np.float64).reshape(3, 3)
        bottle_axis_local = box_rotation.T @ bottle_rotation[:, 2]
        inside = self._placement_inside_box(
            bottle_root_local, bottle_center_local, bottle_axis_local
        )
        qvel = np.asarray(data.joint("task_obj_0_001_bottle_13_free").qvel)
        stable = bool(
            np.linalg.norm(qvel[:3]) <= self._stable_linear_speed
            and np.linalg.norm(qvel[3:]) <= self._stable_angular_speed
        )
        return bottle_root_local, inside, stable

    def compute_reward(
        self, info: dict[str, Any], *args: Any, mujoco_env: Any = None, **kwargs: Any
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Task4 reward requires the MuJoCo environment")
        bottle_root_local, inside, stable = self._read_task_state(mujoco_env)
        self._task4_last_inside = inside
        self._task4_last_stable = stable
        bottle_z = float(np.asarray(info["target"])[2])
        self._was_lifted |= bottle_z >= self._bottle_start_z + self._lift_threshold
        self._task4_inside_steps = 1 if inside else 0
        if inside:
            self._task4_success = True
        distance_xy = float(np.linalg.norm(bottle_root_local[:2]))
        approach = 1.0 - float(np.clip(distance_xy / 0.50, 0.0, 1.0))
        self.reward = (
            1.0
            if self._task4_success
            else 0.15 * float(self._was_lifted) + 0.25 * approach
        )
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._task4_success)

    def evaluation_metrics(self) -> dict[str, Any]:
        return {
            "inside": bool(self._task4_last_inside),
            "stable": bool(self._task4_last_stable),
            "inside_hold_steps": int(self._task4_inside_steps),
            "was_lifted": bool(self._was_lifted),
        }
