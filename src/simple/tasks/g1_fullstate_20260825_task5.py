"""MuJoCo evaluation task for the 2026-08-25 upper-drawer demos."""

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


@TaskRegistry.register("g1_fullstate_20260825_task5")
class G1Fullstate20260825Task5(G1Fullstate20260615Task1):
    """Walk forward and pull open the upper drawer."""

    uid = "g1_fullstate_20260825_task5"
    label = "G1 Fullstate 20260825 Task 5"
    description = "Walk forward and pull open the upper drawer."
    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 500,
        "render_hz": 50,
    }

    recording_env_prefix = "TASK5"
    recording_default_dir = Path(
        "/home/ubuntu/yzh/mujoco_recordings/copy/20260825_task5_scq"
    )
    recording_nq = 53
    recording_semantic_fields = {
        0: "pelvis.floating_base_joint.x",
        49: "right_hand_index_1_joint.angle",
        50: "drawer_lower_drawer.drawer_lower_drawer_slide.slide",
        51: "drawer_middle_drawer.drawer_middle_drawer_slide.slide",
        52: "drawer_upper_drawer.drawer_upper_drawer_slide.slide",
    }

    sensor_cfgs = copy.deepcopy(G1Fullstate20260615Task1.sensor_cfgs)
    sensor_cfgs["head_stereo"].pose["position"] = [0.10, 0.06, 0.70]

    _humanoid_vla_root = Path(
        os.environ.get(
            "HUMANOID_VLA_MJ_ROOT",
            "/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ",
        )
    )
    mujoco_extra_mjcf = [
        {
            "path": _humanoid_vla_root
            / "mujoco/model/task_assets/local_mjcf/drawer/drawer.xml",
            "prefix": "drawer_",
            "pos": [0.92, 0.0, 0.0],
            "quat": [0.707107, 0.0, 0.0, -0.707107],
        }
    ]
    mujoco_object_body_names = {"target": "drawer_upper_drawer"}
    # Count the task as complete once the upper drawer is visibly open.  The
    # 126 Task5 recordings all start at 0 m and finish at >=0.153 m, so 0.10 m
    # avoids reset-time false positives without requiring a near-full pull.
    _success_open_distance = 0.10

    @staticmethod
    def _drawer_translate_from_env() -> np.ndarray:
        raw_value = os.environ.get("TASK5_DRAWER_TRANSLATE", "0.0 0.0 0.0")
        fields = raw_value.split()
        if len(fields) != 3:
            raise ValueError(
                "TASK5_DRAWER_TRANSLATE must contain exactly three floats "
                f"(x y z), got {raw_value!r}"
            )
        try:
            translate = np.asarray([float(field) for field in fields], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(
                "TASK5_DRAWER_TRANSLATE must contain exactly three floats "
                f"(x y z), got {raw_value!r}"
            ) from exc
        if not np.isfinite(translate).all():
            raise ValueError(
                f"TASK5_DRAWER_TRANSLATE must be finite, got {raw_value!r}"
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
            "drawer_lower_drawer_slide": [float(selected.qpos[50])],
            "drawer_middle_drawer_slide": [float(selected.qpos[51])],
            "drawer_upper_drawer_slide": [float(selected.qpos[52])],
        }
        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        drawer_translate = self._drawer_translate_from_env()
        drawer_position = np.asarray(
            self.mujoco_extra_mjcf[0]["pos"], dtype=np.float64
        ) + drawer_translate
        self.mujoco_extra_mjcf[0]["pos"] = drawer_position.tolist()
        self.task5_drawer_translate = drawer_translate.tolist()
        print(
            "[Task5Drawer] "
            f"translate={tuple(self.task5_drawer_translate)} "
            f"world_position={tuple(self.mujoco_extra_mjcf[0]['pos'])}",
            flush=True,
        )

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)
        self._instruction = os.environ.get("TASK5_INSTRUCTION", self.description)
        self._target = None
        self.reward = 0.0
        self._task5_upper_open = float(selected.qpos[52])
        self._task5_success = self._task5_upper_open >= self._success_open_distance
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    def compute_reward(
        self, info: dict[str, Any], *args: Any, mujoco_env: Any = None, **kwargs: Any
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Task5 reward requires the MuJoCo environment")
        opened = float(
            np.asarray(mujoco_env.mjData.joint("drawer_upper_drawer_slide").qpos)[0]
        )
        self._task5_upper_open = opened
        self._task5_success |= opened >= self._success_open_distance
        self.reward = 1.0 if self._task5_success else float(
            np.clip(opened / self._success_open_distance, 0.0, 1.0)
        )
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._task5_success)

    def evaluation_metrics(self) -> dict[str, Any]:
        return {
            "upper_drawer_open_distance": float(self._task5_upper_open),
            "upper_drawer_open": bool(self._task5_success),
        }
