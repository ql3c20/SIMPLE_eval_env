"""Pedal-bin opening evaluation task from the 2026-08-04 recordings."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

from simple.core.layout import Layout
from simple.core.types import Pose
from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_fullstate_20260804_task3")
class G1Fullstate20260804Task3(G1Fullstate20260615Task1):
    uid = "g1_fullstate_20260804_task3"
    label = "G1 Fullstate 20260804 Task 3"
    description = "Step on the pedal to open the trash can in front of you."
    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 800,
        "render_hz": 50,
    }
    recording_env_prefix = "TASK3"
    recording_default_dir = Path("/home/ubuntu/yzh/mujoco_recordings/20260804_task3_new")
    recording_nq = 52
    recording_semantic_fields = {
        0: "pelvis.floating_base_joint.x",
        49: "right_hand_index_1_joint.angle",
        50: "custom_step_trash_can_lid_hinge_joint.angle",
        51: "custom_step_trash_can_pedal_hinge_joint.angle",
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
            "path": _asset_root / "2real_asset/trash/step_trash_can.xml",
            "prefix": "custom_step_trash_can_",
            "pos": [0.78, 0.0, 0.0],
            "quat": [0.707107, 0.0, 0.0, -0.707107],
            "scale": 0.395257,
        }
    ]
    mujoco_object_body_names = {"container": "custom_step_trash_can_bin"}
    # Default evaluation is intentionally below the mechanical maximum: a lid
    # held clearly open at roughly 46 degrees is sufficient for the task.  Keep
    # explicit environment overrides for threshold-ablation runs.
    _lid_threshold = float(os.environ.get("TASK3_LID_THRESHOLD_RAD", "0.8"))
    _hold_steps_required = int(os.environ.get("TASK3_SUCCESS_HOLD_STEPS", "10"))

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
        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self._layout.add_camera("head_stereo", copy.deepcopy(self.sensor_cfgs["head_stereo"]))
        self._instruction = "Step on the pedal to open the trash can in front of you."
        self._target = None
        self.reward = 0.0
        self._task3_lid_steps = 0
        self._task3_success = False
        self._task3_lid_angle = float(selected.qpos[50])
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    def compute_reward(
        self, info: dict[str, Any], *args: Any, mujoco_env: Any = None, **kwargs: Any
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Task3 reward requires the MuJoCo environment")
        self._task3_lid_angle = float(
            mujoco_env.mjData.joint("custom_step_trash_can_lid_hinge_joint").qpos[0]
        )
        self._task3_lid_steps = (
            self._task3_lid_steps + 1 if self._task3_lid_angle >= self._lid_threshold else 0
        )
        if self._task3_lid_steps >= self._hold_steps_required:
            self._task3_success = True
        self.reward = (
            1.0
            if self._task3_success
            else min(0.9, max(0.0, self._task3_lid_angle / self._lid_threshold) * 0.8)
        )
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._task3_success)

    def evaluation_metrics(self) -> dict[str, Any]:
        return {
            "lid_angle_rad": float(self._task3_lid_angle),
            "lid_hold_steps": int(self._task3_lid_steps),
            "lid_threshold_rad": float(self._lid_threshold),
            "lid_hold_steps_required": int(self._hold_steps_required),
        }
