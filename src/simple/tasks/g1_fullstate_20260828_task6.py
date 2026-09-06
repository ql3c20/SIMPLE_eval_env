"""MuJoCo evaluation task for the 2026-08-28 Task6 shelf demos."""

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


@TaskRegistry.register("g1_fullstate_20260828_task6")
class G1Fullstate20260828Task6(G1Fullstate20260615Task1):
    """Pick up the can from one shelf and place it on the shelf below."""

    uid = "g1_fullstate_20260828_task6"
    label = "G1 Fullstate 20260828 Task 6"
    description = "Pick up the object from the shelf and place it on the shelf below."
    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 500,
        "render_hz": 50,
    }

    recording_env_prefix = "TASK6"
    recording_default_dir = Path(
        "/home/ubuntu/yzh/mujoco_recordings/copy/20260828_task6_scq/20260828"
    )
    recording_nq = 57
    recording_semantic_fields = {
        0: "pelvis.floating_base_joint.x",
        49: "right_hand_index_1_joint.angle",
        50: "can_071_base3_on_second_shelf_from_top_free.x",
        56: "can_071_base3_on_second_shelf_from_top_free.qz",
    }

    sensor_cfgs = copy.deepcopy(G1Fullstate20260615Task1.sensor_cfgs)
    sensor_cfgs["head_stereo"].pose["position"] = [0.10, 0.06, 0.50]

    _humanoid_vla_root = Path(
        os.environ.get(
            "HUMANOID_VLA_MJ_ROOT",
            "/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ",
        )
    )
    _asset_root = _humanoid_vla_root / "mujoco/model/task_assets"
    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "2real_asset/shelf/shelf_visual_01pct.xml",
            "prefix": "supplied_shelf_",
            "pos": [1.35, 0.0, 0.0],
            "quat": [0.707107, 0.0, 0.0, 0.707107],
        },
        {
            "path": _asset_root / "local_mjcf/task6_can_base3/can_base3.xml",
            "prefix": "can_071_base3_on_second_shelf_from_top_",
            "pos": [1.24183, -0.01948, 1.03958],
            "quat": [0.707107, 0.707107, 0.0, 0.0],
            "freejoint_body": "object",
            "freejoint_name": "free",
        },
    ]
    mujoco_object_body_names = {
        "target": "can_071_base3_on_second_shelf_from_top_object"
    }

    # The target shelf top is at z~=0.742 m in the recorded MuJoCo scene.
    # Use a bounded band so a can that is still in the air or has fallen to
    # the floor cannot be counted as placed on the shelf below.
    _lower_shelf_z_bounds = (0.72, 0.80)
    _lower_shelf_x_bounds = (1.20, 1.43)
    _lower_shelf_y_bounds = (-0.20, 0.18)
    _upper_shelf_z_min = 0.95
    _hand_release_distance = 0.15
    _hand_reference_bodies = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hand_middle_0_link",
        "right_hand_middle_0_link",
    )
    _can_collision_geom = "can_071_base3_on_second_shelf_from_top_can_collision"
    _target_shelf_geom = "supplied_shelf_rack_collision_shelf_2"
    _stable_linear_speed = 0.15
    _stable_angular_speed = 3.0
    _stable_steps_required = 8

    @staticmethod
    def _can_left_randomization_enabled() -> bool:
        return os.environ.get("TASK6_CAN_RANDOMIZE_LEFT", "0") == "1"

    @staticmethod
    def _can_random_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw in (None, ""):
            return default
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return value

    @classmethod
    def _sample_left_can_from_recordings(
        cls,
        selected_can_pose: np.ndarray,
        recording_initializations: list[Any],
        episode_index: int,
    ) -> tuple[np.ndarray, str | None]:
        if not cls._can_left_randomization_enabled():
            return selected_can_pose.copy(), None
        # The robot faces +X; robot-left is world +Y. Draw from genuinely
        # observed left-side initial poses instead of inventing a rectangular
        # distribution outside the demonstrations. Small bounded jitter
        # expands the finite pose pool while retaining its geometry.
        min_y = cls._can_random_float("TASK6_CAN_LEFT_MIN_Y", 0.05)
        jitter_x = cls._can_random_float("TASK6_CAN_LEFT_JITTER_X", 0.005)
        jitter_y = cls._can_random_float("TASK6_CAN_LEFT_JITTER_Y", 0.005)
        weight_power = cls._can_random_float("TASK6_CAN_LEFT_WEIGHT_POWER", 2.0)
        if jitter_x < 0.0 or jitter_y < 0.0 or weight_power < 0.0:
            raise ValueError(
                "TASK6 can-left jitter and weight power must be non-negative"
            )
        fixed_x_raw = os.environ.get("TASK6_CAN_FIXED_X")
        fixed_y_raw = os.environ.get("TASK6_CAN_FIXED_Y")
        if (fixed_x_raw is None) != (fixed_y_raw is None):
            raise ValueError(
                "TASK6_CAN_FIXED_X and TASK6_CAN_FIXED_Y must be set together"
            )
        if fixed_x_raw is not None:
            fixed_x = cls._can_random_float("TASK6_CAN_FIXED_X", 1.220)
            fixed_y = cls._can_random_float("TASK6_CAN_FIXED_Y", 0.020)
            if not 1.215 <= fixed_x <= 1.475 or not -0.20 <= fixed_y <= 0.20:
                raise ValueError(
                    "Task6 fixed can XY must stay inside the shelf-safe bounds: "
                    "X=[1.215, 1.475], Y=[-0.20, 0.20]"
                )
            rng = np.random.default_rng(
                int(
                    os.environ.get(
                        "TASK6_CAN_RANDOM_SEED",
                        os.environ.get("TASK6_RECORDING_SEED", "0"),
                    )
                )
                + episode_index
            )
            randomized = selected_can_pose.copy()
            randomized[0] = fixed_x + rng.uniform(-jitter_x, jitter_x)
            randomized[1] = fixed_y + rng.uniform(-jitter_y, jitter_y)
            randomized[0] = np.clip(randomized[0], 1.215, 1.475)
            randomized[1] = np.clip(randomized[1], -0.20, 0.20)
            return randomized, "fixed_xy"
        candidates = [
            item for item in recording_initializations if item.qpos[51] >= min_y
        ]
        if not candidates:
            raise ValueError(
                f"No Task6 recording has a can initial Y >= {min_y:.5f}"
            )
        base_seed = int(
            os.environ.get(
                "TASK6_CAN_RANDOM_SEED",
                os.environ.get("TASK6_RECORDING_SEED", "0"),
            )
        )
        rng = np.random.default_rng(base_seed + episode_index)
        candidate_y = np.asarray([item.qpos[51] for item in candidates])
        normalized_leftness = (candidate_y - candidate_y.min()) / max(
            float(np.ptp(candidate_y)), np.finfo(np.float64).eps
        )
        weights = np.power(normalized_leftness + 0.05, weight_power)
        weights /= weights.sum()
        candidate_index = int(rng.choice(len(candidates), p=weights))
        candidate = candidates[candidate_index]
        randomized = candidate.qpos[50:57].copy()
        randomized[0] += rng.uniform(-jitter_x, jitter_x)
        randomized[1] += rng.uniform(-jitter_y, jitter_y)
        # Shelf-safe limits include a can-radius margin. Keep the configured
        # left threshold after jitter as well.
        randomized[0] = np.clip(randomized[0], 1.215, 1.475)
        randomized[1] = np.clip(randomized[1], min_y, 0.20)
        return randomized, candidate.name

    @staticmethod
    def _robot_translation() -> np.ndarray:
        raw = os.environ.get("TASK6_ROBOT_TRANSLATE", "0.0 0.0 0.0")
        try:
            translation = np.asarray(
                [float(value) for value in raw.replace(",", " ").split()],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise ValueError(
                "TASK6_ROBOT_TRANSLATE must contain three finite numbers"
            ) from exc
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError(
                "TASK6_ROBOT_TRANSLATE must contain three finite numbers"
            )
        return translation

    @staticmethod
    def _robot_face_can_enabled() -> bool:
        return os.environ.get("TASK6_ROBOT_FACE_CAN", "0") == "1"

    @staticmethod
    def _quaternion_yaw(quaternion_wxyz: np.ndarray) -> float:
        w, x, y, z = (float(value) for value in quaternion_wxyz)
        return float(
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        )

    @staticmethod
    def _quaternion_with_yaw(
        quaternion_wxyz: np.ndarray, target_yaw: float
    ) -> np.ndarray:
        w, x, y, z = (float(value) for value in quaternion_wxyz)
        roll = np.arctan2(
            2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)
        )
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        cy, sy = np.cos(target_yaw / 2.0), np.sin(target_yaw / 2.0)
        quaternion = np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
        return quaternion / np.linalg.norm(quaternion)

    @staticmethod
    def _wrap_angle_degrees(angle_degrees: float) -> float:
        return float((angle_degrees + 180.0) % 360.0 - 180.0)

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
        robot_qpos = selected.qpos[:50].copy()
        robot_translation = self._robot_translation()
        robot_qpos[:3] += robot_translation
        if np.any(robot_translation):
            print(
                "[TASK6RecordingInit] robot_translate="
                + " ".join(f"{value:.3f}" for value in robot_translation)
            )
        episode_index = int(
            (options or {}).get(
                "episode_index", getattr(self, "_recording_reset_index", 1) - 1
            )
        )
        recorded_can_pose = selected.qpos[50:57].copy()
        can_pose, sampled_can_source = self._sample_left_can_from_recordings(
            recorded_can_pose,
            self._recording_initializations,
            episode_index,
        )
        original_yaw = self._quaternion_yaw(robot_qpos[3:7])
        can_direction = can_pose[:2] - robot_qpos[:2]
        if np.linalg.norm(can_direction) <= np.finfo(np.float64).eps:
            raise ValueError("Task6 robot and can cannot share the same XY position")
        target_yaw = float(np.arctan2(can_direction[1], can_direction[0]))
        self._task6_robot_face_can = self._robot_face_can_enabled()
        self._task6_original_robot_yaw_deg = float(np.degrees(original_yaw))
        self._task6_target_can_bearing_deg = float(np.degrees(target_yaw))
        self._task6_original_heading_error_deg = self._wrap_angle_degrees(
            np.degrees(target_yaw - original_yaw)
        )
        if self._task6_robot_face_can:
            robot_qpos[3:7] = self._quaternion_with_yaw(
                robot_qpos[3:7], target_yaw
            )
        applied_yaw = self._quaternion_yaw(robot_qpos[3:7])
        self._task6_applied_robot_yaw_deg = float(np.degrees(applied_yaw))
        self._task6_applied_heading_error_deg = self._wrap_angle_degrees(
            np.degrees(target_yaw - applied_yaw)
        )
        self.mujoco_initial_robot_qpos = robot_qpos.tolist()
        print(
            "[TASK6RecordingInit] robot_face_can="
            f"{int(self._task6_robot_face_can)}"
            f" original_yaw_deg={self._task6_original_robot_yaw_deg:.3f}"
            f" target_yaw_deg={self._task6_target_can_bearing_deg:.3f}"
            f" applied_error_deg={self._task6_applied_heading_error_deg:.3f}",
            flush=True,
        )
        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[1]["pos"] = can_pose[:3].tolist()
        self.mujoco_extra_mjcf[1]["quat"] = can_pose[3:].tolist()
        self._task6_initial_can_xyz = can_pose[:3].copy()
        self._task6_recorded_can_xyz = recorded_can_pose[:3].copy()
        self._task6_can_randomized_left = self._can_left_randomization_enabled()
        self._task6_sampled_can_source = sampled_can_source
        if self._task6_can_randomized_left:
            print(
                "[TASK6RecordingInit] can_left_random="
                + " ".join(f"{value:.5f}" for value in can_pose[:3]),
                f" source={sampled_can_source}",
                flush=True,
            )

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)
        self._instruction = os.environ.get("TASK6_INSTRUCTION", self.description)
        self._target = None
        self.reward = 0.0
        self._task6_can_xyz = can_pose[:3].copy()
        self._task6_started_on_upper_shelf = bool(
            can_pose[2] >= self._upper_shelf_z_min
        )
        self._task6_stable_steps = 0
        self._task6_linear_speed = 0.0
        self._task6_angular_speed = 0.0
        self._task6_hand_clearance = 0.0
        self._task6_supported_by_lower_shelf = False
        self._task6_success = False
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    @classmethod
    def _on_lower_shelf(cls, xyz: np.ndarray) -> bool:
        x, y, z = (float(value) for value in xyz)
        return (
            cls._lower_shelf_x_bounds[0] <= x <= cls._lower_shelf_x_bounds[1]
            and cls._lower_shelf_y_bounds[0] <= y <= cls._lower_shelf_y_bounds[1]
            and cls._lower_shelf_z_bounds[0] <= z <= cls._lower_shelf_z_bounds[1]
        )

    @classmethod
    def _has_target_shelf_contact(cls, mujoco_env: Any) -> bool:
        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        can_geom = int(model.geom(cls._can_collision_geom).id)
        shelf_geom = int(model.geom(cls._target_shelf_geom).id)
        expected = {can_geom, shelf_geom}
        return any(
            {int(data.contact[index].geom1), int(data.contact[index].geom2)}
            == expected
            for index in range(data.ncon)
        )

    @classmethod
    def _minimum_hand_clearance(cls, mujoco_env: Any) -> float:
        data = mujoco_env.mjData
        can_position = np.asarray(
            data.body("can_071_base3_on_second_shelf_from_top_object").xpos,
            dtype=np.float64,
        )
        return min(
            float(
                np.linalg.norm(
                    can_position
                    - np.asarray(data.body(body_name).xpos, dtype=np.float64)
                )
            )
            for body_name in cls._hand_reference_bodies
        )

    def compute_reward(
        self, info: dict[str, Any], *args: Any, mujoco_env: Any = None, **kwargs: Any
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Task6 reward requires the MuJoCo environment")
        can_joint = mujoco_env.mjData.joint(
            "can_071_base3_on_second_shelf_from_top_free"
        )
        xyz = np.asarray(can_joint.qpos[:3], dtype=np.float64)
        velocity = np.asarray(can_joint.qvel, dtype=np.float64)
        self._task6_can_xyz = xyz.copy()
        self._task6_linear_speed = float(np.linalg.norm(velocity[:3]))
        self._task6_angular_speed = float(np.linalg.norm(velocity[3:]))
        self._task6_hand_clearance = self._minimum_hand_clearance(mujoco_env)
        self._task6_supported_by_lower_shelf = self._has_target_shelf_contact(
            mujoco_env
        )
        # Success is intentionally outcome-based: once the can has moved from
        # the upper shelf into the lower-shelf region, the task is complete.
        # Contact, hand clearance, velocity and dwell time remain available as
        # diagnostics, but are not success requirements.
        placed_on_lower_shelf = (
            self._task6_started_on_upper_shelf
            and self._on_lower_shelf(xyz)
        )
        if placed_on_lower_shelf:
            self._task6_stable_steps += 1
        else:
            self._task6_stable_steps = 0
        self._task6_success |= placed_on_lower_shelf
        self.reward = 1.0 if self._task6_success else float(
            np.clip(
                (1.03958 - xyz[2])
                / (1.03958 - self._lower_shelf_z_bounds[1]),
                0.0,
                0.99,
            )
        )
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._task6_success)

    def evaluation_metrics(self) -> dict[str, Any]:
        return {
            "can_xyz": self._task6_can_xyz.tolist(),
            "initial_can_xyz": self._task6_initial_can_xyz.tolist(),
            "recorded_can_xyz": self._task6_recorded_can_xyz.tolist(),
            "can_randomized_left": bool(self._task6_can_randomized_left),
            "sampled_can_source": self._task6_sampled_can_source,
            "robot_face_can": bool(self._task6_robot_face_can),
            "original_robot_yaw_deg": float(
                self._task6_original_robot_yaw_deg
            ),
            "target_can_bearing_deg": float(
                self._task6_target_can_bearing_deg
            ),
            "original_heading_error_deg": float(
                self._task6_original_heading_error_deg
            ),
            "applied_robot_yaw_deg": float(
                self._task6_applied_robot_yaw_deg
            ),
            "applied_heading_error_deg": float(
                self._task6_applied_heading_error_deg
            ),
            "can_linear_speed": float(self._task6_linear_speed),
            "can_angular_speed": float(self._task6_angular_speed),
            "can_hand_clearance": float(self._task6_hand_clearance),
            "supported_by_lower_shelf": bool(
                self._task6_supported_by_lower_shelf
            ),
            "lower_shelf_stable_steps": int(self._task6_stable_steps),
            "can_on_lower_shelf": bool(self._task6_success),
        }
