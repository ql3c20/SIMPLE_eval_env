"""MuJoCo evaluation task for the 2026-07-29 bottle-to-green-box demos."""

from __future__ import annotations

import copy
import csv
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simple.core.layout import Layout
from simple.core.types import Pose
from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_fullstate_20260729_task4")
class G1Fullstate20260729Task4(G1Fullstate20260615Task1):
    """Walk to the table and place the bottle stably inside the green box."""

    uid = "g1_fullstate_20260729_task4"
    label = "G1 Fullstate 20260729 Task 4"
    description = "Walk forward and put the bottle into the box."

    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 900,
    }

    _humanoid_vla_root = Path(
        os.environ.get(
            "HUMANOID_VLA_MJ_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ_1",
        )
    )
    _asset_root = _humanoid_vla_root / "mujoco/model/task_assets"
    _infra_asset_root = Path(
        os.environ.get(
            "HUMANOID_VLA_MJ_INFRA_ASSET_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ_infra/mujoco/model/task_assets",
        )
    )

    # All 203 snapshots use these authored static poses. The dynamic bottle
    # pose is overwritten from the selected recording's first CSV row.
    _table_pos = [0.95, 0.0, 0.0]
    _box_pos = [0.98, 0.1, 0.75]
    _fallback_bottle_pos = np.asarray(
        [0.6878392620449287, -0.16011817589076813, 0.7497951081922605],
        dtype=np.float64,
    )
    _fallback_bottle_quat = [
        0.6559510180904689,
        0.6559880550470002,
        0.26400106824372627,
        0.26402910723612055,
    ]

    mujoco_extra_mjcf = [
        {
            "path": _infra_asset_root
            / "local_mjcf/primitive_round_table_green_cylinder/round_table.xml",
            "prefix": "task4_table_",
            "pos": _table_pos,
        },
        {
            "path": _asset_root / "local_mjcf/task4_green_box/green_box.xml",
            "prefix": "task4_box_",
            "pos": _box_pos,
        },
        {
            "path": _asset_root / "local_mjcf/task4_bottle/bottle.xml",
            "prefix": "task4_bottle_",
            "pos": _fallback_bottle_pos.tolist(),
            "quat": _fallback_bottle_quat,
            "freejoint_body": "object",
            "freejoint_name": "free",
        },
    ]
    mujoco_object_body_names = {
        "target": "task4_bottle_object",
        "container": "task4_box_object",
    }

    # First robot qpos row from 20260729_151117_g1_sim/data.csv.
    mujoco_initial_robot_qpos = [
        0.033505614264445274,
        0.011148199133491733,
        0.7881787837062328,
        0.9998316091261357,
        -0.003931934265478575,
        0.0014220319356922674,
        0.017868159120294665,
        0.020715711251627925,
        -0.01748153589078198,
        -0.1605917061766047,
        0.14966233258332629,
        -0.17065464717018736,
        -0.0009129600613597508,
        -0.013801788397170011,
        -0.06810893632948492,
        -0.2539165927118012,
        0.15470605311031246,
        -0.13025257682718416,
        0.027300288900318357,
        -0.10820805354101877,
        0.05386075991150192,
        0.0019831127450974555,
        0.04743599072490498,
        0.132376378083604,
        -0.2428071709430502,
        1.2062250957566145,
        -0.5019937860049934,
        -0.19694941333955893,
        0.09401255885297194,
        -7.624501954998919e-05,
        0.0025244763012081074,
        0.00037979230539902405,
        -0.00011600707630963976,
        -2.635663768101226e-05,
        -0.00011721851802292827,
        -2.627486221543311e-05,
        0.16803053158788678,
        -0.18581276176841716,
        -0.10982056999531743,
        1.0274066203184724,
        0.1610555653923137,
        -0.2091385484681488,
        0.23797591541616864,
        0.00033358683609757803,
        -0.02636275508751226,
        -0.0008080122898899366,
        -2.4311274060928966e-06,
        -2.1242306458339682e-07,
        -2.3369696668890395e-06,
        -1.9907993111626424e-07,
    ]

    # Balanced placement check: require the bottle root and collision body to
    # be inside the box, the bottle bottom to be near the floor, and the bottle
    # to remain settled.  Contact tolerances account for MuJoCo penetration and
    # the small mismatch between the scanned and primitive box geometry.
    _bottle_radius = 0.0322801
    _bottle_half_length = 0.121928
    _box_inner_half_x = 0.1224493994
    _box_inner_half_y = 0.167
    _box_floor_top = 0.008
    _box_rim = 0.12
    _root_boundary_tolerance = 0.005
    _wall_safety_margin = 0.002
    _wall_contact_tolerance = 0.005
    _inside_half_x = (
        _box_inner_half_x
        - _bottle_radius
        - _wall_safety_margin
        + _wall_contact_tolerance
    )
    _inside_half_y = (
        _box_inner_half_y
        - _bottle_radius
        - _wall_safety_margin
        + _wall_contact_tolerance
    )
    _floor_penetration_tolerance = 0.012
    _settled_bottom_clearance = 0.027
    _stable_linear_speed = 0.10
    _stable_angular_speed = 2.0
    _inside_steps_required = 8
    _lift_threshold = 0.05

    _bottle_field_re = re.compile(
        r"^qpos:task_obj_0_001_bottle_13_body\."
        r"task_obj_0_001_bottle_13_free\."
        r"(x|y|z|qw|qx|qy|qz)\[qpos(5[0-6])\]$"
    )

    @classmethod
    def _load_recording_initializations(
        cls,
        recordings_dir: Path,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        initializations = []
        for data_path in sorted(recordings_dir.glob("*/data.csv")):
            try:
                with data_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    first_row = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Missing header or first row in {data_path}") from exc

            if len(header) != len(first_row):
                raise ValueError(f"Header/row length mismatch in {data_path}")

            qpos: dict[int, tuple[str, float]] = {}
            for column, value in zip(header, first_row, strict=True):
                match = re.search(r"\[qpos(\d+)\]$", column)
                if match:
                    qpos[int(match.group(1))] = (column, float(value))

            missing = [index for index in range(57) if index not in qpos]
            if missing:
                raise ValueError(f"Missing qpos indices {missing} in {data_path}")

            # Check both numerical indices and semantic joint/body names. This
            # prevents silently reading an unrelated free joint at qpos50..56.
            expected_components = ("x", "y", "z", "qw", "qx", "qy", "qz")
            for offset, component in enumerate(expected_components):
                index = 50 + offset
                match = cls._bottle_field_re.fullmatch(qpos[index][0])
                if match is None or match.group(1) != component:
                    raise ValueError(
                        f"Unexpected bottle qpos field {qpos[index][0]!r} in {data_path}"
                    )
            if "pelvis.floating_base_joint.x" not in qpos[0][0]:
                raise ValueError(f"qpos0 is not the G1 floating root in {data_path}")
            if "right_hand_index_1_joint.angle" not in qpos[49][0]:
                raise ValueError(f"qpos49 is not the final G1 hand joint in {data_path}")

            robot_qpos = np.asarray([qpos[index][1] for index in range(50)])
            bottle_pose = np.asarray([qpos[index][1] for index in range(50, 57)])
            if not np.isfinite(robot_qpos).all() or not np.isfinite(bottle_pose).all():
                raise ValueError(f"NaN/Inf qpos in {data_path}")
            if not np.isclose(np.linalg.norm(robot_qpos[3:7]), 1.0, atol=1e-3):
                raise ValueError(f"Invalid robot root quaternion in {data_path}")
            if not np.isclose(np.linalg.norm(bottle_pose[3:]), 1.0, atol=1e-3):
                raise ValueError(f"Invalid bottle quaternion in {data_path}")
            initializations.append((data_path.parent.name, robot_qpos, bottle_pose))

        if not initializations:
            raise FileNotFoundError(f"No */data.csv recordings found in {recordings_dir}")
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

        reset_index = int(getattr(self, "_task4_reset_index", 0))
        self._task4_reset_index = reset_index + 1
        self.mujoco_initial_robot_qpos = list(type(self).mujoco_initial_robot_qpos)
        bottle_pos = self._fallback_bottle_pos.copy()
        bottle_quat = list(self._fallback_bottle_quat)
        self._task4_recording_name = "fallback_20260729_151117_g1_sim"

        if os.environ.get("TASK4_INIT_FROM_RECORDINGS", "0") == "1":
            recordings_dir = Path(
                os.environ.get(
                    "TASK4_RECORDINGS_DIR",
                    (
                        "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/"
                        "HumanoidVLA_MJ_1/output/20260729_task4"
                    ),
                )
            )
            if not hasattr(self, "_task4_recording_initializations"):
                self._task4_recording_initializations = (
                    self._load_recording_initializations(recordings_dir)
                )
                self._task4_recording_seed = int(
                    os.environ.get("TASK4_RECORDING_SEED", "0")
                )

            pool_size = len(self._task4_recording_initializations)
            recording_index = os.environ.get("TASK4_RECORDING_INDEX")
            if recording_index is not None:
                selected_index = int(recording_index)
                if not 0 <= selected_index < pool_size:
                    raise ValueError(
                        f"TASK4_RECORDING_INDEX={selected_index} is outside "
                        f"the available range 0..{pool_size - 1}"
                    )
            else:
                cycle_index, cycle_offset = divmod(reset_index, pool_size)
                recording_order = np.random.default_rng(
                    self._task4_recording_seed + cycle_index
                ).permutation(pool_size)
                selected_index = int(recording_order[cycle_offset])

            recording_name, robot_qpos, bottle_pose = (
                self._task4_recording_initializations[selected_index]
            )
            self._task4_recording_name = recording_name
            self.mujoco_initial_robot_qpos = robot_qpos.tolist()
            bottle_pos = bottle_pose[:3].copy()
            bottle_quat = bottle_pose[3:].tolist()
            print(
                "[Task4RecordingInit] "
                f"episode_reset={reset_index} recording={recording_name} "
                f"bottle_xyz=({bottle_pos[0]:.4f}, "
                f"{bottle_pos[1]:.4f}, {bottle_pos[2]:.4f})"
            )

        bottle_pos[0] += float(os.environ.get("TASK4_BOTTLE_X_OFFSET", "0"))
        bottle_pos[1] += float(os.environ.get("TASK4_BOTTLE_Y_OFFSET", "0"))

        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[2]["pos"] = bottle_pos.tolist()
        self.mujoco_extra_mjcf[2]["quat"] = bottle_quat

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)

        instruction = "Walk forward and put the bottle into the box."
        if os.environ.get("TASK4_ALLOW_INSTRUCTION_OVERRIDE", "0") == "1":
            instruction = os.environ.get("TASK4_INSTRUCTION", instruction)
        self._instruction = instruction
        self._target = None
        self.reward = 0.0
        self._bottle_start_z = float(bottle_pos[2])
        self._was_lifted = False
        self._task4_inside_steps = 0
        self._task4_success = False
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    @classmethod
    def _placement_inside_box(
        cls,
        bottle_root_local: np.ndarray,
        bottle_center_local: np.ndarray,
        bottle_axis_local: np.ndarray,
    ) -> bool:
        root = np.asarray(bottle_root_local, dtype=np.float64)
        center = np.asarray(bottle_center_local, dtype=np.float64)
        axis = np.asarray(bottle_axis_local, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if (
            root.shape != (3,)
            or center.shape != (3,)
            or axis.shape != (3,)
            or not np.all(np.isfinite(root))
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(axis))
            or axis_norm == 0.0
        ):
            return False
        axis = axis / axis_norm
        vertical_half_extent = (
            abs(float(axis[2])) * cls._bottle_half_length
            + cls._bottle_radius
            * np.sqrt(max(0.0, 1.0 - float(axis[2]) ** 2))
        )
        bottle_bottom = float(center[2]) - vertical_half_extent
        tolerance = cls._root_boundary_tolerance
        return bool(
            abs(float(root[0])) <= cls._box_inner_half_x + tolerance
            and abs(float(root[1])) <= cls._box_inner_half_y + tolerance
            and cls._box_floor_top - tolerance
            <= float(root[2])
            <= cls._box_rim + tolerance
            and abs(float(center[0])) <= cls._inside_half_x
            and abs(float(center[1])) <= cls._inside_half_y
            and cls._box_floor_top - cls._floor_penetration_tolerance
            <= bottle_bottom
            <= cls._box_floor_top + cls._settled_bottom_clearance
        )

    def _read_task_state(
        self,
        mujoco_env: Any,
    ) -> tuple[np.ndarray, bool, bool]:
        data = mujoco_env.mjData
        bottle_body = data.body("task4_bottle_object")
        bottle_geom = data.geom("task4_bottle_collision")
        box_body = data.body("task4_box_object")
        box_rotation = np.asarray(box_body.xmat, dtype=np.float64).reshape(3, 3)
        bottle_root_local = box_rotation.T @ (
            np.asarray(bottle_body.xpos, dtype=np.float64)
            - np.asarray(box_body.xpos, dtype=np.float64)
        )
        bottle_center_local = box_rotation.T @ (
            np.asarray(bottle_geom.xpos, dtype=np.float64)
            - np.asarray(box_body.xpos, dtype=np.float64)
        )
        bottle_rotation = np.asarray(
            bottle_geom.xmat, dtype=np.float64
        ).reshape(3, 3)
        bottle_axis_local = box_rotation.T @ bottle_rotation[:, 2]
        inside = self._placement_inside_box(
            bottle_root_local,
            bottle_center_local,
            bottle_axis_local,
        )

        bottle_qvel = np.asarray(
            data.joint("task4_bottle_free").qvel,
            dtype=np.float64,
        )
        stable = bool(
            np.linalg.norm(bottle_qvel[:3]) <= self._stable_linear_speed
            and np.linalg.norm(bottle_qvel[3:]) <= self._stable_angular_speed
        )
        return bottle_root_local, inside, stable

    def compute_reward(
        self,
        info: dict[str, Any],
        *args: Any,
        mujoco_env: Any = None,
        **kwargs: Any,
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Task4 reward requires the MuJoCo environment")

        bottle_root_local, inside, stable = self._read_task_state(mujoco_env)
        bottle_z = float(np.asarray(info["target"], dtype=np.float64)[2])
        self._was_lifted |= bottle_z >= self._bottle_start_z + self._lift_threshold
        self._task4_inside_steps = (
            self._task4_inside_steps + 1 if inside and stable else 0
        )
        if self._task4_inside_steps >= self._inside_steps_required:
            self._task4_success = True

        distance_xy = float(np.linalg.norm(bottle_root_local[:2]))
        approach = 1.0 - float(np.clip(distance_xy / 0.50, 0.0, 1.0))
        stable_progress = min(
            self._task4_inside_steps / self._inside_steps_required,
            1.0,
        )
        self.reward = (
            1.0
            if self._task4_success
            else 0.15 * float(self._was_lifted)
            + 0.25 * approach
            + 0.50 * stable_progress
        )
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._task4_success)
