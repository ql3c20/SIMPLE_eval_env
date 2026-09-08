"""Fixed MuJoCo evaluation task for the 2026-06-25 bottle-to-bin demo."""

from __future__ import annotations

import csv
import copy
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simple.core.layout import Layout
from simple.core.types import Pose
from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_fullstate_20260625_task2")
class G1Fullstate20260625Task2(G1Fullstate20260615Task1):
    """Pick up the bottle, press the bin pedal, and put the bottle inside."""

    uid = "g1_fullstate_20260625_task2"
    label = "G1 Fullstate 20260625 Task 2"
    description = "Pick up the bottle, open the pedal bin, and put it inside."

    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 900,
    }

    _humanoid_vla_root = Path(
        os.environ.get(
            "HUMANOID_VLA_MJ_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ",
        )
    )
    _asset_root = _humanoid_vla_root / "mujoco/model/task_assets/local_mjcf"
    _nominal_bottle_center_xy = np.asarray(
        [0.6301283280837536, -0.005209913136218], dtype=np.float64
    )
    # Settled free-joint pose from the first frame of 20260625_145857_g1_sim.
    # The scene-authored pose settles to this state before recording starts.
    _nominal_bottle_body_pos = np.asarray(
        [0.6301283280837536, -0.005209913136218, 0.7498682600609945],
        dtype=np.float64,
    )
    _nominal_bottle_body_quat = [
        0.0755299555218114,
        0.0755484960985177,
        -0.7030584907109257,
        -0.703062165953623,
    ]

    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "primitive_round_table_green_cylinder/round_table.xml",
            "prefix": "task2_table_",
            "pos": [0.95, 0.0, 0.0],
        },
        {
            "path": _asset_root / "step_trash_can/trash_can.xml",
            "prefix": "task2_trash_",
            "pos": [0.30, -0.75, 0.0],
            "quat": [0.70710678, 0.0, 0.0, -0.70710678],
        },
        {
            "path": _asset_root / "obj_001_bottle_13_8e13a37c387158c8/bottle.xml",
            "prefix": "task2_bottle_",
            "pos": _nominal_bottle_body_pos.tolist(),
            "quat": _nominal_bottle_body_quat,
            "freejoint_body": "object",
            "freejoint_name": "free",
        },
    ]
    mujoco_object_body_names = {
        "target": "task2_bottle_object",
        "container": "task2_trash_object",
    }

    # First recorded robot qpos row from 20260625_145857_g1_sim/data.csv.
    # Ordering is floating root followed by the G1 43 actuated joints.
    mujoco_initial_robot_qpos = [
        0.0364648166966125, 0.0169539139028861, 0.7886194181518692,
        0.994494630702156, -0.0037909888615654, -0.0171590287151832,
        -0.1033035606433081, 0.0257458459160167, 0.0584054853469112,
        0.1714588780583383, 0.1659461441036304, -0.1504838258715325,
        -0.022510918773875, -0.0482827416426821, -0.0368000250930811,
        -0.1969344583921034, 0.1808720422320429, -0.0943192622020385,
        -0.0102960445637815, -0.0043350424043464, -0.0024059632330908,
        0.0106849326516176, -0.2461254756902047, 0.2139759165081475,
        -0.2906375264484901, 1.375287656403721, 0.0846941428956144,
        0.0584263317979694, -0.1453834144322764, 0.014165367085977,
        -0.0412152636596343, 3.014375737283132e-05,
        -0.0007873757859703, -4.974437576496433e-05,
        -0.0015142339586754, -4.84833258201062e-05,
        -0.1703107007085183, -0.1952182236245829, 0.2505322267088719,
        1.304500073203432, 0.1977267614918172, -0.0022604559083704,
        -0.1166122759934563, 0.0435744919830231, -0.0921991193741168,
        -6.152833539587038e-05, 0.0001412707907469,
        3.008119867004157e-05, 0.0001683366030192,
        3.141239449139311e-05,
    ]

    _lift_threshold = 0.005
    _pedal_threshold = -0.18
    _lid_threshold = 1.0

    @staticmethod
    def _load_recording_initializations(
        recordings_dir: Path,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        initializations = []
        for data_path in sorted(recordings_dir.glob("*/data.csv")):
            with data_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                first_row = next(reader)

            qpos = {}
            for column, value in zip(header, first_row, strict=True):
                match = re.search(r"\[qpos(\d+)\]$", column)
                if match:
                    qpos[int(match.group(1))] = float(value)

            required = list(range(50)) + list(range(52, 59))
            missing = [index for index in required if index not in qpos]
            if missing:
                raise ValueError(f"Missing qpos indices {missing} in {data_path}")

            robot_qpos = np.asarray([qpos[index] for index in range(50)])
            bottle_pose = np.asarray([qpos[index] for index in range(52, 59)])
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

        reset_index = int(getattr(self, "_task2_reset_index", 0))
        self._task2_reset_index = reset_index + 1
        bottle_body_pos = self._nominal_bottle_body_pos.copy()
        bottle_body_quat = list(self._nominal_bottle_body_quat)
        if os.environ.get("TASK2_INIT_FROM_RECORDINGS", "0") == "1":
            recordings_dir = Path(
                os.environ.get(
                    "TASK2_RECORDINGS_DIR",
                    (
                        "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/"
                        "HumanoidVLA_MJ/output/20260625_task2_67"
                    ),
                )
            )
            if not hasattr(self, "_task2_recording_initializations"):
                self._task2_recording_initializations = (
                    self._load_recording_initializations(recordings_dir)
                )
                self._task2_recording_seed = int(
                    os.environ.get("TASK2_RECORDING_SEED", "0")
                )

            pool_size = len(self._task2_recording_initializations)
            recording_index = os.environ.get("TASK2_RECORDING_INDEX")
            if recording_index is not None:
                selected_index = int(recording_index)
                if not 0 <= selected_index < pool_size:
                    raise ValueError(
                        f"TASK2_RECORDING_INDEX={selected_index} is outside "
                        f"the available range 0..{pool_size - 1}"
                    )
            else:
                cycle_index, cycle_offset = divmod(reset_index, pool_size)
                recording_order = np.random.default_rng(
                    self._task2_recording_seed + cycle_index
                ).permutation(pool_size)
                selected_index = int(recording_order[cycle_offset])
            recording_name, robot_qpos, bottle_pose = (
                self._task2_recording_initializations[selected_index]
            )
            self.mujoco_initial_robot_qpos = robot_qpos.tolist()
            bottle_body_pos = bottle_pose[:3].copy()
            bottle_body_quat = bottle_pose[3:].tolist()
            print(
                "[Task2RecordingInit] "
                f"episode_reset={reset_index} recording={recording_name} "
                f"bottle_xyz=({bottle_body_pos[0]:.4f}, "
                f"{bottle_body_pos[1]:.4f}, {bottle_body_pos[2]:.4f})"
            )
        elif os.environ.get("TASK2_RANDOMIZE_BOTTLE", "0") == "1":
            seed_base = int(os.environ.get("TASK2_BOTTLE_SEED", "0"))
            rng = np.random.default_rng(seed_base + reset_index)
            x_range = np.fromstring(
                os.environ.get("TASK2_BOTTLE_X_RANGE", "0.64,0.73"),
                sep=",",
                dtype=np.float64,
            )
            y_range = np.fromstring(
                os.environ.get("TASK2_BOTTLE_Y_RANGE", "-0.23,-0.13"),
                sep=",",
                dtype=np.float64,
            )
            if x_range.size != 2 or y_range.size != 2:
                raise ValueError(
                    "TASK2_BOTTLE_X_RANGE and TASK2_BOTTLE_Y_RANGE must be "
                    "comma-separated min,max pairs"
                )
            sampled_center = np.asarray(
                [
                    rng.uniform(float(x_range[0]), float(x_range[1])),
                    rng.uniform(float(y_range[0]), float(y_range[1])),
                ]
            )
            bottle_body_pos[:2] += sampled_center - self._nominal_bottle_center_xy
            print(
                "[Task2BottleRandomize] "
                f"episode_reset={reset_index} "
                f"center_xy=({sampled_center[0]:.4f}, {sampled_center[1]:.4f})"
            )

        bottle_body_pos[0] += float(os.environ.get("TASK2_BOTTLE_X_OFFSET", "0"))
        bottle_body_pos[1] += float(os.environ.get("TASK2_BOTTLE_Y_OFFSET", "0"))

        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[2]["pos"] = bottle_body_pos.tolist()
        self.mujoco_extra_mjcf[2]["quat"] = bottle_body_quat

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)

        # Keep inference language conditioning identical to the task2 training data.
        self._instruction = (
            "Move forward, pick up the bottle from the table, then turn your body "
            "to the right, step on the trash can pedal, and then throw the bottle "
            "into the trash can."
        )
        self._target = None
        self.reward = 0.0
        self._bottle_start_z = float(bottle_body_pos[2])
        self._was_lifted = False
        self._was_pedal_pressed = False
        self._was_lid_opened = False
        self._task2_success = False
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    def _read_task_state(
        self,
        info: dict[str, Any],
        mujoco_env: Any,
    ) -> tuple[float, float, float, bool]:
        data = mujoco_env.mjData
        bottle_z = float(np.asarray(info["target"], dtype=np.float64)[2])
        pedal = float(data.joint("task2_trash_pedal_hinge_joint").qpos[0])
        lid = float(data.joint("task2_trash_lid_hinge_joint").qpos[0])

        bottle_center = np.asarray(data.geom("task2_bottle_collision").xpos)
        bin_base = data.site("task2_trash_base_site")
        bin_rotation = np.asarray(bin_base.xmat).reshape(3, 3)
        bottle_local = bin_rotation.T @ (bottle_center - np.asarray(bin_base.xpos))
        inside = bool(
            abs(float(bottle_local[0])) < 0.17
            and abs(float(bottle_local[1])) < 0.145
            and 0.05 < float(bottle_local[2]) < 0.50
        )
        return bottle_z, pedal, lid, inside

    def compute_reward(
        self,
        info: dict[str, Any],
        *args: Any,
        mujoco_env: Any = None,
        **kwargs: Any,
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Task2 reward requires the MuJoCo environment")
        bottle_z, pedal, lid, inside = self._read_task_state(info, mujoco_env)

        self._was_lifted |= bottle_z >= self._bottle_start_z + self._lift_threshold
        if self._was_lifted:
            self._was_pedal_pressed |= pedal <= self._pedal_threshold
        if self._was_pedal_pressed:
            self._was_lid_opened |= lid >= self._lid_threshold

        self._task2_success = inside
        self.reward = (
            1.0
            if self._task2_success
            else 0.25
            * sum((self._was_lifted, self._was_pedal_pressed, self._was_lid_opened))
        )
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._task2_success)
