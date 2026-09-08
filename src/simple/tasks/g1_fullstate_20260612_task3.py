"""Fixed MuJoCo evaluation task for the 2026-06-12 pedal-bin demo."""

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


@TaskRegistry.register("g1_fullstate_20260612_task3")
class G1Fullstate20260612Task3(G1Fullstate20260615Task1):
    """Walk to the pedal bin, step on its pedal, and open the lid."""

    uid = "g1_fullstate_20260612_task3"
    label = "G1 Fullstate 20260612 Task 3"
    description = "Move forward, step on the trash can pedal, and open the lid."

    metadata = {
        **G1Fullstate20260615Task1.metadata,
        "max_episode_steps": 800,
    }

    _humanoid_vla_root = Path(
        os.environ.get(
            "HUMANOID_VLA_MJ_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ",
        )
    )
    _asset_path = (
        _humanoid_vla_root
        / "mujoco/model/task_assets/local_mjcf/step_trash_can/trash_can.xml"
    )
    mujoco_extra_mjcf = [
        {
            "path": _asset_path,
            "prefix": "task3_trash_",
            # All 109 saved task3 scene snapshots use this authored pose.
            "pos": [1.0, -0.18, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
        },
    ]
    mujoco_object_body_names: dict[str, str] = {}

    # First robot qpos row from 20260612_143643_g1_sim/data.csv. This is also
    # the fallback when recording-based initialization is disabled.
    mujoco_initial_robot_qpos = [
        -0.2297761883677429,
        -0.09140866166722521,
        0.7865767069586987,
        0.9955573221401874,
        0.0115050288749754,
        0.03031946778556359,
        -0.08839673363221692,
        -0.1773104237499449,
        0.08058286929655339,
        0.2061093282013007,
        0.1397860549428836,
        -0.01044300932448751,
        -0.03422494863688258,
        0.004131754255452096,
        -0.15721920444707,
        -0.425422019573671,
        0.1785911503835388,
        -0.1960485008416845,
        0.08379507956272267,
        0.07458322728468184,
        0.05718248162050339,
        -0.0002946669296864089,
        0.138770677655538,
        0.2569722862234368,
        -0.07089824704361895,
        -0.4179110920044932,
        0.5382091444812317,
        0.05887658620961783,
        0.1132399883131696,
        -0.0008184393365870649,
        0.003934623190703397,
        0.0005930681104904176,
        -0.0003686343645757946,
        -5.589649048128782e-05,
        -0.0003938823127106863,
        -5.956499064867346e-05,
        0.2183211372757524,
        -0.1895012258168656,
        0.09158360003790124,
        -0.4319643550612968,
        -0.2453435350951626,
        -0.0125313419372732,
        -0.3383156548251093,
        0.03484476625756759,
        -0.04763282374785938,
        -0.0002080527130580516,
        0.0001256079684410216,
        1.841392072512719e-05,
        0.0001547402019893499,
        2.271223530122426e-05,
    ]

    # The source task config declares 1.2 rad as the successful lid angle.
    _pedal_threshold = -0.18
    _lid_threshold = 1.2

    @staticmethod
    def _load_recording_initializations(
        recordings_dir: Path,
    ) -> list[tuple[str, np.ndarray]]:
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

            missing = [index for index in range(50) if index not in qpos]
            if missing:
                raise ValueError(f"Missing robot qpos indices {missing} in {data_path}")

            robot_qpos = np.asarray([qpos[index] for index in range(50)])
            initializations.append((data_path.parent.name, robot_qpos))

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

        reset_index = int(getattr(self, "_task3_reset_index", 0))
        self._task3_reset_index = reset_index + 1
        self.mujoco_initial_robot_qpos = list(type(self).mujoco_initial_robot_qpos)
        if os.environ.get("TASK3_INIT_FROM_RECORDINGS", "0") == "1":
            recordings_dir = Path(
                os.environ.get(
                    "TASK3_RECORDINGS_DIR",
                    (
                        "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/"
                        "HumanoidVLA_MJ/output/20260612_task3"
                    ),
                )
            )
            if not hasattr(self, "_task3_recording_initializations"):
                self._task3_recording_initializations = (
                    self._load_recording_initializations(recordings_dir)
                )
                self._task3_recording_seed = int(
                    os.environ.get("TASK3_RECORDING_SEED", "0")
                )

            pool_size = len(self._task3_recording_initializations)
            recording_index = os.environ.get("TASK3_RECORDING_INDEX")
            if recording_index is not None:
                selected_index = int(recording_index)
                if not 0 <= selected_index < pool_size:
                    raise ValueError(
                        f"TASK3_RECORDING_INDEX={selected_index} is outside "
                        f"the available range 0..{pool_size - 1}"
                    )
            else:
                cycle_index, cycle_offset = divmod(reset_index, pool_size)
                recording_order = np.random.default_rng(
                    self._task3_recording_seed + cycle_index
                ).permutation(pool_size)
                selected_index = int(recording_order[cycle_offset])

            recording_name, robot_qpos = self._task3_recording_initializations[
                selected_index
            ]
            self.mujoco_initial_robot_qpos = robot_qpos.tolist()
            print(
                "[Task3RecordingInit] "
                f"episode_reset={reset_index} recording={recording_name}"
            )

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)

        # Allow each Task3 checkpoint to use the exact language annotation from
        # the dataset it was trained on.
        self._instruction = os.environ.get(
            "TASK3_INSTRUCTION",
            "perform the manipulation task in simulation",
        )
        self._target = None
        self.reward = 0.0
        self._task3_success = False
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    def _read_task_state(self, mujoco_env: Any) -> tuple[float, float]:
        data = mujoco_env.mjData
        pedal = float(data.joint("task3_trash_pedal_hinge_joint").qpos[0])
        lid = float(data.joint("task3_trash_lid_hinge_joint").qpos[0])
        return pedal, lid

    def compute_reward(
        self,
        info: dict[str, Any],
        *args: Any,
        mujoco_env: Any = None,
        **kwargs: Any,
    ) -> float:
        if mujoco_env is None:
            raise ValueError("Task3 reward requires the MuJoCo environment")

        pedal, lid = self._read_task_state(mujoco_env)
        self._task3_success |= (
            pedal <= self._pedal_threshold and lid >= self._lid_threshold
        )
        pedal_progress = np.clip(pedal / self._pedal_threshold, 0.0, 1.0)
        lid_progress = np.clip(lid / self._lid_threshold, 0.0, 1.0)
        self.reward = (
            1.0
            if self._task3_success
            else 0.5 * float(pedal_progress) + 0.5 * float(lid_progress)
        )
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        return bool(self._task3_success)
