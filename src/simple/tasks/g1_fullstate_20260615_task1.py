"""Fixed MuJoCo evaluation task for the 2026-06-15 green-cylinder demo."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simple.core.layout import Layout
from simple.core.types import Pose
from simple.tasks.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_fullstate_20260615_task1")
class G1Fullstate20260615Task1(G1WholebodyXMovePickTaskTeleop):
    """Walk to the round table and pick up the green cylinder."""

    uid = "g1_fullstate_20260615_task1"
    label = "G1 Fullstate 20260615 Task 1"
    description = "Move forward to pick up the green cylinder."

    metadata = {
        **G1WholebodyXMovePickTaskTeleop.metadata,
        "dr_level": 0,
        "max_episode_steps": 800,
    }

    # Keep the declared observation space consistent with the actual camera.
    # Changing only the reset-time copy would leave Gym at 640x360 and cause
    # the 640x480 frames (and video writer) to be rejected.
    sensor_cfgs = copy.deepcopy(G1WholebodyXMovePickTaskTeleop.sensor_cfgs)
    sensor_cfgs["head_stereo"].width = 640
    sensor_cfgs["head_stereo"].height = 480
    sensor_cfgs["head_stereo"].baseline = 0.06
    # Convert MuJoCo's vertical fovy to CameraCfg's horizontal FOV for 4:3.
    # Keep an explicit override for controlled camera-ablation runs.
    # The recorded LeRobot video was rendered with the OAK-D W wide camera
    # model used by the HumanoidVLA_MJ converter:
    #   EGO_VIEW = 640x480 @ 50 Hz, HFOV ~= 95 deg, VFOV ~= 70 deg,
    #   MuJoCo cam_fovy = 70 deg, pitch_offset = -15 deg,
    #   apply_distortion = False.
    # Keep explicit env overrides for camera-ablation runs.
    _effective_fovy_deg = float(os.environ.get("TASK1_CAMERA_FOVY_DEG", "70.0"))
    sensor_cfgs["head_stereo"].fov = 2.0 * np.arctan(
        (640.0 / 480.0) * np.tan(np.deg2rad(_effective_fovy_deg) / 2.0)
    )
    _effective_pitch_offset_deg = float(
        os.environ.get("TASK1_CAMERA_PITCH_OFFSET_DEG", "-15.0")
    )
    # HumanoidVLA_MJ's camera-model "pitch_offset=-15 deg" is a collection
    # convention for the OAK-D W head camera.  MuJoCo's camera frame/euler
    # convention is opposite to the intuitive "look down" sign here: directly
    # adding -15 deg to the XML pitch makes the rendered view look farther
    # toward the horizon.  Subtract the collection offset so the default
    # oak_dw_wide setting produces the lower, body/feet-visible ego view.
    _camera_pitch_rad = -0.8 - np.deg2rad(_effective_pitch_offset_deg)
    _camera_yaw_rad = -1.57
    _camera_cp = np.cos(_camera_pitch_rad / 2.0)
    _camera_sp = np.sin(_camera_pitch_rad / 2.0)
    _camera_cy = np.cos(_camera_yaw_rad / 2.0)
    _camera_sy = np.sin(_camera_yaw_rad / 2.0)
    sensor_cfgs["head_stereo"].pose = {
        "position": [0.06, 0.03, 0.45],
        "quaternion": [
            # Native MuJoCo wxyz quaternion for euler xyz:
            # [0, -0.8 rad - TASK1_CAMERA_PITCH_OFFSET_DEG, -1.57].
            # Default offset is the OAK-D W wide collection setting (-15 deg).
            # The no-offset XML camera is:
            # [0.6515477437, 0.2752506882, -0.2754699743, -0.6510290959].
            float(_camera_cp * _camera_cy),
            float(_camera_sp * _camera_sy),
            float(_camera_sp * _camera_cy),
            float(_camera_cp * _camera_sy),
        ],
    }

    # Match the data collection camera rather than SIMPLE's generic G1 camera.
    # Quaternions are native MuJoCo wxyz.
    mujoco_native_head_camera = True
    mujoco_ground_visual = {
        "builtin": "mjBUILTIN_FLAT",
        "rgb1": [1.0, 1.0, 1.0],
        "rgb2": [1.0, 1.0, 1.0],
        "mark": "mjMARK_EDGE",
        "markrgb": [0.0, 0.0, 0.0],
        "width": 64,
        "height": 64,
        "texrepeat": [12.0, 12.0],
        "reflectance": 0.0,
    }

    _humanoid_vla_root = Path(
        os.environ.get(
            "HUMANOID_VLA_MJ_ROOT",
            "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ",
        )
    )
    _asset_root = (
        _humanoid_vla_root
        / "mujoco/model/task_assets/local_mjcf/primitive_round_table_green_cylinder"
    )
    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "round_table.xml",
            "prefix": "task1_table_",
            "pos": [1.3, 0.0, 0.0],
        },
        {
            "path": _asset_root / "green_cylinder.xml",
            "prefix": "task1_cylinder_",
            # Nominal task initialization from scene_43dof.xml.  The first
            # data.csv row is already part-way through the demonstration and
            # places the cylinder almost over the table edge.
            "pos": [1.05, 0.0, 0.825],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "freejoint_body": "object",
            "freejoint_name": "green_cylinder_free",
        },
    ]
    mujoco_object_body_names = {"target": "task1_cylinder_object"}
    initial_target_height = 0.8249867
    _nominal_cylinder_pos = np.asarray([1.05, 0.0, 0.825], dtype=np.float64)
    # First recorded robot qpos row from 20260615_141610_g1_sim/data.csv.
    # Ordering matches the G1 Sonic MJCF: floating root + 43 actuated joints.
    mujoco_initial_robot_qpos = [
        0.0206043354, 0.0134953979, 0.7878130289,
        0.9968620637, 0.0022908446, 0.0789099019, 0.0058314125,
        -0.2189575108, 0.0334457700, 0.1918361194, 0.1584133989,
        -0.0938559860, 0.0064830734, -0.2379501221, -0.1043142336,
        -0.3903334892, 0.2163883240, -0.1179534276, -0.0061890220,
        -0.0225069025, 0.0844717538, 0.0212448637, -0.0815536815,
        0.1128131742, -0.2300022274, 1.2258022296, -0.0443666491,
        0.0226888213, 0.1066635377, 0.0435490960, -0.0265756979,
        -0.0000290518, -0.0004389300, -0.0000769215, -0.0004809546,
        -0.0000777691, -0.0082179868, 0.0029625081, 0.3374948280,
        0.1770043920, 0.3815248960, -0.0762979387, -0.1810697752,
        -0.0001217408, -0.0003242507, -0.0000456221, -0.0000135413,
        -0.0000019872, -0.0000135553, -0.0000019893,
    ]

    def reset(
        self,
        seed: int | None = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        # This task intentionally has one deterministic initialization, matching
        # HumanoidVLA_MJ/20260615/20260615_141610_g1_sim.
        self._layout = Layout()
        self._layout.add_robot(self.robot)
        self._layout.robot.pose = Pose(
            position=[0.0, 0.0, 0.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        )

        # Optional small object randomization for fixed-task robustness tests.
        # Keep it opt-in so the default task exactly matches the recorded
        # 20260615 initialization.  The ranges are intentionally narrow: the
        # learned policy was trained for a table-top cylinder near this nominal
        # location, not for arbitrary table positions.
        reset_index = int(getattr(self, "_task1_reset_index", 0))
        self._task1_reset_index = reset_index + 1
        cylinder_pos = self._nominal_cylinder_pos.copy()
        if os.environ.get("TASK1_RANDOMIZE_OBJECT", "0") == "1":
            seed_base = int(os.environ.get("TASK1_OBJECT_SEED", "0"))
            rng = np.random.default_rng(seed_base + reset_index)
            x_range = np.fromstring(
                os.environ.get("TASK1_OBJECT_X_RANGE", "1.00,1.10"),
                sep=",",
                dtype=np.float64,
            )
            y_range = np.fromstring(
                os.environ.get("TASK1_OBJECT_Y_RANGE", "-0.05,0.05"),
                sep=",",
                dtype=np.float64,
            )
            if x_range.size != 2 or y_range.size != 2:
                raise ValueError(
                    "TASK1_OBJECT_X_RANGE and TASK1_OBJECT_Y_RANGE must be "
                    "comma-separated min,max pairs"
                )
            cylinder_pos[0] = rng.uniform(float(x_range[0]), float(x_range[1]))
            cylinder_pos[1] = rng.uniform(float(y_range[0]), float(y_range[1]))
            print(
                "[Task1ObjectRandomize] "
                f"episode_reset={reset_index} "
                f"pos=({cylinder_pos[0]:.4f}, {cylinder_pos[1]:.4f}, {cylinder_pos[2]:.4f})"
            )
        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[1]["pos"] = cylinder_pos.tolist()

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)

        self._instruction = "move forward to pick up the cylinder"
        self._target = None
        self._init_target_height = self.initial_target_height
        self.reward = 0.0
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        target_height = float(np.asarray(info["target"])[2])
        lift = max(0.0, target_height - self.initial_target_height)
        self.reward = min(1.0, lift / 0.10)
        return self.reward

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self.compute_reward(info, *args, **kwargs) >= self.success_criteria
