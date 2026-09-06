"""Fixed MuJoCo evaluation task for the 2026-06-15 green-cylinder demo."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simple.core.layout import Layout
from simple.core.types import Pose
from simple.tasks.g1_fullstate_recordings import FullstateRecordingMixin
from simple.tasks.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_fullstate_20260615_task1")
class G1Fullstate20260615Task1(
    FullstateRecordingMixin, G1WholebodyXMovePickTaskTeleop
):
    """Walk to the round table and pick up the green cylinder."""

    uid = "g1_fullstate_20260615_task1"
    label = "G1 Fullstate 20260615 Task 1"
    description = "Move forward to pick up the green cylinder."

    metadata = {
        **G1WholebodyXMovePickTaskTeleop.metadata,
        "dr_level": 0,
        "max_episode_steps": 800,
        "render_hz": 50,
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
        "position": [0.06, 0.06, 0.40],
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
            "/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ",
        )
    )
    recording_env_prefix = "TASK1"
    recording_default_dir = Path("/home/ubuntu/yzh/mujoco_recordings/20260615_task1_new")
    recording_nq = 57
    recording_semantic_fields = {
        0: "pelvis.floating_base_joint.x",
        49: "right_hand_index_1_joint.angle",
        50: "green_grasp_cylinder_free.x",
        56: "green_grasp_cylinder_free.qz",
    }
    _asset_root = (
        _humanoid_vla_root
        / "mujoco/model/task_assets/local_mjcf/primitive_round_table_green_cylinder"
    )
    mujoco_extra_mjcf = [
        {
            "path": _asset_root / "round_table.xml",
            "prefix": "primitive_round_table_",
            "pos": [1.3, 0.0, 0.0],
        },
        {
            "path": _asset_root / "green_cylinder.xml",
            "prefix": "green_grasp_cylinder_",
            # Nominal task initialization from scene_43dof.xml.  The first
            # data.csv row is already part-way through the demonstration and
            # places the cylinder almost over the table edge.
            "pos": [1.05, 0.0, 0.825],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "freejoint_body": "object",
            "freejoint_name": "free",
        },
    ]
    mujoco_object_body_names = {"target": "green_grasp_cylinder_object"}
    initial_target_height = 0.8249867
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
        selected = self._select_recording(options)
        self._layout = Layout()
        self._layout.add_robot(self.robot)
        self._layout.robot.pose = Pose(
            position=[0.0, 0.0, 0.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        )

        camera_cfg = copy.deepcopy(self.sensor_cfgs["head_stereo"])
        self._layout.add_camera("head_stereo", camera_cfg)

        self.mujoco_initial_robot_qpos = selected.qpos[:50].tolist()
        cylinder_pose = selected.qpos[50:57]
        self.mujoco_extra_mjcf = copy.deepcopy(type(self).mujoco_extra_mjcf)
        self.mujoco_extra_mjcf[1]["pos"] = cylinder_pose[:3].tolist()
        self.mujoco_extra_mjcf[1]["quat"] = cylinder_pose[3:].tolist()

        self._instruction = "walk forward and then pick up the green cylinder"
        self._target = None
        self._init_target_height = float(cylinder_pose[2])
        self._init_target_xy = np.asarray(cylinder_pose[:2], dtype=np.float64).copy()
        self.reward = 0.0
        self._task1_lift_steps = 0
        self._task1_grasp_steps = 0
        self._task1_grasp_anchor_position: np.ndarray | None = None
        self._task1_last_grasp_sides: tuple[str, ...] = ()
        self._task1_last_horizontal_move = 0.0
        self._task1_last_grasp_lift = 0.0
        self._task1_last_grasp_horizontal_move = 0.0
        self._task1_success = False
        self.robot.reset(spawn_pose=self._layout.robot.pose)

    @staticmethod
    def _valid_grasp_sides(contact_body_names: set[str]) -> tuple[str, ...]:
        """Return hands forming a thumb-opposition grasp on the cylinder.

        A single hand/cylinder contact is only a touch and must not qualify.
        The palm collision belongs to ``*_wrist_yaw_link`` in this G1 MJCF, so
        either the palm or an index/middle finger can oppose the thumb.
        """
        valid: list[str] = []
        for side in ("left", "right"):
            bodies = {
                name for name in contact_body_names if name.startswith(f"{side}_")
            }
            has_thumb = any("_hand_thumb_" in name for name in bodies)
            has_opposition = any(
                "_hand_index_" in name
                or "_hand_middle_" in name
                or name == f"{side}_wrist_yaw_link"
                for name in bodies
            )
            if has_thumb and has_opposition:
                valid.append(side)
        return tuple(valid)

    def _grasp_sides(self, mujoco_env: Any) -> tuple[str, ...]:
        if mujoco_env is None:
            raise ValueError("Task1 grasp success requires mujoco_env")
        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        hand_contacts: set[str] = set()
        target_prefix = "green_grasp_cylinder_"
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = model.geom(int(contact.geom1))
            geom2 = model.geom(int(contact.geom2))
            body1 = model.body(int(geom1.bodyid)).name or ""
            body2 = model.body(int(geom2.bodyid)).name or ""
            if body1.startswith(target_prefix) and body2.startswith(("left_", "right_")):
                hand_contacts.add(body2)
            elif body2.startswith(target_prefix) and body1.startswith(("left_", "right_")):
                hand_contacts.add(body1)
        return self._valid_grasp_sides(hand_contacts)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        target_position = np.asarray(info["target"], dtype=np.float64)[:3]
        target_height = float(target_position[2])
        self._last_target_height = target_height
        lift = max(0.0, target_height - self._init_target_height)
        horizontal_move = float(np.linalg.norm(target_position[:2] - self._init_target_xy))
        self._task1_last_horizontal_move = horizontal_move

        grasp_sides = self._grasp_sides(kwargs.get("mujoco_env"))
        self._task1_last_grasp_sides = grasp_sides
        if grasp_sides:
            if self._task1_grasp_anchor_position is None:
                # Displacement starts only after a real thumb-opposition grasp.
                # If the cylinder was pushed first, that earlier motion cannot
                # be reused to satisfy the success criterion.
                self._task1_grasp_anchor_position = target_position.copy()
            self._task1_grasp_steps += 1
            grasp_delta = target_position - self._task1_grasp_anchor_position
            grasp_lift = max(0.0, float(grasp_delta[2]))
            grasp_horizontal_move = float(np.linalg.norm(grasp_delta[:2]))
        else:
            # Any loss of the grasp breaks continuity.  A later re-grasp gets a
            # new anchor and must carry the cylinder another 3 cm itself.
            self._task1_grasp_anchor_position = None
            self._task1_grasp_steps = 0
            grasp_lift = 0.0
            grasp_horizontal_move = 0.0

        self._task1_last_grasp_lift = grasp_lift
        self._task1_last_grasp_horizontal_move = grasp_horizontal_move
        threshold_reached = grasp_lift >= 0.05 or grasp_horizontal_move >= 0.09
        self._task1_lift_steps = self._task1_lift_steps + 1 if threshold_reached else 0
        if grasp_sides and threshold_reached:
            self._task1_success = True
        grasp_progress = max(grasp_lift, grasp_horizontal_move)
        self.reward = 1.0 if self._task1_success else min(0.9, grasp_progress / 0.03)
        return self.reward

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return bool(self._task1_success)

    def evaluation_metrics(self) -> dict[str, Any]:
        return {
            "lift_m": float(max(0.0, self._last_target_height - self._init_target_height))
            if hasattr(self, "_last_target_height")
            else 0.0,
            "horizontal_move_m": float(self._task1_last_horizontal_move),
            "grasped": bool(self._task1_last_grasp_sides),
            "grasp_sides": list(self._task1_last_grasp_sides),
            "grasp_hold_steps": int(self._task1_grasp_steps),
            "grasp_carried_lift_m": float(self._task1_last_grasp_lift),
            "grasp_carried_horizontal_m": float(
                self._task1_last_grasp_horizontal_move
            ),
            "lift_hold_steps": int(self._task1_lift_steps),
        }
