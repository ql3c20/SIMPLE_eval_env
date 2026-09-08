"""Strict Task2/3/4 SIMPLE eval tasks for MuJoCo physics + Isaac Ego RGB."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from simple.evals.task234_reproduction import (
    EGO_CAMERA,
    TASK_ASSET_RELATIVE_PATHS,
    TaskReproductionSpec,
    ego_camera_quaternion_wxyz,
    get_task_spec,
)
from simple.scenes.hssd import HssdSuite
from simple.sensors import StereoCameraCfg
from simple.tasks.g1_fullstate_20260612_task3 import G1Fullstate20260612Task3
from simple.tasks.g1_fullstate_20260625_task2 import G1Fullstate20260625Task2
from simple.tasks.g1_fullstate_20260729_task4 import G1Fullstate20260729Task4
from simple.tasks.registry import TaskRegistry


def _camera_cfg() -> StereoCameraCfg:
    # CameraCfg.fov is horizontal.  The frozen contract specifies vertical FOV.
    horizontal_fov = 2.0 * np.arctan(
        (EGO_CAMERA.width / EGO_CAMERA.height)
        * np.tan(np.deg2rad(EGO_CAMERA.vertical_fov_deg) / 2.0)
    )
    return StereoCameraCfg(
        uid="task234_isaac_ego",
        mount="eye_in_head",
        width=EGO_CAMERA.width,
        height=EGO_CAMERA.height,
        focal_length=1.0,
        fov=float(horizontal_fov),
        near=EGO_CAMERA.near_clip,
        far=100.0,
        baseline=0.0,
        pose={
            "position": list(EGO_CAMERA.eye),
            "quaternion": ego_camera_quaternion_wxyz().tolist(),
        },
    )


def _hssd_scene(spec: TaskReproductionSpec) -> HssdSuite:
    hssd = spec.hssd
    name = "102344250" if hssd.scene_uid == "scene13" else "102344280"
    return HssdSuite(
        {
            "uid": hssd.scene_uid,
            "data_dir": "scenes/hssd",
            "name": name,
            "default_prim_path": "/World/scene",
            "scale": hssd.scale,
            "hide_ceilings": True,
            "hide_walls": False,
            "center_offset_limit_up": list(hssd.target_surface_center),
            "center_offset_limit_down": list(hssd.target_surface_center),
            "center_orientation_limit_up": list(hssd.rotation_xyz_deg),
            "center_orientation_limit_down": list(hssd.rotation_xyz_deg),
            "surface": {
                "category": "insertion_surface",
                "prim_path": f"/World/{hssd.insertion_surface}",
            },
        }
    )


class _Task234IsaacEvalMixin:
    reproduction_spec: TaskReproductionSpec
    sensor_cfgs = {"head_stereo": _camera_cfg()}

    @property
    def observation_space(self) -> spaces.Space:
        observation_space = super().observation_space
        assert isinstance(observation_space, spaces.Dict)
        return spaces.Dict(
            {
                key: value
                for key, value in observation_space.spaces.items()
                if key != "head_stereo_right"
            }
        )

    def _configure_reproduction(self) -> None:
        spec = self.reproduction_spec
        asset_root = Path(
            os.environ.get(
                "SIMPLE_TASK_ASSETS_ROOT",
                "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets",
            )
        ).resolve()
        task_asset_paths = tuple(
            asset_root / relative_path
            for relative_path in TASK_ASSET_RELATIVE_PATHS[spec.task_number]
        )
        missing = [path for path in task_asset_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Task{spec.task_number} inference assets are missing: {missing}. "
                "Set SIMPLE_TASK_ASSETS_ROOT to the downloaded asset directory."
            )

        # Keep the original SIMPLE task construction, initialization, language
        # instruction and success checks.  Only replace its source MJCF paths
        # with the explicitly downloaded inference assets.
        self.mujoco_extra_mjcf = copy.deepcopy(self.mujoco_extra_mjcf)
        if len(self.mujoco_extra_mjcf) != len(task_asset_paths):
            raise ValueError(
                f"Task{spec.task_number} expected {len(task_asset_paths)} SIMPLE "
                f"MJCF fragments, got {len(self.mujoco_extra_mjcf)}"
            )
        for fragment, asset_path in zip(
            self.mujoco_extra_mjcf, task_asset_paths, strict=True
        ):
            fragment["path"] = asset_path

        print(
            f"[Task{spec.task_number}Inference] task_assets={asset_root}"
        )
        self.mujoco_expected_dimensions = spec.expected_dimensions()
        self.isaac_reproduction_spec = spec
        self.isaac_hssd_usd_relative_path = spec.hssd.usd_relative_path
        self.isaac_policy_camera_parent_prim = EGO_CAMERA.parent_prim
        self.isaac_policy_camera_name = "head_stereo_left"
        self._layout.scene = _hssd_scene(spec)

        # super().reset() created the cameras from our frozen class-level config.
        # Retain only one policy render product; a GUI/world camera is never used
        # as policy input.
        self._layout.cameras.pop("head_stereo_right", None)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._configure_reproduction()

@TaskRegistry.register("g1_fullstate_20260805_task2_isaac_eval")
class G1Fullstate20260805Task2IsaacEval(
    _Task234IsaacEvalMixin, G1Fullstate20260625Task2
):
    uid = "g1_fullstate_20260805_task2_isaac_eval"
    label = "G1 Task2 MuJoCo Physics + Isaac Ego Eval"
    reproduction_spec = get_task_spec(2)
    metadata = {
        **G1Fullstate20260625Task2.metadata,
        "render_hz": 50,
        "control_hz": 50,
        "isaac_read_only_mirror": True,
    }


@TaskRegistry.register("g1_fullstate_20260804_task3_isaac_eval")
class G1Fullstate20260804Task3IsaacEval(
    _Task234IsaacEvalMixin, G1Fullstate20260612Task3
):
    uid = "g1_fullstate_20260804_task3_isaac_eval"
    label = "G1 Task3 MuJoCo Physics + Isaac Ego Eval"
    reproduction_spec = get_task_spec(3)
    metadata = {
        **G1Fullstate20260612Task3.metadata,
        "render_hz": 50,
        "control_hz": 50,
        "isaac_read_only_mirror": True,
    }


@TaskRegistry.register("g1_fullstate_20260729_task4_isaac_eval")
class G1Fullstate20260729Task4IsaacEval(
    _Task234IsaacEvalMixin, G1Fullstate20260729Task4
):
    uid = "g1_fullstate_20260729_task4_isaac_eval"
    label = "G1 Task4 MuJoCo Physics + Isaac Ego Eval"
    reproduction_spec = get_task_spec(4)
    metadata = {
        **G1Fullstate20260729Task4.metadata,
        "render_hz": 50,
        "control_hz": 50,
        "isaac_read_only_mirror": True,
    }
