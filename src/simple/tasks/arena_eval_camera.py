"""Shared camera configuration for HumanoidArena SIMPLE evaluations."""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import numpy as np


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _positive_env_number(name: str, default: str, cast: type) -> Any:
    raw = os.environ.get(name, default)
    try:
        value = cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive {cast.__name__}; got {raw!r}") from exc
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive; got {raw!r}")
    return value


def configure_arena_eval_cameras(task: Any) -> None:
    """Give one task instance an optional wide recording-only head camera.

    The policy camera is never modified.  The default 1280x720 recording
    camera retains the Arena camera's 15 mm vertical aperture and expands the
    horizontal aperture from 20 mm to 26.666... mm.
    """
    task.sensor_cfgs = copy.deepcopy(type(task).sensor_cfgs)
    task.metadata = dict(task.metadata)
    if not _env_flag("ARENA_AUX_VIDEO_ENABLED", False):
        return

    width = _positive_env_number("ARENA_AUX_VIDEO_WIDTH", "1280", int)
    height = _positive_env_number("ARENA_AUX_VIDEO_HEIGHT", "720", int)
    horizontal_aperture = _positive_env_number(
        "ARENA_AUX_VIDEO_HORIZONTAL_APERTURE", "26.666666666666668", float
    )
    policy_camera = task.sensor_cfgs["front_camera"]
    video_camera = copy.deepcopy(policy_camera)
    video_camera.uid = "ArenaEvalVideoCamera"
    video_camera.width = width
    video_camera.height = height
    video_camera.fov = 2.0 * np.arctan(
        horizontal_aperture / (2.0 * video_camera.focal_length)
    )
    task.sensor_cfgs["video_camera"] = video_camera
    task.metadata["video_camera_keys"] = ("video_camera",)


def add_arena_eval_cameras(task: Any) -> None:
    """Add all configured Arena cameras to the current task layout."""
    task._layout.add_camera(  # noqa: SLF001 - helper owns task layout assembly
        "front_camera", copy.deepcopy(task.sensor_cfgs["front_camera"])
    )
    if "video_camera" in task.sensor_cfgs:
        task._layout.add_camera(  # noqa: SLF001
            "video_camera", copy.deepcopy(task.sensor_cfgs["video_camera"])
        )
