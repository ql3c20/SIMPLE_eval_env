from __future__ import annotations

from types import SimpleNamespace

import cv2
import gymnasium as gym
import numpy as np

from simple.envs.wrappers.video_recorder import (
    VideoRecorder,
    task_video_recorder_options,
)


class _OneCameraEnv(gym.Env):
    metadata = {}

    def __init__(self, success: bool):
        self.observation_space = gym.spaces.Dict(
            {
                "front_camera": gym.spaces.Box(
                    0, 255, shape=(12, 16, 3), dtype=np.uint8
                ),
                "joint_qpos": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._success = success

    @staticmethod
    def _observation():
        return {
            "front_camera": np.zeros((12, 16, 3), dtype=np.uint8),
            "joint_qpos": np.zeros(2, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 0.0, True, False, {}


def test_task_video_options_are_one_based_only_for_flat_layout() -> None:
    football = SimpleNamespace(
        metadata={
            "video_output_layout": "classified_flat",
            "video_camera_keys": ("front_camera",),
        }
    )
    legacy = SimpleNamespace(metadata={})

    assert task_video_recorder_options(football, 0) == {
        "name_prefix": "ep1",
        "output_layout": "classified_flat",
        "camera_keys": ("front_camera",),
    }
    assert task_video_recorder_options(legacy, 0)["name_prefix"] == "episode_0"


def test_flat_recorder_classifies_success_without_episode_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "simple.envs.video_writer.is_ffmpeg_installed", lambda: False
    )
    env = VideoRecorder(
        _OneCameraEnv(success=True),
        video_folder=str(tmp_path),
        framerate=10,
        name_prefix="ep1",
        output_layout="classified_flat",
        camera_keys=("front_camera",),
    )

    env.reset()
    env.step(np.zeros(1, dtype=np.float32))
    env.release()

    output = tmp_path / "success" / "ep1_success.mp4"
    assert output.is_file()
    assert not (tmp_path / "ep1").exists()
    assert not (tmp_path / ".video_tmp").exists()
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    capture.release()


def test_flat_recorder_classifies_failure_with_requested_episode_number(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "simple.envs.video_writer.is_ffmpeg_installed", lambda: False
    )
    env = VideoRecorder(
        _OneCameraEnv(success=False),
        video_folder=str(tmp_path),
        framerate=10,
        name_prefix="ep2",
        output_layout="classified_flat",
        camera_keys=("front_camera",),
    )

    env.reset()
    env.release()

    assert (tmp_path / "failure" / "ep2_failure.mp4").is_file()
    assert not (tmp_path / "ep2").exists()
