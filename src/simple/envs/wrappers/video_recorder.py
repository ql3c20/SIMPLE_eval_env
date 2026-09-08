"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import shutil
import gymnasium as gym
from simple.envs.video_writer import VideoWriter
from datetime import datetime


def task_video_recorder_options(task, episode_index: int) -> dict:
    """Translate optional task video metadata into recorder arguments."""
    output_layout = task.metadata.get("video_output_layout", "per_episode")
    camera_keys = task.metadata.get("video_camera_keys")
    if camera_keys is not None:
        camera_keys = tuple(camera_keys)
    name_prefix = (
        f"ep{episode_index + 1}"
        if output_layout == "classified_flat"
        else f"episode_{episode_index}"
    )
    return {
        "name_prefix": name_prefix,
        "output_layout": output_layout,
        "camera_keys": camera_keys,
    }


class VideoRecorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        video_folder:str = "video",
        framerate:int = 10,
        # camera: List[str] = ["mujoco", "front_left", "wrist"],
        name_prefix:str|None = None, 
        write_png:bool = False,
        output_layout: str = "per_episode",
        camera_keys: tuple[str, ...] | None = None,
    ):
        if output_layout not in {"per_episode", "classified_flat"}:
            raise ValueError(f"Unsupported video output layout: {output_layout}")
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            name_prefix=name_prefix,
            write_png=write_png,
            output_layout=output_layout,
            camera_keys=camera_keys,
        )
        gym.Wrapper.__init__(self, env)

        os.makedirs(video_folder, exist_ok=True)
        
        # self._elapsed_steps = None
        self.work_dir = video_folder
        
        if name_prefix is None:
            now = datetime.now()
            name_prefix = now.isoformat().replace(":", "-").replace(".", "-")
            if self.unwrapped.__module__.startswith("simple.envs"):
                self.sim_mode = self.unwrapped.sim_mode # type: ignore
                name_prefix = f"{self.unwrapped.task.uid}_{name_prefix}" # type: ignore

        self.name_prefix = name_prefix
        self.write_png = write_png
        self.framerate = framerate
        self.output_layout = output_layout
        self.camera_keys = camera_keys
        self.video_writers = {}

    def reset(self, **kwargs):
        observations, info = super().reset(**kwargs)

        if kwargs.get("options") is not None:
            if  kwargs["options"].get("task_id") is not None:
                if self.output_layout == "per_episode":
                    # Preserve the historical reset-time task-id override for
                    # legacy recorders.  Flat layout names are already the
                    # required one-based epN identifiers.
                    self.name_prefix = kwargs["options"]["task_id"]
                    video_folder = f"{self.work_dir}/{self.name_prefix}"
                    if os.path.exists(video_folder):
                        shutil.rmtree(video_folder, ignore_errors=True)
                        print(f"Overwriting existing videos at {video_folder} folder")

                    os.makedirs(video_folder, exist_ok=True)

        self.video_writers = {}
        for key, subspace in self.unwrapped.observation_space.items():
            if len(subspace.shape) == 3 and subspace.shape[-1] == 3: # only record image observations
                if self.camera_keys is not None and key not in self.camera_keys:
                    continue
                if self.output_layout == "classified_flat":
                    raw_filename = f"{self.work_dir}/.video_tmp/{self.name_prefix}__{key}.mp4"
                else:
                    raw_filename = f"{self.work_dir}/{self.name_prefix}/{key}.mp4"
                self.video_writers[key] = VideoWriter(
                    raw_filename,
                    self.framerate, 
                    subspace.shape[:2][::-1], 
                    write_png=self.write_png
                )
                self.video_writers[key].write(observations[key])

        if self.output_layout == "classified_flat" and len(self.video_writers) != 1:
            raise ValueError(
                "classified_flat video output requires exactly one RGB camera; "
                f"found {tuple(self.video_writers)}"
            )

        self._is_released = False
        return observations, info

    def render(self):
        pass

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        for key, video_writer in self.video_writers.items():
            video_writer.write(observation[key])

        return observation, reward, terminated, truncated, info
    
    def release(self):
        if not self._is_released:
            success = bool(self.unwrapped._success)  # type: ignore
            for video_writer in self.video_writers.values():
                output_filename = None
                if self.output_layout == "classified_flat":
                    category = "success" if success else "failure"
                    suffix = "success" if success else "failure"
                    output_filename = (
                        f"{self.work_dir}/{category}/{self.name_prefix}_{suffix}.mp4"
                    )
                video_writer.release(success, output_filename=output_filename)
            if self.output_layout == "classified_flat":
                temp_dir = f"{self.work_dir}/.video_tmp"
                if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            self._is_released = True

    def close(self):
        """Closes the wrapper then the video recorder."""
        self.release()
        super().close()
