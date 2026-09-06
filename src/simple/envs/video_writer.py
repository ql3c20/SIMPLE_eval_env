"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import cv2
import shutil
import numpy as np
from PIL import Image

from simple.utils import is_ffmpeg_installed


class VideoWriter:
    def __init__(self, filename, 
                 framerate, 
                 resolution, # (w,h)
                 write_png = False):

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if os.path.exists(filename):
            print("remove existing file:", filename)
            os.unlink(filename)

        self.resize = False
        resolution = (resolution[0]//2, resolution[1]//2) if self.resize else resolution
        self.resolution = tuple(int(value) for value in resolution)
        # resolution = [H, W]
        self.video_writer = cv2.VideoWriter(
            filename, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            framerate, 
            self.resolution
        )
        if not self.video_writer.isOpened():
            raise RuntimeError(
                f"Failed to open video writer for {filename} at "
                f"{self.resolution[0]}x{self.resolution[1]} @ {framerate} FPS"
            )

        self.frame_idx = 0
        self.filename = filename
        self.write_png = write_png

        self.is_ffmpeg_installed = is_ffmpeg_installed()
        if not self.is_ffmpeg_installed:
            import warnings
            warnings.warn("Warning: FFmpeg is not installed. Video files may not be playable inside VSCode.", stacklevel=2)

    def write(self, image):
        assert image.dtype == np.uint8
        if self.resize:
            h, w = image.shape[:2]
            image = cv2.resize(image, (w//2, h//2))
        actual_resolution = tuple(int(value) for value in image.shape[:2][::-1])
        if actual_resolution != self.resolution:
            raise ValueError(
                f"Video frame size changed for {self.filename}: expected "
                f"{self.resolution[0]}x{self.resolution[1]}, got "
                f"{actual_resolution[0]}x{actual_resolution[1]}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(image)
        if self.write_png:
            # Image.fromarray(image).save(f"{self.filename}_{self.frame_idx:03d}.png")
            cv2.imwrite(f"{self.filename}_{self.frame_idx:03d}.png", image)
        self.frame_idx += 1

    def release(self, success = True):
        if getattr(self, "_released", False):
            return self.filename

        self.video_writer.release()
        suffix = "success" if success else "failed"

        stem, extension = os.path.splitext(self.filename)
        newfilename = f"{stem}_{suffix}{extension}"
        if os.path.exists(newfilename):
            print(f"remove existing file: {newfilename}")
            os.remove(newfilename)

        transcoded = False
        if self.is_ffmpeg_installed:
            import subprocess

            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    self.filename,
                    "-vcodec",
                    "libx264",
                    newfilename,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            transcoded = completed.returncode == 0 and os.path.exists(newfilename)
            if transcoded:
                os.remove(self.filename)
            else:
                print(
                    f"FFmpeg conversion failed; preserving the original codec as {newfilename}"
                )

        # The success/failure label is useful independently of FFmpeg.  When
        # FFmpeg is unavailable (or conversion fails), keep the original codec
        # and atomically rename the completed recording instead.
        if not transcoded and os.path.exists(self.filename):
            os.replace(self.filename, newfilename)

        self.filename = newfilename
        self._released = True
        return newfilename
