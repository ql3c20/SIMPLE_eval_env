"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from .policy_debugger import PolicyDebugger
from .video_recorder import VideoRecorder, task_video_recorder_options
from .episode_extractor import EpisodeExtractor

try:
    from .data_recoder import DataRecorder
except ModuleNotFoundError as exc:
    if exc.name != "envlogger":
        raise
    # Dataset recording is optional for policy evaluation/video capture.
    DataRecorder = None
