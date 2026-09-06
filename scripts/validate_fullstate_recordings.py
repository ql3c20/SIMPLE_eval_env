#!/usr/bin/env python3
"""Validate every recording used by one fixed fullstate evaluation task."""

from __future__ import annotations

import argparse
from pathlib import Path

from simple.tasks.g1_fullstate_20260615_task1 import G1Fullstate20260615Task1
from simple.tasks.g1_fullstate_20260729_task4 import G1Fullstate20260729Task4
from simple.tasks.g1_fullstate_20260804_task3 import G1Fullstate20260804Task3
from simple.tasks.g1_fullstate_20260805_task2 import G1Fullstate20260805Task2
from simple.tasks.g1_fullstate_20260825_task5 import G1Fullstate20260825Task5
from simple.tasks.g1_fullstate_20260828_task6 import G1Fullstate20260828Task6


TASK_CLASSES = {
    1: G1Fullstate20260615Task1,
    2: G1Fullstate20260805Task2,
    3: G1Fullstate20260804Task3,
    4: G1Fullstate20260729Task4,
    5: G1Fullstate20260825Task5,
    6: G1Fullstate20260828Task6,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=TASK_CLASSES, required=True)
    parser.add_argument("--recordings-dir", type=Path, required=True)
    args = parser.parse_args()

    task_class = TASK_CLASSES[args.task]
    recordings = task_class._load_recording_initializations(args.recordings_dir)
    scene_hashes = {recording.scene_sha256 for recording in recordings}
    if len(scene_hashes) != 1:
        raise RuntimeError(f"Expected one scene hash, got {sorted(scene_hashes)}")
    print(
        f"[recording-preflight] task={args.task} recordings={len(recordings)} "
        f"nq={task_class.recording_nq} scene_sha256={next(iter(scene_hashes))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
