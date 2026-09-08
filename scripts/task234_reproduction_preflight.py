#!/usr/bin/env python3
"""Fail-fast inference-asset validation for Task2/3/4 SIMPLE eval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from simple.evals.task234_reproduction import (
    EGO_CAMERA,
    get_task_spec,
    inference_asset_paths,
    missing_inference_assets,
    write_camera_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("2", "3", "4", "all"), default="all")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="SIMPLE data directory containing scenes/hssd",
    )
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument(
        "--task-assets-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SIMPLE_TASK_ASSETS_ROOT",
                "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets",
            )
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_numbers = (2, 3, 4) if args.task == "all" else (int(args.task),)
    failures: list[Path] = []
    for task_number in task_numbers:
        spec = get_task_spec(task_number)
        absent = missing_inference_assets(
            spec,
            simple_data_root=args.data_root,
            task_assets_root=args.task_assets_root,
        )
        if absent:
            failures.extend(absent)
            print(f"[Task{task_number}] MISSING")
            for path in absent:
                print(f"  - {path}")
            continue

        paths = inference_asset_paths(
            spec,
            simple_data_root=args.data_root,
            task_assets_root=args.task_assets_root,
        )
        print(f"[Task{task_number}] OK checkpoint-only inference assets")
        for path in paths:
            print(f"  - {path}")
        if args.manifest_dir is not None:
            write_camera_manifest(
                args.manifest_dir / f"task{task_number}_ego_manifest.json", spec
            )

    print(
        "Ego: "
        f"parent={EGO_CAMERA.parent_prim} eye={EGO_CAMERA.eye} "
        f"forward={EGO_CAMERA.forward} up={EGO_CAMERA.up} "
        f"pitch={EGO_CAMERA.pitch_deg}deg vfov={EGO_CAMERA.vertical_fov_deg}deg "
        f"near={EGO_CAMERA.near_clip} resolution="
        f"{EGO_CAMERA.width}x{EGO_CAMERA.height}"
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
