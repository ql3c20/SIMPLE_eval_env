#!/usr/bin/env python3
"""Relay only the latest mujoco_sim_state DDS sample to localhost UDP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23331)
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--dds-python-path",
        type=Path,
        default=os.environ.get("MUJOCO_DDS_PYTHON_PATH"),
        help="Directory containing dds_types.py",
    )
    return parser.parse_args()


def resolve_dds_python_path(requested: Path | None) -> Path:
    candidates = [requested] if requested is not None else []
    candidates.extend(
        [
            Path(
                "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/"
                "HumanoidVLA_MJ/example/python"
            ),
            Path(
                "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/"
                "HumanoidVLA_MJ_1/example/python"
            ),
        ]
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "dds_types.py").is_file():
            return candidate
    raise FileNotFoundError(
        "dds_types.py not found; pass --dds-python-path or set "
        "MUJOCO_DDS_PYTHON_PATH"
    )


def main() -> int:
    args = parse_args()
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    dds_python_path = resolve_dds_python_path(args.dds_python_path)
    sys.path.insert(0, os.fspath(dds_python_path))

    from cyclonedds.core import Policy, Qos
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from dds_types import MujocoSimStateDDS

    qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
    participant = DomainParticipant(args.domain_id)
    reader = DataReader(
        participant,
        Topic(participant, "mujoco_sim_state", MujocoSimStateDDS),
        qos=qos,
    )
    destination = (args.host, args.port)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / args.rate
    deadline = time.monotonic()
    last_sample = None
    print(
        f"[Task234Relay] DDS mujoco_sim_state -> udp://{args.host}:{args.port} "
        f"rate={args.rate:g}Hz qos=BestEffort/KeepLast(1)"
    )
    while True:
        for sample in reader.take():
            last_sample = sample
        if last_sample is not None:
            nq = int(last_sample.nq)
            nv = int(last_sample.nv)
            payload = {
                "nq": nq,
                "nv": nv,
                "sim_time": float(last_sample.sim_time),
                "running": int(last_sample.running),
                "qpos": [float(value) for value in last_sample.qpos[:nq]],
                "qvel": [float(value) for value in last_sample.qvel[:nv]],
            }
            udp.sendto(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                destination,
            )
        deadline += period
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            deadline = time.monotonic()


if __name__ == "__main__":
    raise SystemExit(main())
