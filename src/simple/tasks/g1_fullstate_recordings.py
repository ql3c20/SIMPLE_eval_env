"""Shared recording initialization utilities for the HumanoidVLA tasks."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_QPOS_INDEX_RE = re.compile(r"\[qpos(\d+)\]$")


@dataclass(frozen=True)
class RecordingInitialization:
    name: str
    data_path: Path
    scene_path: Path
    scene_sha256: str
    qpos_columns: tuple[str, ...]
    qpos: np.ndarray


class FullstateRecordingMixin:
    """Strict, reproducible first-frame loading shared by Task1--Task4."""

    recording_env_prefix: str
    recording_default_dir: Path
    recording_nq: int
    recording_semantic_fields: Mapping[int, str]

    @classmethod
    def _load_recording_initializations(
        cls, recordings_dir: Path
    ) -> list[RecordingInitialization]:
        recordings_dir = recordings_dir.expanduser().resolve()
        initializations: list[RecordingInitialization] = []
        scene_hashes: dict[str, list[str]] = {}
        for data_path in sorted(recordings_dir.glob("*/data.csv")):
            scene_path = data_path.parent / "model_snapshot/mujoco/model/g1/scene_43dof.xml"
            if not scene_path.is_file():
                raise FileNotFoundError(f"Recording scene snapshot is missing: {scene_path}")
            try:
                with data_path.open("r", encoding="utf-8", newline="") as stream:
                    reader = csv.reader(stream)
                    header = next(reader)
                    first_row = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Recording needs a header and first row: {data_path}") from exc
            if len(header) != len(first_row):
                raise ValueError(f"Header/row length mismatch in {data_path}")

            indexed: dict[int, tuple[str, float]] = {}
            for column, value in zip(header, first_row, strict=True):
                match = _QPOS_INDEX_RE.search(column)
                if match:
                    index = int(match.group(1))
                    if index in indexed:
                        raise ValueError(f"Duplicate qpos{index} in {data_path}")
                    indexed[index] = (column, float(value))
            expected_indices = list(range(cls.recording_nq))
            if sorted(indexed) != expected_indices:
                missing = sorted(set(expected_indices) - set(indexed))
                extra = sorted(set(indexed) - set(expected_indices))
                raise ValueError(
                    f"Unexpected qpos layout in {data_path}: missing={missing}, extra={extra}"
                )
            for index, required_text in cls.recording_semantic_fields.items():
                column = indexed[index][0]
                if required_text not in column:
                    raise ValueError(
                        f"qpos{index} field {column!r} does not contain "
                        f"{required_text!r} in {data_path}"
                    )
            qpos = np.asarray([indexed[index][1] for index in expected_indices], dtype=np.float64)
            if not np.isfinite(qpos).all():
                raise ValueError(f"NaN/Inf qpos in {data_path}")
            quaternion_starts = [3]
            if cls.recording_nq in (57, 59):
                quaternion_starts.append(cls.recording_nq - 4)
            for start in quaternion_starts:
                norm = float(np.linalg.norm(qpos[start : start + 4]))
                if not np.isclose(norm, 1.0, atol=1e-3):
                    raise ValueError(
                        f"Invalid quaternion qpos[{start}:{start + 4}] "
                        f"norm={norm:.6f} in {data_path}"
                    )
            scene_sha256 = hashlib.sha256(scene_path.read_bytes()).hexdigest()
            scene_hashes.setdefault(scene_sha256, []).append(data_path.parent.name)
            initializations.append(
                RecordingInitialization(
                    name=data_path.parent.name,
                    data_path=data_path,
                    scene_path=scene_path,
                    scene_sha256=scene_sha256,
                    qpos_columns=tuple(indexed[index][0] for index in expected_indices),
                    qpos=qpos,
                )
            )
        if not initializations:
            raise FileNotFoundError(
                f"No recording directories containing data.csv found in {recordings_dir}"
            )
        if len(scene_hashes) != 1:
            details = {digest: names[:3] for digest, names in scene_hashes.items()}
            raise ValueError(f"Recordings do not share one scene_43dof.xml hash: {details}")
        return initializations

    def _select_recording(
        self, options: dict[str, Any] | None
    ) -> RecordingInitialization:
        prefix = self.recording_env_prefix
        recordings_dir = Path(
            os.environ.get(f"{prefix}_RECORDINGS_DIR", str(self.recording_default_dir))
        )
        cache_key = str(recordings_dir.expanduser().resolve())
        if getattr(self, "_recording_cache_key", None) != cache_key:
            self._recording_initializations = self._load_recording_initializations(recordings_dir)
            self._recording_cache_key = cache_key
        episode_index = int(
            (options or {}).get("episode_index", getattr(self, "_recording_reset_index", 0))
        )
        self._recording_reset_index = episode_index + 1
        if episode_index < 0:
            raise ValueError(f"episode_index must be non-negative, got {episode_index}")
        pool_size = len(self._recording_initializations)
        fixed_index = os.environ.get(f"{prefix}_RECORDING_INDEX")
        seed = int(os.environ.get(f"{prefix}_RECORDING_SEED", "0"))
        if fixed_index not in (None, ""):
            selected_index = int(fixed_index)
            if not 0 <= selected_index < pool_size:
                raise ValueError(
                    f"{prefix}_RECORDING_INDEX={selected_index} is outside 0..{pool_size - 1}"
                )
        else:
            cycle, offset = divmod(episode_index, pool_size)
            order = np.random.default_rng(seed + cycle).permutation(pool_size)
            selected_index = int(order[offset])
        selected = self._recording_initializations[selected_index]
        self.selected_recording_name = selected.name
        self.selected_recording_index = selected_index
        self.selected_recording_seed = seed
        self.selected_recording_scene_path = selected.scene_path
        self.selected_recording_scene_sha256 = selected.scene_sha256
        self.external_isaac_mjcf = selected.scene_path
        print(
            f"[{prefix}RecordingInit] episode={episode_index} "
            f"recording_index={selected_index} recording={selected.name} "
            f"scene_sha256={selected.scene_sha256[:12]}",
            flush=True,
        )
        return selected

    def evaluation_metrics(self) -> dict[str, Any]:
        return {}
