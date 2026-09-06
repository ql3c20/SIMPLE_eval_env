from __future__ import annotations

from pathlib import Path

import numpy as np

from simple.tasks.g1_fullstate_recordings import (
    FullstateRecordingMixin,
    RecordingInitialization,
)


class _FakeSelector(FullstateRecordingMixin):
    recording_env_prefix = "TESTRECORDING"
    recording_default_dir = Path("/unused")
    recording_nq = 1
    recording_semantic_fields = {}

    @classmethod
    def _load_recording_initializations(cls, recordings_dir):
        return [
            RecordingInitialization(
                name=f"recording_{index}",
                data_path=Path(f"/{index}/data.csv"),
                scene_path=Path(f"/{index}/scene.xml"),
                scene_sha256="same-scene",
                qpos_columns=("qpos0",),
                qpos=np.asarray([index], dtype=np.float64),
            )
            for index in range(8)
        ]


def test_seeded_selection_is_reproducible_and_without_replacement(monkeypatch):
    monkeypatch.setenv("TESTRECORDING_RECORDING_SEED", "42")
    left, right = _FakeSelector(), _FakeSelector()
    left_order = [left._select_recording({"episode_index": i}).name for i in range(8)]
    right_order = [right._select_recording({"episode_index": i}).name for i in range(8)]
    assert left_order == right_order
    assert len(set(left_order)) == 8


def test_fixed_recording_index(monkeypatch):
    monkeypatch.setenv("TESTRECORDING_RECORDING_INDEX", "3")
    selector = _FakeSelector()
    assert [selector._select_recording({"episode_index": i}).name for i in range(4)] == [
        "recording_3"
    ] * 4
