from __future__ import annotations

from pathlib import Path

import numpy as np

from simple.baselines.kimodo_adapter import KimodoAdapterConfig, KimodoPolicyAdapter


def test_policy_only_first_frame_is_pinned_to_current_qpos(tmp_path, monkeypatch):
    cfg = KimodoAdapterConfig(
        source_fps=50.0,
        kimodo_fps=50.0,
        output_fps=50.0,
        keyframe_step=1,
        work_dir=tmp_path,
        keep_work=True,
        server_url=None,
        anchor_mode="policy_only",
        policy_only_initial_qpos=True,
        episode_subdir=False,
    )
    adapter = KimodoPolicyAdapter(cfg)
    adapter.begin_episode()

    def fake_run_kimodo(**kwargs):
        generated = np.zeros((4, 36), dtype=np.float32)
        generated[:, 2] = 0.75
        generated[:, 3] = 1.0
        generated[0, 0:2] = [9.0, -9.0]
        np.savetxt(
            Path(kwargs["output_stem"]).with_suffix(".csv"),
            generated,
            delimiter=",",
        )

    monkeypatch.setattr(adapter, "_run_kimodo", fake_run_kimodo)
    policy_action = np.zeros((4, 44), dtype=np.float32)
    current_qpos = np.zeros(36, dtype=np.float32)
    current_qpos[:3] = [0.12, -0.08, 0.79]
    current_qpos[3] = 1.0
    current_qpos[7:] = np.linspace(-0.2, 0.2, 29, dtype=np.float32)

    _, qpos50 = adapter.policy44_to_simple36(
        policy_action,
        prev_qpos_mujoco=current_qpos,
    )

    np.testing.assert_array_equal(qpos50[0], current_qpos)
