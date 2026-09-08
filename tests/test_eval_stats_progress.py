from __future__ import annotations

from pathlib import Path

import pytest

from simple.evals.stats import EvalStatsProgress


def _lines(root: Path) -> list[str]:
    return (root / "eval_stats.txt").read_text().splitlines()


@pytest.mark.parametrize("success", [False, True])
def test_single_episode_writes_final_before_worker_teardown(
    tmp_path: Path, success: bool
) -> None:
    progress = EvalStatsProgress(str(tmp_path), num_workers=1)
    progress.handle(0, {"event": "worker_init", "total_episodes": 1})
    progress.handle(
        0,
        {"event": "episode_end", "episode": "episode_0", "success": success},
    )

    expected = "success rate: 1.00 " if success else "success rate: 0.00 "
    assert _lines(tmp_path) == [expected]


def test_interrupted_run_keeps_partial_success_rate(tmp_path: Path) -> None:
    progress = EvalStatsProgress(str(tmp_path), num_workers=1)
    progress.handle(0, {"event": "worker_init", "total_episodes": 10})
    for index, success in enumerate((True, False, True, False, False)):
        progress.handle(
            0,
            {
                "event": "episode_end",
                "episode": f"episode_{index}",
                "success": success,
            },
        )

    assert _lines(tmp_path)[-1] == (
        "partial success rate: 0.40 (2/5, requested=10) "
    )
    assert not any(line.startswith("success rate:") for line in _lines(tmp_path))


def test_multiworker_writes_one_global_final_rate(tmp_path: Path) -> None:
    progress = EvalStatsProgress(str(tmp_path), num_workers=2)
    progress.handle(0, {"event": "worker_init", "total_episodes": 2})
    progress.handle(1, {"event": "worker_init", "total_episodes": 2})
    for worker_id, episode, success in (
        (0, "episode_0", True),
        (1, "episode_1", False),
        (0, "episode_2", True),
        (1, "episode_3", True),
    ):
        progress.handle(
            worker_id,
            {"event": "episode_end", "episode": episode, "success": success},
        )

    progress.ensure_final(progress.stats)
    lines = _lines(tmp_path)
    assert lines[-1] == "success rate: 0.75 "
    assert sum(line.startswith("success rate:") for line in lines) == 1
