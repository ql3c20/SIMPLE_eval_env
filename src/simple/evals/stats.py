"""Durable progress summaries for long-running SIMPLE evaluations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def append_eval_stats_line(eval_dir: str, line: str) -> None:
    """Append and fsync one line so completed episodes survive teardown stalls."""

    path = Path(eval_dir) / "eval_stats.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as file:
        file.write(line)
        file.flush()
        os.fsync(file.fileno())


class EvalStatsProgress:
    """Aggregate episode events and persist partial/final success rates."""

    def __init__(self, eval_dir: str, num_workers: int):
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}")
        self.eval_dir = eval_dir
        self.num_workers = num_workers
        self.expected_by_worker: dict[int, int] = {}
        self.stats: dict[str, bool] = {}
        self.final_written = False
        self._last_progress_line: str | None = None

    def handle(self, worker_id: int, payload: Mapping[str, Any]) -> None:
        event = payload.get("event")
        if event == "worker_init" and "total_episodes" in payload:
            self.expected_by_worker[worker_id] = int(payload["total_episodes"])
        elif event == "episode_end":
            episode = str(payload["episode"])
            if "success" not in payload:
                raise KeyError(f"episode_end event is missing success: {payload}")
            self.stats[episode] = bool(payload["success"])
        else:
            return
        self._write_progress_if_needed()

    def _expected_total(self) -> int | None:
        if len(self.expected_by_worker) != self.num_workers:
            return None
        return sum(self.expected_by_worker.values())

    def _write_progress_if_needed(self) -> None:
        if self.final_written or not self.stats:
            return
        completed = len(self.stats)
        successes = sum(self.stats.values())
        success_rate = successes / completed
        expected = self._expected_total()
        if expected is not None and completed >= expected:
            append_eval_stats_line(self.eval_dir, f"success rate: {success_rate:.2f} \n")
            self.final_written = True
            return

        expected_text = "?" if expected is None else str(expected)
        line = (
            f"partial success rate: {success_rate:.2f} "
            f"({successes}/{completed}, requested={expected_text}) \n"
        )
        if line != self._last_progress_line:
            append_eval_stats_line(self.eval_dir, line)
            self._last_progress_line = line

    def ensure_final(self, stats: Mapping[str, bool]) -> float:
        """Write the legacy final line once and return its success rate."""

        self.stats.update({str(key): bool(value) for key, value in stats.items()})
        success_rate = (
            sum(self.stats.values()) / len(self.stats) if self.stats else 0.0
        )
        if not self.final_written:
            append_eval_stats_line(self.eval_dir, f"success rate: {success_rate:.2f} \n")
            self.final_written = True
        return success_rate
