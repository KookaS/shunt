"""Declarative dataset registry + the default hyperparameter grid for the sweep."""

# The committable trajectory planes are registered here; the terminal `results.csv` bootstrap
# turns the existing measured grid into degenerate length-1 trajectories so the module runs
# end-to-end on REAL data before any paid multi-step collection (no synthetic data).

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from benchmark.escalation.normalize.base import build_trajectory, make_step
from benchmark.escalation.replay import GridPoint

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.escalation.schema import Trajectory

# escalate_after_n x stale_window x ladder — the offline sweep / optimal-stop search grid.
DEFAULT_GRID: Final[list[GridPoint]] = [
    GridPoint(escalate_after_n=n, stale_window=w, ladder=ladder)
    for n in (2, 3)
    for w in (5, 10, 20)
    for ladder in ("effort_then_rank", "rank_only")
]

_TRUE = "True"


@dataclass(frozen=True)
class DatasetSpec:
    """One registered trajectory source."""

    name: str
    framework: str
    license: str | None
    path: Path
    committable: bool


def results_csv_bootstrap(
    results_path: Path, dataset_revision: str | None = None
) -> list[Trajectory]:
    """Each REAL results.csv row → a degenerate length-1 trajectory (one terminal decision).

    A length-1 stream CANNOT fire the recurrence trigger; this bootstrap validates the
    prefix-labeler, the metrics, and the plots on real (coarse) terminal data — not the trigger.
    """
    trajectories: list[Trajectory] = []
    with results_path.open(encoding="utf-8", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            resolved = row.get("pass") == _TRUE
            step = make_step(
                step_index=0,
                observation="",
                action="terminal",
                tool="terminal",
                args=None,
                result=row.get("pass", ""),
                metadata={
                    "model": row.get("model", ""),
                    "challenge_id": row.get("challenge_id", ""),
                },
                status="ok" if resolved else "error",
            )
            meta = {
                "trajectory_id": f"{row.get('challenge_id', 'row')}:{row.get('model', '')}:{i}",
                "dataset": "results_csv_bootstrap",
                "terminal_resolved": resolved,
                "instance_id": row.get("challenge_id"),
                "dataset_revision": dataset_revision,
            }
            trajectories.append(build_trajectory([step], meta, "results_bootstrap"))
    return trajectories


def is_degenerate(traj: Trajectory) -> bool:
    """A length-1 trajectory on which the recurrence trigger structurally cannot fire."""
    return traj.header.n_steps < 2


__all__ = [
    "DEFAULT_GRID",
    "DatasetSpec",
    "is_degenerate",
    "results_csv_bootstrap",
]
