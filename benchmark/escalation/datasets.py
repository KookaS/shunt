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

# The sweep varies BOTH `escalate_after_n` AND `stale_window`, because the two are coupled and the
# old one-axis grid was the reason the escalation edge was invisible. Measured on the rebuilt
# corpus (2026-08-02, 727 stamped runs, base rate 0.421):
#
#   - The recurrence trigger has a REAL, null-clearing edge, but only at HIGH recurrence
#     thresholds: escalate_after_n=15 -> P(fail|fired)=0.538 (lift 1.28), n=20 -> 0.582 (1.38),
#     n=30 -> 0.706 (1.68), all outside the challenge-block permutation null (p=0.005). The
#     shipped default n=2 fires on 727/727 trajectories (reproduction failures recur at step 1-2),
#     so it reads precision == base rate and hides the edge. A grid that stopped at n=3 could only
#     ever measure the mask, never the signal.
#   - `stale_window` is NOT inert at high n: `_in_window` admits at most `stale_window` events, so
#     reaching n recurrences needs a window at least that wide. The grid sweeps {10, 1000} so the
#     binding is visible rather than assumed. `ladder` stays pinned: the detection metric reads
#     whether the policy fired, not which rung it climbed, so it cannot move the headline.
#   - `escalate_after_n=2` stays in the grid as the SHIPPED cell: the report must show what ships,
#     not only what scores better (run_eval guarantees it is measured and flagged, never adopted).
_PINNED_LADDER: Final[str] = "effort_then_rank"

# n below the shipped default is deliberately excluded: escalate_after_n=1 fires on the FIRST
# verified failure, which is failure-biased (intermediate fail-then-fix is normal) and which the
# shipped config comment already rules out. n=2..30 spans the mask (all-fire) through the
# null-clearing edge with room to see the precision/recall trade-off.
_N_LADDER: Final[tuple[int, ...]] = (2, 5, 8, 10, 15, 20, 30)
# 10 is the shipped window; 1000 is "the whole session" (a run's median length is ~31 steps), so a
# recurrence can never retire inside a session. The two values show the window's binding at high n.
_STALE_WINDOWS: Final[tuple[int, ...]] = (10, 1000)

DEFAULT_GRID: Final[list[GridPoint]] = [
    GridPoint(escalate_after_n=n, stale_window=sw, ladder=_PINNED_LADDER)
    for n in _N_LADDER
    for sw in _STALE_WINDOWS
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
