"""Declarative dataset registry + the default hyperparameter grid for the sweep."""

# The committable trajectory planes are registered here; the terminal `results.csv` bootstrap
# turns the existing measured grid into degenerate length-1 trajectories so the module runs
# end-to-end on REAL data before any paid multi-step collection (no synthetic data).

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from benchmark.escalation.normalize.base import build_trajectory, make_step
from benchmark.escalation.replay import GridPoint
from benchmark.routing import integrity

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.escalation.schema import Trajectory

# The sweep varies BOTH `escalate_after_n` AND `stale_window`, because the two are coupled and the
# old one-axis grid was the reason the escalation edge was invisible. Measured on the rebuilt
# corpus (2026-08-02, 727 stamped runs, base rate 0.421):
#
#   - The recurrence trigger has a REAL, null-clearing edge in the AS-SHIPPED family only at HIGH
#     recurrence thresholds: escalate_after_n=15 -> P(fail|fired)=0.538 (lift 1.28), n=20 -> 0.582
#     (1.38), n=30 -> 0.706 (1.68), all outside the challenge-block permutation null (p=0.005).
#     The n=2 cell (the old shipped default) fires on 727/727 trajectories (reproduction
#     failures recur at step 1-2), so it reads precision == base rate and hides the edge. The
#     EDIT-GATED family (failures after the agent's first edit) clears the null at and just
#     above the n=2 cell (best cell n=3, AUROC 0.724) — see
#     `policy_eval` and the `policy_cells_edit_gated` family in the report. A grid that stopped
#     at n=3 could only ever measure the mask, never the signal.
#   - `stale_window` is NOT inert at high n: `_in_window` admits at most `stale_window` events, so
#     reaching n recurrences needs a window at least that wide. The grid sweeps {10, 1000} so the
#     binding is visible rather than assumed. `ladder` stays pinned: the detection metric reads
#     whether the policy fired, not which rung it climbed, so it cannot move the headline.
#   - `escalate_after_n=2` stays in the grid as the SHIPPED cell: the report must show what ships,
#     not only what scores better (run_eval guarantees it is measured and flagged, never adopted).
#   - The ladder is DENSE because the knob is a threshold, and the whole point of the sweep is the
#     full precision/recall mapping a user picks an operating point from. Sparser grids (the old
#     2/5/8/10/15/20/30) hide where the curve bends; this one spans the floor (n=1) through the
#     run-length-selection tail (n=50, past the corpus's ~31-step median) with enough points that
#     the PR/ROC operating-characteristic figures trace the real curve rather than a few vertices.
_PINNED_LADDER: Final[str] = "effort_then_rank"

# n=1 fires on the FIRST verified failure, which is failure-biased (intermediate fail-then-fix is
# normal) and which the shipped config comment rules out — but it is kept IN the sweep so the
# mapping shows the floor and a reader can see it degrade to the base rate, exactly as n=2 does
# as-shipped. The high end (40, 50) probes past the corpus's ~31-step median run length, where any
# remaining precision is run-length selection (the `len-only` column on the sweep table shows the
# ceiling). Production should pick a LOW n — escalate early, before a doomed run burns its budget —
# which is what the edit-gated rows at n=2..5 show is the discriminative range.
_N_LADDER: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)
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
    # REPLICATE POLICY: rep 0 ONLY. This is a MEASUREMENT question — each row becomes one
    # terminal decision in the escalation corpus — so a cell must contribute exactly one
    # trajectory however many times it was observed, or a re-run cell would silently carry
    # more weight in every prefix-eval statistic than a cell measured once.
    trajectories: list[Trajectory] = []
    for i, row in enumerate(integrity.rep_zero_rows(results_path)):
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
