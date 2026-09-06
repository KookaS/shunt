"""Ranker-defer session cascade: a predicted defer score picks the opening rung.

The finetuned cross-encoder's predicted defer probability — not an oracle difficulty
label — decides whether the session opens one rung above the cheapest model.
"""

# THE ONLY DIFFERENCE FROM session_cascade IS "SOMETIMES SKIP THE CHEAP RUNG".
# Session-Cascade opens every session at the cheapest rung; this row opens at rung 1
# (the next model above cheapest) when the ranker says the task will NOT be solved by the
# cheap rung (predicted defer probability above a pre-registered threshold), and at the
# cheap rung otherwise. The session ladder then climbs from whichever rung opened, exactly
# as Session-Cascade does. A task with no defer prediction (never defer-labelled, i.e.
# never measured on the cheap rung) opens cheap — the same degradation as the base row.
#
# PREDICTION PROVENANCE. The committed per-task table (`ranker_predicted_defer.csv`) is a
# projection of one real-model run: an out-of-fold predicted defer probability from a
# FINETUNED 33M cross-encoder (ms-marco-MiniLM-L-12-v2), scored over the 190 defer-labelled
# tasks (the cheap-rung measured set of defer_labels.csv). The small derived projection is
# committed so a clean checkout reproduces every figure that plots the row; the raw
# per-task predictions stay out of tree. No committed results.csv row is touched.
#
# WHY THIS IS THE LIVE-ABLE ROW THE ORACLE DIFFICULTY ROWS ARE NOT. knn_difficulty reads an
# oracle judge difficulty that requires a paid judge call per task at inference. This row's
# signal is a local CPU model — the only missing surface is the mechanism plumbing
# (a local defer scorer invoked at the task boundary before the opening rung is chosen, with
# a cold-start fallback to session_cascade), which is why strategy_class.py gives it a
# path_to_live rather than marking it a pure oracle-bound control. It is STILL not live
# today: no `router.strategy` value produces it, and the measured finetuned defer signal was
# at chance, so the row is a research row, not a deployable one.
#
# COST MODEL. The prediction is a local CPU call the eval does not price, so the row folds
# $0.0 prediction cost: no judge bill is invented (a paid per-task label would be the one
# thing that could make this row MORE expensive than Session-Cascade by construction).

from __future__ import annotations

import csv
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final

from .session_cascade import DEFAULT_LADDER, SessionCascadeStrategy

# Resolved from this module (like config.py's path convention), never from the cwd.
_DEFER_TABLE: Final[Path] = (
    Path(__file__).resolve().parents[1] / "data" / "ranker_predicted_defer.csv"
)
# Pre-registered defer threshold: above it the cheap rung is predicted to fail, so the
# session opens one rung higher. A single config knob (`ranker_defer.defer_threshold`).
_DEFER_THRESHOLD: Final[float] = 0.5


@lru_cache(maxsize=1)
def _load_table() -> dict[str, float]:
    """The committed predicted-defer table (task -> predicted defer probability)."""
    rows: dict[str, float] = {}
    with _DEFER_TABLE.open(newline="") as f:
        for rec in csv.DictReader(f):
            rows[rec["challenge_id"]] = float(rec["pred_defer"])
    return rows


def predicted_defer(task_id: str) -> float | None:
    """The task's finetuned cross-encoder predicted defer probability, or None unlabelled."""
    return _load_table().get(task_id)


class RankerDeferCascadeStrategy(SessionCascadeStrategy):
    """Open the session ladder one rung above the cheapest model when the ranker predicts
    the cheap rung will defer (p > threshold); open cheap otherwise. The ladder that
    follows is Session-Cascade's, verbatim."""

    def __init__(
        self,
        defer_threshold: float = _DEFER_THRESHOLD,
        escalate_after_n: int = 2,
        stale_window: int = 10,
        ladder: str = DEFAULT_LADDER,
    ) -> None:
        super().__init__(
            escalate_after_n=escalate_after_n,
            stale_window=stale_window,
            ladder=ladder,
        )
        self._defer_threshold = defer_threshold

    @property
    def name(self) -> str:
        return "Ranker-Defer-cascade"

    def _initial_rank_floor(self, task_id: str, matrix: dict, rungs: Sequence[str]) -> int:
        p = predicted_defer(task_id)
        if p is not None and p > self._defer_threshold:
            # Rung 1 = the next model above cheap, clamped to a one-model ladder.
            return min(1, len(rungs) - 1)
        return 0
