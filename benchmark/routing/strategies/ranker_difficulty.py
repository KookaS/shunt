"""Ranker-predicted-difficulty strategies: route on a cross-encoder difficulty score.

The difficulty label comes from a REAL, fit-free local-CPU cross-encoder prediction
(ms-marco-MiniLM-L-12-v2) scoring the task's problem statement, instead of the LLM judge
or the embedding neighbourhood. Everything downstream — neighbour set, pick rule, session
ladder — is the knn_difficulty machinery unchanged.
"""

# WHY THIS FAMILY EXISTS. The difficulty kNN rows route on a per-task judge label rather
# than embeddings; this family asks whether a PREDICTED difficulty label (a model output,
# obtainable at inference) buys what the ORACLE judge label bought, or whether the whole
# gain sits in the oracle's access to the measured outcome.
#
# PREDICTION PROVENANCE. The committed per-task table (`ranker_predicted_difficulty.csv`)
# is a projection of one real-model run: the zero-shot cross-encoder above scored each of
# the 500 SWE-bench problem statements and `pred_difficulty` is its raw difficulty output
# for the task. Fit-free (no training labels), ran once; the small derived projection is
# committed so a clean checkout reproduces every figure that plots a ranker row, while the
# raw per-task predictions stay out of tree. No committed results.csv row is touched.
#
# NOT LIVE — research/control row. The zero-shot difficulty signal on this corpus measured
# weak and the defer channel measured at chance, so strategy_class.py classifies this
# family control/blocked; these rows price the mechanism question, they are not a
# deployable router.
#
# COST MODEL. The prediction is a local CPU call the eval does not price, so these rows
# fold judge_cost_total = $0.0: no judge bill is invented, unlike the knn_difficulty rows
# which carry the MEASURED per-task judge cost.

from __future__ import annotations

import csv
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final

from . import Strategy
from ._cascade_common import cheapest_priced_model, measured_models_by_price
from .session_cascade import SessionCascadeStrategy
from .tier_classifier import predict_model

# Resolved from this module (like config.py's path convention), never from the cwd.
_DIFFICULTY_TABLE: Final[Path] = (
    Path(__file__).resolve().parents[1] / "data" / "ranker_predicted_difficulty.csv"
)
# Two tasks within this predicted-difficulty distance share a band (the radius-band
# neighbourhood).
#
# CALIBRATED FOR A DIFFERENT SCALE, AND CURRENTLY UNREACHED. 0.5 is the judge-difficulty
# family's width, inherited unchanged. There the labels are integers 1-5 with sd ~0.75, so
# the band is under a standard deviation wide and selects a genuine neighbourhood. The
# committed predicted-difficulty distribution is an order of magnitude tighter (500 tasks,
# sd 0.139, range 2.04-3.47), so +/-0.5 is +/-3.6 sd and the median task's band holds 493 of
# the other 499 tasks — a near-global neighbourhood, not a band. Nothing reaches it today:
# `_band_neighbourhood` is False on every subclass below, so `band=True` is never passed and
# no scored row is affected. Recalibrate to THIS distribution's scale before enabling a
# banded ranker row; do not inherit the judge width a second time.
_BAND_WIDTH: Final[float] = 0.5


@lru_cache(maxsize=1)
def _load_table() -> dict[str, float]:
    """The committed predicted-difficulty table (task -> pred_difficulty)."""
    rows: dict[str, float] = {}
    with _DIFFICULTY_TABLE.open(newline="") as f:
        for rec in csv.DictReader(f):
            rows[rec["challenge_id"]] = float(rec["pred_difficulty"])
    return rows


def predicted_difficulty(task_id: str) -> float | None:
    """The task's R0 cross-encoder predicted difficulty, or None when it has no row."""
    return _load_table().get(task_id)


def neighbor_ids(task_id: str, matrix: dict, k: int, *, band: bool = False) -> list[str]:
    """Nearest task ids in predicted-difficulty space (|d_i - d_q|), self excluded.

    ``band=True`` restricts to a radius band (|d_i - d_q| <= 0.5 around the query's
    prediction) — every task the ranker called about as hard, with no k cap.
    """
    dq = predicted_difficulty(task_id)
    if dq is None:
        return []
    scored: list[tuple[float, str]] = []
    for tid in sorted(matrix.get("results", {}).keys()):
        d = predicted_difficulty(tid)
        if tid == task_id or d is None:
            continue
        if band and abs(d - dq) > _BAND_WIDTH:
            continue
        scored.append((abs(d - dq), tid))
    scored.sort()
    return [tid for _dist, tid in scored[:k]] if not band else [tid for _dist, tid in scored]


def pick(
    task_id: str,
    matrix: dict,
    k: int,
    success_rate_threshold: float,
    min_samples: int,
    *,
    band: bool = False,
) -> str:
    """Weakest (cheapest) model whose neighbour pass rate clears the bar; strongest else.

    Reuses ``tier_classifier.predict_model`` — the same weakest-eligible rule the semantic
    and judge-difficulty tiers use, so the families differ only in the label source, never
    the decision rule. Unlabelled tasks have no neighbours and open at the cheap end.
    """
    if not matrix.get("results"):
        return cheapest_priced_model(matrix)
    models = measured_models_by_price(matrix)
    nids = neighbor_ids(task_id, matrix, k, band=band)
    if not nids:
        return models[0] if models else cheapest_priced_model(matrix)
    return predict_model(nids, matrix, models, success_rate_threshold, min_samples)


class RankerDifficultyStrategy(Strategy):
    """Single-shot: the k nearest tasks in predicted-difficulty space vote; route to the
    cheapest model whose neighbour pass rate clears the bar. A control — no
    `router.strategy` value names it, because a live install would escalate on top."""

    def __init__(
        self,
        k: int = 20,
        success_rate_threshold: float = 0.6,
        min_samples: int = 3,
    ) -> None:
        self._k = k
        self._threshold = success_rate_threshold
        self._min_samples = min_samples

    @property
    def name(self) -> str:
        return "Ranker-Difficulty"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        del task_meta
        return pick(task_id, matrix, self._k, self._threshold, self._min_samples)


class _RankerDifficultySessionCascade(SessionCascadeStrategy):
    """Shared base for the ranker-difficulty session-cadence rows: the predicted-difficulty
    pick opens the ladder, the shipped session cadence climbs it."""

    _band_neighbourhood: bool = False

    def __init__(
        self,
        k: int = 20,
        success_rate_threshold: float = 0.6,
        min_samples: int = 3,
        escalate_after_n: int = 2,
        stale_window: int = 10,
        ladder: str = "effort_then_rank",
    ) -> None:
        super().__init__(
            escalate_after_n=escalate_after_n,
            stale_window=stale_window,
            ladder=ladder,
        )
        self._k = k
        self._threshold = success_rate_threshold
        self._min_samples = min_samples

    def _initial_rank_floor(self, task_id: str, matrix: dict, rungs: Sequence[str]) -> int:
        chosen = pick(
            task_id,
            matrix,
            self._k,
            self._threshold,
            self._min_samples,
            band=self._band_neighbourhood,
        )
        return rungs.index(chosen) if chosen in rungs else 0


class RankerDifficultyCascadeStrategy(_RankerDifficultySessionCascade):
    """Ranker-difficulty pick + the session ladder: opens where the predicted-difficulty
    neighbours point."""

    @property
    def name(self) -> str:
        return "Ranker-Difficulty-cascade"
