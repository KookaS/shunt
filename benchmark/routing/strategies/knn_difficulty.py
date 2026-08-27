"""Judge-difficulty kNN strategies: route on an LLM-judge difficulty label, not embeddings."""

# WHY THIS FAMILY EXISTS. The semantic kNN embeds the task text; the probe and the learnability
# study showed the embedding carries essentially no difficulty signal (LOO R^2 -0.04, inside its
# null) while a difficulty LABEL does (human tag +0.13; the gpt-5.6-terra judge +0.027 LOO). These
# strategies route on that label: the judge rates the task 1-5, neighbours are tasks judged
# equally hard, and the cheapest model whose measured pass rate in that neighbourhood clears the
# bar is picked. The committed per-task labels live in `judge_difficulty.json`, derived by
# derive_judge_difficulty.py from the gitignored probe JSONLs.
#
# NOT LIVE. A difficulty strategy needs a judge call per task at inference (a provider + key, a
# task-boundary call, and a difficulty index over labelled history) — see docs/routing.md for
# what moving one to inference requires. They are scored and plotted like any benchmark row, but
# strategy_class.py marks them so the frontier cannot headline them.
#
# COST MODEL. Each difficulty strategy sets `judge_cost_total` (the MEASURED per-task judge cost
# from the committed table, ~$0.0016 measured for a real gpt-5.6-terra call) on every
# `select`, mirroring
# how the cascades report `cascade_total_cost`. summary.evaluate_billed reads it and folds it into
# the decision cost AND returns it per task, so the row's TotalCost, its bootstrap CIs and its
# cache-aware total all include the judge bill — the frontier's x position is judge + model runs
# with escalation, not model runs alone.

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final

from . import Strategy
from ._cascade_common import cheapest_priced_model, measured_models_by_price
from .session_cascade import SessionCascadeStrategy
from .tier_classifier import predict_model

# Resolved from this module (like config.py's path convention), never from the cwd: the
# difficulty family is scored by report.py / run_eval / the pipeline, which all run from
# the repo root, but a caller elsewhere must still resolve the committed table.
_DIFFICULTY_TABLE: Final[Path] = (
    Path(__file__).resolve().parents[1] / "data" / "judge_difficulty.json"
)
# Two tasks within this difficulty distance share a band (the radius-band neighbourhood).
_BAND_WIDTH: Final[float] = 0.5


@lru_cache(maxsize=1)
def _load_table() -> dict[str, dict]:
    """The committed derived difficulty table (task -> {difficulty, judge_cost_usd})."""
    payload = json.loads(_DIFFICULTY_TABLE.read_text())
    return dict(payload["tasks"])


def difficulty(task_id: str) -> float | None:
    """The task's judge difficulty (mean over the probe's round-2 runs), or None unlabelled."""
    row = _load_table().get(task_id)
    return float(row["difficulty"]) if row else None


def judge_cost(task_id: str) -> float:
    """The MEASURED per-task judge cost from the committed table (0 when unlabelled)."""
    row = _load_table().get(task_id)
    return float(row["judge_cost_usd"]) if row else 0.0


def neighbor_ids(task_id: str, matrix: dict, k: int, *, band: bool = False) -> list[str]:
    """Nearest task ids in difficulty space (|d_i - d_q|), self excluded.

    ``band=True`` restricts to a radius band (|d_i - d_q| <= 0.5 around the query's label) —
    every task a judge called about as hard, with no k cap.
    """
    dq = difficulty(task_id)
    if dq is None:
        return []
    scored: list[tuple[float, str]] = []
    for tid in sorted(matrix.get("results", {}).keys()):
        d = difficulty(tid)
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
    tier uses, so the two families differ only in the neighbourhood, never the decision rule."""
    if not matrix.get("results"):
        return cheapest_priced_model(matrix)
    models = measured_models_by_price(matrix)
    nids = neighbor_ids(task_id, matrix, k, band=band)
    if not nids:
        return models[0] if models else cheapest_priced_model(matrix)
    return predict_model(nids, matrix, models, success_rate_threshold, min_samples)


class knnDifficultyStrategy(Strategy):  # noqa: N801 (kNN is the established algorithm name)
    """Single-shot: the k nearest tasks in judge-difficulty space vote; route to the cheapest
    model whose neighbour pass rate clears the bar. A control — no `router.strategy` value
    names it, because a live install would escalate on top of the pick."""

    def __init__(
        self,
        k: int = 20,
        success_rate_threshold: float = 0.6,
        min_samples: int = 3,
    ) -> None:
        self._k = k
        self._threshold = success_rate_threshold
        self._min_samples = min_samples
        self.judge_cost_total: float = 0.0

    @property
    def name(self) -> str:
        return "kNN-difficulty"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        del task_meta
        self.judge_cost_total = judge_cost(task_id)
        return pick(task_id, matrix, self._k, self._threshold, self._min_samples)


class _DifficultySessionCascade(SessionCascadeStrategy):
    """Shared base for the difficulty session-cadence rows: the difficulty pick opens the
    ladder, the shipped session cadence climbs it. Subclasses choose the pick's neighbourhood
    via ``_band_neighbourhood``."""

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
        self.judge_cost_total: float = 0.0

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        # The judge bill is paid per task the moment the difficulty label is bought, so it is
        # charged here — BEFORE the base `select`, which can return early on an empty rung list
        # without ever reaching `_initial_rank_floor`. Charging here guarantees a fresh value on
        # every task instead of leaking the previous task's bill into a ladder-less cell.
        self.judge_cost_total = judge_cost(task_id)
        return super().select(task_id, task_meta, matrix)

    def _initial_rank_floor(self, task_id: str, matrix: dict, rungs: Sequence[str]) -> int:
        self.judge_cost_total = judge_cost(task_id)
        chosen = pick(
            task_id,
            matrix,
            self._k,
            self._threshold,
            self._min_samples,
            band=self._band_neighbourhood,
        )
        return rungs.index(chosen) if chosen in rungs else 0


class knnDifficultyCascadeStrategy(_DifficultySessionCascade):  # noqa: N801
    """kNN-difficulty pick + the session ladder: opens where the difficulty neighbours point."""

    @property
    def name(self) -> str:
        return "kNN-difficulty-cascade"


class DifficultyBandCascadeStrategy(_DifficultySessionCascade):
    """The "just judge label + escalation" rule: same-difficulty-band members vote, the cheapest
    in-band model whose pass rate clears the bar opens the ladder."""

    _band_neighbourhood: bool = True

    @property
    def name(self) -> str:
        return "Difficulty-Band-cascade"
