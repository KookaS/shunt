from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from shunt.models import TIER_ORDER

logger = logging.getLogger(__name__)


class ModelInfo(Protocol):
    """Minimal view of a model config the router reads — just its name."""

    name: str


class ModelPoolProtocol(Protocol):
    """Minimal model-pool interface the router depends on.

    Kept structural so benchmarks can pass fakes without importing the
    concrete ``ModelPool``.
    """

    def get_tier_models(self, tier: str) -> Sequence[ModelInfo]: ...

    def is_healthy(self, name: str) -> bool: ...


_DEFAULT_COLD_START_MODEL = "qwen3.7-plus"
# Raw library default for a bare SelectionRule(). Production does NOT use this: the server
# builds the rule from KnnPolicy.success_rate_threshold (shipped 0.6), the single source.
_DEFAULT_MIN_SUCCESS_RATE = 0.7
_DEFAULT_MIN_SAMPLES = 3


@dataclass
class NeighborResult:
    """A single neighbor returned by the outcome-index kNN query."""

    model: str
    outcome: bool
    cost: float
    verification_confidence: float
    distance: float
    session_id: str
    truncation_rate: float = 0.0


def _confidence_weight(n: NeighborResult) -> float:
    """Weight a neighbor's influence by confidence and (1 - distance).

    Clamped to ``[0.0, ∞)`` so that very distant neighbors (distance > 1)
    don't produce negative weights.
    """
    return n.verification_confidence * max(0.0, 1.0 - n.distance)


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish effective sample size ``nₑ = (Σwᵢ)² / Σ(wᵢ²)`` over per-outcome weights."""
    # Equals the raw count when weights are uniform (the backward-compat invariant); a spread
    # of confidences drives nₑ below the count. Non-positive weights (zeroed by the confidence
    # clamp) drop out; an all-zero vector yields 0.0. Non-finite weights are dropped too — a
    # NaN/inf would poison the ratio (the DB NOT NULL wall keeps them out, but keep the pure
    # function total regardless), mirroring the ``math.isfinite`` guard in exploration._cost_key.
    total = 0.0
    sq = 0.0
    for w in weights:
        if not math.isfinite(w) or w <= 0.0:
            continue
        total += w
        sq += w * w
    if sq <= 0.0:
        return 0.0
    return (total * total) / sq


class SelectionRule:
    """V1 selection rule: from kNN neighbours, keep models above
    ``min_success_rate`` with enough samples and pick the cheapest; escalate if
    none qualify. Cold-start (delegated via ``cold_start_active``) overrides.
    """

    def __init__(
        self,
        cold_start_threshold: int = 20,
        cold_start_model: str = _DEFAULT_COLD_START_MODEL,
        min_success_rate: float = _DEFAULT_MIN_SUCCESS_RATE,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._cold_start_threshold = cold_start_threshold
        self._cold_start_model = cold_start_model
        self._min_success_rate = min_success_rate
        self._min_samples = min_samples

    @property
    def cold_start_threshold(self) -> int:
        return self._cold_start_threshold

    @property
    def cold_start_model(self) -> str:
        return self._cold_start_model

    @property
    def min_success_rate(self) -> float:
        return self._min_success_rate

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
        cold_start_active: bool,
    ) -> tuple[str, str]:
        if cold_start_active:
            return (self._cold_start_model, "cold_start")

        if not neighbors:
            # The silent-degradation case: no evidence at all, so `_escalate` returns the
            # cheapest model and labels it `exploration_untested` — indistinguishable in
            # the response from a learned choice unless the log says so here.
            logger.debug(
                "select: no model cleared the threshold — neighbourhood is EMPTY, "
                "falling through to the cheapest untested model"
            )
            return self._escalate(set(), model_pool)

        groups: dict[str, list[NeighborResult]] = {}
        for n in neighbors:
            groups.setdefault(n.model, []).append(n)

        eligible: list[tuple[str, float]] = []

        for model, group in groups.items():
            weights = [_confidence_weight(n) for n in group]
            total_weight = sum(weights)
            if total_weight <= 0:
                continue

            weighted_success = (
                sum(w * (1.0 if n.outcome else 0.0) for w, n in zip(weights, group, strict=True))
                / total_weight
            )

            weighted_cost = (
                sum(w * n.cost for w, n in zip(weights, group, strict=True)) / total_weight
            )
            # A neighbour with UNKNOWN cost surfaces as +inf; a zero-weight one makes the
            # term `0 * inf = nan`, and nan sorts as neither cheap nor dear (all comparisons
            # False), so a nan group could be returned as "cheapest". Collapse any non-finite
            # weighted cost to +inf — never cheapest — matching the engine's guard.
            if not math.isfinite(weighted_cost):
                weighted_cost = math.inf

            passes = weighted_success >= self._min_success_rate and len(group) >= self._min_samples
            logger.debug(
                "select: model=%s samples=%d/%d success=%.3f/%.3f cost=%.6f -> %s",
                model,
                len(group),
                self._min_samples,
                weighted_success,
                self._min_success_rate,
                weighted_cost,
                "ELIGIBLE" if passes else "rejected",
            )
            if passes:
                eligible.append((model, weighted_cost))

        if eligible:
            eligible.sort(key=lambda x: x[1])
            return (eligible[0][0], "cheapest_above_threshold")

        tested_models = set(groups.keys())
        # Not a neutral fallback: this returns the CHEAPEST untested model, so a thin or
        # empty neighbourhood makes the router look like always_cheap.
        logger.debug(
            "select: no model cleared the threshold; escalating past tested=%s",
            sorted(tested_models) or "none",
        )
        return self._escalate(tested_models, model_pool)

    def _escalate(
        self,
        tested_models: set[str],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        for tier in TIER_ORDER:
            for m in model_pool.get_tier_models(tier):
                if m.name not in tested_models:
                    return (m.name, "exploration_untested")

        for tier in reversed(TIER_ORDER):
            models = model_pool.get_tier_models(tier)
            if models:
                return (models[-1].name, "safe_fallback")

        return (self._cold_start_model, "safe_fallback")
