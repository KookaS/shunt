from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


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


_TIER_ORDER = ("cheap", "mid", "frontier")
_DEFAULT_COLD_START_MODEL = "qwen3.7-plus"
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

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
        cold_start_active: bool,
    ) -> tuple[str, str]:
        if cold_start_active:
            return (self._cold_start_model, "cold_start")

        if not neighbors:
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

            if weighted_success >= self._min_success_rate and len(group) >= self._min_samples:
                eligible.append((model, weighted_cost))

        if eligible:
            eligible.sort(key=lambda x: x[1])
            return (eligible[0][0], "cheapest_above_threshold")

        tested_models = set(groups.keys())
        return self._escalate(tested_models, model_pool)

    def _escalate(
        self,
        tested_models: set[str],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        for tier in _TIER_ORDER:
            for m in model_pool.get_tier_models(tier):
                if m.name not in tested_models:
                    return (m.name, "exploration_untested")

        for tier in reversed(_TIER_ORDER):
            models = model_pool.get_tier_models(tier)
            if models:
                return (models[-1].name, "safe_fallback")

        return (self._cold_start_model, "safe_fallback")
