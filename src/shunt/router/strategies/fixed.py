"""Fixed live strategies — one model class per session, chosen by tier (the router
never prices; ``always_cheap`` = lowest tier, ``always_frontier`` = highest tier).
"""

from __future__ import annotations

from collections.abc import Iterable

from shunt.models import TIER_ORDER
from shunt.router.selection import ModelInfo, ModelPoolProtocol, NeighborResult


def _first_in_tier_order(model_pool: ModelPoolProtocol, tiers: Iterable[str]) -> str | None:
    """First healthy model scanning *tiers* in order; else the first model of any tier."""
    ordered = list(tiers)
    for tier in ordered:
        for model in model_pool.get_tier_models(tier):
            if model_pool.is_healthy(model.name):
                return model.name
    return _any_model(model_pool, ordered)


def _any_model(model_pool: ModelPoolProtocol, tiers: list[str]) -> str | None:
    """First model of any tier (health ignored) — a last resort when none are healthy."""
    for tier in tiers:
        models: list[ModelInfo] = list(model_pool.get_tier_models(tier))
        if models:
            return models[0].name
    return None


class AlwaysCheapStrategy:
    """Route every session to the cheapest (lowest-tier) healthy model."""

    @property
    def consults_neighbors(self) -> bool:
        """Fixed strategy — decision is pool-only, so no warmup/embedding/exploration."""
        return False

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """Pick the lowest-tier healthy model; neighbors are irrelevant to a fixed strategy."""
        chosen = _first_in_tier_order(model_pool, TIER_ORDER)
        if chosen is None:
            raise ValueError("no models configured for always_cheap routing")
        return (chosen, "always_cheap")


class AlwaysFrontierStrategy:
    """Route every session to the strongest (highest-tier) healthy model."""

    @property
    def consults_neighbors(self) -> bool:
        """Fixed strategy — decision is pool-only, so no warmup/embedding/exploration."""
        return False

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """Pick the highest-tier healthy model; neighbors are irrelevant to a fixed strategy."""
        chosen = _first_in_tier_order(model_pool, reversed(TIER_ORDER))
        if chosen is None:
            raise ValueError("no models configured for always_frontier routing")
        return (chosen, "always_frontier")
