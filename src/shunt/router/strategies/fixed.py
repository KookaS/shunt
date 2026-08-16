"""Fixed live strategies — one model class per session, chosen by capability rank
(``always_cheap`` = lowest rank / cheapest price, ``always_frontier`` = highest rank).
"""

from __future__ import annotations

from shunt.router.selection import ModelPoolProtocol, NeighborResult


def _first_healthy(model_pool: ModelPoolProtocol, *, strongest_first: bool) -> str | None:
    """First healthy model scanning the rank order; else the endpoint (health ignored)."""
    ranked = list(model_pool.ranked_models())
    if not ranked:
        return None
    scan = list(reversed(ranked)) if strongest_first else ranked
    for model in scan:
        if model_pool.is_healthy(model.name):
            return model.name
    return scan[0].name  # none healthy → the intended endpoint anyway


class AlwaysCheapStrategy:
    """Route every session to the cheapest (lowest-rank) healthy model."""

    @property
    def consults_neighbors(self) -> bool:
        """Fixed strategy — decision is pool-only, so no warmup/embedding/exploration."""
        return False

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """Pick the lowest-rank healthy model; neighbors are irrelevant to a fixed strategy."""
        chosen = _first_healthy(model_pool, strongest_first=False)
        if chosen is None:
            raise ValueError("no models configured for always_cheap routing")
        return (chosen, "always_cheap")


class AlwaysFrontierStrategy:
    """Route every session to the strongest (highest-rank) healthy model."""

    @property
    def consults_neighbors(self) -> bool:
        """Fixed strategy — decision is pool-only, so no warmup/embedding/exploration."""
        return False

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """Pick the highest-rank healthy model; neighbors are irrelevant to a fixed strategy."""
        chosen = _first_healthy(model_pool, strongest_first=True)
        if chosen is None:
            raise ValueError("no models configured for always_frontier routing")
        return (chosen, "always_frontier")
