"""The shared live-path routing-strategy seam: one per-session decision over a
live kNN neighborhood, returning ``(model_name, reason)``.
"""

from __future__ import annotations

from typing import Protocol

from shunt.router.selection import ModelPoolProtocol, NeighborResult


class RoutingStrategy(Protocol):
    """A single cache-safe per-session model decision from live neighbors + pool."""

    @property
    def consults_neighbors(self) -> bool:
        """Whether the decision depends on the kNN neighborhood (thus warmup + exploration).

        ``False`` for fixed strategies — the engine then skips cold-start, embedding, the
        index query, and exploration, so they route deterministically from the first turn.
        """
        ...

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """Return ``(model_name, reason)`` for one session (cold-start handled upstream)."""
        ...
