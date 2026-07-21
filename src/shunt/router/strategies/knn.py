"""The default live strategy: kNN cheapest-above-threshold, wrapping ``SelectionRule``
(the algorithm lives once in ``selection.py`` — this only adapts it to the protocol).
"""

from __future__ import annotations

from shunt.router.selection import ModelPoolProtocol, NeighborResult, SelectionRule


class KnnStrategy:
    """Adapts ``SelectionRule`` to the ``RoutingStrategy`` protocol (cold-start off)."""

    def __init__(self, selection_rule: SelectionRule) -> None:
        self._selection_rule = selection_rule

    @property
    def consults_neighbors(self) -> bool:
        """kNN routes on the neighborhood, so it needs warmup + may explore."""
        return True

    def select(
        self,
        neighbors: list[NeighborResult],
        model_pool: ModelPoolProtocol,
    ) -> tuple[str, str]:
        """Delegate to the wrapped ``SelectionRule`` with cold-start disabled."""
        return self._selection_rule.select(neighbors, model_pool, cold_start_active=False)
