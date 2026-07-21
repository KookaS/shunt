"""Name → strategy builder registry, keyed by the live-eligible strategy names so
``router.strategy`` maps to a concrete ``RoutingStrategy``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from shunt.router.selection import SelectionRule
from shunt.router.strategies.base import RoutingStrategy
from shunt.router.strategies.fixed import AlwaysCheapStrategy, AlwaysFrontierStrategy
from shunt.router.strategies.knn import KnnStrategy

# Strategies whose selection consults the kNN neighborhood + exploration knobs. The
# fixed strategies ignore them, so exploration is off for those (see server wiring).
EXPLORATORY_STRATEGIES: Final[frozenset[str]] = frozenset({"knn"})

_StrategyBuilder = Callable[[SelectionRule], RoutingStrategy]


def _build_knn(selection_rule: SelectionRule) -> RoutingStrategy:
    return KnnStrategy(selection_rule)


def _build_always_cheap(selection_rule: SelectionRule) -> RoutingStrategy:
    return AlwaysCheapStrategy()


def _build_always_frontier(selection_rule: SelectionRule) -> RoutingStrategy:
    return AlwaysFrontierStrategy()


_BUILDERS: Final[dict[str, _StrategyBuilder]] = {
    "knn": _build_knn,
    "always_cheap": _build_always_cheap,
    "always_frontier": _build_always_frontier,
}


def build_strategy(name: str, selection_rule: SelectionRule) -> RoutingStrategy:
    """Construct the live strategy for *name*; the ``SelectionRule`` carries the knn knobs."""
    builder = _BUILDERS.get(name)
    if builder is None:
        allowed = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"unknown live routing strategy {name!r}; live-eligible: {allowed}")
    return builder(selection_rule)
