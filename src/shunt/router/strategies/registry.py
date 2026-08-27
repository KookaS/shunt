"""Name → strategy builder registry, keyed by the live-eligible strategy names so
``router.strategy`` maps to a concrete ``RoutingStrategy``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from shunt.router.selection import SelectionRule
from shunt.router.strategies.base import RoutingStrategy
from shunt.router.strategies.fixed import (
    AlwaysCheapStrategy,
    AlwaysFrontierStrategy,
    SessionCascadeStrategy,
)
from shunt.router.strategies.knn import KnnStrategy

# Strategies whose selection consults the kNN neighborhood + exploration knobs. The
# fixed strategies ignore them, so exploration is off for those (see server wiring).
# This set is keyed by the id `router.strategy` names, so it MUST move with a rename: it was
# `{"knn"}` before the id became `knn_cascade`, then `knn_semantic_cascade`; leaving it behind
# would have switched Thompson exploration off for every install with no error and no failing
# test.
EXPLORATORY_STRATEGIES: Final[frozenset[str]] = frozenset({"knn_semantic_cascade"})

_StrategyBuilder = Callable[[SelectionRule], RoutingStrategy]


def _build_knn_cascade(selection_rule: SelectionRule) -> RoutingStrategy:
    # `KnnStrategy` already carries `participates_in_escalation = True`, so this builder is the
    # kNN pick PLUS the ladder — which is what `knn_semantic_cascade` names and what has always
    # shipped.
    return KnnStrategy(selection_rule)


def _build_always_cheap(selection_rule: SelectionRule) -> RoutingStrategy:
    return AlwaysCheapStrategy()


def _build_always_frontier(selection_rule: SelectionRule) -> RoutingStrategy:
    return AlwaysFrontierStrategy()


def _build_session_cascade(selection_rule: SelectionRule) -> RoutingStrategy:
    # `always_cheap`'s base pick plus the escalation layer — the live spelling of the
    # benchmark's Session-Cascade row. It is a distinct CLASS rather than `AlwaysCheapStrategy`
    # under a second name because the two differ on `participates_in_escalation`, and that
    # difference is load-bearing in both directions: always_cheap is the pinned control every
    # routing comparison is read against, and the ladder is all this strategy is.
    return SessionCascadeStrategy()


_BUILDERS: Final[dict[str, _StrategyBuilder]] = {
    "knn_semantic_cascade": _build_knn_cascade,
    "always_cheap": _build_always_cheap,
    "always_frontier": _build_always_frontier,
    "session_cascade": _build_session_cascade,
}


def build_strategy(name: str, selection_rule: SelectionRule) -> RoutingStrategy:
    """Construct the live strategy for *name*; the ``SelectionRule`` carries the knn knobs."""
    builder = _BUILDERS.get(name)
    if builder is None:
        allowed = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"unknown live routing strategy {name!r}; live-eligible: {allowed}")
    return builder(selection_rule)
