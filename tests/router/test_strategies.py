"""Tests for the shared live-path routing-strategy layer (src/shunt/router/strategies)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from shunt.router.selection import NeighborResult, SelectionRule
from shunt.router.strategies import RoutingStrategy, build_strategy
from shunt.router.strategies.fixed import (
    AlwaysCheapStrategy,
    AlwaysFrontierStrategy,
    SessionCascadeStrategy,
)
from shunt.router.strategies.knn import KnnStrategy


@dataclass
class _Model:
    name: str


class FakePool:
    """Minimal ModelPoolProtocol: models weakest→strongest (rank order) + an unhealthy set."""

    def __init__(self, ranked: list[str], unhealthy: frozenset[str] = frozenset()) -> None:
        self._ranked = ranked
        self._unhealthy = unhealthy

    def ranked_models(self) -> list[_Model]:
        return [_Model(n) for n in self._ranked]

    def rank_of(self, name: str) -> int | None:
        return self._ranked.index(name) if name in self._ranked else None

    def models_from_rank(self, i: int) -> list[_Model]:
        return [_Model(n) for n in self._ranked[max(i, 0) :]]

    def is_healthy(self, name: str) -> bool:
        return name not in self._unhealthy


def _neighbor(model: str, outcome: bool = True, cost: float = 1.0) -> NeighborResult:
    return NeighborResult(
        model=model,
        outcome=outcome,
        cost=cost,
        verification_confidence=1.0,
        distance=0.1,
        session_id="s",
    )


class TestAlwaysCheap:
    def test_picks_lowest_rank_healthy_model(self) -> None:
        pool = FakePool(["c1", "c2", "f1"])
        model, reason = AlwaysCheapStrategy().select([], pool)
        assert model == "c1"
        assert reason == "always_cheap"

    def test_skips_unhealthy_and_escalates_rank(self) -> None:
        pool = FakePool(["c1", "m1"], unhealthy=frozenset({"c1"}))
        model, _ = AlwaysCheapStrategy().select([], pool)
        assert model == "m1"

    def test_ignores_neighbors(self) -> None:
        pool = FakePool(["c1", "f1"])
        model, _ = AlwaysCheapStrategy().select([_neighbor("f1")], pool)
        assert model == "c1"

    def test_falls_back_to_any_when_none_healthy(self) -> None:
        pool = FakePool(["c1"], unhealthy=frozenset({"c1"}))
        model, _ = AlwaysCheapStrategy().select([], pool)
        assert model == "c1"


class TestAlwaysFrontier:
    def test_picks_highest_rank_healthy_model(self) -> None:
        pool = FakePool(["c1", "f1", "f2"])
        model, reason = AlwaysFrontierStrategy().select([], pool)
        assert model == "f2"
        assert reason == "always_frontier"

    def test_skips_unhealthy_and_steps_down_rank(self) -> None:
        pool = FakePool(["h1", "f1"], unhealthy=frozenset({"f1"}))
        model, _ = AlwaysFrontierStrategy().select([], pool)
        assert model == "h1"


class TestKnn:
    def test_delegates_to_selection_rule(self) -> None:
        pool = FakePool(["model-a"])
        neighbors = [_neighbor("model-a") for _ in range(3)]
        rule = SelectionRule(min_success_rate=0.6, min_samples=3)
        model, reason = KnnStrategy(rule).select(neighbors, pool)
        assert model == "model-a"
        assert reason == "cheapest_above_threshold"


class TestRegistry:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("knn_cascade", KnnStrategy),
            ("always_cheap", AlwaysCheapStrategy),
            ("always_frontier", AlwaysFrontierStrategy),
            ("session_cascade", SessionCascadeStrategy),
        ],
    )
    def test_builds_each_live_strategy(self, name: str, cls: type) -> None:
        rule = SelectionRule()
        strategy = build_strategy(name, rule)
        assert isinstance(strategy, cls)

    def test_returns_routing_strategy(self) -> None:
        strategy: RoutingStrategy = build_strategy("always_cheap", SelectionRule())
        pool = FakePool(["c1"])
        assert strategy.select([], pool)[0] == "c1"

    def test_unknown_strategy_raises(self) -> None:
        # `knn` is the RETIRED spelling. The alias is resolved one layer up, in
        # `parse_router_policy`; the registry itself knows only live ids, so a caller that
        # bypassed the policy layer with the old name must be refused rather than served.
        with pytest.raises(ValueError, match="unknown"):
            build_strategy("knn", SelectionRule())

    def test_the_control_contract_and_the_cascade_are_opposites(self) -> None:
        # The invariant `test_a_fixed_strategy_is_a_pinned_control_and_never_escalates`
        # depends on, stated where the strategies are built. always_cheap anchors every
        # routing comparison, so a verified failure must never move it; session_cascade picks
        # the same model and must move. `consults_neighbors` cannot express that difference —
        # it is False for both — which is why the engine branches on this predicate instead.
        rule = SelectionRule()
        control = build_strategy("always_cheap", rule)
        cascade = build_strategy("session_cascade", rule)
        assert control.consults_neighbors is cascade.consults_neighbors is False
        assert control.participates_in_escalation is False
        assert build_strategy("always_frontier", rule).participates_in_escalation is False
        assert cascade.participates_in_escalation is True

    def test_the_cascade_reports_its_own_reason_not_the_controls(self) -> None:
        # Same model, different token. Filtering decisions on `always_cheap` is how an
        # analysis finds control sessions; if the cascade shared the token it would be
        # counted as control, which is the confound the separate strategy exists to avoid.
        pool = FakePool(["c1", "c2"])
        assert build_strategy("always_cheap", SelectionRule()).select([], pool) == (
            "c1",
            "always_cheap",
        )
        assert build_strategy("session_cascade", SelectionRule()).select([], pool) == (
            "c1",
            "session_cascade",
        )

    def test_every_live_strategy_is_buildable(self) -> None:
        # Wall against drift: a name that passes policy validation but has no builder
        # would crash at server boot. Both lists must agree.
        from shunt.router.policy import LIVE_STRATEGIES

        for name in LIVE_STRATEGIES:
            assert build_strategy(name, SelectionRule()) is not None


class TestConsultsNeighbors:
    def test_knn_consults_neighbors(self) -> None:
        assert KnnStrategy(SelectionRule()).consults_neighbors is True

    def test_fixed_strategies_do_not(self) -> None:
        assert AlwaysCheapStrategy().consults_neighbors is False
        assert AlwaysFrontierStrategy().consults_neighbors is False


class TestExplorationStaysWiredToTheStrategyThatExplores:
    """`EXPLORATORY_STRATEGIES` is keyed by strategy ID, so a rename can silently unwire it."""

    def test_the_knn_cascade_id_is_exploratory(self) -> None:
        # A rename that moves the kNN cascade's ID without moving this set switches Thompson
        # exploration off for every install that selected it — no error, no other failing
        # test, because `server.py` simply skips the wiring for a name it does not find.
        from shunt.router.strategies import EXPLORATORY_STRATEGIES

        assert "knn_cascade" in EXPLORATORY_STRATEGIES

    def test_the_default_strategy_is_deliberately_not_exploratory(self) -> None:
        # The shipped default is `session_cascade`, whose base pick is fixed at the cheapest
        # model — there is no per-task choice to explore over, so exploration is inert under
        # it BY DESIGN. Pinned so the inertness stays a decision rather than an accident: if
        # the default ever moves back to a neighbour-consulting strategy, this fails and the
        # move gets to be deliberate.
        from shunt.proxy.server import _effective_exploration
        from shunt.router.policy import RouterPolicy
        from shunt.router.strategies import EXPLORATORY_STRATEGIES

        policy = RouterPolicy()
        assert policy.strategy not in EXPLORATORY_STRATEGIES
        assert policy.exploration.enabled is True  # the block is ON; the strategy ignores it
        assert _effective_exploration(policy) is None

    def test_every_exploratory_strategy_consults_neighbours(self) -> None:
        # Exploration samples over the kNN neighbourhood, so a strategy that never looks at
        # neighbours cannot explore — listing one here would arm a layer with no input.
        from shunt.router.strategies import EXPLORATORY_STRATEGIES

        for name in EXPLORATORY_STRATEGIES:
            assert build_strategy(name, SelectionRule()).consults_neighbors is True
