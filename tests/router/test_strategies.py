"""Tests for the shared live-path routing-strategy layer (src/shunt/router/strategies)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from shunt.router.selection import NeighborResult, SelectionRule
from shunt.router.strategies import RoutingStrategy, build_strategy
from shunt.router.strategies.fixed import AlwaysCheapStrategy, AlwaysFrontierStrategy
from shunt.router.strategies.knn import KnnStrategy


@dataclass
class _Model:
    name: str


class FakePool:
    """Minimal ModelPoolProtocol: tier→names plus an unhealthy set."""

    def __init__(
        self, tiers: dict[str, list[str]], unhealthy: frozenset[str] = frozenset()
    ) -> None:
        self._tiers = tiers
        self._unhealthy = unhealthy

    def get_tier_models(self, tier: str) -> list[_Model]:
        return [_Model(n) for n in self._tiers.get(tier, [])]

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
    def test_picks_lowest_tier_healthy_model(self) -> None:
        pool = FakePool({"cheap": ["c1", "c2"], "frontier": ["f1"]})
        model, reason = AlwaysCheapStrategy().select([], pool)
        assert model == "c1"
        assert reason == "always_cheap"

    def test_skips_unhealthy_and_escalates_tier(self) -> None:
        pool = FakePool({"cheap": ["c1"], "mid": ["m1"]}, unhealthy=frozenset({"c1"}))
        model, _ = AlwaysCheapStrategy().select([], pool)
        assert model == "m1"

    def test_ignores_neighbors(self) -> None:
        pool = FakePool({"cheap": ["c1"], "frontier": ["f1"]})
        model, _ = AlwaysCheapStrategy().select([_neighbor("f1")], pool)
        assert model == "c1"

    def test_falls_back_to_any_when_none_healthy(self) -> None:
        pool = FakePool({"cheap": ["c1"]}, unhealthy=frozenset({"c1"}))
        model, _ = AlwaysCheapStrategy().select([], pool)
        assert model == "c1"


class TestAlwaysFrontier:
    def test_picks_highest_tier_healthy_model(self) -> None:
        pool = FakePool({"cheap": ["c1"], "frontier": ["f1", "f2"]})
        model, reason = AlwaysFrontierStrategy().select([], pool)
        assert model == "f1"
        assert reason == "always_frontier"

    def test_skips_unhealthy_and_steps_down_tier(self) -> None:
        pool = FakePool({"high": ["h1"], "frontier": ["f1"]}, unhealthy=frozenset({"f1"}))
        model, _ = AlwaysFrontierStrategy().select([], pool)
        assert model == "h1"


class TestKnn:
    def test_delegates_to_selection_rule(self) -> None:
        pool = FakePool({"cheap": ["model-a"]})
        neighbors = [_neighbor("model-a") for _ in range(3)]
        rule = SelectionRule(min_success_rate=0.6, min_samples=3)
        model, reason = KnnStrategy(rule).select(neighbors, pool)
        assert model == "model-a"
        assert reason == "cheapest_above_threshold"


class TestRegistry:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("knn", KnnStrategy),
            ("always_cheap", AlwaysCheapStrategy),
            ("always_frontier", AlwaysFrontierStrategy),
        ],
    )
    def test_builds_each_live_strategy(self, name: str, cls: type) -> None:
        rule = SelectionRule()
        strategy = build_strategy(name, rule)
        assert isinstance(strategy, cls)

    def test_returns_routing_strategy(self) -> None:
        strategy: RoutingStrategy = build_strategy("always_cheap", SelectionRule())
        pool = FakePool({"cheap": ["c1"]})
        assert strategy.select([], pool)[0] == "c1"

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            build_strategy("knn_cascade", SelectionRule())

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
