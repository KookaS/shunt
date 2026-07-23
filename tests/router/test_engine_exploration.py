from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from shunt.router.budget import ConservativeGate, ExplorationBudget
from shunt.router.engine import RouterEngine
from shunt.router.exploration import CandidateStats, ExplorationDecision
from shunt.router.policy import ExplorationPolicy

from .conftest import FakeModelPool
from .test_engine import MockOutcomeIndex, RecordingEmbedder, _neighbor


class _FakeSampler:
    """Deterministic stub — the real sampler's randomness is tested in test_exploration."""

    def __init__(self, decision: ExplorationDecision) -> None:
        self._decision = decision
        self.calls: list[Any] = []

    def select(
        self,
        candidates: list[CandidateStats],
        threshold: float,
        propensity_mc_samples: int = 100,
    ) -> ExplorationDecision:
        self.calls.append((candidates, threshold, propensity_mc_samples))
        return self._decision


def _engine(
    *,
    neighbors: list[Any],
    decision: ExplorationDecision,
    budget: ExplorationBudget | None = None,
    gate: ConservativeGate | None = None,
) -> tuple[RouterEngine, _FakeSampler]:
    sampler = _FakeSampler(decision)
    engine = RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b"),
        session_manager=MagicMock(),
        outcome_index=MockOutcomeIndex(count=50, neighbors=neighbors),
        embedder=RecordingEmbedder(),
        cold_start_threshold=20,
        exploration=ExplorationPolicy(enabled=True, propensity_mc_samples=0),
        sampler=sampler,
        budget=budget or ExplorationBudget(0.4),
        conservative_gate=gate or ConservativeGate(alpha=0.1),
    )
    return engine, sampler


def test_disabled_by_default_keeps_selection_rule() -> None:
    neighbors = [_neighbor("model-a", outcome=True, cost=1.0) for _ in range(5)]
    engine = RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b"),
        session_manager=MagicMock(),
        outcome_index=MockOutcomeIndex(count=50, neighbors=neighbors),
        embedder=RecordingEmbedder(),
    )
    _model, reason, _prov = engine.decide("s1", "p")
    assert reason == "cheapest_above_threshold"


def test_exploratory_upshift_is_taken_with_propensity() -> None:
    neighbors = [_neighbor("model-a", cost=1.0), _neighbor("model-b", cost=5.0)]
    decision = ExplorationDecision(
        model="model-b",  # pricier than greedy → an upshift, always allowed
        greedy_model="model-a",
        sampled_rates={"model-a": 0.5, "model-b": 0.9},
        propensity=0.3,
        is_exploratory=True,
    )
    engine, _s = _engine(neighbors=neighbors, decision=decision)
    model, reason, prov = engine.decide("s1", "p")
    assert model == "model-b"
    assert reason == "exploration"
    assert prov["router_propensity"] == 0.3
    # an upshift (pricier than greedy) is NOT a downshift: the gate must not learn from it.
    assert prov["downshift"] is False


def test_conservative_gate_blocks_downshift_without_slack() -> None:
    neighbors = [_neighbor("model-a", cost=5.0), _neighbor("model-b", cost=1.0)]
    decision = ExplorationDecision(
        model="model-b",  # cheaper/weaker than greedy → a downshift
        greedy_model="model-a",
        sampled_rates={"model-a": 0.4, "model-b": 0.7},
        propensity=0.3,
        is_exploratory=True,
    )
    engine, _s = _engine(neighbors=neighbors, decision=decision, gate=ConservativeGate(alpha=0.0))
    model, reason, prov = engine.decide("s1", "p")
    assert model == "model-a"  # blocked → greedy
    assert reason == "conservative_fallback"
    assert prov["router_propensity"] == 1.0


def test_banked_slack_unlocks_downshift() -> None:
    neighbors = [_neighbor("model-a", cost=5.0), _neighbor("model-b", cost=1.0)]
    decision = ExplorationDecision(
        model="model-b",
        greedy_model="model-a",
        sampled_rates={"model-a": 0.4, "model-b": 0.7},
        propensity=0.3,
        is_exploratory=True,
    )
    engine, _s = _engine(neighbors=neighbors, decision=decision, gate=ConservativeGate(alpha=0.1))
    _m1, reason1, _p1 = engine.decide("s1", "p")
    assert reason1 == "conservative_fallback"  # no slack yet
    engine.record_outcome(downshift=True, success=True)  # bank downshift slack
    model2, reason2, prov2 = engine.decide("s2", "p")
    assert reason2 == "exploration"
    assert model2 == "model-b"
    # the taken cheaper-than-greedy exploration is flagged downshift so the capture
    # read-back feeds the ConservativeGate the evidence it gates future downshifts on.
    assert prov2["downshift"] is True


def test_budget_exhaustion_falls_back_to_selection_rule() -> None:
    neighbors = [_neighbor("model-a", outcome=True, cost=1.0) for _ in range(5)]
    spent = ExplorationBudget(0.4)
    spent.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)
    # explore 10 > 0.4*10 → exhausted
    spent.record(baseline_cost=0.0, actual_cost=10.0, is_exploratory=True)
    decision = ExplorationDecision(
        model="model-b",
        greedy_model="model-a",
        sampled_rates={"model-a": 0.5, "model-b": 0.9},
        propensity=0.3,
        is_exploratory=True,
    )
    engine, sampler = _engine(neighbors=neighbors, decision=decision, budget=spent)
    _model, reason, _prov = engine.decide("s1", "p")
    assert reason == "cheapest_above_threshold"  # exploration skipped
    assert sampler.calls == []  # sampler never consulted when budget exhausted


def test_budget_reopens_as_exploit_spend_accrues() -> None:
    # Regression for the latch-off bug: once the cap trips, main-path (exploit) routing
    # must still feed the budget so the cumulative cap can re-open — else exploration
    # dies permanently after the first trip.
    neighbors = [_neighbor("model-a", outcome=True, cost=1.0) for _ in range(5)]
    budget = ExplorationBudget(0.4)
    budget.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)
    budget.record(baseline_cost=0.0, actual_cost=5.0, is_exploratory=True)  # tripped: 5 > 0.4*10
    assert budget.can_explore() is False
    decision = ExplorationDecision(
        model="model-a",
        greedy_model="model-a",
        sampled_rates={"model-a": 0.9},
        propensity=0.9,
        is_exploratory=False,
    )
    engine, _s = _engine(neighbors=neighbors, decision=decision, budget=budget)
    for i in range(10):
        engine.decide(f"s{i}", "p")  # each exploit routing records ~cost 1.0 to the budget
    assert budget.can_explore() is True  # exploit baseline grew → cap re-opened


def test_sampler_choosing_greedy_is_marked_exploit_not_exploration() -> None:
    neighbors = [_neighbor("model-a", cost=1.0), _neighbor("model-b", cost=5.0)]
    decision = ExplorationDecision(
        model="model-a",
        greedy_model="model-a",
        sampled_rates={"model-a": 0.9, "model-b": 0.2},
        propensity=0.8,
        is_exploratory=False,
    )
    engine, _s = _engine(neighbors=neighbors, decision=decision)
    model, reason, _prov = engine.decide("s1", "p")
    assert model == "model-a"
    assert reason == "exploration_exploit"
