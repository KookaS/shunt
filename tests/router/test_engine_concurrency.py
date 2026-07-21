"""Many distinct sessions through one RouterEngine; check shared-state integrity."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from shunt.router.budget import ConservativeGate, ExplorationBudget
from shunt.router.embedder import Embedder
from shunt.router.engine import RouterEngine
from shunt.router.exploration import CandidateStats, ExplorationDecision
from shunt.router.policy import ExplorationPolicy
from shunt.router.selection import NeighborResult

from .conftest import FakeModelPool
from .test_engine import MockOutcomeIndex, _neighbor

_N_SESSIONS = 64


class _CountingEmbedder(Embedder):
    """Per-session-distinct vectors; counts how often the slow embed path actually ran."""

    def __init__(self) -> None:
        super().__init__(lazy=True)
        self.calls = 0
        self._lock = threading.Lock()

    def embed(self, text: str) -> Any:
        with self._lock:
            self.calls += 1
        seed = abs(hash(text)) % 997
        return np.full(768, 0.001 * (seed + 1), dtype=np.float32)


class _ConcurrentIndex(MockOutcomeIndex):
    """Thread-safe neighbor source that also records the per-thread query count."""

    def __init__(self, neighbors: list[NeighborResult]) -> None:
        super().__init__(count=100, neighbors=neighbors)
        self._lock = threading.Lock()

    def query(self, embedding: Any, k: int = 20) -> list[NeighborResult]:
        with self._lock:
            self.queries.append((embedding, k))
        return self._neighbors


class _AlwaysExploreSampler:
    """Deterministic: always an exploratory *upshift* from cheap model-a to pricier model-b."""

    def select(
        self,
        candidates: list[CandidateStats],
        threshold: float,
        propensity_mc_samples: int = 100,
    ) -> ExplorationDecision:
        return ExplorationDecision(
            model="model-b",
            greedy_model="model-a",
            is_exploratory=True,
            propensity=0.5,
            sampled_rates={"model-a": 0.4, "model-b": 0.6},
        )


def _neighbors() -> list[NeighborResult]:
    return [_neighbor("model-a", outcome=True, cost=1.0) for _ in range(5)] + [
        _neighbor("model-b", outcome=True, cost=2.0) for _ in range(5)
    ]


def _run_all(engine: RouterEngine) -> list[tuple[str, str, dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(
            pool.map(lambda i: engine.decide(f"session-{i}", f"prompt {i}"), range(_N_SESSIONS))
        )


def test_distinct_sessions_each_get_their_own_cached_embedding() -> None:
    embedder = _CountingEmbedder()
    index = _ConcurrentIndex(_neighbors())
    engine = RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b"),
        session_manager=MagicMock(),
        outcome_index=index,
        embedder=embedder,
        cold_start_threshold=20,
    )
    results = _run_all(engine)

    assert len(results) == _N_SESSIONS
    assert all(model for model, _reason, _prov in results)
    # One embedding per distinct session, none lost or shared.
    assert len(index.queries) == _N_SESSIONS
    assert embedder.calls == _N_SESSIONS
    for i in range(_N_SESSIONS):
        cached = engine.cached_embedding(f"session-{i}")
        assert cached is not None
        np.testing.assert_array_equal(cached, embedder.embed(f"prompt {i}"))


def test_shared_budget_counts_every_concurrent_decision_exactly_once() -> None:
    # Lost-update check: the budget is a plain float pair mutated under the engine lock.
    # Every one of the N concurrent decisions must land in the cumulative baseline.
    budget = ExplorationBudget(explore_budget_frac=0.4)
    engine = RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b"),
        session_manager=MagicMock(),
        outcome_index=_ConcurrentIndex(_neighbors()),
        embedder=_CountingEmbedder(),
        cold_start_threshold=20,
    )
    engine._budget = budget  # exploit-path budget accounting without the sampler
    _run_all(engine)

    # Each exploit decision records baseline == the chosen model's weighted cost (1.0),
    # so the cumulative baseline is exactly N — no lost increments, no double counting.
    assert budget._exploit_cost == float(_N_SESSIONS)
    assert budget.explore_ratio == 0.0
    assert budget.can_explore() is True


def test_shared_budget_cap_binds_under_concurrency() -> None:
    # Every decision explores a +1.0 upshift against a 1.0 baseline, i.e. ratio → 1.0 ≫ 0.4.
    # The cap must actually close, and the extra recorded must never exceed one per
    # exploratory decision (which a torn read-modify-write would break).
    budget = ExplorationBudget(explore_budget_frac=0.4)
    gate = ConservativeGate(alpha=0.1)
    engine = RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b"),
        session_manager=MagicMock(),
        outcome_index=_ConcurrentIndex(_neighbors()),
        embedder=_CountingEmbedder(),
        cold_start_threshold=20,
        exploration=ExplorationPolicy(enabled=True, propensity_mc_samples=0),
        sampler=_AlwaysExploreSampler(),
        budget=budget,
        conservative_gate=gate,
    )
    results = _run_all(engine)

    reasons = [reason for _model, reason, _prov in results]
    n_explored = reasons.count("exploration")
    assert n_explored > 0
    assert budget._explore_cost == float(n_explored)  # exactly +1.0 extra per exploration
    assert budget._exploit_cost == float(_N_SESSIONS)  # one baseline per decision
    assert budget.can_explore() is False  # the cap closed and stays closed
    # Once the cap closed, the remaining decisions fell through to the selection rule.
    assert n_explored < _N_SESSIONS


def test_conservative_gate_slack_survives_concurrent_outcome_reports() -> None:
    gate = ConservativeGate(alpha=0.1)
    engine = RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b"),
        session_manager=MagicMock(),
        outcome_index=_ConcurrentIndex(_neighbors()),
        embedder=_CountingEmbedder(),
        cold_start_threshold=20,
        exploration=ExplorationPolicy(enabled=True, propensity_mc_samples=0),
        sampler=_AlwaysExploreSampler(),
        budget=ExplorationBudget(0.4),
        conservative_gate=gate,
    )
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _i: engine.record_outcome(downshift=True, success=True), range(200)))
    # Up-exploration outcomes must not move the gate at all.
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _i: engine.record_outcome(downshift=False, success=False), range(200)))

    # `record_outcome` IS serialized by the engine lock (engine.py), so `self._slack
    # += 1.0` — a read-modify-write — cannot lose an update. A lost update here would
    # mean the gate under-counts verified downshift evidence.
    assert gate.slack == 200.0
    assert gate.allows_downshift() is True
