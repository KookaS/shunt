from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

from shunt.capture.worker import CaptureWorker
from shunt.db.store import OutcomeStore
from shunt.proxy.server import _persist_router_state
from shunt.router.budget import ConservativeGate, ExplorationBudget
from shunt.router.engine import RouterEngine
from shunt.router.exploration import ExplorationDecision
from shunt.router.policy import ExplorationPolicy

from .conftest import FakeModelPool
from .test_engine import MockOutcomeIndex, RecordingEmbedder
from .test_engine_exploration import _FakeSampler


def _exploring_engine(*, budget: ExplorationBudget, gate: ConservativeGate) -> RouterEngine:
    decision = ExplorationDecision(
        model="model-a",
        greedy_model="model-a",
        sampled_rates={},
        propensity=1.0,
        is_exploratory=False,
    )
    return RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b"),
        session_manager=MagicMock(),
        outcome_index=MockOutcomeIndex(count=50, neighbors=[]),
        embedder=RecordingEmbedder(),
        exploration=ExplorationPolicy(enabled=True, propensity_mc_samples=0),
        sampler=_FakeSampler(decision),
        budget=budget,
        conservative_gate=gate,
    )


def test_engine_snapshot_round_trips_budget_and_gate() -> None:
    budget = ExplorationBudget(0.4)
    budget.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)
    budget.record(baseline_cost=0.0, actual_cost=6.0, is_exploratory=True)  # over 0.4*10
    gate = ConservativeGate(alpha=1.0)
    gate.record_outcome(downshift=True, success=True)
    engine = _exploring_engine(budget=budget, gate=gate)

    snap = engine.snapshot_exploration_state()
    assert snap["budget"]["exploit_cost"] == 10.0
    assert snap["gate"]["slack"] == 1.0

    restored = _exploring_engine(budget=ExplorationBudget(0.4), gate=ConservativeGate(alpha=1.0))
    restored.restore_exploration_state(snap)
    assert restored.snapshot_exploration_state() == snap


def test_engine_snapshot_empty_when_exploration_disabled() -> None:
    engine = RouterEngine(
        model_pool=FakeModelPool("model-a", "model-b"),
        session_manager=MagicMock(),
        outcome_index=MockOutcomeIndex(count=50, neighbors=[]),
        embedder=RecordingEmbedder(),
    )
    assert engine.snapshot_exploration_state() == {}
    engine.restore_exploration_state({"budget": {"exploit_cost": 9.0}})  # no-op, no crash


def test_full_restart_cycle_through_store(tmp_path: Path) -> None:
    db = str(tmp_path / "outcomes.db")
    store = OutcomeStore(db_path=db)
    gate = ConservativeGate(alpha=1.0)
    engine = _exploring_engine(budget=ExplorationBudget(0.4), gate=gate)
    engine.record_outcome(downshift=True, success=True)  # slack -> 1.0

    _persist_router_state(engine, store)  # clean-shutdown persist
    store.close()

    # New process: fresh store + engine, restore from disk.
    reopened = OutcomeStore(db_path=db)
    fresh = _exploring_engine(budget=ExplorationBudget(0.4), gate=ConservativeGate(alpha=1.0))
    fresh.restore_exploration_state(reopened.load_router_state())
    assert fresh.snapshot_exploration_state()["gate"]["slack"] == 1.0
    reopened.close()


def test_periodic_sweep_persists_without_clean_shutdown(tmp_path: Path) -> None:
    db = str(tmp_path / "outcomes.db")
    store = OutcomeStore(db_path=db)
    engine = _exploring_engine(budget=ExplorationBudget(0.4), gate=ConservativeGate(alpha=1.0))
    engine.record_outcome(downshift=True, success=True)  # slack -> 1.0, no shutdown yet

    worker = CaptureWorker(
        coordinator=MagicMock(),
        session_manager=MagicMock(),
        sweep_interval=0.05,
        on_sweep=lambda: _persist_router_state(engine, store),
    )
    worker.start()
    try:
        deadline = time.time() + 2.0
        persisted = None
        while time.time() < deadline:
            persisted = store.load_router_state()
            if persisted is not None:
                break
            time.sleep(0.05)
    finally:
        worker.stop()

    assert persisted is not None  # a sweep fired without any clean shutdown
    assert persisted["gate"]["slack"] == 1.0
    store.close()
