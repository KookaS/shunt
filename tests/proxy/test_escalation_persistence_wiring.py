"""Server persistence wires the escalation snapshot to the store and back on restart.

Shutdown/sweep persist the snapshot; startup restores it. The inline lifespan restore
is asserted at the store round-trip level (booting the full app offline is too heavy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from shunt.db.store import OutcomeStore
from shunt.proxy.server import _persist_router_state
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig


def test_persist_flushes_the_escalation_snapshot_to_the_store() -> None:
    # Shutdown and the periodic sweep both route through _persist_router_state.
    engine = MagicMock()
    store = MagicMock()
    _persist_router_state(engine, store)
    store.save_escalation_state.assert_called_once_with(
        engine.snapshot_escalation_state.return_value
    )


def test_persist_skips_the_store_when_the_snapshot_is_empty() -> None:
    engine = MagicMock()
    engine.snapshot_escalation_state.return_value = {}
    engine.snapshot_exploration_state.return_value = {}
    store = MagicMock()
    _persist_router_state(engine, store)
    store.save_escalation_state.assert_not_called()


@dataclass
class _M:
    name: str


class _TieredPool:
    def __init__(self) -> None:
        self._tiers = {
            "cheap": [_M("qwen")],
            "mid": [_M("glm")],
            "high": [],
            "frontier": [],
        }

    def get_tier_models(self, tier: str) -> list[_M]:
        return self._tiers.get(tier, [])

    def is_healthy(self, name: str) -> bool:
        return True


@dataclass
class _Session:
    tool_identity: str


class _SessionManager:
    def get_session(self, session_id: str) -> _Session:
        return _Session(tool_identity="toolA")


class _Index:
    def count_labeled(self) -> int:
        return 100

    def count_total_labeled(self) -> int:
        return 100

    def effective_labeled(self) -> float:
        return 100.0

    def effective_tier2(self) -> float:
        return 100.0

    def model_priors(self) -> dict[str, tuple[float, float]]:
        return {}

    def query(self, embedding: np.ndarray, k: int = 20) -> list:  # type: ignore[type-arg]
        return []


class _Embedder:
    def embed(self, text: str) -> np.ndarray:  # type: ignore[type-arg]
        return np.zeros(8, dtype=np.float32)


def _engine() -> RouterEngine:
    return RouterEngine(
        model_pool=_TieredPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2),
        task_key_resolver=lambda _s: "repoA",
    )


def _fail(eng: RouterEngine) -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key="t::a",
        exit_code=1,
        blocking=True,
        confirmed=True,
    )


def test_escalation_state_round_trips_through_the_store(tmp_path: Path) -> None:
    # save(snapshot()) on shutdown then restore(load()) on startup: one accrued red survives
    # a restart, so a single further recurrence still escalates.
    store = OutcomeStore(db_path=str(tmp_path / "shunt.db"))
    try:
        eng = _engine()
        eng.decide("s1", "task")
        _fail(eng)  # one accrued failure, below the threshold
        store.save_escalation_state(eng.snapshot_escalation_state())

        restarted = _engine()
        restarted.restore_escalation_state(store.load_escalation_state())
        restarted.decide("s2", "task")
        _fail(restarted)  # the second recurrence across the restart
        model, reason, _prov = restarted.decide("s3", "task")
        assert model == "glm"  # the restored red counted; cheap -> mid
        assert reason == "auto_escalation"
    finally:
        store.close()
