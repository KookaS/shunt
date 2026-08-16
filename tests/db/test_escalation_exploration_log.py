"""R5 persistence: a flagged-checkpoint propensity survives to where an estimator reads it."""

# The whole point of R5 is the propensity record. If it does not reach storage joined to the
# verified outcome, the randomization was spent for nothing — so this walks the real path:
# RouterEngine.decide() -> provenance -> OutcomeStore -> the estimator's row shape.

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from benchmark.escalation.ope import (
    NOT_IDENTIFIED,
    always_escalate,
    estimate_policy_value,
    rows_from_records,
)
from shunt.db.store import OutcomeStore
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig

_SECRET = "sk-live-abcdefghijklmnop"
_SECRET_CHECK_ID = f"/home/olivier/secretrepo/tests/test_auth.py::test_key[{_SECRET}]"


@pytest.fixture
def store(tmp_path: Any) -> Any:
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


class _Pool:
    def __init__(self) -> None:
        self._ranked = ["cheap", "mid", "top"]

    def ranked_models(self) -> list[Any]:
        return [type("M", (), {"name": n})() for n in self._ranked]

    def rank_of(self, name: str) -> int | None:
        return self._ranked.index(name) if name in self._ranked else None

    def models_from_rank(self, i: int) -> list[Any]:
        return [type("M", (), {"name": n})() for n in self._ranked[max(i, 0) :]]

    def is_healthy(self, name: str) -> bool:
        return True


class _SessionManager:
    def get_session(self, session_id: str) -> Any:
        return type("S", (), {"tool_identity": "toolA"})()


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

    def query(self, embedding: np.ndarray, k: int = 20) -> list[Any]:
        return []


class _Embedder:
    def embed(self, text: str) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


def _engine(epsilon: float, seed: int | None = 5) -> RouterEngine:
    return RouterEngine(
        model_pool=_Pool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(
            enabled=True,
            escalate_after_n=2,
            exploration_epsilon=epsilon,
            exploration_seed=seed,
        ),
        task_key_resolver=lambda _session: "repoA",
    )


def _fail(engine: RouterEngine, dedup_key: str = "t::a") -> None:
    engine.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key=dedup_key,
        exit_code=1,
        is_infra_failure=False,
        confirmed=True,
    )


def _flagged_provenance(
    engine: RouterEngine, session_id: str, dedup_key: str = "t::a"
) -> dict[str, Any]:
    """Drive the engine to a flagged checkpoint and return that decision's provenance."""
    engine.decide(f"{session_id}-warm", "task")
    _fail(engine, dedup_key)
    engine.decide(f"{session_id}-mid", "task")
    _fail(engine, dedup_key)
    _model, _reason, provenance = engine.decide(session_id, "task")
    return provenance


def test_the_engine_stamps_the_propensity_onto_the_decision_provenance() -> None:
    provenance = _flagged_provenance(_engine(0.5), "s3")
    record = provenance["escalation_exploration"]
    assert 0.0 < record["propensity"] < 1.0
    assert record["randomized"] is True
    assert record["seed"] == 5
    assert record["epsilon"] == 0.5
    assert record["features"]["countable_failures"] == 2.0


def test_a_deterministic_engine_stamps_a_propensity_of_one() -> None:
    record = _flagged_provenance(_engine(0.0), "s3")["escalation_exploration"]
    assert record["propensity"] == 1.0
    assert record["randomized"] is False


def test_the_record_round_trips_through_the_store_joined_to_the_verified_outcome(
    store: OutcomeStore,
) -> None:
    provenance = _flagged_provenance(_engine(0.4), "s3")
    store.store_session(
        session_id="s3",
        prompt_text="task",
        embedding=np.zeros(8, dtype=np.float32),
        model_chosen="cheap",
        cost=1.0,
        cache_stats={},
        duration=1.0,
        decision_provenance=provenance,
    )
    store.store_outcome(
        session_id="s3",
        tier1_outcome="failure",
        tier1_confidence=0.7,
        tier2_outcome="success",
        tier2_confidence=0.95,
        aggregated_confidence=0.95,
    )
    rows = store.escalation_exploration_rows()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s3"
    assert rows[0]["outcome"] == "success"  # tier-2 (verified) wins over tier-1
    estimator_rows = rows_from_records(rows)
    assert estimator_rows[0].reward == 1.0
    assert 0.0 < estimator_rows[0].propensity < 1.0


def test_a_secret_bearing_checkpoint_id_is_redacted_in_the_db_and_the_export(
    store: OutcomeStore,
) -> None:
    # `checkpoint_id` IS a failing-check id, the one behaviour field that carries arbitrary text
    # (a parametrized test id can embed a secret). `StepRecord.committable` scrubs it; the
    # exploration log must scrub it too, or a key reaches sqlite and the documented OPE export.
    provenance = _flagged_provenance(_engine(0.4), "s3", _SECRET_CHECK_ID)
    assert _SECRET not in json.dumps(provenance)

    store.store_session(
        session_id="s3",
        prompt_text="task",
        embedding=np.zeros(8, dtype=np.float32),
        model_chosen="cheap",
        cost=1.0,
        cache_stats={},
        duration=1.0,
        decision_provenance=provenance,
    )
    stored = store._conn.execute(  # noqa: SLF001 (the persisted column is the thing under test)
        "SELECT decision_provenance FROM sessions WHERE session_id = 's3'"
    ).fetchone()[0]
    assert _SECRET not in stored

    checkpoint_id = store.escalation_exploration_rows()[0]["checkpoint_id"]
    assert _SECRET not in checkpoint_id
    assert "<redacted>" in checkpoint_id
    # The non-secret structure survives — redaction scrubs the key, it does not blank the id.
    assert checkpoint_id.startswith("/home/olivier/secretrepo/tests/test_auth.py::test_key[")


def test_sessions_without_an_escalation_record_are_not_returned(store: OutcomeStore) -> None:
    store.store_session(
        session_id="plain",
        prompt_text="an escalation_exploration mention in the PROMPT must not match",
        embedding=np.zeros(8, dtype=np.float32),
        model_chosen="cheap",
        cost=1.0,
        cache_stats={},
        duration=1.0,
        decision_provenance={"downshift": False},
    )
    assert store.escalation_exploration_rows() == []


def test_a_deterministic_live_log_is_refused_end_to_end(store: OutcomeStore) -> None:
    # The structural guard, walked from the live path: deterministic logging cannot produce
    # an escalation-value estimate, however many sessions it accumulates.
    engine = _engine(0.0)
    for i in range(6):
        provenance = _flagged_provenance(engine, f"s{i}")
        store.store_session(
            session_id=f"s{i}",
            prompt_text="task",
            embedding=np.zeros(8, dtype=np.float32),
            model_chosen="cheap",
            cost=1.0,
            cache_stats={},
            duration=1.0,
            decision_provenance=provenance,
        )
        store.store_outcome(
            session_id=f"s{i}",
            tier1_outcome="success",
            tier1_confidence=0.7,
            tier2_outcome="success",
            tier2_confidence=0.95,
            aggregated_confidence=0.95,
        )
    result = estimate_policy_value(
        rows_from_records(store.escalation_exploration_rows()), always_escalate
    )
    assert result.status == NOT_IDENTIFIED
    assert result.dr_estimate is None
