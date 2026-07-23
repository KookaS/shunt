"""The quarantined Tier-1 (weak wire prior) consumer branch of CaptureCoordinator.

A structured wire signal is recorded as a ``wire_tier1`` event but MUST stay out of the
trusted Tier-2 neighbourhood and ``count_verified_outcomes`` until a Tier-2 corroborates.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from shunt.capture.coordinator import CaptureCoordinator, WorkDirResolver
from shunt.db.store import OutcomeEvent, OutcomeStore
from shunt.proxy.wire_signals import WIRE_TERMINAL_STOP, WIRE_TOOL_ERROR_COUNT
from shunt.session import Session
from shunt.verifiers.base import VerifierResult


class _FakeVerifier:
    def __init__(self, result: VerifierResult, *, record_calls: list[str] | None = None) -> None:
        self._result = result
        self._calls = record_calls if record_calls is not None else []

    def verify(self, text: str = "", work_dir: str | None = None) -> VerifierResult:
        self._calls.append(work_dir or "")
        return self._result


def _emb(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


@pytest.fixture
def store(tmp_path) -> OutcomeStore:  # type: ignore[no-untyped-def]
    s = OutcomeStore(db_path=str(tmp_path / "t.db"))
    yield s
    s.close()


def _store_embedded_session(store: OutcomeStore, sid: str, emb: np.ndarray) -> None:
    store.store_session(
        session_id=sid,
        prompt_text="fix the bug",
        embedding=emb,
        model_chosen="deepseek-v4-flash",
        cost=0.01,
        cache_stats={},
        duration=1.0,
    )


def _closed_session(sid: str, metadata: dict | None = None) -> Session:
    now = datetime.now(UTC)
    sess = Session(session_id=sid, tool_identity="tool-a", start_time=now)
    sess.end_time = now
    if metadata:
        sess.metadata.update(metadata)
    return sess


def _event_rows(store: OutcomeStore, sid: str) -> list[dict]:
    with store._lock:  # noqa: SLF001
        rows = store._conn.execute(  # noqa: SLF001
            "SELECT tier, source, outcome FROM outcome_events "
            "WHERE session_id = ? ORDER BY event_id",
            (sid,),
        ).fetchall()
    return [dict(r) for r in rows]


def test_wire_tier1_recorded_but_quarantined(store: OutcomeStore) -> None:
    sid = "sess-wire"
    emb = _emb()
    _store_embedded_session(store, sid, emb)
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(),  # manual-only: no work_dir → no Tier-2
        verifier=_FakeVerifier(VerifierResult(outcome="success", confidence=1.0)),
        store=store,
    )

    coord.capture(_closed_session(sid, {WIRE_TERMINAL_STOP: "end_turn"}))

    # A Tier-1 wire event was recorded in the append-only log (observability).
    rows = _event_rows(store, sid)
    assert rows == [{"tier": 1, "source": "wire_tier1", "outcome": "weak_success"}]
    # ...but it is quarantined: no verified count, no materialized view, not a neighbour.
    assert store.count_verified_outcomes() == 0
    assert store.get_outcome(sid) is None
    assert sid not in {h[0] for h in store.query_index(emb, k=5)}


def test_wire_tier1_failure_from_tool_error(store: OutcomeStore) -> None:
    sid = "sess-toolerr"
    _store_embedded_session(store, sid, _emb())
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(),
        verifier=_FakeVerifier(VerifierResult(outcome="unknown", confidence=0.0)),
        store=store,
    )

    coord.capture(_closed_session(sid, {WIRE_TOOL_ERROR_COUNT: 3}))

    assert _event_rows(store, sid) == [{"tier": 1, "source": "wire_tier1", "outcome": "failure"}]


def test_unknown_tier2_with_wire_signal_writes_quarantined_tier1(store: OutcomeStore) -> None:
    # work_dir IS set but the off-wire verifier returns unknown; a genuine wire signal still
    # yields a weak, quarantined Tier-1 prior (never a fabricated Tier-2 from unknown).
    sid = "sess-unknown-wire"
    _store_embedded_session(store, sid, _emb())
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=_FakeVerifier(VerifierResult(outcome="unknown", confidence=0.0)),
        store=store,
    )

    coord.capture(_closed_session(sid, {WIRE_TERMINAL_STOP: "end_turn"}))

    assert _event_rows(store, sid) == [
        {"tier": 1, "source": "wire_tier1", "outcome": "weak_success"}
    ]
    assert store.count_verified_outcomes() == 0


def test_ac3_no_workdir_no_signal_writes_nothing(store: OutcomeStore) -> None:
    sid = "sess-nothing"
    _store_embedded_session(store, sid, _emb())
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(),
        verifier=_FakeVerifier(VerifierResult(outcome="success", confidence=1.0)),
        store=store,
    )

    coord.capture(_closed_session(sid))  # empty metadata: no structured signal

    assert _event_rows(store, sid) == []
    assert store.get_outcome(sid) is None


def test_wire_tier1_then_tier2_coexist_and_promote(store: OutcomeStore) -> None:
    sid = "sess-promote"
    emb = _emb()
    _store_embedded_session(store, sid, emb)

    # 1) manual-only close records the quarantined weak wire prior.
    CaptureCoordinator(
        resolver=WorkDirResolver(),
        verifier=_FakeVerifier(VerifierResult(outcome="success", confidence=1.0)),
        store=store,
    ).capture(_closed_session(sid, {WIRE_TERMINAL_STOP: "end_turn"}))
    assert store.count_verified_outcomes() == 0
    assert sid not in {h[0] for h in store.query_index(emb, k=5)}

    # 2) a later off-wire Tier-2 on the SAME session coexists (distinct idempotency_key)
    #    and promotes it into the trusted neighbourhood.
    CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=_FakeVerifier(VerifierResult(outcome="failure", confidence=0.9)),
        store=store,
    ).capture(_closed_session(sid))

    rows = _event_rows(store, sid)
    assert len(rows) == 2
    assert {r["source"] for r in rows} == {"wire_tier1", "auto_tier2"}
    outcome = store.get_outcome(sid)
    assert outcome is not None
    assert outcome["tier2_outcome"] == "failure"
    assert outcome["tier1_outcome"] == "weak_success"  # the wire prior is preserved
    assert store.count_verified_outcomes() == 1
    assert sid in {h[0] for h in store.query_index(emb, k=5)}


def test_wire_tier1_idempotent(store: OutcomeStore) -> None:
    sid = "sess-idem"
    _store_embedded_session(store, sid, _emb())
    meta = {WIRE_TERMINAL_STOP: "end_turn"}
    for _ in range(2):  # a worker retry of the same close
        CaptureCoordinator(
            resolver=WorkDirResolver(),
            verifier=_FakeVerifier(VerifierResult(outcome="success", confidence=1.0)),
            store=store,
        ).capture(_closed_session(sid, meta))
    assert len(_event_rows(store, sid)) == 1


def test_store_quarantines_tier1_only_projection(store: OutcomeStore) -> None:
    # Store-level guarantee: appending a Tier-1-only event never materializes an `outcomes`
    # row nor indexes the session; a later Tier-2 event does both.
    sid = "sess-store"
    emb = _emb()
    _store_embedded_session(store, sid, emb)

    store.append_outcome_event(
        OutcomeEvent(
            sid,
            tier=1,
            source="wire_tier1",
            outcome="weak_success",
            confidence=0.3,
            run_signature="wire",
        )
    )
    assert store.get_outcome(sid) is None
    assert store._index.count == 0  # noqa: SLF001
    assert store.get_labeled_embeddings() == []

    store.append_outcome_event(
        OutcomeEvent(
            sid,
            tier=2,
            source="auto_tier2",
            outcome="success",
            confidence=0.9,
            run_signature="run-1",
        )
    )
    assert store.get_outcome(sid) is not None
    assert store._index.count == 1  # noqa: SLF001
    assert [s for s, _ in store.get_labeled_embeddings()] == [sid]
