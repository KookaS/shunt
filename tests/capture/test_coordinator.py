from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from shunt.capture.coordinator import CaptureCoordinator, WorkDirResolver
from shunt.db.store import OutcomeStore
from shunt.session import Session
from shunt.verifiers.base import VerifierResult


class _FakeVerifier:
    """Stands in for AutoDetectVerifier — returns a scripted result, never runs pytest."""

    def __init__(self, result: VerifierResult, *, record_calls: list[str] | None = None) -> None:
        self._result = result
        self._calls = record_calls if record_calls is not None else []

    def verify(self, text: str = "", work_dir: str | None = None) -> VerifierResult:
        self._calls.append(work_dir or "")
        return self._result


def _emb(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


@pytest.fixture
def store(tmp_path: pytest.TempPathFactory) -> OutcomeStore:
    s = OutcomeStore(db_path=str(tmp_path / "t.db"))  # type: ignore[operator]
    yield s
    s.close()


def _closed_session(sid: str = "s1", tool_identity: str = "tool-a") -> Session:
    now = datetime.now(UTC)
    sess = Session(session_id=sid, tool_identity=tool_identity, start_time=now)
    sess.end_time = now
    return sess


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


def test_ac2_failure_is_labelled_and_becomes_neighbour(store: OutcomeStore) -> None:
    sid = "sess-fail"
    emb = _emb()
    _store_embedded_session(store, sid, emb)
    verifier = _FakeVerifier(VerifierResult(outcome="failure", confidence=0.7))
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=verifier,
        store=store,
    )

    coord.capture(_closed_session(sid))

    outcome = store.get_outcome(sid)
    assert outcome is not None
    assert outcome["tier2_outcome"] == "failure"
    assert outcome["outcome_source"] == "auto_tier2"
    assert store.count_verified_outcomes() == 1
    # the labelled session is now a kNN neighbour of its own embedding
    hits = {h[0] for h in store.query_index(emb, k=5)}
    assert sid in hits


def test_ac3_unknown_writes_nothing(store: OutcomeStore) -> None:
    sid = "sess-unknown"
    _store_embedded_session(store, sid, _emb())
    verifier = _FakeVerifier(VerifierResult(outcome="unknown", confidence=0.0))
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=verifier,
        store=store,
    )

    coord.capture(_closed_session(sid))

    assert store.get_outcome(sid) is None
    assert store.count_verified_outcomes() == 0
    assert _event_count(store) == 0


def test_ac3_no_work_dir_writes_nothing_and_skips_verify(store: OutcomeStore) -> None:
    sid = "sess-nowd"
    _store_embedded_session(store, sid, _emb())
    calls: list[str] = []
    verifier = _FakeVerifier(VerifierResult(outcome="success", confidence=0.8), record_calls=calls)
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(),  # no work_dir configured
        verifier=verifier,
        store=store,
    )

    coord.capture(_closed_session(sid))

    assert calls == []  # verifier never invoked without a work_dir (no wrong-repo run)
    assert store.get_outcome(sid) is None
    assert _event_count(store) == 0


def test_ac7_double_capture_is_idempotent(store: OutcomeStore) -> None:
    sid = "sess-retry"
    _store_embedded_session(store, sid, _emb())
    verifier = _FakeVerifier(VerifierResult(outcome="success", confidence=0.8))
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=verifier,
        store=store,
    )

    coord.capture(_closed_session(sid))
    coord.capture(_closed_session(sid))  # worker retry of the same closed session

    assert _event_count(store) == 1


def _event_count(store: OutcomeStore) -> int:
    with store._lock:  # noqa: SLF001 (test reaches into the store to prove zero rows)
        return int(
            store._conn.execute("SELECT COUNT(*) AS c FROM outcome_events").fetchone()["c"]  # noqa: SLF001
        )


def test_ac6_capture_event_inherits_session_fingerprint(store: OutcomeStore) -> None:
    # End-to-end test: the fingerprint the persist site wrote onto the session row is
    # copied verbatim onto the auto_tier2 outcome event (model version segmentation).
    from shunt.db.store import SessionProvenance

    sid = "sess-fp"
    store.store_session(
        session_id=sid,
        prompt_text="p",
        embedding=_emb(),
        model_chosen="deepseek-v4-flash",
        cost=0.01,
        cache_stats={},
        duration=1.0,
        provenance=SessionProvenance(model_fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus"),
    )
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=_FakeVerifier(VerifierResult(outcome="success", confidence=1.0)),
        store=store,
    )

    coord.capture(_closed_session(sid))

    row = store._conn.execute(
        "SELECT model_fingerprint FROM outcome_events WHERE session_id = ?", (sid,)
    ).fetchone()
    assert row is not None
    assert row["model_fingerprint"] == "alibaba/qwen3.7-plus@qwen3.7-plus"


class _RecordingCallback:
    """Stands in for engine.record_outcome — captures the (downshift, success) it receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, bool]] = []

    def __call__(self, *, downshift: bool, success: bool) -> None:
        self.calls.append({"downshift": downshift, "success": success})


def _coord(
    store: OutcomeStore, result: VerifierResult, cb: _RecordingCallback
) -> CaptureCoordinator:
    return CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=_FakeVerifier(result),
        store=store,
        record_outcome_callback=cb,
    )


def test_ac1_verified_failure_records_outcome(store: OutcomeStore) -> None:
    sid = "s-rec-fail"
    _store_embedded_session(store, sid, _emb())
    cb = _RecordingCallback()
    coord = _coord(store, VerifierResult(outcome="failure", confidence=0.7), cb)
    coord.capture(_closed_session(sid))
    assert cb.calls == [{"downshift": False, "success": False}]


def test_ac1_verified_success_records_outcome(store: OutcomeStore) -> None:
    sid = "s-rec-ok"
    _store_embedded_session(store, sid, _emb())
    cb = _RecordingCallback()
    coord = _coord(store, VerifierResult(outcome="success", confidence=0.9), cb)
    coord.capture(_closed_session(sid))
    assert cb.calls == [{"downshift": False, "success": True}]


def test_records_downshift_from_stored_provenance(store: OutcomeStore) -> None:
    # The routed decision's downshift flag lives on the durable session row, so capture
    # can attribute the verified outcome to it without an in-memory pending queue.
    sid = "s-down"
    store.store_session(
        session_id=sid,
        prompt_text="p",
        embedding=_emb(),
        model_chosen="cheap",
        cost=0.01,
        cache_stats={},
        duration=1.0,
        decision_provenance={"downshift": True, "selection_rule_used": "exploration"},
    )
    cb = _RecordingCallback()
    coord = _coord(store, VerifierResult(outcome="success", confidence=1.0), cb)
    coord.capture(_closed_session(sid))
    assert cb.calls == [{"downshift": True, "success": True}]


def test_tier1_only_capture_does_not_record(store: OutcomeStore) -> None:
    # A quarantined Tier-1 wire prior must NOT feed the gate — only verified Tier-2 does.
    from shunt.proxy.wire_signals import WIRE_TOOL_ERROR_COUNT

    sid = "s-t1"
    _store_embedded_session(store, sid, _emb())
    cb = _RecordingCallback()
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(),  # no work_dir → no Tier-2 run
        verifier=_FakeVerifier(VerifierResult(outcome="unknown", confidence=0.0)),
        store=store,
        record_outcome_callback=cb,
    )
    sess = _closed_session(sid)
    sess.metadata[WIRE_TOOL_ERROR_COUNT] = 2  # a genuine structured wire signal
    coord.capture(sess)
    assert _event_count(store) == 1  # the Tier-1 prior WAS appended (quarantined)
    assert cb.calls == []  # but the gate never learned from it


def test_idempotent_capture_records_once(store: OutcomeStore) -> None:
    sid = "s-idem"
    _store_embedded_session(store, sid, _emb())
    cb = _RecordingCallback()
    coord = _coord(store, VerifierResult(outcome="success", confidence=0.8), cb)
    coord.capture(_closed_session(sid))
    coord.capture(_closed_session(sid))  # worker retry of the same closed session
    assert len(cb.calls) == 1  # deduped write → single record, no double-count


def test_no_callback_capture_still_labels(store: OutcomeStore) -> None:
    sid = "s-nocb"
    _store_embedded_session(store, sid, _emb())
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=_FakeVerifier(VerifierResult(outcome="success", confidence=0.9)),
        store=store,  # record_outcome_callback defaults None (no-engine deployment)
    )
    coord.capture(_closed_session(sid))
    outcome = store.get_outcome(sid)
    assert outcome is not None
    assert outcome["tier2_outcome"] == "success"


def test_conservative_gate_learns_from_capture(store: OutcomeStore) -> None:
    # End-to-end: a captured verified downshift outcome moves the real engine's gate slack,
    # proving record_outcome now has a live production caller.
    from unittest.mock import MagicMock

    from shunt.router.budget import ConservativeGate
    from shunt.router.engine import RouterEngine
    from shunt.router.policy import ExplorationPolicy

    gate = ConservativeGate(alpha=0.1)

    class _FakeIndex:
        def count_labeled(self) -> int:
            return 0

        def count_total_labeled(self) -> int:
            return 0

        def query(self, embedding: np.ndarray, k: int = 20) -> list:  # type: ignore[type-arg]
            return []

    engine = RouterEngine(
        model_pool=MagicMock(),
        session_manager=MagicMock(),
        outcome_index=_FakeIndex(),
        embedder=MagicMock(),
        exploration=ExplorationPolicy(enabled=True),
        conservative_gate=gate,
    )

    sid = "s-e2e"
    store.store_session(
        session_id=sid,
        prompt_text="p",
        embedding=_emb(),
        model_chosen="cheap",
        cost=0.01,
        cache_stats={},
        duration=1.0,
        decision_provenance={"downshift": True, "selection_rule_used": "exploration"},
    )
    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=_FakeVerifier(VerifierResult(outcome="success", confidence=1.0)),
        store=store,
        record_outcome_callback=engine.record_outcome,
    )

    assert gate.slack == 0.0
    coord.capture(_closed_session(sid))
    assert gate.slack == 1.0  # a verified downshift success banked slack


def test_fresh_tier2_insert_notifies_refit_scheduler(store: OutcomeStore) -> None:
    sid = "sess-refit"
    _store_embedded_session(store, sid, _emb())
    notes: list[bool] = []

    class _SpyScheduler:
        def note_capture(self) -> bool:
            notes.append(True)
            return False

    coord = CaptureCoordinator(
        resolver=WorkDirResolver(work_dir="/repo"),
        verifier=_FakeVerifier(VerifierResult(outcome="success", confidence=1.0)),
        store=store,
        refit_scheduler=_SpyScheduler(),  # type: ignore[arg-type]
    )

    coord.capture(_closed_session(sid))
    assert notes == [True]  # a fresh Tier-2 insert triggers exactly one refit notification

    # A duplicate capture dedups on the idempotency_key ⇒ no second notification.
    coord.capture(_closed_session(sid))
    assert notes == [True]
