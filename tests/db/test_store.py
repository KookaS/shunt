from __future__ import annotations

import json

import numpy as np
import pytest

from shunt.db.store import OutcomeStore


def _random_emb(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


@pytest.fixture
def store(tmp_path: pytest.TempPathFactory) -> OutcomeStore:
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    db_path = str(store_dir / "test.db")
    s = OutcomeStore(db_path=db_path)
    yield s
    s.close()


def test_store_and_get_session(store: OutcomeStore) -> None:
    sid = "session-1"
    store.store_session(
        session_id=sid,
        prompt_text="Hello",
        embedding=_random_emb(),
        model_chosen="model-a",
        cost=0.5,
        cache_stats={"tokens_cached": 100},
        duration=10.0,
    )
    session = store.get_session(sid)
    assert session is not None
    assert session["session_id"] == sid
    assert session["prompt_text"] == "Hello"
    assert session["model_chosen"] == "model-a"
    assert session["cost"] == 0.5
    assert session["session_duration_seconds"] == 10.0
    assert json.loads(session["cache_stats"]) == {"tokens_cached": 100}


def test_get_session_nonexistent(store: OutcomeStore) -> None:
    assert store.get_session("no-such-session") is None


def test_store_and_get_outcome(store: OutcomeStore) -> None:
    sid = "session-1"
    store.store_session(sid, "prompt", _random_emb(), "m", 0.1, {}, 1.0)
    store.store_outcome(
        session_id=sid,
        tier1_outcome="cheap_model_ok",
        tier1_confidence=0.85,
        aggregated_confidence=0.85,
    )
    outcome = store.get_outcome(sid)
    assert outcome is not None
    assert outcome["session_id"] == sid
    assert outcome["tier1_outcome"] == "cheap_model_ok"
    assert outcome["tier1_confidence"] == 0.85
    assert outcome["aggregated_confidence"] == 0.85
    assert outcome["tier2_outcome"] is None
    assert outcome["human_label"] is None


def test_get_outcome_nonexistent(store: OutcomeStore) -> None:
    assert store.get_outcome("no-such-session") is None


def test_store_outcome_update(store: OutcomeStore) -> None:
    sid = "session-1"
    store.store_session(sid, "prompt", _random_emb(), "m", 0.1, {}, 1.0)
    store.store_outcome(sid, "cheap_model_ok", 0.8, aggregated_confidence=0.8)
    store.store_outcome(sid, "expensive_model_better", 0.95, aggregated_confidence=0.95)
    outcome = store.get_outcome(sid)
    assert outcome is not None
    assert outcome["tier1_outcome"] == "expensive_model_better"
    assert outcome["aggregated_confidence"] == 0.95


def test_store_session_with_decision_provenance(store: OutcomeStore) -> None:
    sid = "session-dp"
    dp = {"strategy": "knn", "neighbors": 5, "threshold": 0.7}
    store.store_session(
        session_id=sid,
        prompt_text="test",
        embedding=_random_emb(),
        model_chosen="m",
        cost=0.1,
        cache_stats={},
        duration=1.0,
        decision_provenance=dp,
    )
    session = store.get_session(sid)
    assert session is not None
    assert json.loads(session["decision_provenance"]) == dp


def test_get_sessions_pagination(store: OutcomeStore) -> None:
    for i in range(10):
        store.store_session(
            session_id=f"s{i:02d}",
            prompt_text=f"prompt{i}",
            embedding=_random_emb(),
            model_chosen="m",
            cost=i * 0.1,
            cache_stats={},
            duration=float(i),
        )
    all_sessions = store.get_sessions(limit=100, offset=0)
    assert len(all_sessions) == 10

    page1 = store.get_sessions(limit=3, offset=0)
    assert len(page1) == 3

    page2 = store.get_sessions(limit=3, offset=3)
    assert len(page2) == 3

    # verify sorted by timestamp DESC
    for i in range(1, len(page1)):
        assert page1[i - 1]["timestamp"] >= page1[i]["timestamp"]


def test_get_all_embeddings(store: OutcomeStore) -> None:
    for i in range(3):
        store.store_session(
            session_id=f"emb-s{i}",
            prompt_text="test",
            embedding=_random_emb(),
            model_chosen="m",
            cost=0.1,
            cache_stats={},
            duration=1.0,
        )
    embeddings = store.get_all_embeddings()
    assert len(embeddings) == 3
    for sid, blob in embeddings:
        assert isinstance(sid, str)
        assert isinstance(blob, bytes)
        assert len(blob) > 0


def test_get_all_embeddings_filters_none(store: OutcomeStore) -> None:
    store.store_session(
        session_id="no-emb",
        prompt_text="test",
        embedding=None,
        model_chosen="m",
        cost=0.1,
        cache_stats={},
        duration=1.0,
    )
    embeddings = store.get_all_embeddings()
    assert len(embeddings) == 0


def test_update_human_label(store: OutcomeStore) -> None:
    sid = "label-test"
    store.store_session(sid, "prompt", _random_emb(), "m", 0.1, {}, 1.0)
    store.store_outcome(sid, "cheap_model_ok", 0.8, aggregated_confidence=0.8)

    store.update_human_label(sid, "correct")
    outcome = store.get_outcome(sid)
    assert outcome is not None
    assert outcome["human_label"] == "correct"
    assert outcome["human_label_timestamp"] is not None


def test_get_stats(store: OutcomeStore) -> None:
    stats = store.get_stats()
    assert stats["session_count"] == 0
    assert stats["outcome_count"] == 0
    assert stats["total_cost"] == 0.0
    assert stats["labeled_count"] == 0
    assert stats["index_size"] == 0

    for i in range(5):
        store.store_session(
            session_id=f"stat-s{i}",
            prompt_text="test",
            embedding=_random_emb() if i < 3 else None,
            model_chosen="m",
            cost=(i + 1) * 0.5,
            cache_stats={},
            duration=1.0,
        )

    for i in range(3):
        store.store_outcome(f"stat-s{i}", "ok", 0.8, aggregated_confidence=0.8)

    store.update_human_label("stat-s0", "correct")

    stats = store.get_stats()
    assert stats["session_count"] == 5
    assert stats["outcome_count"] == 3
    assert stats["total_cost"] == pytest.approx(7.5)
    assert stats["labeled_count"] == 1
    assert stats["index_size"] == 3


def test_store_session_with_timestamp(store: OutcomeStore) -> None:
    ts = "2024-06-15T12:00:00+00:00"
    store.store_session(
        session_id="ts-test",
        prompt_text="test",
        embedding=_random_emb(),
        model_chosen="m",
        cost=0.1,
        cache_stats={},
        duration=1.0,
        timestamp=ts,
    )
    session = store.get_session("ts-test")
    assert session is not None
    assert session["timestamp"] == ts


def test_roundtrip_outcome_with_all_fields(store: OutcomeStore) -> None:
    sid = "full-outcome"
    store.store_session(sid, "prompt", _random_emb(), "m", 0.1, {}, 1.0)
    store.store_outcome(
        session_id=sid,
        tier1_outcome="cheap",
        tier1_confidence=0.7,
        tier2_outcome="expensive",
        tier2_confidence=0.9,
        aggregated_confidence=0.85,
        human_label="correct",
        time_decay_weight=1.5,
    )
    o = store.get_outcome(sid)
    assert o is not None
    assert o["tier2_outcome"] == "expensive"
    assert o["tier2_confidence"] == 0.9
    assert o["human_label"] == "correct"
    assert o["time_decay_weight"] == 1.5


def test_persist_and_reload_index(tmp_path: pytest.TempPathFactory) -> None:
    store_dir = tmp_path / "persist"
    store_dir.mkdir()
    db_path = str(store_dir / "test.db")
    s1 = OutcomeStore(db_path=db_path)
    emb = _random_emb()
    s1.store_session("persist-s1", "hello", emb, "m", 0.1, {}, 1.0)
    # An outcome is what puts a session in the index; without it there is nothing to save.
    s1.store_outcome("persist-s1", tier1_outcome="success", tier1_confidence=0.9)
    s1.persist_index()
    s1.close()

    s2 = OutcomeStore(db_path=db_path)
    # should load from persisted index
    results = s2._index.query(emb, k=1)
    assert len(results) == 1
    idx, dist = results[0]
    assert s2._index.get_session_id(idx) == "persist-s1"
    assert dist < 0.01
    s2.close()
