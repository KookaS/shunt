"""Schema-v2 store behaviour: cost-UNKNOWN discriminator, append-only idempotency,
tombstone→rebuild, and a real v1 database migrating cleanly through OutcomeStore."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from shunt.db.store import OutcomeEvent, OutcomeStore, SessionProvenance


def _emb(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


@pytest.fixture
def store(tmp_path: Path) -> OutcomeStore:
    s = OutcomeStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


# ── cost_known discriminator ──────────────────────────────────────────────────
def test_cost_unknown_is_distinct_from_real_zero(store: OutcomeStore) -> None:
    store.store_session(
        "known-zero",
        "p",
        _emb(),
        "m",
        0.0,
        {},
        1.0,
        provenance=SessionProvenance(cost_known=True),
    )
    store.store_session(
        "unknown",
        "p",
        _emb(),
        "m",
        5.0,
        {},
        1.0,
        provenance=SessionProvenance(cost_known=False),
    )

    known = store.get_session("known-zero")
    unknown = store.get_session("unknown")
    assert known is not None and unknown is not None
    assert known["cost_known"] == 1
    assert known["cost"] == 0.0
    assert unknown["cost_known"] == 0

    # Cost aggregation excludes the UNKNOWN row entirely — its 5.0 is not summed and it
    # is not treated as 0.0 either; it is simply not a known cost.
    stats = store.get_stats()
    assert stats["total_cost"] == pytest.approx(0.0)
    assert stats["cost_unknown_count"] == 1


def test_cost_known_defaults_true_for_existing_callers(store: OutcomeStore) -> None:
    # Existing callers pass no provenance → cost_known defaults to 1 (known).
    store.store_session("s", "p", _emb(), "m", 2.5, {}, 1.0)
    row = store.get_session("s")
    assert row is not None
    assert row["cost_known"] == 1
    assert store.get_stats()["total_cost"] == pytest.approx(2.5)


def test_selection_propensity_and_fingerprint_persist(store: OutcomeStore) -> None:
    store.store_session(
        "s",
        "p",
        _emb(),
        "m",
        1.0,
        {},
        1.0,
        provenance=SessionProvenance(
            selection_propensity=0.25,
            model_fingerprint="kimi-k3@2026-07",
        ),
    )
    row = store.get_session("s")
    assert row is not None
    assert row["selection_propensity"] == pytest.approx(0.25)
    assert row["model_fingerprint"] == "kimi-k3@2026-07"


# ── append-only idempotency ───────────────────────────────────────────────────
def test_append_event_idempotent_no_double_count(store: OutcomeStore) -> None:
    store.store_session("s", "p", _emb(), "m", 1.0, {}, 1.0)
    event = OutcomeEvent(
        session_id="s",
        tier=2,
        source="auto_tier2",
        outcome="success",
        confidence=1.0,
        run_signature="run-abc",
    )
    assert store.append_outcome_event(event) is True
    # Same idempotency_key (same session|source|run_signature) → no-op.
    assert store.append_outcome_event(event) is False

    n = store._conn.execute(
        "SELECT COUNT(*) FROM outcome_events WHERE session_id = 's'"
    ).fetchone()[0]
    assert n == 1
    # And the materialized view reflects the single verified outcome once.
    o = store.get_outcome("s")
    assert o is not None
    assert o["tier2_outcome"] == "success"
    assert o["outcome_source"] == "auto_tier2"
    assert store.count_verified_outcomes() == 1


def test_human_beats_auto_tier2_in_materialized_view(store: OutcomeStore) -> None:
    store.store_session("s", "p", _emb(), "m", 1.0, {}, 1.0)
    store.append_outcome_event(
        OutcomeEvent(
            "s",
            2,
            "auto_tier2",
            "failure",
            0.8,
            "auto-1",
        )
    )
    store.append_outcome_event(
        OutcomeEvent(
            "s",
            2,
            "human",
            "success",
            1.0,
            "human-1",
        )
    )
    o = store.get_outcome("s")
    assert o is not None
    assert o["tier2_outcome"] == "success"
    assert o["outcome_source"] == "human"


# ── Tombstone → rebuild ───────────────────────────────────────────────────────
def test_tombstone_then_rebuild_drops_session(store: OutcomeStore) -> None:
    emb = _emb()
    store.store_session("s", "p", emb, "m", 1.0, {}, 1.0)
    store.append_outcome_event(OutcomeEvent("s", 2, "auto_tier2", "success", 1.0, "run-1"))

    # Indexed and labeled after the outcome lands.
    assert store._index.count == 1
    assert [sid for sid, _ in store.get_labeled_embeddings()] == ["s"]

    # A correction appends a tombstone event (never mutates history).
    store.append_outcome_event(
        OutcomeEvent(
            "s",
            2,
            "auto_tier2",
            "success",
            1.0,
            "tombstone-1",
            tombstoned=True,
        )
    )
    # Materialization drops the view row immediately.
    assert store.get_outcome("s") is None
    assert store.get_labeled_embeddings() == []

    # The HNSW index still holds the vector (no in-place delete); rebuild is the truth-up.
    store.rebuild_index()
    assert store._index.count == 0


def test_rebuild_index_reflects_only_live_labels(store: OutcomeStore) -> None:
    for i in range(3):
        store.store_session(f"s{i}", "p", _emb(), "m", 1.0, {}, 1.0)
        store.append_outcome_event(OutcomeEvent(f"s{i}", 2, "auto_tier2", "success", 1.0, f"r{i}"))
    store.append_outcome_event(
        OutcomeEvent(
            "s1",
            2,
            "auto_tier2",
            "success",
            1.0,
            "tomb-s1",
            tombstoned=True,
        )
    )
    store.rebuild_index()
    assert store._index.count == 2
    assert sorted(sid for sid, _ in store.get_labeled_embeddings()) == ["s0", "s2"]


# ── Real v1 DB on disk migrates through OutcomeStore ──────────────────────────
def test_v1_disk_db_opens_and_migrates(tmp_path: Path) -> None:
    db_path = str(tmp_path / "v1.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, prompt_text TEXT NOT NULL, "
        "embedding_blob BLOB, model_chosen TEXT NOT NULL, cost REAL NOT NULL, "
        "cache_stats TEXT NOT NULL, session_duration_seconds REAL NOT NULL, "
        "timestamp TEXT NOT NULL, decision_provenance TEXT)"
    )
    conn.execute(
        "CREATE TABLE outcomes (session_id TEXT PRIMARY KEY, tier1_outcome TEXT NOT NULL, "
        "tier1_confidence REAL NOT NULL, tier2_outcome TEXT, tier2_confidence REAL, "
        "aggregated_confidence REAL NOT NULL, human_label TEXT, human_label_timestamp TEXT, "
        "time_decay_weight REAL DEFAULT 1.0, created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, '2024-01-01')")
    conn.execute(
        "INSERT INTO sessions (session_id, prompt_text, model_chosen, cost, cache_stats, "
        "session_duration_seconds, timestamp) VALUES ('old', 'p', 'm', 1.25, '{}', 1.0, 't')"
    )
    conn.commit()
    conn.close()

    # Opening the store runs run_migrations; the old row must survive with defaults.
    store = OutcomeStore(db_path=db_path)
    try:
        row = store.get_session("old")
        assert row is not None
        assert row["cost"] == 1.25
        assert row["cost_known"] == 1
        assert row["selection_propensity"] is None
        assert row["model_fingerprint"] is None
        # The append-only machinery works on the migrated DB.
        store.append_outcome_event(OutcomeEvent("old", 2, "auto_tier2", "success", 1.0, "r1"))
        assert store.get_outcome("old") is not None
    finally:
        store.close()
