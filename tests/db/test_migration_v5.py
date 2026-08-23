"""Schema-v5 migration: backfill ``outcome_events`` from pre-v2 ``outcomes`` rows.

v2 created the append-only log without seeding it, so a store written before v2 holds
materialized outcomes — and kNN index members — the log never saw.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from shunt.db.schema import (
    CREATE_OUTCOME_EVENTS_SESSION_INDEX,
    CREATE_OUTCOME_EVENTS_TABLE,
    CREATE_OUTCOMES_TABLE,
    CREATE_ROUTER_STATE_TABLE,
    CREATE_SESSIONS_TABLE,
    CREATE_TIMESTAMP_INDEX,
    MIGRATE_V2_COST_KNOWN,
    MIGRATE_V2_MODEL_FINGERPRINT,
    MIGRATE_V2_OUTCOME_SOURCE,
    MIGRATE_V2_SELECTION_PROPENSITY,
    MIGRATE_V4_EXTERNAL_SESSION_ID,
    SCHEMA_VERSION,
    get_current_version,
    run_migrations,
)
from shunt.db.store import OutcomeStore

_LEGACY = "5a9b847e-legacy"
_HEALTHY = "healthy-session"


def test_schema_version_is_five() -> None:
    assert SCHEMA_VERSION == 5


def _v4_schema(conn: sqlite3.Connection) -> None:
    """A v4-shaped DB, stamped 1..4 — the shape the owner's live store carries."""
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute(CREATE_SESSIONS_TABLE)
    conn.execute(CREATE_OUTCOMES_TABLE)
    conn.execute(CREATE_TIMESTAMP_INDEX)
    for stmt in (
        MIGRATE_V2_COST_KNOWN,
        MIGRATE_V2_SELECTION_PROPENSITY,
        MIGRATE_V2_MODEL_FINGERPRINT,
        MIGRATE_V2_OUTCOME_SOURCE,
        MIGRATE_V4_EXTERNAL_SESSION_ID,
    ):
        conn.execute(stmt)
    conn.execute(CREATE_OUTCOME_EVENTS_TABLE)
    conn.execute(CREATE_OUTCOME_EVENTS_SESSION_INDEX)
    conn.execute(CREATE_ROUTER_STATE_TABLE)
    for version in (1, 2, 3, 4):
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
            (version,),
        )
    conn.commit()


def _session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "INSERT INTO sessions (session_id, prompt_text, embedding_blob, model_chosen, cost, "
        "cache_stats, session_duration_seconds, timestamp) "
        "VALUES (?, 'p', X'00', 'm', 0.0, '{}', 1.0, '2026-07-20T11:27:23+00:00')",
        (session_id,),
    )


def _outcome(conn: sqlite3.Connection, session_id: str, tier2: str | None = "success") -> None:
    conn.execute(
        "INSERT INTO outcomes (session_id, tier1_outcome, tier1_confidence, tier2_outcome, "
        "tier2_confidence, aggregated_confidence, created_at) "
        "VALUES (?, 'success', 1.0, ?, 0.9, 1.0, '2026-07-20T11:27:23.408753+00:00')",
        (session_id, tier2),
    )


def _events(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM outcome_events WHERE session_id = ? ORDER BY event_id", (session_id,)
    ).fetchall()


def _legacy_store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _v4_schema(conn)
    _session(conn, _LEGACY)
    _outcome(conn, _LEGACY)
    conn.commit()
    return conn


def test_a_pre_v2_outcome_with_no_event_is_backfilled() -> None:
    conn = _legacy_store()
    run_migrations(conn)
    rows = _events(conn, _LEGACY)
    assert len(rows) == 1
    assert rows[0]["tier"] == 2
    assert rows[0]["outcome"] == "success"
    assert rows[0]["confidence"] == 0.9
    assert rows[0]["tombstoned"] == 0
    assert rows[0]["source"] == "human"
    # Dated to the outcome's own created_at, not to migration time — the log's chronology
    # must stay truthful or every recency read over it lies.
    assert rows[0]["created_at"] == "2026-07-20T11:27:23.408753+00:00"
    assert get_current_version(conn) == 5


def test_a_tier1_only_legacy_outcome_backfills_as_a_tier1_event() -> None:
    conn = sqlite3.connect(":memory:")
    _v4_schema(conn)
    _session(conn, _LEGACY)
    _outcome(conn, _LEGACY, tier2=None)
    conn.commit()
    run_migrations(conn)
    rows = _events(conn, _LEGACY)
    assert [rows[0]["tier"], rows[0]["confidence"]] == [1, 1.0]


def test_a_healthy_store_is_untouched() -> None:
    conn = sqlite3.connect(":memory:")
    _v4_schema(conn)
    _session(conn, _HEALTHY)
    _outcome(conn, _HEALTHY)
    conn.execute(
        "INSERT INTO outcome_events (session_id, tier, source, outcome, confidence, "
        "idempotency_key, tombstoned, created_at) "
        "VALUES (?, 2, 'auto_tier2', 'success', 0.9, 'k1', 0, '2026-08-01T00:00:00+00:00')",
        (_HEALTHY,),
    )
    conn.commit()
    run_migrations(conn)
    rows = _events(conn, _HEALTHY)
    assert len(rows) == 1
    assert rows[0]["idempotency_key"] == "k1"


def test_a_tombstoned_session_is_not_resurrected() -> None:
    # `_NOT_TOMBSTONED` reads the LATEST event; a backfill appended after a tombstone would
    # silently un-erase the session and put it back in the index.
    conn = sqlite3.connect(":memory:")
    _v4_schema(conn)
    _session(conn, _HEALTHY)
    _outcome(conn, _HEALTHY)
    conn.execute(
        "INSERT INTO outcome_events (session_id, tier, source, outcome, confidence, "
        "idempotency_key, tombstoned, created_at) "
        "VALUES (?, 2, 'human', 'success', 0.9, 'k1', 1, '2026-08-01T00:00:00+00:00')",
        (_HEALTHY,),
    )
    conn.commit()
    run_migrations(conn)
    rows = _events(conn, _HEALTHY)
    assert len(rows) == 1
    assert rows[0]["tombstoned"] == 1


def test_an_outcome_whose_session_is_gone_is_skipped() -> None:
    conn = sqlite3.connect(":memory:")
    _v4_schema(conn)
    _outcome(conn, "orphan")
    conn.commit()
    run_migrations(conn)
    assert _events(conn, "orphan") == []


def test_running_the_migration_twice_is_a_no_op() -> None:
    conn = _legacy_store()
    run_migrations(conn)
    first = [tuple(row) for row in _events(conn, _LEGACY)]
    run_migrations(conn)
    assert [tuple(row) for row in _events(conn, _LEGACY)] == first
    # And again with the version stamp forced back, so the guard is not the only defence.
    conn.execute("DELETE FROM schema_version WHERE version = 5")
    conn.commit()
    run_migrations(conn)
    assert [tuple(row) for row in _events(conn, _LEGACY)] == first


def test_a_fresh_store_migrates_cleanly_to_five() -> None:
    conn = sqlite3.connect(":memory:")
    run_migrations(conn)
    assert get_current_version(conn) == SCHEMA_VERSION
    conn.row_factory = sqlite3.Row
    versions = [row["version"] for row in conn.execute("SELECT version FROM schema_version")]
    assert versions == [1, 2, 3, 4, 5]


def test_the_census_stops_inverting_once_the_legacy_row_is_backfilled(tmp_path: Path) -> None:
    # The exact live rig: an `outcomes` row with tier2 set and zero events makes
    # `tier2` (counted off `outcomes`) exceed `labeled` (counted off `outcome_events`).
    store = OutcomeStore(db_path=str(tmp_path / "census.db"))
    try:
        for i in range(2):
            store.store_session(f"live-{i}", "p", np.zeros(64, dtype=np.float32), "m", 0.1, {}, 1.0)
            store.store_outcome(f"live-{i}", "success", 1.0, tier2_outcome="success")
        store._conn.execute("DELETE FROM outcome_events WHERE session_id = 'live-0'")
        store._conn.execute("DELETE FROM schema_version WHERE version = 5")
        store._conn.commit()

        broken = store.stratum_census().live
        assert (broken.labeled, broken.tier2) == (1, 2)

        run_migrations(store._conn)

        fixed = store.stratum_census().live
        assert fixed.tier2 <= fixed.labeled
        assert (fixed.labeled, fixed.tier2, fixed.indexed) == (2, 2, 2)
    finally:
        store.close()
