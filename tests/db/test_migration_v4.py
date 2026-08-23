"""Schema-v4 migration: external session identity (per-conversation keying + resume).

A v3 DB gains the nullable ``external_session_id`` column and its index without touching
old rows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
    SCHEMA_VERSION,
    get_current_version,
    run_migrations,
)
from shunt.db.store import OutcomeStore


def _inmemory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _v3_schema(conn: sqlite3.Connection) -> None:
    """Build a v3-shaped DB: v1 tables + v2 columns/events + the router_state table."""
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute(CREATE_SESSIONS_TABLE)
    conn.execute(CREATE_OUTCOMES_TABLE)
    conn.execute(CREATE_TIMESTAMP_INDEX)
    conn.execute(MIGRATE_V2_COST_KNOWN)
    conn.execute(MIGRATE_V2_SELECTION_PROPENSITY)
    conn.execute(MIGRATE_V2_MODEL_FINGERPRINT)
    conn.execute(MIGRATE_V2_OUTCOME_SOURCE)
    conn.execute(CREATE_OUTCOME_EVENTS_TABLE)
    conn.execute(CREATE_OUTCOME_EVENTS_SESSION_INDEX)
    conn.execute(CREATE_ROUTER_STATE_TABLE)
    for v in (1, 2, 3):
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))", (v,)
        )
    conn.commit()


def test_v3_db_migrates_adds_column_and_index() -> None:
    conn = _inmemory()
    _v3_schema(conn)
    conn.execute(
        "INSERT INTO sessions (session_id, prompt_text, model_chosen, cost, cache_stats, "
        "session_duration_seconds, timestamp) VALUES ('old', 'p', 'm', 1.0, '{}', 1.0, 't')"
    )
    conn.commit()
    assert get_current_version(conn) == 3

    run_migrations(conn)

    assert get_current_version(conn) == SCHEMA_VERSION
    cols = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(sessions)")}
    assert cols["external_session_id"] == "TEXT"
    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sessions_external_id'"
    )
    assert idx.fetchone() is not None
    # Old rows survive; the new column is NULL for them.
    row = conn.execute(
        "SELECT cost, external_session_id FROM sessions WHERE session_id = 'old'"
    ).fetchone()
    assert row[0] == 1.0
    assert row[1] is None


def test_v4_migration_is_idempotent() -> None:
    conn = _inmemory()
    run_migrations(conn)
    run_migrations(conn)
    versions = [
        row[0] for row in conn.execute("SELECT version FROM schema_version ORDER BY version")
    ]
    assert versions == [1, 2, 3, 4, 5]
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    assert "external_session_id" in cols


def test_store_session_persists_external_session_id(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "v4a.db"))
    try:
        store.store_session("s1", "p", None, "m", 1.0, {}, 1.0, external_session_id="ses-X")
        row = store.get_session("s1")
        assert row is not None
        assert row["external_session_id"] == "ses-X"
    finally:
        store.close()


def test_get_session_by_external_id_returns_latest(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "v4b.db"))
    try:
        store.store_session(
            "s-old",
            "p",
            None,
            "model-a",
            1.0,
            {},
            1.0,
            timestamp="2026-01-01T00:00:00+00:00",
            external_session_id="ses-A",
        )
        store.store_session(
            "s-new",
            "p",
            None,
            "model-b",
            1.0,
            {},
            1.0,
            timestamp="2026-01-02T00:00:00+00:00",
            external_session_id="ses-A",
        )
        row = store.get_session_by_external_id("ses-A")
        assert row is not None
        assert row["session_id"] == "s-new"
        assert row["model_chosen"] == "model-b"
        assert row["external_session_id"] == "ses-A"
        assert store.get_session_by_external_id("ses-missing") is None
    finally:
        store.close()
