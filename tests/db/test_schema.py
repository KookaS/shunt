from __future__ import annotations

import sqlite3

import pytest

from shunt.db.schema import SCHEMA_VERSION, get_current_version, run_migrations


def _inmemory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def test_schema_version_is_three() -> None:
    assert SCHEMA_VERSION == 3


def test_run_migrations_creates_tables() -> None:
    conn = _inmemory()
    run_migrations(conn)

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cursor.fetchall()}
    assert "sessions" in tables
    assert "outcomes" in tables
    assert "schema_version" in tables


def test_sessions_table_columns() -> None:
    conn = _inmemory()
    run_migrations(conn)

    cursor = conn.execute("PRAGMA table_info(sessions)")
    cols = {row[1]: row[2] for row in cursor.fetchall()}
    assert cols["session_id"] == "TEXT"
    assert cols["prompt_text"] == "TEXT"
    assert cols["embedding_blob"] == "BLOB"
    assert cols["model_chosen"] == "TEXT"
    assert cols["cost"] == "REAL"
    assert cols["cache_stats"] == "TEXT"
    assert cols["session_duration_seconds"] == "REAL"
    assert cols["timestamp"] == "TEXT"
    assert cols["decision_provenance"] == "TEXT"


def test_outcomes_table_columns() -> None:
    conn = _inmemory()
    run_migrations(conn)

    cursor = conn.execute("PRAGMA table_info(outcomes)")
    cols = {row[1]: (row[2], row[5]) for row in cursor.fetchall()}
    assert cols["session_id"][0] == "TEXT"
    assert cols["tier1_outcome"][0] == "TEXT"
    assert cols["tier1_confidence"][0] == "REAL"
    assert cols["tier2_outcome"][0] == "TEXT"
    assert cols["tier2_confidence"][0] == "REAL"
    assert cols["aggregated_confidence"][0] == "REAL"
    assert cols["human_label"][0] == "TEXT"
    assert cols["human_label_timestamp"][0] == "TEXT"
    assert cols["created_at"][0] == "TEXT"


def test_time_decay_weight_default() -> None:
    conn = _inmemory()
    run_migrations(conn)

    cursor = conn.execute("PRAGMA table_info(outcomes)")
    for row in cursor.fetchall():
        if row[1] == "time_decay_weight":
            assert row[4] == "1.0" or row[4] == "1"
            break
    else:
        raise AssertionError("time_decay_weight column not found")


def test_timestamp_index_exists() -> None:
    conn = _inmemory()
    run_migrations(conn)

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sessions_timestamp'"
    )
    assert cursor.fetchone() is not None


def test_version_tracking() -> None:
    conn = _inmemory()
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)")
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, '2024-01-01')")
    conn.commit()
    assert get_current_version(conn) == 1


def test_get_current_version_returns_zero_when_empty() -> None:
    conn = _inmemory()
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)")
    conn.commit()
    assert get_current_version(conn) == 0


def test_migration_idempotent() -> None:
    conn = _inmemory()
    run_migrations(conn)
    run_migrations(conn)
    cursor = conn.execute("SELECT version FROM schema_version ORDER BY version")
    versions = [row[0] for row in cursor.fetchall()]
    assert versions == [1, 2, 3]


def test_v2_sessions_provenance_columns() -> None:
    conn = _inmemory()
    run_migrations(conn)
    cursor = conn.execute("PRAGMA table_info(sessions)")
    cols = {row[1]: (row[2], row[3], row[4]) for row in cursor.fetchall()}
    assert cols["cost_known"][0] == "INTEGER"
    assert cols["cost_known"][1] == 1  # notnull flag
    assert cols["cost_known"][2] == "1"  # default value
    assert cols["selection_propensity"][0] == "REAL"
    assert cols["model_fingerprint"][0] == "TEXT"


def test_v2_outcomes_source_column() -> None:
    conn = _inmemory()
    run_migrations(conn)
    cursor = conn.execute("PRAGMA table_info(outcomes)")
    cols = {row[1]: (row[2], row[3]) for row in cursor.fetchall()}
    assert cols["outcome_source"][0] == "TEXT"
    assert cols["outcome_source"][1] == 1  # NOT NULL


def test_v2_outcome_events_table_and_index() -> None:
    conn = _inmemory()
    run_migrations(conn)
    cursor = conn.execute("PRAGMA table_info(outcome_events)")
    cols = {row[1]: row[2] for row in cursor.fetchall()}
    for col in (
        "event_id",
        "session_id",
        "tier",
        "source",
        "outcome",
        "confidence",
        "model_fingerprint",
        "idempotency_key",
        "tombstoned",
        "created_at",
    ):
        assert col in cols, f"missing outcome_events column: {col}"

    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_outcome_events_session'"
    )
    assert idx.fetchone() is not None


def test_v2_idempotency_key_unique() -> None:
    conn = _inmemory()
    run_migrations(conn)
    conn.execute(
        "INSERT INTO sessions (session_id, prompt_text, model_chosen, cost, "
        "cache_stats, session_duration_seconds, timestamp) "
        "VALUES ('s1', 'p', 'm', 0.0, '{}', 1.0, 'now')"
    )
    conn.execute(
        "INSERT INTO outcome_events (session_id, tier, source, outcome, confidence, "
        "idempotency_key, created_at) VALUES ('s1', 2, 'auto_tier2', 'success', 1.0, 'k1', 'now')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO outcome_events (session_id, tier, source, outcome, confidence, "
            "idempotency_key, created_at) VALUES ('s1', 2, 'human', 'failure', 1.0, 'k1', 'now')"
        )


def test_v1_db_migrates_cleanly() -> None:
    """A real v1 database on disk must open, migrate to v2, and keep its old rows."""
    conn = _inmemory()
    # Build a v1-shaped DB: create ONLY the v1 tables and stamp version 1.
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
        "session_duration_seconds, timestamp) "
        "VALUES ('old-s', 'old prompt', 'm', 0.42, '{}', 1.0, 't')"
    )
    conn.execute(
        "INSERT INTO outcomes (session_id, tier1_outcome, tier1_confidence, "
        "aggregated_confidence, created_at) VALUES ('old-s', 'ok', 0.9, 0.9, 't')"
    )
    conn.commit()

    assert get_current_version(conn) == 1
    run_migrations(conn)
    assert get_current_version(conn) == 3

    # Old rows survive and the new cost_known column defaults to 1 (known).
    row = conn.execute(
        "SELECT cost, cost_known FROM sessions WHERE session_id = 'old-s'"
    ).fetchone()
    assert row[0] == 0.42
    assert row[1] == 1
    o = conn.execute("SELECT outcome_source FROM outcomes WHERE session_id = 'old-s'").fetchone()
    assert o[0] == "human"  # default backfill
