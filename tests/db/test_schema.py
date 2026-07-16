from __future__ import annotations

import sqlite3

from shunt.db.schema import SCHEMA_VERSION, get_current_version, run_migrations


def _inmemory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def test_schema_version_is_one() -> None:
    assert SCHEMA_VERSION == 1


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
    cursor = conn.execute("SELECT COUNT(*) as cnt FROM schema_version")
    assert cursor.fetchone()[0] == 1
