"""Schema-v6 migration: the prompt-prefix conversation key (resume without a session id).

A v5 DB gains the nullable ``prefix_digest`` column and its index without touching old rows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from shunt.db.schema import (
    CREATE_EXTERNAL_ID_INDEX,
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


def _inmemory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _v5_schema(conn: sqlite3.Connection) -> None:
    """Build a v5-shaped DB: v1 tables + v2 columns/events + router_state + the v4 column."""
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
    conn.execute(MIGRATE_V4_EXTERNAL_SESSION_ID)
    conn.execute(CREATE_EXTERNAL_ID_INDEX)
    for v in (1, 2, 3, 4, 5):
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))", (v,)
        )
    conn.commit()


def test_v5_db_migrates_adds_column_and_index() -> None:
    conn = _inmemory()
    _v5_schema(conn)
    conn.execute(
        "INSERT INTO sessions (session_id, prompt_text, model_chosen, cost, cache_stats, "
        "session_duration_seconds, timestamp) VALUES ('old', 'p', 'm', 1.0, '{}', 1.0, 't')"
    )
    conn.commit()
    assert get_current_version(conn) == 5

    run_migrations(conn)

    assert get_current_version(conn) == SCHEMA_VERSION
    cols = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(sessions)")}
    assert cols["prefix_digest"] == "TEXT"
    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sessions_prefix_digest'"
    )
    assert idx.fetchone() is not None
    # Old rows survive; the new column is NULL for them, so a pre-v6 row is never a
    # prefix-resume candidate.
    row = conn.execute(
        "SELECT cost, prefix_digest FROM sessions WHERE session_id = 'old'"
    ).fetchone()
    assert row[0] == 1.0
    assert row[1] is None


def test_v6_migration_is_idempotent() -> None:
    conn = _inmemory()
    run_migrations(conn)
    run_migrations(conn)
    versions = [
        row[0] for row in conn.execute("SELECT version FROM schema_version ORDER BY version")
    ]
    assert versions == [1, 2, 3, 4, 5, 6]
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    assert "prefix_digest" in cols


def test_store_session_persists_prefix_digest(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "v6a.db"))
    try:
        store.store_session("s1", "p", None, "m", 1.0, {}, 1.0, prefix_digest="d" * 64)
        row = store.get_session("s1")
        assert row is not None
        assert row["prefix_digest"] == "d" * 64
    finally:
        store.close()


def test_get_session_by_prefix_digest_returns_latest(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "v6b.db"))
    try:
        for session_id, ts in (("s-old", "2026-01-01T00:00:00+00:00"), ("s-new", "2026-01-02T00")):
            store.store_session(
                session_id,
                "p",
                None,
                "model-a",
                1.0,
                {},
                1.0,
                timestamp=ts,
                prefix_digest="d" * 64,
            )
        row = store.get_session_by_prefix_digest("d" * 64)
        assert row is not None
        assert row["session_id"] == "s-new"
        assert store.get_session_by_prefix_digest("e" * 64) is None
    finally:
        store.close()


def test_an_ambiguous_digest_resolves_to_nothing(tmp_path: Path) -> None:
    """Two conversations sharing one digest: refuse, rather than label the wrong one."""
    store = OutcomeStore(db_path=str(tmp_path / "v6c.db"))
    try:
        store.store_session("s-a", "p", None, "model-a", 1.0, {}, 1.0, prefix_digest="d" * 64)
        store.store_session("s-b", "p", None, "model-b", 1.0, {}, 1.0, prefix_digest="d" * 64)
        assert store.get_session_by_prefix_digest("d" * 64) is None
    finally:
        store.close()


def test_the_external_id_term_of_the_guard_is_inert_on_production_rows(tmp_path: Path) -> None:
    """Only the model term of the ambiguity guard can ever refuse on data Shunt writes."""
    # `server._resolve_session` computes a prefix digest ONLY when no external session id was
    # declared, so the two columns are mutually exclusive on every row this codebase writes and
    # `COUNT(DISTINCT external_session_id)` is 0 for any real digest lineage. The earlier version
    # of this test stored rows carrying BOTH columns to make that term fire — a state production
    # cannot reach, so it proved nothing about the shipped guard. Reviving that term needs a
    # signal the schema does not carry today; that is an owner decision, not a test's to fake.
    store = OutcomeStore(db_path=str(tmp_path / "v6d.db"))
    try:
        for session_id, ts in (("s-a", "2026-01-01T00:00:00+00:00"), ("s-b", "2026-01-02T00")):
            store.store_session(
                session_id, "p", None, "model-a", 1.0, {}, 1.0, timestamp=ts, prefix_digest="d" * 64
            )
        rows = store._conn.execute(
            "SELECT COUNT(DISTINCT external_session_id) FROM sessions WHERE prefix_digest = ?",
            ("d" * 64,),
        ).fetchone()
        assert rows[0] == 0  # the term the guard reads is structurally always 0
        resolved = store.get_session_by_prefix_digest("d" * 64)
        assert resolved is not None
        assert resolved["session_id"] == "s-b"  # the lineage still resolves, on the model term
    finally:
        store.close()
