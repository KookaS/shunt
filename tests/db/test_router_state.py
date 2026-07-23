from __future__ import annotations

import sqlite3
from pathlib import Path

from shunt.db.schema import SCHEMA_VERSION, get_current_version, run_migrations
from shunt.db.store import OutcomeStore


def _inmemory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def test_schema_version_is_three() -> None:
    assert SCHEMA_VERSION == 3


def test_run_migrations_creates_router_state_table() -> None:
    conn = _inmemory()
    run_migrations(conn)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "router_state" in tables
    assert get_current_version(conn) == 3


def test_run_migrations_idempotent() -> None:
    conn = _inmemory()
    run_migrations(conn)
    run_migrations(conn)  # second run must be a no-op, not an error
    assert get_current_version(conn) == 3
    versions = [
        row[0] for row in conn.execute("SELECT version FROM schema_version ORDER BY version")
    ]
    assert versions == [1, 2, 3]


def test_v2_db_migrates_to_v3_with_rows_intact() -> None:
    # Simulate a pre-existing v2 database: migrate only up to v2, seed a session row,
    # then a full migration must reach v3 and leave the old row untouched.
    conn = _inmemory()
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    from shunt.db.schema import (
        CREATE_OUTCOME_EVENTS_SESSION_INDEX,
        CREATE_OUTCOME_EVENTS_TABLE,
        CREATE_OUTCOMES_TABLE,
        CREATE_SESSIONS_TABLE,
        CREATE_TIMESTAMP_INDEX,
        MIGRATE_V2_COST_KNOWN,
        MIGRATE_V2_MODEL_FINGERPRINT,
        MIGRATE_V2_OUTCOME_SOURCE,
        MIGRATE_V2_SELECTION_PROPENSITY,
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
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, datetime('now'))")
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (2, datetime('now'))")
    conn.execute(
        "INSERT INTO sessions (session_id, prompt_text, model_chosen, cost, cache_stats, "
        "session_duration_seconds, timestamp) VALUES ('s1', 'p', 'm', 1.0, '{}', 0.0, 't')"
    )
    conn.commit()

    run_migrations(conn)

    assert get_current_version(conn) == 3
    row = conn.execute("SELECT prompt_text FROM sessions WHERE session_id = 's1'").fetchone()
    assert row["prompt_text"] == "p"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "router_state" in tables


def test_store_save_and_load_round_trip(tmp_path: Path) -> None:
    db = str(tmp_path / "outcomes.db")
    store = OutcomeStore(db_path=db)
    payload = {"budget": {"exploit_cost": 12.5, "decisions": 3}, "gate": {"slack": 2.0}}
    store.save_router_state(payload)
    store.close()

    reopened = OutcomeStore(db_path=db)  # a fresh process would reopen the same file
    assert reopened.load_router_state() == payload
    reopened.close()


def test_escalation_state_round_trips_under_own_key(tmp_path: Path) -> None:
    # B4: escalation state persists under its own KV key, independent of exploration state —
    # neither write clobbers the other, and a fresh process reloads it intact.
    db = str(tmp_path / "outcomes.db")
    store = OutcomeStore(db_path=db)
    exploration = {"budget": {"exploit_cost": 9.0, "decisions": 4}}
    escalation = {
        "failure_log": {
            "/repo": [
                {
                    "decision_index": 0,
                    "dedup_key": "t::a",
                    "exit_code": 1,
                    "success": False,
                    "confirmed": True,
                    "blocking": True,
                }
            ]
        },
        "decision_index": {"/repo": 2},
    }
    store.save_router_state(exploration)
    store.save_escalation_state(escalation)
    store.close()

    reopened = OutcomeStore(db_path=db)
    assert reopened.load_escalation_state() == escalation
    assert reopened.load_router_state() == exploration  # not clobbered
    reopened.close()


def test_missing_escalation_state_is_none(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "esc.db"))
    assert store.load_escalation_state() is None
    store.close()


def test_store_load_missing_returns_none(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
    assert store.load_router_state() is None
    store.close()


def test_store_load_corrupt_value_returns_none(tmp_path: Path) -> None:
    db = str(tmp_path / "outcomes.db")
    store = OutcomeStore(db_path=db)
    store.save_router_state({"budget": {"exploit_cost": 1.0}})
    # Corrupt the stored JSON directly; a reload must degrade to None, never raise.
    store._conn.execute(  # type: ignore[attr-defined]
        "UPDATE router_state SET value = '{not json' WHERE key = 'exploration_state'"
    )
    store._conn.commit()  # type: ignore[attr-defined]
    assert store.load_router_state() is None
    store.close()


def test_fingerprint_round_trips_without_clobbering_exploration_state(tmp_path: Path) -> None:
    # Kill-gate: the fingerprint uses its OWN KV key, so writing it must never reset the
    # exploration budget/gate — a clobber would silently zero the exploration budget.
    db = str(tmp_path / "outcomes.db")
    store = OutcomeStore(db_path=db)
    exploration = {"budget": {"exploit_cost": 9.0, "decisions": 4}, "gate": {"slack": 3.0}}
    fingerprint = {"repo": "jinaai/x", "dim": 768, "max_chars": 4000, "revision": None}
    store.save_router_state(exploration)
    store.save_embedding_fingerprint(fingerprint)
    # Both survive independently, in either write order.
    assert store.load_router_state() == exploration
    assert store.load_embedding_fingerprint() == fingerprint
    # Overwriting the fingerprint leaves the exploration state untouched.
    store.save_embedding_fingerprint({"repo": "other", "dim": 512, "max_chars": 8000})
    assert store.load_router_state() == exploration
    rows = store._conn.execute("SELECT COUNT(*) AS c FROM router_state").fetchone()["c"]  # type: ignore[attr-defined]
    assert rows == 2  # two distinct keys, not one clobbered row
    store.close()


def test_missing_fingerprint_is_none(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "fp.db"))
    assert store.load_embedding_fingerprint() is None
    store.close()


def test_store_save_upserts(tmp_path: Path) -> None:
    db = str(tmp_path / "outcomes.db")
    store = OutcomeStore(db_path=db)
    store.save_router_state({"gate": {"slack": 1.0}})
    store.save_router_state({"gate": {"slack": 5.0}})  # overwrite, single row
    assert store.load_router_state() == {"gate": {"slack": 5.0}}
    count = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS c FROM router_state"
    ).fetchone()["c"]
    assert count == 1
    store.close()
