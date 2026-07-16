from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    embedding_blob BLOB,
    model_chosen TEXT NOT NULL,
    cost REAL NOT NULL,
    cache_stats TEXT NOT NULL,
    session_duration_seconds REAL NOT NULL,
    timestamp TEXT NOT NULL,
    decision_provenance TEXT
)
"""

CREATE_OUTCOMES_TABLE = """
CREATE TABLE IF NOT EXISTS outcomes (
    session_id TEXT PRIMARY KEY,
    tier1_outcome TEXT NOT NULL,
    tier1_confidence REAL NOT NULL,
    tier2_outcome TEXT,
    tier2_confidence REAL,
    aggregated_confidence REAL NOT NULL,
    human_label TEXT,
    human_label_timestamp TEXT,
    time_decay_weight REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
"""

CREATE_TIMESTAMP_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp)
"""

CREATE_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""


def get_current_version(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    if cursor.fetchone() is None:
        return 0
    cursor = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    return int(cursor.fetchone()[0])


def run_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_VERSION_TABLE)
    conn.commit()

    current = get_current_version(conn)

    if current < 1:
        conn.execute(CREATE_SESSIONS_TABLE)
        conn.execute(CREATE_OUTCOMES_TABLE)
        conn.execute(CREATE_TIMESTAMP_INDEX)
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, datetime('now'))")
        conn.commit()
