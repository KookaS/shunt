from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 3

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
    -- Persisted but NEVER read: no routing path applies it, so a two-year-old
    -- outcome weighs exactly as much as one from five minutes ago. Reserved for
    -- non-stationarity handling (model versions churn); until something reads it,
    -- do not treat a stored value as having any effect on routing.
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

# ── schema v2: append-only outcome log + cost/propensity/fingerprint provenance ──
# The `outcome_events` table is the source of truth; `outcomes` becomes a materialized
# current-view and the HNSW index a rebuildable projection (see architecture §Q4/§Q5).
CREATE_OUTCOME_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS outcome_events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT    NOT NULL,
    tier              INTEGER NOT NULL,
    source            TEXT    NOT NULL,
    outcome           TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    model_fingerprint TEXT,
    idempotency_key   TEXT    NOT NULL,
    tombstoned        INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL,
    UNIQUE(idempotency_key),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
"""

CREATE_OUTCOME_EVENTS_SESSION_INDEX = """
CREATE INDEX IF NOT EXISTS idx_outcome_events_session ON outcome_events(session_id)
"""

# 0 ⇒ cost is UNKNOWN (unreported by the provider), distinct from a real 0.0.
MIGRATE_V2_COST_KNOWN = "ALTER TABLE sessions ADD COLUMN cost_known INTEGER NOT NULL DEFAULT 1"
MIGRATE_V2_SELECTION_PROPENSITY = "ALTER TABLE sessions ADD COLUMN selection_propensity REAL"
MIGRATE_V2_MODEL_FINGERPRINT = "ALTER TABLE sessions ADD COLUMN model_fingerprint TEXT"
MIGRATE_V2_OUTCOME_SOURCE = (
    "ALTER TABLE outcomes ADD COLUMN outcome_source TEXT NOT NULL DEFAULT 'human'"
)


# ── schema v3: durable router state (exploration budget + conservative-gate slack) ──
# A small key-value table so the cost cap and downshift slack are not reset to zero on
# every restart. Lives in the outcome DB to share its path/lock/WAL — one JSON scalar
# per key, written under the same transactional discipline as outcomes.
CREATE_ROUTER_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS router_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
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

    if current < 2:
        # Additive, backward-compatible: a v1 DB on disk gains columns via defaults and
        # one new table. Pre-v2 rows keep working (cost_known defaults to 1 = known).
        conn.execute(MIGRATE_V2_COST_KNOWN)
        conn.execute(MIGRATE_V2_SELECTION_PROPENSITY)
        conn.execute(MIGRATE_V2_MODEL_FINGERPRINT)
        conn.execute(MIGRATE_V2_OUTCOME_SOURCE)
        conn.execute(CREATE_OUTCOME_EVENTS_TABLE)
        conn.execute(CREATE_OUTCOME_EVENTS_SESSION_INDEX)
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (2, datetime('now'))")
        conn.commit()

    if current < 3:
        # Additive: a v1/v2 DB on disk gains one new table; no existing row is touched.
        conn.execute(CREATE_ROUTER_STATE_TABLE)
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (3, datetime('now'))")
        conn.commit()
