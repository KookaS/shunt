from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 6

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


# ── schema v4: per-conversation external session identity ──
# Tools like opencode send a per-conversation id (`X-Session-Id`); the router keys
# sessions on it when present and persists it here so a resumed or forked conversation
# can reuse its locked model. Nullable: header-less (ip, user_agent) traffic keeps NULL.
MIGRATE_V4_EXTERNAL_SESSION_ID = "ALTER TABLE sessions ADD COLUMN external_session_id TEXT"
CREATE_EXTERNAL_ID_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_external_id ON sessions(external_session_id)
"""


# ── schema v5: backfill `outcome_events` from pre-v2 `outcomes` rows ──
# v2 introduced the append-only log with CREATE TABLE IF NOT EXISTS and no backfill, so a
# store carrying `outcomes` rows written before v2 has a materialized outcome — and a live
# kNN index member — that the "source of truth" log has never heard of. Rebuilding from the
# log would silently disagree with the view. This seeds exactly one synthetic event per such
# row, dated to the outcome's own `created_at` so the log's chronology stays truthful.
#
# Idempotent three ways over: it is gated on `current < 5`; the SELECT is restricted to
# sessions with no event at all; and `idempotency_key` is deterministic under the table's
# UNIQUE constraint, so a re-run inserts nothing. The join to `sessions` keeps the FK honest
# and drops outcome rows whose session is gone.
MIGRATE_V5_BACKFILL_OUTCOME_EVENTS = """
INSERT OR IGNORE INTO outcome_events
    (session_id, tier, source, outcome, confidence, model_fingerprint,
     idempotency_key, tombstoned, created_at)
SELECT o.session_id,
       CASE WHEN o.tier2_outcome IS NOT NULL THEN 2 ELSE 1 END,
       COALESCE(o.outcome_source, 'human'),
       COALESCE(o.tier2_outcome, o.tier1_outcome),
       CASE WHEN o.tier2_outcome IS NOT NULL
            THEN COALESCE(o.tier2_confidence, o.tier1_confidence)
            ELSE o.tier1_confidence END,
       NULL,
       o.session_id || '|' || COALESCE(o.outcome_source, 'human') || '|backfill-v5',
       0,
       o.created_at
FROM outcomes o
JOIN sessions s ON s.session_id = o.session_id
WHERE NOT EXISTS (SELECT 1 FROM outcome_events e WHERE e.session_id = o.session_id)
"""


# ── schema v6: prompt-prefix session identity (resume for clients that send no id) ──
# Claude Code and other header-less clients cannot be resumed by `external_session_id`,
# so the router derives a stable per-conversation digest from the request's opening
# prefix (first system block + first user block, volatile injections normalised) bound
# to the client identity and a digest of the resolved repo. Persisted here so a restarted
# or expired conversation can find its own prior row. Nullable: header-keyed and
# prefix-less traffic keeps NULL. The stored value is a one-way digest — no prompt text
# and no filesystem path is recoverable from it.
MIGRATE_V6_PREFIX_DIGEST = "ALTER TABLE sessions ADD COLUMN prefix_digest TEXT"
CREATE_PREFIX_DIGEST_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_prefix_digest ON sessions(prefix_digest)
"""


def get_current_version(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    if cursor.fetchone() is None:
        return 0
    cursor = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    return int(cursor.fetchone()[0])


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Whether *table* already carries *column* — the ALTER's second guard."""
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))


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

    if current < 4:
        # Additive, backward-compatible: a v1/v2/v3 DB on disk gains one nullable column
        # plus an index; pre-v4 rows keep working (external_session_id defaults to NULL).
        conn.execute(MIGRATE_V4_EXTERNAL_SESSION_ID)
        conn.execute(CREATE_EXTERNAL_ID_INDEX)
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (4, datetime('now'))")
        conn.commit()

    if current < 5:
        # Repair-only: touches no healthy row (every post-v2 outcome already has an event).
        conn.execute(MIGRATE_V5_BACKFILL_OUTCOME_EVENTS)
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (5, datetime('now'))")
        conn.commit()

    if current < 6:
        # Additive, backward-compatible: one nullable column plus its index; every pre-v6
        # row keeps NULL and is simply never a prefix-resume candidate.
        # The version stamp is the primary guard; the column check keeps the ALTER safe when a
        # stamp is rolled back (a repair re-runs the earlier steps against a newer table).
        if not _has_column(conn, "sessions", "prefix_digest"):
            conn.execute(MIGRATE_V6_PREFIX_DIGEST)
        conn.execute(CREATE_PREFIX_DIGEST_INDEX)
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (6, datetime('now'))")
        conn.commit()
