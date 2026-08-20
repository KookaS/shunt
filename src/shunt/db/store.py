from __future__ import annotations

import datetime as _datetime
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from shunt.proxy.redaction import redact_secrets

from .index import HNSWIndex
from .loop_health import (
    LiveCostAggregates,
    LoopHealthSnapshot,
    StratumCensus,
    StratumStages,
)
from .schema import run_migrations

# Human labels win over automatic Tier-2, which win over the quarantined wire prior.
# Drives which event's outcome/source becomes the materialized current view (§Q4).
# benchmark_seed rows are synthetic prior-only data: zero priority means a live/human
# outcome for the same session always overrides the seeded one.
_SOURCE_PRIORITY: Final[dict[str, int]] = {
    "wire_tier1": 0,
    "benchmark_seed": 0,
    "auto_tier2": 1,
    "human": 2,
}

# The origin discriminator for every live-only aggregate read below. Seeded rows are the
# replayed benchmark corpus and carry a `bench:`-prefixed session_id; the format is
# `bench:<sha256(task_id)[:12]>:<model>` and its OWNER is `benchmark/routing/seed_live.py`
# (the `cell_id` builder). Change it there and this clause must change with it — grep
# `bench:` from either end to find the other.
_LIVE_CLAUSE: Final[str] = "session_id NOT LIKE 'bench:%'"

# A session whose most recent outcome_event is a tombstone is NOT an index member. Written
# once because two readers need it — `_labeled_embeddings_locked` builds the index from it and
# `stratum_census` counts it — and a drifted second copy would report a stage count the index
# itself disagrees with.
#
# Vacuously TRUE for a session with no events at all, and that is INTENDED: this is a purely
# SUBTRACTIVE erasure filter ("has never been erased"), and both readers establish membership
# positively first (a JOIN to `outcomes` / `o.session_id IS NOT NULL`) before applying it. A
# session that was never labeled cannot have been erased. The census's `labeled` stage instead
# asks the POSITIVE question (does an untombstoned event exist), which is why the two disagreed
# on pre-v2 rows the event log never received — repaired by the v5 backfill, not by weakening
# this predicate.
_NOT_TOMBSTONED: Final[str] = (
    "NOT EXISTS (SELECT 1 FROM outcome_events e WHERE e.session_id = s.session_id "
    "AND e.tombstoned = 1 AND e.event_id = (SELECT MAX(event_id) FROM outcome_events e2 "
    "WHERE e2.session_id = s.session_id))"
)

# Single row holding the serialized exploration budget + conservative-gate slack.
_ROUTER_STATE_KEY: Final[str] = "exploration_state"

# The corpus's embedding-space fingerprint (repo, dim, max_chars, revision?). A second
# consumer of the generic router_state KV, under its own key — never clobbers the above.
_EMBEDDING_FINGERPRINT_KEY: Final[str] = "embedding_fingerprint"

# The auto-escalation failure log + per-task decision counters — a third KV consumer under its
# own key, so a restart does not wipe accrued same-failure counts.
_ESCALATION_STATE_KEY: Final[str] = "escalation_state"

# The last-applied benchmark seed marker (fingerprint + results digest) — a fourth KV consumer
# under its own key, so the importer can skip re-importing an unchanged results.csv.
_SEED_STATE_KEY: Final[str] = "seed_state"


@dataclass(frozen=True)
class SessionProvenance:
    """Decision-time provenance persisted on the session row.

    Grouped into one value object so ``store_session`` stays under the arg limit;
    ``cost_known=False`` marks cost UNKNOWN (unreported), distinct from a real 0.0.
    """

    cost_known: bool = True
    selection_propensity: float | None = None
    model_fingerprint: str | None = None


@dataclass(frozen=True)
class OutcomeEvent:
    """One immutable observation appended to the append-only ``outcome_events`` log.

    ``outcome`` is one of ``success | weak_success | failure`` on the capture path —
    never ``unknown`` (the caller drops unknowns; the store never invents one).
    """

    session_id: str
    tier: int
    source: str
    outcome: str
    confidence: float
    run_signature: str
    model_fingerprint: str | None = None
    tombstoned: bool = False
    # Wall-clock when None (the capture path). A replayed corpus passes a fixed stamp so
    # the rows it writes are reproducible build-to-build.
    created_at: str | None = None


def _exploration_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten one session row into an exploration record + its verified outcome, or None."""
    try:
        provenance = json.loads(row["decision_provenance"] or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    record = provenance.get("escalation_exploration")
    if not isinstance(record, dict):  # the LIKE filter can match a prompt, not just the key
        return None
    return {
        **record,
        "session_id": row["session_id"],
        "outcome": row["tier2_outcome"] or row["tier1_outcome"],
    }


def _live_clause(alias: str = "") -> str:
    """``_LIVE_CLAUSE`` qualified by a table alias, for reads that JOIN on ``session_id``."""
    if not alias:
        return _LIVE_CLAUSE
    return _LIVE_CLAUSE.replace("session_id", f"{alias}.session_id", 1)


def _window_start(window_days: int | None) -> str | None:
    """UTC ISO cutoff *window_days* before now, or None for the whole store."""
    # Compared against `sessions.timestamp` as a STRING: every writer stamps
    # `datetime.now(utc).isoformat()`, so the rows are fixed-width UTC ISO-8601 and
    # lexicographic order is chronological order.
    if window_days is None:
        return None
    now = _datetime.datetime.now(_datetime.timezone.utc)  # noqa: UP017
    return (now - _datetime.timedelta(days=window_days)).isoformat()


def _routing_ope_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten one live routing decision into the record shape ``analysis.ope`` reads."""
    # The provenance carries `candidate_model_scores` (the routing features) at top level, so
    # it is spread rather than nested; Tier-2 wins over Tier-1 and an unlabelled decision keeps
    # outcome=None so the estimator EXCLUDES it rather than inventing a reward.
    try:
        provenance = json.loads(row["decision_provenance"] or "{}")
    except (json.JSONDecodeError, TypeError):
        provenance = {}
    if not isinstance(provenance, dict):
        provenance = {}
    return {
        **provenance,
        "session_id": row["session_id"],
        "model_chosen": row["model_chosen"],
        "selection_propensity": row["selection_propensity"],
        "timestamp": row["timestamp"],
        "outcome": row["tier2_outcome"] or row["tier1_outcome"],
    }


def _stages(row: dict[str, Any], stratum: str) -> StratumStages:
    """One GROUP BY row of the stratum census as its five stage counts."""
    return StratumStages(
        stratum=stratum,
        stored=int(row["stored"]),
        embedded=int(row["embedded"] or 0),
        labeled=int(row["labeled"] or 0),
        tier2=int(row["tier2"] or 0),
        indexed=int(row["in_index"] or 0),
    )


def _empty_stages(stratum: str) -> StratumStages:
    return StratumStages(stratum=stratum, stored=0, embedded=0, labeled=0, tier2=0, indexed=0)


def _latest_tier(events: list[sqlite3.Row], tier: int) -> sqlite3.Row | None:
    """The most recent event of a given tier (events must arrive ordered by event_id)."""
    match = [e for e in events if e["tier"] == tier]
    return match[-1] if match else None


def _best_tier2(events: list[sqlite3.Row]) -> sqlite3.Row | None:
    """The winning Tier-2 event: human beats auto_tier2; ties break to the latest."""
    tier2 = [e for e in events if e["tier"] == 2]
    if not tier2:
        return None
    return max(tier2, key=lambda e: (_SOURCE_PRIORITY.get(e["source"], 0), e["event_id"]))


def _project_events(session_id: str, events: list[sqlite3.Row]) -> dict[str, Any] | None:
    """Project the append-only log into one materialized `outcomes` row, or None if erased."""
    non_tomb = [e for e in events if not e["tombstoned"]]
    if events[-1]["tombstoned"] or not non_tomb:
        return None
    tier1 = _latest_tier(non_tomb, 1)
    tier2 = _best_tier2(non_tomb)
    winner = tier2 or tier1
    if winner is None:
        return None
    base = tier1 or winner
    return {
        "session_id": session_id,
        "tier1_outcome": base["outcome"],
        "tier1_confidence": base["confidence"],
        "tier2_outcome": tier2["outcome"] if tier2 else None,
        "tier2_confidence": tier2["confidence"] if tier2 else None,
        "aggregated_confidence": winner["confidence"],
        "outcome_source": winner["source"],
    }


DEFAULT_DB_DIR = Path.home() / ".local" / "share" / "shunt"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "outcomes.db"


def default_db_path() -> str:
    """The outcome DB path the store opens by default — public so `doctor` can stat it."""
    env_path = os.environ.get("SHUNT_DATA_DIR")
    if env_path:
        return os.path.join(env_path, "outcomes.db")
    return str(DEFAULT_DB_PATH)


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _now_iso() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()  # noqa: UP017


class ReindexEmbedder(Protocol):
    """The embedder surface ``reindex_corpus`` needs: a fingerprint and per-text embedding."""

    def fingerprint(self) -> dict[str, Any]: ...

    def embed(self, text: str) -> np.ndarray: ...


class OutcomeStoreUnavailableError(RuntimeError):
    """The outcome database could not be opened — actionable, unlike the raw driver error."""


class OutcomeStore:
    def __init__(
        self,
        db_path: str | None = None,
        index_path: str | None = None,
        hnsw_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._db_path = db_path or default_db_path()
        self._lock = threading.Lock()

        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            run_migrations(self._conn)
        except (sqlite3.Error, OSError) as exc:
            # A corrupt or unwritable store used to abort startup with a bare driver
            # error naming neither the file nor a remedy, which reads like a crash.
            raise OutcomeStoreUnavailableError(
                f"could not open the outcome database at {self._db_path}: {exc}. "
                "Check the path is writable, or move/delete the file to start fresh "
                "(it holds learned routing history only). Relocate with SHUNT_DATA_DIR."
            ) from exc

        if hnsw_kwargs is None:
            hnsw_kwargs = {}
        self._index = HNSWIndex(**hnsw_kwargs)

        # `.hnsw2`: a `.hnsw` file written by an earlier build holds EVERY session, not
        # just the labeled ones. Loading it would silently restore the dilution this
        # index shape exists to prevent, so the old name is left to rot and we rebuild.
        self._index_path = index_path or (self._db_path + ".hnsw2")

        if os.path.exists(self._index_path):
            try:
                self._index.load(self._index_path)
            except Exception:
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        # Always rebuild — including to an empty index when every labeled session was
        # tombstoned. `build([])` clears the index (HNSW has no in-place delete), which
        # is exactly how a tombstone drops a session from the neighbourhood.
        #
        # Hold the store lock across BOTH the labeled-set snapshot AND the build. A
        # concurrent store_outcome commits under this same lock and only then adds to the
        # index lock-free (store.py:293/:362); holding it here forces that commit to
        # serialize AFTER the build, so its later index add lands on the rebuilt index
        # instead of being overwritten by a stale snapshot. Without this the row stays
        # durable in the DB but silently drops out of the kNN index until the next rebuild.
        with self._lock:
            embeddings = self._labeled_embeddings_locked()
            np_embeddings = [(sid, _blob_to_embedding(blob)) for sid, blob in embeddings]
            self._index.build(np_embeddings)

    def rebuild_index(self) -> None:
        """Public truth-up: rebuild the HNSW index from the non-tombstoned labeled log."""
        self._rebuild_index()

    def store_session(  # noqa: PLR0913 (config-heavy session-row writer, one arg per column)
        self,
        session_id: str,
        prompt_text: str,
        embedding: np.ndarray | None,
        model_chosen: str,
        cost: float,
        cache_stats: dict[str, Any],
        duration: float,
        timestamp: str | None = None,
        decision_provenance: dict[str, Any] | None = None,
        provenance: SessionProvenance | None = None,
        external_session_id: str | None = None,
    ) -> None:
        prov = provenance or SessionProvenance()
        with self._lock:
            embedding_blob = _embedding_to_blob(embedding) if embedding is not None else None
            ts = timestamp or _now_iso()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (session_id, prompt_text, embedding_blob, model_chosen, cost,
                     cache_stats, session_duration_seconds, timestamp, decision_provenance,
                     cost_known, selection_propensity, model_fingerprint, external_session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    # Redacted at rest: this is the only free-text sink that reached the DB
                    # raw, and a user prompt routinely pastes a key or token. The embedding
                    # blob is computed upstream from the live text, so redaction changes what
                    # a REINDEX re-embeds (a secret is noise to the router), never the
                    # decision this session already made.
                    redact_secrets(prompt_text),
                    embedding_blob,
                    model_chosen,
                    cost,
                    json.dumps(cache_stats),
                    duration,
                    ts,
                    json.dumps(decision_provenance) if decision_provenance is not None else None,
                    1 if prov.cost_known else 0,
                    prov.selection_propensity,
                    prov.model_fingerprint,
                    external_session_id,
                ),
            )
            self._conn.commit()

        # Deliberately NOT indexed here. Only a session carrying an outcome can be a
        # neighbour, and outcomes arrive later — indexing every session let ordinary
        # traffic crowd the labeled ones out of the k nearest, so selection fell through
        # to the cheapest model. The embedding is durable in `sessions`, so
        # `store_outcome` indexes it the moment the session becomes usable.

    def store_outcome(  # noqa: PLR0913 (config-heavy outcome-row writer, one arg per column)
        self,
        session_id: str,
        tier1_outcome: str,
        tier1_confidence: float,
        tier2_outcome: str | None = None,
        tier2_confidence: float | None = None,
        aggregated_confidence: float = 0.0,
        human_label: str | None = None,
        time_decay_weight: float = 1.0,
    ) -> None:
        # Compatibility wrapper. Routes through the append-only log (source='human') so the
        # log stays the complete source of truth for tombstone/rebuild, then writes the rich
        # `outcomes` row verbatim — preserving human_label / time_decay_weight / free-form
        # outcome strings that the event-projection view does not carry.
        with self._lock:
            self._insert_event_locked(
                OutcomeEvent(
                    session_id=session_id,
                    tier=2 if tier2_outcome is not None else 1,
                    source="human",
                    outcome=tier2_outcome if tier2_outcome is not None else tier1_outcome,
                    confidence=(
                        tier2_confidence if tier2_confidence is not None else tier1_confidence
                    ),
                    run_signature=uuid.uuid4().hex,
                )
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO outcomes
                    (session_id, tier1_outcome, tier1_confidence, tier2_outcome,
                     tier2_confidence, aggregated_confidence, human_label,
                     time_decay_weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tier1_outcome,
                    tier1_confidence,
                    tier2_outcome,
                    tier2_confidence,
                    aggregated_confidence,
                    human_label,
                    time_decay_weight,
                    _now_iso(),
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT embedding_blob FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

        # The session becomes a usable neighbour exactly now, so this is when it joins
        # the index. Inside no lock: HNSWIndex guards its own slot allocation.
        if row is not None and row["embedding_blob"] is not None:
            self._index.add(session_id, _blob_to_embedding(row["embedding_blob"]))

    def _insert_event_locked(self, event: OutcomeEvent) -> bool:
        """Append one immutable event; idempotent via the UNIQUE key. Caller holds the lock."""
        idempotency_key = f"{event.session_id}|{event.source}|{event.run_signature}"
        cursor = self._conn.execute(
            """
            INSERT INTO outcome_events
                (session_id, tier, source, outcome, confidence, model_fingerprint,
                 idempotency_key, tombstoned, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(idempotency_key) DO NOTHING
            """,
            (
                event.session_id,
                event.tier,
                event.source,
                event.outcome,
                event.confidence,
                event.model_fingerprint,
                idempotency_key,
                1 if event.tombstoned else 0,
                event.created_at or _now_iso(),
            ),
        )
        return cursor.rowcount > 0

    def append_outcome_event(self, event: OutcomeEvent) -> bool:
        """Append one capture-path event to the source-of-truth log, then materialize.

        Idempotent: a duplicate ``run_signature`` is a no-op — no double-count.
        Returns True if a new row was inserted, False on a deduplicated conflict.
        """
        with self._lock:
            inserted = self._insert_event_locked(event)
            self._conn.commit()
        self._materialize_outcome(event.session_id, created_at=event.created_at)
        return inserted

    def _materialize_outcome(self, session_id: str, created_at: str | None = None) -> None:
        """Rebuild the `outcomes` current-view row for a session from its event log."""
        with self._lock:
            events = self._conn.execute(
                "SELECT * FROM outcome_events WHERE session_id = ? ORDER BY event_id",
                (session_id,),
            ).fetchall()
            if not events:
                return
            row = _project_events(session_id, events)
            if row is None:
                # Erased (latest event tombstoned): drop the view row. The index still
                # holds the vector — a rebuild_index() call is the truth-up (HNSW can't
                # delete in place).
                self._conn.execute("DELETE FROM outcomes WHERE session_id = ?", (session_id,))
                self._conn.commit()
                return
            if row["tier2_outcome"] is None:
                # A Tier-1-only projection (a weak wire prior with no verified Tier-2) stays
                # in the append-only log for observability/corroboration but is kept OUT of
                # the materialized view AND the trusted kNN index until a Tier-2 corroborates
                # it — it must never become a routing neighbour. A later Tier-2 event re-runs
                # this and materializes both tiers.
                return
            self._upsert_materialized_outcome(row, created_at)
            self._conn.commit()
            emb = self._conn.execute(
                "SELECT embedding_blob FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if emb is not None and emb["embedding_blob"] is not None:
            self._index.add(session_id, _blob_to_embedding(emb["embedding_blob"]))

    def _upsert_materialized_outcome(self, row: dict[str, Any], created_at: str | None) -> None:
        """UPSERT the projected row, preserving human_label/time_decay_weight on conflict."""
        self._conn.execute(
            """
            INSERT INTO outcomes
                (session_id, tier1_outcome, tier1_confidence, tier2_outcome,
                 tier2_confidence, aggregated_confidence, outcome_source, created_at)
            VALUES (:session_id, :tier1_outcome, :tier1_confidence, :tier2_outcome,
                    :tier2_confidence, :aggregated_confidence, :outcome_source, :created_at)
            ON CONFLICT(session_id) DO UPDATE SET
                tier1_outcome = excluded.tier1_outcome,
                tier1_confidence = excluded.tier1_confidence,
                tier2_outcome = excluded.tier2_outcome,
                tier2_confidence = excluded.tier2_confidence,
                aggregated_confidence = excluded.aggregated_confidence,
                outcome_source = excluded.outcome_source
            """,
            {**row, "created_at": created_at or _now_iso()},
        )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def get_session_by_external_id(self, external_session_id: str) -> dict[str, Any] | None:
        """The most recent session row for *external_session_id*, or None if never seen."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE external_session_id = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (external_session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def get_outcome(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM outcomes WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def get_sessions(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    # ORDER BY rowid is load-bearing on both embedding readers below, not tidiness: HNSWIndex
    # assigns index slots by list position, so the row order these return IS the slot layout the
    # persisted .hnsw2 inherits, and published figures read neighbour ids out of it. SQLite
    # guarantees no order without ORDER BY -- a new index or a version bump could silently
    # reorder the scan and change a committed figure with no other symptom. Do not remove.
    def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT session_id, embedding_blob FROM sessions "
                "WHERE embedding_blob IS NOT NULL ORDER BY rowid"
            )
            return [(row["session_id"], row["embedding_blob"]) for row in cursor.fetchall()]

    def get_labeled_embeddings(self) -> list[tuple[str, bytes]]:
        """Embeddings of non-tombstoned labeled sessions — i.e. exactly the index members.

        A session whose most recent event is a tombstone is excluded so a rebuilt index
        reflects the truthful log (materialization also drops its `outcomes` row).
        """
        with self._lock:
            return self._labeled_embeddings_locked()

    def _labeled_embeddings_locked(self) -> list[tuple[str, bytes]]:
        """Body of get_labeled_embeddings; caller already holds the lock (rebuild reuses it)."""
        cursor = self._conn.execute(
            "SELECT s.session_id, s.embedding_blob FROM sessions s "
            "JOIN outcomes o ON o.session_id = s.session_id "
            "WHERE s.embedding_blob IS NOT NULL AND " + _NOT_TOMBSTONED + " ORDER BY s.rowid"
        )
        return [(row["session_id"], row["embedding_blob"]) for row in cursor.fetchall()]

    # Both counts join `sessions` and require an embedding. These decide when cold start
    # ends — i.e. when the kNN index is trusted to answer — so an outcome on a session
    # that was never embedded (any fixed strategy returns before embedding) can never be
    # anyone's neighbour. Counting it ended cold start against an empty index.
    def count_outcomes(self) -> int:
        """Count embedded sessions carrying any labeled outcome."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) AS c FROM outcomes o "
                "JOIN sessions s ON s.session_id = o.session_id "
                "WHERE s.embedding_blob IS NOT NULL"
            )
            return int(cursor.fetchone()["c"])

    def count_verified_outcomes(self) -> int:
        """Count embedded sessions with a Tier-2 (verified) outcome."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) AS c FROM outcomes o "
                "JOIN sessions s ON s.session_id = o.session_id "
                "WHERE o.tier2_outcome IS NOT NULL AND s.embedding_blob IS NOT NULL"
            )
            return int(cursor.fetchone()["c"])

    def labeled_outcome_rows(self, *, tier2_only: bool = False) -> list[dict[str, Any]]:
        """Outcome rows (+ session ``model_chosen``) for embedded, indexable sessions."""
        # The read-back seam weights these for the effective-sample-size gate and per-model
        # priors without a second query. tier2_only restricts to verified (Tier-2) sessions —
        # the same population count_verified_outcomes gates on.
        clause = " AND o.tier2_outcome IS NOT NULL" if tier2_only else ""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT o.*, s.model_chosen, s.decision_provenance FROM outcomes o "
                "JOIN sessions s ON s.session_id = o.session_id "
                "WHERE s.embedding_blob IS NOT NULL" + clause
            )
            return [dict(row) for row in cursor.fetchall()]

    def escalation_exploration_rows(self) -> list[dict[str, Any]]:
        """Flagged-checkpoint escalation records joined to their verified outcome (OPE input)."""
        # The propensity rides the session's decision_provenance, exactly as the routing
        # selection propensity does; the verified outcome is joined here so the estimator gets
        # (state, action, propensity, reward) in one row. Tier-2 (verified) wins over Tier-1;
        # an unlabelled decision keeps outcome=None so the estimator can EXCLUDE it rather than
        # invent a reward.
        with self._lock:
            cursor = self._conn.execute(
                "SELECT s.session_id, s.decision_provenance, s.timestamp, "
                "o.tier1_outcome, o.tier2_outcome FROM sessions s "
                "LEFT JOIN outcomes o ON o.session_id = s.session_id "
                "WHERE s.decision_provenance LIKE '%escalation_exploration%' "
                "ORDER BY s.timestamp"
            )
            rows = [dict(row) for row in cursor.fetchall()]
        return [row for row in (_exploration_row(r) for r in rows) if row is not None]

    def routing_ope_rows(self) -> list[dict[str, Any]]:
        """Live routing decisions joined to their verified outcome (routing-arm OPE input)."""
        # `selection_propensity` is written only by the live router — a replayed seed row never
        # sets it — so the NOT NULL filter already excludes the seeded stratum. `_LIVE_CLAUSE`
        # is stated anyway so the live-only guarantee survives a seeder that starts writing it.
        with self._lock:
            cursor = self._conn.execute(
                "SELECT s.session_id, s.model_chosen, s.selection_propensity, "
                "s.decision_provenance, s.timestamp, o.tier1_outcome, o.tier2_outcome "
                "FROM sessions s LEFT JOIN outcomes o ON o.session_id = s.session_id "
                f"WHERE s.selection_propensity IS NOT NULL AND {_live_clause('s')} "
                "ORDER BY s.timestamp"
            )
            rows = [dict(row) for row in cursor.fetchall()]
        return [_routing_ope_row(row) for row in rows]

    def live_cost_aggregates(self, window_days: int | None = None) -> LiveCostAggregates:
        """Live inference cost by model over a window — seeded (`bench:`) rows excluded."""
        start = _window_start(window_days)
        where = f"WHERE {_LIVE_CLAUSE}"
        params: list[Any] = []
        if start is not None:
            where += " AND timestamp >= ?"
            params.append(start)
        with self._lock:
            known = self._conn.execute(
                "SELECT model_chosen, COUNT(*) n, COALESCE(SUM(cost), 0) total FROM sessions "
                f"{where} AND cost_known = 1 GROUP BY model_chosen ORDER BY model_chosen",
                params,
            ).fetchall()
            # Counted, never summed and never zero-filled: cost_known=0 means the provider
            # reported NO cost, which is UNKNOWN and deliberately distinct from a real 0.0.
            unknown = self._conn.execute(
                f"SELECT COUNT(*) c FROM sessions {where} AND cost_known = 0", params
            ).fetchone()["c"]
        by_model = [(r["model_chosen"], int(r["n"]), float(r["total"])) for r in known]
        return LiveCostAggregates(
            by_model=by_model,
            total=sum(total for _, _, total in by_model),
            n_cost_known=sum(n for _, n, _ in by_model),
            n_cost_unknown=int(unknown),
            window_days=window_days,
        )

    def stratum_census(self) -> StratumCensus:
        """Per-stratum lifecycle counts: stored, embedded, labeled, tier-2, indexed."""
        # NOT a funnel: `labeled` is counted off `outcome_events` and `tier2` off `outcomes`,
        # two tables with no shared predicate, so the five stages do not nest. `indexed` reuses
        # `_NOT_TOMBSTONED`, the same predicate the index is built from, so at least that stage
        # cannot claim a membership the kNN index disagrees with.
        with self._lock:
            rows = self._conn.execute(
                f"SELECT ({_live_clause('s')}) is_live, COUNT(*) stored, "
                "SUM(s.embedding_blob IS NOT NULL) embedded, "
                "SUM(EXISTS (SELECT 1 FROM outcome_events e3 "
                "WHERE e3.session_id = s.session_id AND e3.tombstoned = 0)) labeled, "
                "SUM(o.tier2_outcome IS NOT NULL) tier2, "
                "SUM(s.embedding_blob IS NOT NULL AND o.session_id IS NOT NULL "
                f"AND {_NOT_TOMBSTONED}) in_index "
                "FROM sessions s LEFT JOIN outcomes o ON o.session_id = s.session_id "
                "GROUP BY is_live"
            ).fetchall()
        by_stratum = {int(row["is_live"]): dict(row) for row in rows}
        return StratumCensus(
            seeded=(
                _stages(by_stratum[0], "seeded") if 0 in by_stratum else _empty_stages("seeded")
            ),
            live=_stages(by_stratum[1], "live") if 1 in by_stratum else _empty_stages("live"),
        )

    def query_index(self, embedding: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        """Return ``(session_id, distance)`` for the *k* nearest embedded sessions."""
        hits = self._index.query(embedding, k)
        out: list[tuple[str, float]] = []
        for idx, distance in hits:
            session_id = self._index.get_session_id(idx)
            if session_id is not None:
                out.append((session_id, float(distance)))
        return out

    def update_human_label(self, session_id: str, label: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE outcomes
                SET human_label = ?, human_label_timestamp = ?
                WHERE session_id = ?
                """,
                (label, _now_iso(), session_id),
            )
            self._conn.commit()

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) as count FROM sessions")
            session_count = cursor.fetchone()["count"]

            # cost_known=0 is UNKNOWN (provider never reported cost), not a real 0.0 —
            # excluded from the sum so an unreported cost can't masquerade as free.
            cursor = self._conn.execute(
                "SELECT COALESCE(SUM(cost), 0) as total_cost FROM sessions WHERE cost_known = 1"
            )
            total_cost = cursor.fetchone()["total_cost"]

            # `total_cost` above is the WHOLE-STORE stat and stays that way. On a seeded rig it
            # is dominated by replayed benchmark spend, so the split below is published beside
            # it and callers name the stratum they mean instead of guessing.
            cursor = self._conn.execute(
                f"SELECT COALESCE(SUM(cost), 0) as c FROM sessions WHERE cost_known = 1 "
                f"AND {_LIVE_CLAUSE}"
            )
            live_total_cost = cursor.fetchone()["c"]

            cursor = self._conn.execute(
                f"SELECT COALESCE(SUM(cost), 0) as c FROM sessions WHERE cost_known = 1 "
                f"AND NOT ({_LIVE_CLAUSE})"
            )
            seeded_total_cost = cursor.fetchone()["c"]

            cursor = self._conn.execute(
                "SELECT COUNT(*) as count FROM sessions WHERE cost_known = 0"
            )
            cost_unknown_count = cursor.fetchone()["count"]

            cursor = self._conn.execute("SELECT COUNT(*) as count FROM outcomes")
            outcome_count = cursor.fetchone()["count"]

            cursor = self._conn.execute(
                "SELECT COUNT(*) as count FROM outcomes WHERE human_label IS NOT NULL"
            )
            labeled_count = cursor.fetchone()["count"]

        return {
            "session_count": session_count,
            "outcome_count": outcome_count,
            "total_cost": total_cost,
            "live_total_cost": live_total_cost,
            "seeded_total_cost": seeded_total_cost,
            "cost_unknown_count": cost_unknown_count,
            "labeled_count": labeled_count,
            "index_size": self._index.count,
        }

    def loop_health_snapshot(self, recent_window: int = 100) -> LoopHealthSnapshot:
        """Raw aggregates for loop-health telemetry — aggregate counts only, no prompt_text."""
        with self._lock:
            total = int(self._conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"])
            eligible = int(
                self._conn.execute(
                    "SELECT COUNT(*) c FROM sessions WHERE embedding_blob IS NOT NULL"
                ).fetchone()["c"]
            )
            verified = int(
                self._conn.execute(
                    "SELECT COUNT(*) c FROM outcomes o JOIN sessions s ON s.session_id = "
                    "o.session_id WHERE o.tier2_outcome IS NOT NULL "
                    "AND s.embedding_blob IS NOT NULL"
                ).fetchone()["c"]
            )
            any_labeled = int(
                self._conn.execute(
                    "SELECT COUNT(*) c FROM outcomes o JOIN sessions s ON s.session_id = "
                    "o.session_id WHERE s.embedding_blob IS NOT NULL"
                ).fetchone()["c"]
            )
            # Live rows only, exactly as `routing_ope_rows` reads them: today's seeder writes
            # no `selection_propensity`, so NOT NULL excludes the seeded stratum incidentally.
            # `_LIVE_CLAUSE` states the guarantee so a seeder that starts writing one cannot
            # mix strata into the support floor or /admin/loop-health.
            prop_rows = self._conn.execute(
                "SELECT model_chosen, COUNT(*) n, AVG(selection_propensity) mean_p, "
                "MIN(selection_propensity) min_p FROM sessions "
                f"WHERE selection_propensity IS NOT NULL AND {_LIVE_CLAUSE} "
                "GROUP BY model_chosen"
            ).fetchall()
            # Reward-independent: the collapse alarm's input touches only model_chosen —
            # no join to `outcomes`, so no reward can quiet the alarm.
            # Live rows only. A replayed benchmark corpus is imported in one burst and would
            # otherwise own the entire recency window, so the collapse alarm would be reading
            # the benchmark matrix's model distribution as recent router behaviour.
            recent = self._conn.execute(
                f"SELECT model_chosen FROM sessions WHERE {_LIVE_CLAUSE} "
                "ORDER BY timestamp DESC LIMIT ?",
                (recent_window,),
            ).fetchall()
            # Live rows only: this feeds `router_cost` on /admin/loop-health, and seeded cost is
            # replayed benchmark spend, not this router's economics.
            cost_rows = self._conn.execute(
                "SELECT model_chosen, COUNT(*) n, COALESCE(SUM(cost), 0) total FROM sessions "
                f"WHERE cost_known = 1 AND {_LIVE_CLAUSE} GROUP BY model_chosen"
            ).fetchall()
            cost_unknown_rows = self._conn.execute(
                "SELECT model_chosen, COUNT(*) n FROM sessions "
                f"WHERE cost_known = 0 AND {_LIVE_CLAUSE} GROUP BY model_chosen"
            ).fetchall()
        return LoopHealthSnapshot(
            total_sessions=total,
            eligible_sessions=eligible,
            verified_labeled=verified,
            any_labeled=any_labeled,
            model_propensities=[
                (r["model_chosen"], int(r["n"]), float(r["mean_p"]), float(r["min_p"]))
                for r in prop_rows
            ],
            recent_choices=[r["model_chosen"] for r in recent],
            cost_by_model=[(r["model_chosen"], int(r["n"]), float(r["total"])) for r in cost_rows],
            cost_unknown_by_model=[(r["model_chosen"], int(r["n"])) for r in cost_unknown_rows],
        )

    def _save_state(self, key: str, state: dict[str, Any]) -> None:
        """UPSERT one JSON value under *key* in the generic router_state KV table."""
        payload = json.dumps(state)
        with self._lock:
            self._conn.execute(
                "INSERT INTO router_state (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, payload, _now_iso()),
            )
            self._conn.commit()

    def _load_state(self, key: str) -> dict[str, Any] | None:
        """Read the JSON value under *key*; missing or corrupt → None (caller starts fresh)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM router_state WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            loaded = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def save_router_state(self, state: dict[str, Any]) -> None:
        """Durably persist the router's mutable exploration state (budget cap + gate slack)."""
        self._save_state(_ROUTER_STATE_KEY, state)

    def load_router_state(self) -> dict[str, Any] | None:
        """Read the persisted exploration state; missing or corrupt → None."""
        return self._load_state(_ROUTER_STATE_KEY)

    def save_escalation_state(self, state: dict[str, Any]) -> None:
        """Durably persist the auto-escalation failure log + per-task decision counters."""
        self._save_state(_ESCALATION_STATE_KEY, state)

    def load_escalation_state(self) -> dict[str, Any] | None:
        """Read the persisted escalation state; missing or corrupt → None."""
        return self._load_state(_ESCALATION_STATE_KEY)

    def save_embedding_fingerprint(self, fingerprint: dict[str, Any]) -> None:
        """Persist the corpus embedding-space fingerprint (its own key, distinct from state)."""
        self._save_state(_EMBEDDING_FINGERPRINT_KEY, fingerprint)

    def load_embedding_fingerprint(self) -> dict[str, Any] | None:
        """Read the stored embedding-space fingerprint; None on a fresh/legacy DB."""
        return self._load_state(_EMBEDDING_FINGERPRINT_KEY)

    def save_seed_state(self, state: dict[str, Any]) -> None:
        """Persist the last-applied benchmark seed marker (fingerprint + results digest)."""
        self._save_state(_SEED_STATE_KEY, state)

    def load_seed_state(self) -> dict[str, Any] | None:
        """Read the persisted seed marker; missing or corrupt → None."""
        return self._load_state(_SEED_STATE_KEY)

    def reindex_corpus(self, embedder: ReindexEmbedder) -> dict[str, Any]:
        """Re-embed the whole corpus into *embedder*'s space, atomically (offline command)."""
        # Blobs rewritten in one txn, index built to a temp file then os.replaced, and the
        # new fingerprint written LAST as the commit marker — a crash before it leaves the
        # old fingerprint, so the next boot refuses the half-migrated space and asks again.
        new_fp = embedder.fingerprint()
        old_fp = self.load_embedding_fingerprint()
        n = self._rewrite_embeddings_txn(embedder)
        # Rebuild the in-memory index from the freshly-written blobs, then swap the file.
        self._rebuild_index()
        if self._index.count > 0:
            self._index.save_atomic(self._index_path)
        else:
            self._discard_index_files()
        # Fingerprint LAST: only now are the DB and the index both fully in the new space.
        self.save_embedding_fingerprint(new_fp)
        return {"reindexed": n, "old_fingerprint": old_fp, "new_fingerprint": new_fp}

    def _rewrite_embeddings_txn(self, embedder: ReindexEmbedder) -> int:
        """Re-embed every already-embedded session's prompt_text in ONE transaction."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, prompt_text FROM sessions WHERE embedding_blob IS NOT NULL"
            ).fetchall()
            try:
                self._conn.execute("BEGIN")
                for row in rows:
                    blob = _embedding_to_blob(embedder.embed(row["prompt_text"]))
                    self._conn.execute(
                        "UPDATE sessions SET embedding_blob = ? WHERE session_id = ?",
                        (blob, row["session_id"]),
                    )
                self._conn.commit()
            except Exception:
                # Any failure mid-embed rolls the whole batch back to the OLD space — never
                # a half-A/half-B corpus. The stale fingerprint (unwritten) keeps boot safe.
                self._conn.rollback()
                raise
        return len(rows)

    def _discard_index_files(self) -> None:
        """Remove the on-disk index so an empty new-space corpus boots to a clean rebuild."""
        for suffix in ("", ".ids.json"):
            path = self._index_path + suffix
            if os.path.exists(path):
                os.remove(path)

    def persist_index(self) -> None:
        self._index.save(self._index_path)

    def close(self) -> None:
        self._conn.close()
