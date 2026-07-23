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

from .index import HNSWIndex
from .loop_health import LoopHealthSnapshot
from .schema import run_migrations

# Human labels win over automatic Tier-2, which win over the quarantined wire prior.
# Drives which event's outcome/source becomes the materialized current view (§Q4).
_SOURCE_PRIORITY: Final[dict[str, int]] = {"wire_tier1": 0, "auto_tier2": 1, "human": 2}

# Single row holding the serialized exploration budget + conservative-gate slack.
_ROUTER_STATE_KEY: Final[str] = "exploration_state"

# The corpus's embedding-space fingerprint (repo, dim, max_chars, revision?). A second
# consumer of the generic router_state KV, under its own key — never clobbers the above.
_EMBEDDING_FINGERPRINT_KEY: Final[str] = "embedding_fingerprint"

# The auto-escalation failure log + per-task decision counters — a third KV consumer under its
# own key, so a restart does not wipe accrued same-failure counts.
_ESCALATION_STATE_KEY: Final[str] = "escalation_state"


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


def _get_default_db_path() -> str:
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
        self._db_path = db_path or _get_default_db_path()
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
                     cost_known, selection_propensity, model_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    prompt_text,
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
                _now_iso(),
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
        self._materialize_outcome(event.session_id)
        return inserted

    def _materialize_outcome(self, session_id: str) -> None:
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
            self._upsert_materialized_outcome(row)
            self._conn.commit()
            emb = self._conn.execute(
                "SELECT embedding_blob FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if emb is not None and emb["embedding_blob"] is not None:
            self._index.add(session_id, _blob_to_embedding(emb["embedding_blob"]))

    def _upsert_materialized_outcome(self, row: dict[str, Any]) -> None:
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
            {**row, "created_at": _now_iso()},
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

    def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT session_id, embedding_blob FROM sessions WHERE embedding_blob IS NOT NULL"
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
            "WHERE s.embedding_blob IS NOT NULL "
            "AND NOT EXISTS ("
            "    SELECT 1 FROM outcome_events e "
            "    WHERE e.session_id = s.session_id AND e.tombstoned = 1 "
            "    AND e.event_id = ("
            "        SELECT MAX(event_id) FROM outcome_events e2 "
            "        WHERE e2.session_id = s.session_id"
            "    )"
            ")"
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
            prop_rows = self._conn.execute(
                "SELECT model_chosen, COUNT(*) n, AVG(selection_propensity) mean_p, "
                "MIN(selection_propensity) min_p FROM sessions "
                "WHERE selection_propensity IS NOT NULL GROUP BY model_chosen"
            ).fetchall()
            # Reward-independent: the collapse alarm's input touches only model_chosen —
            # no join to `outcomes`, so no reward can quiet the alarm.
            recent = self._conn.execute(
                "SELECT model_chosen FROM sessions ORDER BY timestamp DESC LIMIT ?",
                (recent_window,),
            ).fetchall()
            cost_rows = self._conn.execute(
                "SELECT model_chosen, COUNT(*) n, COALESCE(SUM(cost), 0) total FROM sessions "
                "WHERE cost_known = 1 GROUP BY model_chosen"
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
