from __future__ import annotations

import datetime as _datetime
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .index import HNSWIndex
from .schema import run_migrations

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
        embeddings = self.get_labeled_embeddings()
        if not embeddings:
            return
        np_embeddings: list[tuple[str, np.ndarray]] = []
        for sid, blob in embeddings:
            np_embeddings.append((sid, _blob_to_embedding(blob)))
        self._index.build(np_embeddings)

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
    ) -> None:
        with self._lock:
            embedding_blob = _embedding_to_blob(embedding) if embedding is not None else None
            ts = timestamp or _now_iso()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (session_id, prompt_text, embedding_blob, model_chosen, cost,
                     cache_stats, session_duration_seconds, timestamp, decision_provenance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        with self._lock:
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
        """Embeddings of sessions carrying an outcome — i.e. exactly the index members."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT s.session_id, s.embedding_blob FROM sessions s "
                "JOIN outcomes o ON o.session_id = s.session_id "
                "WHERE s.embedding_blob IS NOT NULL"
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

            cursor = self._conn.execute("SELECT COALESCE(SUM(cost), 0) as total_cost FROM sessions")
            total_cost = cursor.fetchone()["total_cost"]

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
            "labeled_count": labeled_count,
            "index_size": self._index.count,
        }

    def persist_index(self) -> None:
        self._index.save(self._index_path)

    def close(self) -> None:
        self._conn.close()
