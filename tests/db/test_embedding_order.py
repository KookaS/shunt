"""The embedding readers that feed index construction must return rowid order."""

# HNSWIndex assigns index slots by list position, so these readers' row order IS the slot
# layout the persisted .hnsw2 inherits, and published figures read neighbour ids out of it.
# Without ORDER BY, SQLite is free to return any order — a covering index over the selected
# columns is enough to make it scan in a different one, silently changing a committed figure.

from __future__ import annotations

from typing import Final

import numpy as np
import pytest

from shunt.db.store import OutcomeStore

_DIM = 8
# Descending ids: insertion order is the reverse of session_id order, so an index-driven
# scan is distinguishable from the rowid scan the fix pins.
_IDS: Final[list[str]] = [f"s-{9 - i}" for i in range(10)]


@pytest.fixture
def store(tmp_path: object) -> OutcomeStore:
    st = OutcomeStore(db_path=str(tmp_path / "s.db"), hnsw_kwargs={"dim": _DIM})  # type: ignore[operator]
    for i, sid in enumerate(_IDS):
        st.store_session(
            session_id=sid,
            prompt_text="p",
            embedding=np.random.default_rng(i).random(_DIM, dtype=np.float32),
            model_chosen="cheap-model",
            cost=0.0,
            cache_stats={},
            duration=0.0,
        )
        st.store_outcome(sid, tier1_outcome="success", tier1_confidence=0.9)
    return st


def _rowid_order(store: OutcomeStore) -> list[str]:
    rows = store._conn.execute("SELECT session_id FROM sessions ORDER BY rowid").fetchall()
    return [row["session_id"] for row in rows]


def test_embedding_readers_return_rowid_order(store: OutcomeStore) -> None:
    expected = _rowid_order(store)
    assert expected == _IDS, "insertion order should be rowid order"
    assert [sid for sid, _ in store.get_all_embeddings()] == expected
    assert [sid for sid, _ in store.get_labeled_embeddings()] == expected


def test_a_covering_index_cannot_reorder_the_index_slots(store: OutcomeStore) -> None:
    # The concrete way the order drifts in principle: a later index over exactly the selected
    # columns lets SQLite scan it instead of the table, yielding session_id order.
    store._conn.execute("CREATE INDEX idx_probe ON sessions(session_id, embedding_blob)")

    assert [sid for sid, _ in store.get_all_embeddings()] == _IDS
    assert [sid for sid, _ in store.get_labeled_embeddings()] == _IDS
