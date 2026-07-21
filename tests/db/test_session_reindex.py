"""A multi-turn session must contribute exactly one neighbour, not one per turn."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shunt.db.store import OutcomeStore


def _store(tmp_path: Path) -> OutcomeStore:
    return OutcomeStore(db_path=str(tmp_path / "sessions.db"))


def _persist(store: OutcomeStore, session_id: str, embedding: np.ndarray, cost: float) -> None:
    store.store_session(
        session_id=session_id,
        prompt_text="a task",
        embedding=embedding,
        model_chosen="model-a",
        cost=cost,
        cache_stats={},
        duration=1.0,
    )
    # Only labeled sessions join the index, so these tests must label to observe it.
    store.store_outcome(session_id, tier1_outcome="success", tier1_confidence=0.9)


def test_five_turns_of_one_session_index_once(tmp_path: Path) -> None:
    # The proxy persists on EVERY turn, passing the session's cached embedding. Without
    # per-session dedup a 5-turn session became 5 identical neighbours — enough for one
    # session to satisfy the selection rule's min_samples (3) on its own, and enough to
    # multiply its weight in the cost/success aggregation by 5.
    store = _store(tmp_path)
    embedding = np.ones(384, dtype=np.float32)
    for turn in range(5):
        _persist(store, "s1", embedding, cost=float(turn))

    neighbors = store.query_index(embedding, k=10)
    assert [sid for sid, _ in neighbors] == ["s1"]


def test_distinct_sessions_still_accumulate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rng = np.random.default_rng(0)
    for i in range(4):
        vec = rng.random(384).astype(np.float32)
        _persist(store, f"s{i}", vec, cost=1.0)
        _persist(store, f"s{i}", vec, cost=2.0)  # a second turn each

    neighbors = store.query_index(rng.random(384).astype(np.float32), k=10)
    assert sorted(sid for sid, _ in neighbors) == ["s0", "s1", "s2", "s3"]
