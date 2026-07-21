"""The kNN index must contain exactly the sessions that can be neighbours."""

# Every session was indexed on persist, but only sessions carrying an outcome are usable
# as neighbours — and outcomes arrive separately. So ordinary traffic crowded the labeled
# sessions out of the k nearest, `query()` returned mostly nothing, and selection fell
# through to `_escalate(set())` — the cheapest model, reported as `exploration_untested`.
# The router silently became always-cheap the more it was used.

from __future__ import annotations

import numpy as np
import pytest

from shunt.db.outcome_index import OutcomeIndexAdapter
from shunt.db.store import OutcomeStore

_DIM = 16
_LABELED = 40
_NOISE = 2000


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(_DIM, dtype=np.float32)


@pytest.fixture
def store(tmp_path: object) -> OutcomeStore:
    return OutcomeStore(db_path=str(tmp_path / "s.db"), hnsw_kwargs={"dim": _DIM})  # type: ignore[operator]


def _persist(store: OutcomeStore, session_id: str, seed: int) -> None:
    store.store_session(
        session_id=session_id,
        prompt_text="p",
        embedding=_vector(seed),
        model_chosen="cheap-model",
        cost=0.0,
        cache_stats={},
        duration=0.0,
    )


def test_labeled_neighbours_survive_a_flood_of_unlabeled_traffic(store: OutcomeStore) -> None:
    for i in range(_LABELED):
        _persist(store, f"lab-{i}", i)
        store.store_outcome(f"lab-{i}", tier1_outcome="success", tier1_confidence=0.9)
    # The traffic an ordinary day produces: persisted, never flagged.
    for i in range(_NOISE):
        _persist(store, f"noise-{i}", 10_000 + i)

    adapter = OutcomeIndexAdapter(store)
    neighbours = adapter.query(_vector(0), k=20)

    assert len(neighbours) == 20, "unlabeled traffic crowded out every labeled neighbour"
    assert adapter.count_total_labeled() == _LABELED


def test_an_unlabeled_session_is_never_returned_as_a_neighbour(store: OutcomeStore) -> None:
    _persist(store, "labeled", 1)
    store.store_outcome("labeled", tier1_outcome="success", tier1_confidence=0.9)
    _persist(store, "unlabeled", 2)

    adapter = OutcomeIndexAdapter(store)
    got = {n.session_id for n in adapter.query(_vector(2), k=10)}
    assert got == {"labeled"}


def test_a_late_arriving_outcome_makes_its_session_reachable(store: OutcomeStore) -> None:
    # Outcomes arrive after the session closes, so indexing must happen then, not before.
    _persist(store, "late", 3)
    adapter = OutcomeIndexAdapter(store)
    assert adapter.query(_vector(3), k=5) == []

    store.store_outcome("late", tier1_outcome="success", tier1_confidence=0.8)
    assert [n.session_id for n in adapter.query(_vector(3), k=5)] == ["late"]
