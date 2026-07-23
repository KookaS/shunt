"""Batch offline re-fit trigger: fires at the session-count cadence and rebuilds the
kNN index from the append-only log (so a tombstone drops out, a fresh label appears)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shunt.capture.refit import RefitScheduler
from shunt.db.store import OutcomeEvent, OutcomeStore


class _CountingStore:
    def __init__(self) -> None:
        self.rebuilds = 0

    def rebuild_index(self) -> None:
        self.rebuilds += 1


class TestCadence:
    def test_fires_every_n(self) -> None:
        store = _CountingStore()
        sched = RefitScheduler(store, every_n=3)
        results = [sched.note_capture() for _ in range(7)]
        # Fires on the 3rd and 6th capture only.
        assert results == [False, False, True, False, False, True, False]
        assert store.rebuilds == 2

    def test_zero_disables_trigger(self) -> None:
        store = _CountingStore()
        sched = RefitScheduler(store, every_n=0)
        assert [sched.note_capture() for _ in range(5)] == [False] * 5
        assert store.rebuilds == 0


def _vec(seed: float, dim: int = 8) -> np.ndarray:
    return np.full(dim, seed, dtype=np.float32)


def _add_tier2(store: OutcomeStore, sid: str, *, tombstoned: bool = False) -> None:
    store.store_session(sid, f"p {sid}", _vec(0.1 + len(sid)), "m", 1.0, {}, 1.0)
    store.append_outcome_event(
        OutcomeEvent(
            session_id=sid,
            tier=2,
            source="auto_tier2",
            outcome="success",
            confidence=1.0,
            run_signature=f"sig-{sid}",
            tombstoned=tombstoned,
        )
    )


class TestRebuildReflectsLog:
    def test_refit_drops_tombstoned_and_matches_labeled_log(self, tmp_path: Path) -> None:
        store = OutcomeStore(db_path=str(tmp_path / "test.db"))
        try:
            for i in range(3):
                _add_tier2(store, f"s{i}")
            # Tombstone one via a later tombstoned event — HNSW can't delete in place, so
            # the live index is now STALE (still holds the vector) until a re-fit.
            store.append_outcome_event(
                OutcomeEvent(
                    session_id="s0",
                    tier=2,
                    source="auto_tier2",
                    outcome="success",
                    confidence=1.0,
                    run_signature="sig-s0-tomb",
                    tombstoned=True,
                )
            )
            labeled = store.get_labeled_embeddings()
            assert len(labeled) == 2  # s0 excluded from the truthful log
            assert store._index.count == 3  # stale: still holds s0

            fired = RefitScheduler(store, every_n=1).note_capture()

            assert fired is True
            # After re-fit the index matches the non-tombstoned labeled log exactly.
            assert store._index.count == len(store.get_labeled_embeddings()) == 2
        finally:
            store.close()
