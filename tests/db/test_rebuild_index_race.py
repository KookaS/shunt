"""A session stored concurrently with a rebuild must not be lost from the kNN index."""

# Regression: _rebuild_index snapshots the labeled set then builds. If a store_outcome's
# lock-free index add landed between snapshot and build, the full rebuild overwrote it and
# the session went missing from kNN (durable in the DB, invisible to routing) until the next
# rebuild. The fix holds the store lock across snapshot AND build, so a concurrent write
# serialises AFTER the build and its add lands on the rebuilt index.

from __future__ import annotations

import threading
import time

import numpy as np

from shunt.db.store import OutcomeStore
from tests.fake_embedder import FakeEmbedder


def _label(store: OutcomeStore, sid: str, emb: np.ndarray) -> None:
    store.store_session(
        session_id=sid,
        prompt_text=sid,
        embedding=emb,
        model_chosen="m",
        cost=1.0,
        cache_stats={},
        duration=1.0,
    )
    store.store_outcome(
        session_id=sid,
        tier1_outcome="success",
        tier1_confidence=0.7,
        tier2_outcome="success",
        tier2_confidence=0.95,
        aggregated_confidence=0.9,
    )


def test_store_during_rebuild_is_not_lost_from_index(tmp_path):  # type: ignore[no-untyped-def]
    store = OutcomeStore(db_path=str(tmp_path / "race.db"))
    try:
        fe = FakeEmbedder()
        _label(store, "A", fe.embed("alpha alpha alpha"))

        entered = threading.Event()
        release = threading.Event()
        orig_build = store._index.build

        def slow_build(items):  # type: ignore[no-untyped-def]
            entered.set()  # rebuild is now inside build(), holding the store lock
            assert release.wait(5)
            return orig_build(items)

        store._index.build = slow_build  # type: ignore[method-assign]

        rebuild = threading.Thread(target=store.rebuild_index)
        rebuild.start()
        assert entered.wait(5)

        done: dict[str, bool] = {}

        def store_b() -> None:
            _label(store, "B", fe.embed("beta beta beta"))
            done["ok"] = True

        writer = threading.Thread(target=store_b)
        writer.start()
        # B's store_session needs the store lock the rebuild holds, so it must be blocked
        # while build() is open — proving the serialisation the fix introduces.
        time.sleep(0.25)
        assert "ok" not in done, "concurrent write was not serialised behind the rebuild"

        release.set()
        rebuild.join(5)
        writer.join(5)
        assert done.get("ok"), "concurrent write never completed"

        # B is durable AND present in the rebuilt index (its own vector finds it nearest).
        hits = store.query_index(fe.embed("beta beta beta"), k=5)
        assert hits and hits[0][0] == "B", "B was lost from the index after the rebuild race"
    finally:
        store.close()
