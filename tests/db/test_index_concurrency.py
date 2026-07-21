"""Concurrent `add()` must not lose vectors or desync the slot map."""

# `next_id = len(self._ids)` read, then wrote, without a lock: two threads computed the
# same slot and one overwrote the other, while both appended to `_ids`. The id list then
# overstated the real element count and `get_session_id(slot)` resolved to a *different*
# session than the vector that matched it — so an outcome was credited to the wrong
# neighbour, corrupting routing evidence rather than merely dropping it.

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from shunt.db.index import HNSWIndex

_DIM = 16
_THREADS = 8
_PER_THREAD = 100


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(_DIM, dtype=np.float32)


def test_concurrent_adds_keep_ids_slots_and_element_count_in_step() -> None:
    index = HNSWIndex(dim=_DIM)
    index.add("seed", _vector(0))

    def worker(thread_id: int) -> None:
        for i in range(_PER_THREAD):
            n = thread_id * _PER_THREAD + i
            index.add(f"s-{n}", _vector(n + 1))

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        list(pool.map(worker, range(_THREADS)))

    expected = _THREADS * _PER_THREAD + 1
    assert len(index._ids) == expected
    assert len(index._slots) == expected
    assert index._index is not None
    assert index._index.element_count == expected


def test_every_slot_resolves_to_the_session_whose_vector_occupies_it() -> None:
    index = HNSWIndex(dim=_DIM)

    def worker(thread_id: int) -> None:
        for i in range(_PER_THREAD):
            n = thread_id * _PER_THREAD + i
            index.add(f"s-{n}", _vector(n + 1))

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        list(pool.map(worker, range(_THREADS)))

    # Query each vector back; the nearest hit must resolve to its own session id.
    for n in range(0, _THREADS * _PER_THREAD, 37):
        hits = index.query(_vector(n + 1), k=1)
        assert hits, f"vector {n} vanished from the index"
        assert index.get_session_id(hits[0][0]) == f"s-{n}"


def test_save_is_atomic_against_a_concurrent_add(tmp_path: Path) -> None:
    # save() writes the vector file and the ids sidecar as two separate steps.
    # Unlocked, an add() landing in the gap produces a sidecar claiming one more
    # id than the vector file holds — a silent id->slot desync that survives a
    # reload and maps queries onto the wrong session.
    #
    # The real window is sub-millisecond, so a plain thread race passes with or
    # without the lock and proves nothing. We hold the window open deliberately:
    # save_index() is wrapped to block until the adder thread has run.
    index = HNSWIndex(dim=4)
    index.build([("s0", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))])
    path = str(tmp_path / "idx" / "index.bin")

    in_save = threading.Event()
    may_finish = threading.Event()

    class _SlowSave:
        """Delegates to the real hnswlib index but holds save_index open."""

        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def save_index(self, target: str) -> None:
            self._inner.save_index(target)  # type: ignore[attr-defined]
            in_save.set()
            may_finish.wait(timeout=5)

    index._index = _SlowSave(index._index)  # type: ignore[assignment]

    added = threading.Event()

    def adder() -> None:
        in_save.wait(timeout=5)
        # With the lock held by save(), this blocks until save() completes.
        index.add("s1", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        added.set()
        may_finish.set()

    thread = threading.Thread(target=adder)
    thread.start()
    index.save(path)
    may_finish.set()
    thread.join(timeout=5)
    assert added.is_set()

    reloaded = HNSWIndex(dim=4)
    reloaded.load(path)
    # The sidecar must never claim ids the persisted vector file does not hold.
    assert reloaded.count == len(reloaded._ids)
    assert all(reloaded.get_session_id(i) is not None for i in range(reloaded.count))
