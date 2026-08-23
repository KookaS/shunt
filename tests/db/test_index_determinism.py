"""Two builds of the same corpus must return the same neighbour ids, not just distances."""

# The seed corpus stores one embedding per task shared by its ~6 model rows, so exact ties are
# structural. hnswlib's default multi-threaded ``add_items`` races on insertion order and breaks
# those ties differently per build; the index pins a single thread so it cannot.

from __future__ import annotations

import numpy as np

from shunt.db.index import HNSWIndex

_TASKS = 30
_ARMS = 6  # one embedding per task, shared by its model rows — where the ties come from
_K = 5


def _corpus() -> list[tuple[str, np.ndarray]]:
    """A tie-heavy corpus of the seed store's shape: `_ARMS` distinct ids per vector."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((_TASKS, 32)).astype(np.float32)
    return [(f"bench:t{t:03d}:m{a}", vectors[t]) for t in range(_TASKS) for a in range(_ARMS)]


def _neighbour_ids(corpus: list[tuple[str, np.ndarray]]) -> list[tuple[str, ...]]:
    index = HNSWIndex()
    index.build(corpus)
    out: list[tuple[str, ...]] = []
    for _, vector in corpus:
        hits = index.query(vector, k=_K)
        out.append(tuple(str(index.get_session_id(i)) for i, _ in hits))
    return out


def test_two_builds_return_the_same_neighbour_ids() -> None:
    corpus = _corpus()
    first = _neighbour_ids(corpus)
    second = _neighbour_ids(corpus)
    unstable = [i for i, (a, b) in enumerate(zip(first, second, strict=True)) if set(a) != set(b)]
    assert not unstable, f"{len(unstable)}/{len(corpus)} queries returned different neighbour ids"
