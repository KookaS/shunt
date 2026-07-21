from __future__ import annotations

import json
import os
import threading

import hnswlib
import numpy as np


class HNSWIndex:
    def __init__(
        self,
        dim: int = 0,
        M: int = 16,  # noqa: N803 (hnswlib graph-degree parameter; kept for **hnsw_kwargs passthrough)
        ef_construction: int = 200,
        ef_search: int = 50,
        max_elements: int = 1000,
        space: str = "cosine",
    ) -> None:
        self._dim = dim
        self._M = M
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._max_elements = max_elements
        self._space = space
        self._index: hnswlib.Index | None = None
        self._ids: list[str] = []
        # session_id -> slot. A session is persisted once per *turn*, so without this an
        # N-turn session would add N identical vectors, inflating every neighbourhood and
        # letting one session satisfy the selection rule's min_samples on its own.
        self._slots: dict[str, int] = {}
        # The index is mutated from request handlers, so slot allocation must be atomic:
        # `next_id = len(self._ids)` read-then-write let two threads claim one slot.
        # Reentrant because add() falls through to build() on an empty index.
        self._lock = threading.RLock()

    def build(self, embeddings: list[tuple[str, np.ndarray]]) -> None:
        with self._lock:
            self._build_locked(embeddings)

    def _build_locked(self, embeddings: list[tuple[str, np.ndarray]]) -> None:
        if not embeddings:
            self._index = None
            self._ids = []
            return

        # Dedup at the source, keeping the newest embedding per session. Previously
        # duplicates were all inserted and only the id->slot MAP was collapsed — and
        # build kept the last slot while load() kept the first, so a later add()
        # overwrote a different copy than the one a query would return. One session
        # could then satisfy min_samples on its own, the exact failure the map exists
        # to prevent.
        deduped: dict[str, np.ndarray] = {}
        for session_id, embedding in embeddings:
            deduped[session_id] = embedding
        items = list(deduped.items())

        dim = items[0][1].shape[0]
        n = len(items)
        max_elements = max(n, self._max_elements)

        self._dim = dim
        self._max_elements = max_elements

        self._index = hnswlib.Index(space=self._space, dim=self._dim)
        self._index.init_index(
            max_elements=max_elements,
            M=self._M,
            ef_construction=self._ef_construction,
        )

        ids = np.arange(n, dtype=np.intp)
        data = np.array([e[1] for e in items], dtype=np.float32)

        self._index.add_items(data, ids)
        self._index.set_ef(self._ef_search)

        self._ids = [e[0] for e in items]
        self._slots = {sid: i for i, sid in enumerate(self._ids)}

    def add(self, session_id: str, embedding: np.ndarray) -> None:
        with self._lock:
            if self._index is None:
                self._build_locked([(session_id, embedding)])
                return

            slot = self._slots.get(session_id)
            if slot is not None:
                # Re-persist of a session already indexed (a later turn): overwrite the
                # vector in place rather than appending a duplicate neighbour.
                self._index.add_items(
                    embedding.astype(np.float32).reshape(1, -1), np.array([slot], dtype=np.intp)
                )
                return

            next_id = len(self._ids)
            if next_id >= self._max_elements:
                self._max_elements = max(self._max_elements * 2, next_id + 1)
                self._index.resize_index(self._max_elements)

            embedding_f32 = embedding.astype(np.float32).reshape(1, -1)
            self._index.add_items(embedding_f32, np.array([next_id], dtype=np.intp))
            self._ids.append(session_id)
            self._slots[session_id] = next_id

    def query(self, embedding: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        with self._lock:
            if self._index is None or self._index.element_count == 0:
                return []

            k = min(k, self._index.element_count)
            embedding_f32 = embedding.astype(np.float32).reshape(1, -1)
            labels, distances = self._index.knn_query(embedding_f32, k)

            return list(zip(labels[0].tolist(), distances[0].tolist(), strict=True))

    def get_session_id(self, idx: int) -> str | None:
        with self._lock:
            if idx < 0 or idx >= len(self._ids):
                return None
            return self._ids[idx]

    def save(self, path: str) -> None:
        # Under the lock like every other mutator. The vector file and the ids
        # sidecar are two separate writes: an add() landing between them wrote a
        # sidecar claiming N+1 ids over a vector file holding N, and the desync
        # survived a reload as a wrong id -> slot mapping.
        with self._lock:
            if self._index is None:
                raise ValueError("Cannot save empty index")

            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._index.save_index(path)

            ids_path = path + ".ids.json"
            with open(ids_path, "w") as f:
                json.dump(
                    {"ids": self._ids, "dim": self._dim, "max_elements": self._max_elements}, f
                )

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        ids_path = path + ".ids.json"
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"Index ids file not found: {ids_path}")

        with open(ids_path) as f:
            meta = json.load(f)

        with self._lock:
            self._ids = meta["ids"]
            # An index written before per-session dedup may hold duplicate ids; keep the
            # first slot per session so re-persists overwrite rather than append a copy.
            self._slots = {}
            for i, sid in enumerate(self._ids):
                self._slots.setdefault(sid, i)
            self._dim = meta["dim"]
            self._max_elements = meta["max_elements"]
            n = len(self._ids)

            self._index = hnswlib.Index(space=self._space, dim=self._dim)
            self._index.load_index(path, max_elements=max(n, self._max_elements))
            self._index.set_ef(self._ef_search)

    @property
    def count(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.element_count)
