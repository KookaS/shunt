from __future__ import annotations

import json
import os

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

    def build(self, embeddings: list[tuple[str, np.ndarray]]) -> None:
        if not embeddings:
            self._index = None
            self._ids = []
            return

        dim = embeddings[0][1].shape[0]
        n = len(embeddings)
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
        data = np.array([e[1] for e in embeddings], dtype=np.float32)

        self._index.add_items(data, ids)
        self._index.set_ef(self._ef_search)

        self._ids = [e[0] for e in embeddings]

    def add(self, session_id: str, embedding: np.ndarray) -> None:
        if self._index is None:
            self.build([(session_id, embedding)])
            return

        next_id = len(self._ids)
        if next_id >= self._max_elements:
            self._max_elements = max(self._max_elements * 2, next_id + 1)
            self._index.resize_index(self._max_elements)

        embedding_f32 = embedding.astype(np.float32).reshape(1, -1)
        self._index.add_items(embedding_f32, np.array([next_id], dtype=np.intp))
        self._ids.append(session_id)

    def query(self, embedding: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if self._index is None or self._index.element_count == 0:
            return []

        k = min(k, self._index.element_count)
        embedding_f32 = embedding.astype(np.float32).reshape(1, -1)
        labels, distances = self._index.knn_query(embedding_f32, k)

        return list(zip(labels[0].tolist(), distances[0].tolist(), strict=True))

    def get_session_id(self, idx: int) -> str | None:
        if idx < 0 or idx >= len(self._ids):
            return None
        return self._ids[idx]

    def save(self, path: str) -> None:
        if self._index is None:
            raise ValueError("Cannot save empty index")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._index.save_index(path)

        ids_path = path + ".ids.json"
        with open(ids_path, "w") as f:
            json.dump({"ids": self._ids, "dim": self._dim, "max_elements": self._max_elements}, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        ids_path = path + ".ids.json"
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"Index ids file not found: {ids_path}")

        with open(ids_path) as f:
            meta = json.load(f)

        self._ids = meta["ids"]
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
