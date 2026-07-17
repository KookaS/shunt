"""kNN-informed cascade routing: order models by kNN-expected success, then try
them in cascade until one passes.
"""

from __future__ import annotations

import hnswlib
import numpy as np

from . import Strategy

# ---------------------------------------------------------------------------
# Lazy fastembed loader
# ---------------------------------------------------------------------------
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER  # noqa: PLW0603, SH001 (lazy embedder singleton; loaded once on first use)
    if _EMBEDDER is None:
        from fastembed import TextEmbedding

        _EMBEDDER = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")
    return _EMBEDDER


def _embed_texts(texts: list[str]) -> np.ndarray:
    return np.array(list(_get_embedder().embed(texts)), dtype=np.float32)


# ---------------------------------------------------------------------------
# Pure cascade-order algorithm (testable without ML)
# ---------------------------------------------------------------------------
def compute_cascade_order(
    neighbor_results: dict[str, list[tuple[float, bool]]],
    pricing: dict[str, float],
    max_tries: int = 3,
    min_samples: int = 3,
    success_rate_threshold: float = 0.7,
) -> list[tuple[str, float]]:
    """Compute cascade order from per-model neighbor outcomes
    (``{model: [(distance, passed), ...]}`` + ``{model: cost_per_1M}``),
    returning ``(model, score)`` tuples by descending score.
    """
    max_price = max(pricing.values()) if pricing else 1.0

    scored: list[tuple[str, float]] = []
    for model, outcomes in neighbor_results.items():
        if len(outcomes) < min_samples:
            continue

        total_weight = 0.0
        weighted_passes = 0.0
        for dist, passed in outcomes:
            conf = 1.0 - min(dist, 1.0)
            weighted_passes += conf * (1.0 if passed else 0.0)
            total_weight += conf

        weighted_rate = weighted_passes / total_weight if total_weight > 0 else 0.0

        if weighted_rate < success_rate_threshold:
            continue

        price = pricing.get(model, 0.0)
        score = weighted_rate - 0.01 * (price / max_price)
        scored.append((model, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:max_tries]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class kNNCascadeStrategy(Strategy):  # noqa: N801 (kNN is the established algorithm name)
    """kNN-informed cascade: rank neighbour-seen models by distance-weighted
    success minus a price penalty, try the top ``max_tries`` in order against
    the matrix, and fall back to the frontier model if none pass.
    """

    def __init__(
        self,
        k: int = 20,
        max_tries: int = 3,
        success_rate_threshold: float = 0.7,
        min_samples: int = 3,
    ):
        self._k = k
        self._max_tries = max_tries
        self._success_rate_threshold = success_rate_threshold
        self._min_samples = min_samples

        self._task_ids: list[str] | None = None
        self._embeddings: np.ndarray | None = None
        self._index: hnswlib.Index | None = None
        self._pricing: dict[str, float] | None = None
        self._ready = False

        # Cascade metadata (reset per :meth:`select` call)
        self.cascade_total_cost: float = 0.0
        self.cascade_tried_models: list[str] = []
        self.cascade_order: list[tuple[str, float]] = []

    @property
    def name(self) -> str:
        return "kNN-cascade"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        if not matrix.get("results"):
            self.cascade_tried_models = []
            self.cascade_total_cost = 0.0
            self.cascade_order = []
            return _fallback_model(matrix)
        if not self._ready:
            self._build(matrix)

        self.cascade_tried_models = []
        self.cascade_total_cost = 0.0

        order = self._get_cascade_order(task_id, task_meta, matrix)
        self.cascade_order = order

        task_results = matrix.get("results", {}).get(task_id, {})

        for model, _score in order:
            self.cascade_tried_models.append(model)
            outcome = task_results.get(model, {})
            self.cascade_total_cost += outcome.get("cost", 0.0)
            if outcome.get("pass", False):
                return model

        # Fallback to frontier (most expensive model)
        pricing = self._pricing
        assert pricing is not None
        frontier = max(pricing, key=lambda m: pricing[m])
        self.cascade_tried_models.append(frontier)
        frontier_outcome = task_results.get(frontier, {})
        self.cascade_total_cost += frontier_outcome.get("cost", 0.0)
        return frontier

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_cascade_order(
        self, task_id: str, task_meta: dict, matrix: dict
    ) -> list[tuple[str, float]]:
        assert self._index is not None
        assert self._embeddings is not None
        assert self._pricing is not None
        assert self._task_ids is not None

        description = task_meta.get("description", task_id)
        query_emb = _embed_texts([description])

        k_search = min(self._k + 1, len(self._task_ids))
        labels, distances = self._index.knn_query(query_emb.reshape(1, -1), k_search)

        neighbor_results: dict[str, list[tuple[float, bool]]] = {}
        for label, dist in zip(labels[0], distances[0], strict=True):
            nid = self._task_ids[label]
            distance = float(dist)
            # Skip self (identical embedding has distance ~0)
            if distance < 0.001:
                continue
            for model, outcome in matrix["results"].get(nid, {}).items():
                neighbor_results.setdefault(model, []).append(
                    (distance, bool(outcome.get("pass", False)))
                )

        return compute_cascade_order(
            neighbor_results,
            self._pricing,
            max_tries=self._max_tries,
            min_samples=self._min_samples,
            success_rate_threshold=self._success_rate_threshold,
        )

    def _build(self, matrix: dict) -> None:
        """Build HNSW index over all task descriptions."""
        task_ids = sorted(matrix.get("results", {}).keys())
        self._task_ids = task_ids
        self._pricing = _model_pricing(matrix)

        # Embed all task descriptions
        descriptions = [matrix["tasks"].get(tid, {}).get("description", tid) for tid in task_ids]
        self._embeddings = _embed_texts(descriptions)

        # Build HNSW index
        dim = self._embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(task_ids), ef_construction=100, M=16)
        # num_threads=1: pin the neighbour graph for bit-reproducible regeneration
        # (multi-threaded add_items wobbles at ~1e-4), matching knn_blended._build_hnsw.
        index.add_items(self._embeddings, np.arange(len(task_ids)), num_threads=1)
        index.set_ef(50)
        self._index = index

        self._ready = True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _model_pricing(matrix: dict) -> dict[str, float]:
    """Return ``{model: total_cost_per_1M}`` from the matrix."""
    models = matrix.get("models", {})
    return {m: models[m].get("input_price", 0) + models[m].get("output_price", 0) for m in models}


def _fallback_model(matrix: dict) -> str:
    """Cheapest priced model, or a fixed default when the matrix has no data.

    With an empty results matrix the cascade has no neighbours to rank, so it
    degrades to a cheap default rather than embedding an empty task list.
    """
    pricing = _model_pricing(matrix)
    return min(pricing, key=lambda m: pricing[m]) if pricing else "deepseek-v4-flash"
