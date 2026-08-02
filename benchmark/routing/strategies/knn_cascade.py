"""kNN-informed cascade routing: shortlist models whose kNN-expected success clears
a bar, then try them cheapest-first until one passes.
"""

from __future__ import annotations

import hnswlib
import numpy as np

from . import Strategy, routing_text
from ._cascade_common import cheapest_priced_model, frontier_model, model_pricing

# `_embed_texts` reaches the SHIPPED Embedder through the one loader the rest of the
# routing family shares (same precedent as tier_classifier). A local `TextEmbedding(...)`
# here hardcoded the model name past `load_embedding_config()`, so flipping
# `embedding.yaml`'s active row (or SHUNT_EMBEDDER_MODEL) would silently score kNN and
# kNN-cascade in two different embedding spaces; it also skipped the durable cache_dir,
# the 4000-char clip, `fingerprint()`, and the SHUNT_DISALLOW_REAL_EMBEDDER test wall.
from .knn import _embed_texts


# ---------------------------------------------------------------------------
# Pure cascade-order algorithm (testable without ML)
# ---------------------------------------------------------------------------
def _weighted_success_rate(outcomes: list[tuple[float, bool]]) -> float:
    """Distance-weighted share of neighbour runs that passed (nearer neighbours count more)."""
    total_weight = 0.0
    weighted_passes = 0.0
    for dist, passed in outcomes:
        conf = 1.0 - min(dist, 1.0)
        weighted_passes += conf * (1.0 if passed else 0.0)
        total_weight += conf
    return weighted_passes / total_weight if total_weight > 0 else 0.0


def compute_cascade_order(
    neighbor_results: dict[str, list[tuple[float, bool]]],
    pricing: dict[str, float],
    max_tries: int = 3,
    min_samples: int = 3,
    success_rate_threshold: float = 0.7,
) -> list[str]:
    """Cheapest first among the models whose neighbour success rate clears the bar.

    Quality gates *eligibility* (``min_samples`` + ``success_rate_threshold``); price alone
    decides the *order*, because a failed cheap attempt escalates rather than losing the task.
    """
    # The rate must never rank the shortlist. A blended quality-minus-price score cannot
    # express "cheapest that clears the bar": whatever weight the price term carries, a
    # rate gap wide enough outvotes an arbitrarily large price gap, and the shipped
    # version put the frontier first on 79% of tasks in a cascade documented as cheap-first.
    # `model in pricing` is the LIKE-FOR-LIKE bar: Price-Cascade draws its candidates from
    # `matrix["models"]` (priced by construction), so admitting an unpriced model here would
    # let kNN-cascade route somewhere its own baseline structurally cannot, and the published
    # head-to-head is only a comparison if both draw from the same universe. Sorting an
    # unpriced model last was not enough — it stayed inside the `max_tries` shortlist.
    eligible = [
        model
        for model, outcomes in neighbor_results.items()
        if model in pricing
        and len(outcomes) >= min_samples
        and _weighted_success_rate(outcomes) >= success_rate_threshold
    ]
    # Name breaks exact price ties so the order is deterministic across dict orderings.
    eligible.sort(key=lambda model: (pricing[model], model))
    return eligible[:max_tries]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class kNNCascadeStrategy(Strategy):  # noqa: N801 (kNN is the established algorithm name)
    """kNN-informed cascade: shortlist the neighbour-seen models whose distance-weighted
    success clears the bar, try the ``max_tries`` cheapest in ascending price order, and
    fall back to the frontier model if none pass.
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
        # False when the cascade path (any tried model, or the frontier fallback)
        # lands on an unmeasured matrix cell — the true cost/outcome is unknown, so
        # this decision is a coverage gap, not a real fail@$0.
        self.cascade_scorable: bool = True

    @property
    def name(self) -> str:
        return "kNN-cascade"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        if not matrix.get("results"):
            self.cascade_tried_models = []
            self.cascade_total_cost = 0.0
            self.cascade_scorable = False
            return cheapest_priced_model(matrix)
        if not self._ready:
            self._build(matrix)

        self.cascade_tried_models = []
        self.cascade_total_cost = 0.0
        self.cascade_scorable = True

        order = self._get_cascade_order(task_id, task_meta, matrix)

        task_results = matrix.get("results", {}).get(task_id, {})

        for model in order:
            self.cascade_tried_models.append(model)
            outcome = task_results.get(model, {})
            if not outcome:
                self.cascade_scorable = False
            self.cascade_total_cost += outcome.get("cost", 0.0)
            if outcome.get("pass", False):
                return model

        # Escalate to the frontier — the SAME target Price-Cascade uses, so the zero-ML
        # baseline and this router are compared on one escalation rule, not two.
        frontier = frontier_model(matrix) or cheapest_priced_model(matrix)
        # Re-running a model the shortlist already failed is a second API call with a
        # guaranteed-identical outcome — bill it once and return it as the (failed) pick.
        if frontier in self.cascade_tried_models:
            return frontier
        self.cascade_tried_models.append(frontier)
        frontier_outcome = task_results.get(frontier, {})
        if not frontier_outcome:
            self.cascade_scorable = False
        self.cascade_total_cost += frontier_outcome.get("cost", 0.0)
        return frontier

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_cascade_order(self, task_id: str, task_meta: dict, matrix: dict) -> list[str]:
        assert self._index is not None
        assert self._embeddings is not None
        assert self._pricing is not None
        assert self._task_ids is not None

        query_emb = _embed_texts([routing_text(task_id, task_meta)])

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
        """Build HNSW index over every task's routing text."""
        task_ids = sorted(matrix.get("results", {}).keys())
        self._task_ids = task_ids
        self._pricing = model_pricing(matrix)

        # Embed each task's routing text (problem statement, else the short description)
        texts = [routing_text(tid, matrix["tasks"].get(tid, {})) for tid in task_ids]
        self._embeddings = _embed_texts(texts)

        # Build HNSW index
        dim = self._embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(task_ids), ef_construction=100, M=16)
        # num_threads=1: pin the neighbour graph for bit-reproducible regeneration
        # (multi-threaded add_items wobbles at ~1e-4).
        index.add_items(self._embeddings, np.arange(len(task_ids)), num_threads=1)
        index.set_ef(50)
        self._index = index

        self._ready = True
