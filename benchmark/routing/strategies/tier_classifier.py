"""Single-shot rank classifier: predict the crossover model from kNN, route there once.

Unlike kNN-semantic-cascade it never probes cheaper models first — one embedding lookup picks the
capability-rank position and routes straight to that model.
"""

from __future__ import annotations

from statistics import mean

import hnswlib

from benchmark import config

from . import Strategy, routing_text
from .knn import _build_index, _embed_texts

_FALLBACK_MODEL = "deepseek-v4-flash"


def predict_model(
    neighbor_ids: list[str],
    matrix: dict,
    models: list[str],
    threshold: float,
    min_samples: int,
) -> str:
    """Weakest ranked model (``models`` is weakest -> strongest) whose MEASURED neighbours pass
    at ``>= threshold``; the strongest model when none clears the bar; a fixed default when
    ``models`` is empty."""
    if not models:
        return _FALLBACK_MODEL
    results = matrix.get("results", {})
    for model in models:  # weakest -> strongest: the first likely to solve it wins
        # An IMPUTED cell is a monotone-ladder fill (impute.py `to_cell`), near-exclusively
        # pass=True — not a verification. Exclude it from BOTH the pass vote and the
        # min_samples count, exactly as the kNN row zeroes an imputed neighbour's
        # verification_confidence (knn.py): at 0.0 it contributes no weight and is not a
        # measured sample. A genuinely measured cell keeps full weight. A raw matrix with
        # no `imputed` key therefore behaves as before (`.get(..., False)` == measured).
        measured = [
            results[nid][model]
            for nid in neighbor_ids
            if model in results.get(nid, {}) and not results[nid][model].get("imputed", False)
        ]
        passes = [bool(cell.get("pass")) for cell in measured]
        if len(passes) >= min_samples and mean(passes) >= threshold:
            return model
    return models[-1]


class TierClassifier(Strategy):
    """kNN over the shipped jina embeddings predicts the crossover rank position; route to
    that model in one shot (no cascade). Params: ``k``, ``success_rate_threshold``,
    ``min_samples``.
    """

    def __init__(
        self, k: int = 20, success_rate_threshold: float = 0.6, min_samples: int = 3
    ) -> None:
        self._k = k
        self._threshold = success_rate_threshold
        self._min_samples = min_samples
        self._models: list[str] = [r.model for r in config.capability_rank().ordered]
        self._task_ids: list[str] | None = None
        self._index: hnswlib.Index | None = None
        self._ready = False

    @property
    def name(self) -> str:
        return "kNN-semantic-tier"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        if not matrix.get("results"):
            return self._models[0] if self._models else _FALLBACK_MODEL
        if not self._ready:
            self._build(matrix)
        neighbor_ids = self._neighbors(task_id, task_meta)
        return predict_model(neighbor_ids, matrix, self._models, self._threshold, self._min_samples)

    def _neighbors(self, task_id: str, task_meta: dict) -> list[str]:
        assert self._index is not None and self._task_ids is not None
        emb = _embed_texts([routing_text(task_id, task_meta)])
        k_search = min(self._k + 1, len(self._task_ids))
        labels, distances = self._index.knn_query(emb.reshape(1, -1), k_search)
        out: list[str] = []
        for label, dist in zip(labels[0], distances[0], strict=True):
            if float(dist) < 0.001:
                continue  # skip self (identical embedding ~ distance 0)
            out.append(self._task_ids[label])
            if len(out) >= self._k:
                break
        return out

    def _build(self, matrix: dict) -> None:
        task_ids = sorted(matrix.get("results", {}).keys())
        self._task_ids = task_ids
        texts = [routing_text(tid, matrix["tasks"].get(tid, {})) for tid in task_ids]
        self._index = _build_index(_embed_texts(texts))
        self._ready = True
