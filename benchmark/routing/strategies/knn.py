"""kNN routing strategy for the offline benchmark — embeds prompts into an HNSW index."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hnswlib
import numpy as np

import shunt.models
from shunt.models.config import ModelPool
from shunt.router.cold_start import ColdStartStrategy
from shunt.router.embedder import Embedder
from shunt.router.engine import RouterEngine
from shunt.router.selection import NeighborResult, SelectionRule

from . import Strategy

if TYPE_CHECKING:
    import numpy.typing as npt

# The shipped default model config — pinned so benchmark tiering is reproducible
# regardless of a dev's ambient SHUNT_CONFIG_DIR / ~/.config/shunt/models.yaml.
_BUNDLED_MODEL_CONFIG = str(Path(shunt.models.__file__).parent / "default_config.yaml")

# ---------------------------------------------------------------------------
# Embedder — the shipped router's Embedder (honors SHUNT_EMBEDDER_MODEL +
# fallback), lazily instantiated so the model download is deferred until first use.
# ---------------------------------------------------------------------------
_EMBEDDER: Embedder | None = None


def _get_embedder() -> Embedder:
    global _EMBEDDER  # noqa: PLW0603, SH001 (lazy embedder singleton; loaded once on first use)
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


def _embed_texts(texts: list[str]) -> np.ndarray:
    return np.array(_get_embedder().embed_batch(texts), dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_pricing(matrix: dict) -> dict[str, float]:
    """Return {model: total_cost_per_1M} from the matrix pricing section."""
    models = matrix.get("models", {})
    return {m: models[m].get("input_price", 0) + models[m].get("output_price", 0) for m in models}


def _fallback_model(matrix: dict) -> str:
    """Cheapest priced model, or a fixed default when the matrix has no data.

    kNN needs cached neighbour outcomes; with an empty results matrix it cannot
    embed or query, so it degrades to a cheap default instead of crashing.
    """
    pricing = _model_pricing(matrix)
    return min(pricing, key=lambda m: pricing[m]) if pricing else "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# OutcomeIndex backed by the benchmark matrix + HNSW index
# ---------------------------------------------------------------------------
class MatrixOutcomeIndex:
    """Implements the ``OutcomeIndex`` protocol over a benchmark matrix + HNSW index.

    All tasks in the matrix are considered labeled. The query excludes the
    query task itself (distance < 0.001) so that KNN is leave-one-out.
    """

    def __init__(
        self,
        task_ids: list[str],
        embeddings: np.ndarray,
        index: hnswlib.Index,
        matrix: dict,
    ) -> None:
        self._task_ids = task_ids
        self._embeddings = embeddings
        self._index = index
        self._matrix = matrix

    def count_labeled(self) -> int:
        return len(self._task_ids)

    def count_total_labeled(self) -> int:
        return len(self._task_ids)

    def query(self, embedding: npt.NDArray, k: int = 20) -> list[NeighborResult]:
        """Return k neighbours (excluding self) as per-model NeighborResults."""
        k_search = min(k + 1, len(self._task_ids))
        labels, distances = self._index.knn_query(embedding.reshape(1, -1), k_search)

        results: list[NeighborResult] = []
        for label, dist in zip(labels[0], distances[0], strict=True):
            if len(results) >= k:
                break
            nid = self._task_ids[label]
            distance = float(dist)
            # Skip self (identical embedding has distance ~0)
            if distance < 0.001:
                continue
            neighbor_results = self._matrix["results"].get(nid, {})
            for model, outcome in neighbor_results.items():
                results.append(
                    NeighborResult(
                        model=model,
                        outcome=bool(outcome.get("pass", False)),
                        cost=float(outcome.get("cost", 0.0)),
                        verification_confidence=1.0,
                        distance=distance,
                        session_id=nid,
                        truncation_rate=0.0,
                    )
                )
        return results


# ---------------------------------------------------------------------------
# Minimal stand-ins for live-router dependencies
# ---------------------------------------------------------------------------
class _DummySessionManager:
    """Satisfies ``RouterEngine``'s session-manager slot without side effects."""

    def get_session(self, session_id: str) -> None:
        return None


class _LookupEmbedder:
    """Precomputed embeddings keyed by prompt text, computed on demand for uncached texts."""

    def __init__(self, lookup: dict[str, np.ndarray]) -> None:
        self._lookup = lookup

    def embed(self, text: str) -> np.ndarray:
        cached = self._lookup.get(text)
        if cached is None:
            cached = _embed_texts([text])[0]
            self._lookup[text] = cached
        return cached


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class kNNStrategy(Strategy):  # noqa: N801 (kNN is the established algorithm name)
    """kNN routing via HNSW with distance-weighted cheapest-above-threshold,
    using the live ``RouterEngine.decide()`` for selection. Params: ``k`` (20),
    ``success_rate_threshold`` (0.7), ``min_samples`` (3).
    """

    def __init__(
        self,
        k: int = 20,
        success_rate_threshold: float = 0.7,
        min_samples: int = 3,
    ):
        self._k = k
        self._success_rate_threshold = success_rate_threshold
        self._min_samples = min_samples

        # Lazy-initialized state
        self._task_ids: list[str] | None = None
        self._embeddings: np.ndarray | None = None
        self._index: hnswlib.Index | None = None
        self._pricing: dict[str, float] | None = None
        self._engine: RouterEngine | None = None
        self._ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "kNN"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        if not matrix.get("results"):
            return _fallback_model(matrix)
        if not self._ready:
            self._build(matrix)

        assert self._engine is not None
        model_name, _reason, _provenance = self._engine.decide(
            session_id=task_id,
            prompt_text=task_meta.get("description", task_id),
        )
        return model_name

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _build(self, matrix: dict) -> None:
        task_ids = sorted(matrix.get("results", {}).keys())
        self._task_ids = task_ids
        self._pricing = _model_pricing(matrix)

        # Embed all task descriptions
        descriptions = [matrix["tasks"].get(tid, {}).get("description", tid) for tid in task_ids]
        self._embeddings = _embed_texts(descriptions)

        # Build HNSW index over ALL embeddings (index holds everything;
        # self-exclusion is handled at query time)
        dim = self._embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(task_ids), ef_construction=100, M=16)
        # num_threads=1: pin the neighbour graph so regenerated metrics/plots are
        # bit-reproducible (multi-threaded add_items wobbles at ~1e-4), matching
        # knn_blended._build_hnsw.
        index.add_items(self._embeddings, np.arange(len(task_ids)), num_threads=1)
        index.set_ef(50)
        self._index = index

        # Build the live-router abstractions
        outcome_index = MatrixOutcomeIndex(
            task_ids=task_ids,
            embeddings=self._embeddings,
            index=index,
            matrix=matrix,
        )

        # The SHIPPED model pool — tiers come from src/shunt config, not a
        # benchmark-local price heuristic, so the benchmark measures production tiering.
        # Pinned to the bundled default_config.yaml so an ambient SHUNT_CONFIG_DIR /
        # ~/.config/shunt override can't silently change the benchmark's tiering.
        model_pool = ModelPool(_BUNDLED_MODEL_CONFIG)

        selection_rule = SelectionRule(
            min_success_rate=self._success_rate_threshold,
            min_samples=self._min_samples,
        )

        # Precomputed embedding lookup so the engine never re-embeds
        lookup = dict(zip(descriptions, list(self._embeddings), strict=True))
        embedder = _LookupEmbedder(lookup)

        self._engine = RouterEngine(
            model_pool=model_pool,
            session_manager=_DummySessionManager(),
            outcome_index=outcome_index,
            embedder=embedder,
            selection_rule=selection_rule,
            cold_start_strategy=ColdStartStrategy(threshold_tier2=0, threshold_tier1=0),
            cold_start_threshold=0,
        )

        self._ready = True
