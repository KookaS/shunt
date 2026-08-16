"""kNN routing strategy for the offline benchmark — embeds prompts into an HNSW index."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import hnswlib
import numpy as np

from benchmark import config
from shunt.models.config import ModelPool, default_registry_path
from shunt.router.cold_start import ColdStartStrategy
from shunt.router.embedder import Embedder
from shunt.router.engine import RouterEngine
from shunt.router.selection import NeighborResult, SelectionRule

from . import Strategy, routing_text

if TYPE_CHECKING:
    import numpy.typing as npt

# The shipped default model config — pinned so benchmark tiering is reproducible
# regardless of a dev's ambient SHUNT_CONFIG_DIR / ~/.config/shunt/models.yaml.
_BUNDLED_MODEL_CONFIG = str(default_registry_path())

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


# A SWE-bench `problem_statement` runs to a few thousand characters, ~11x the old
# identifier label it replaced. Transformer attention is O(batch x seq^2), so one
# `embed_batch` over all 500 asks onnxruntime for a buffer tens of GB wide and the
# process is SIGKILLed. The fix is a batch budget in `batch x chars^2`, not a fixed
# batch count: a fixed count is either slow on short texts or fatal on long ones.
# Embeddings are per-text, so regrouping and reordering is bit-identical to the single
# call that used to fit.
_EMBED_BUDGET: int = 32_000_000


def _length_batches(texts: list[str]) -> list[list[int]]:
    """Indices grouped so `len(batch) * max_chars^2` stays under the budget."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        width = max(len(texts[i]), 1)
        if current and (len(current) + 1) * width * width > _EMBED_BUDGET:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


# The report builds several kNN-family strategies over ONE corpus, and each used to embed
# the same texts again. That was ~free on a 106-char label and is minutes per strategy on a
# real problem statement. Keyed by the exact text, so a corpus change misses by construction.
# Lazily created behind an accessor for the same reason `_EMBEDDER` above is: a module-level
# mutable literal is denied (SH001), and this mirrors the one opt-out already in this file
# rather than inventing a second convention for the same problem.
_EMBED_CACHE: dict[str, np.ndarray] | None = None


def _embed_cache() -> dict[str, np.ndarray]:
    global _EMBED_CACHE  # noqa: PLW0603, SH001 (lazy per-process embedding cache)
    if _EMBED_CACHE is None:
        _EMBED_CACHE = {}
    return _EMBED_CACHE


def _embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    cache = _embed_cache()
    pending = [t for t in dict.fromkeys(texts) if t not in cache]
    if pending:
        embedder = _get_embedder()
        for batch in _length_batches(pending):
            vectors = embedder.embed_batch([pending[i] for i in batch])
            for i, vector in zip(batch, vectors, strict=True):
                cache[pending[i]] = np.asarray(vector, dtype=np.float32)
    return np.stack([cache[t] for t in texts])


def _build_index(embeddings: np.ndarray) -> hnswlib.Index:
    """A cosine HNSW index over ``embeddings`` (ef_construction=100, M=16, ef=50).

    ``num_threads=1`` pins the neighbour graph so regenerated metrics/plots are
    bit-reproducible (multi-threaded add_items wobbles at ~1e-4).
    """
    n = len(embeddings)
    index = hnswlib.Index(space="cosine", dim=int(embeddings.shape[1]))
    index.init_index(max_elements=n, ef_construction=100, M=16)
    index.add_items(embeddings, np.arange(n), num_threads=1)
    index.set_ef(50)
    return index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_pricing(matrix: dict) -> dict[str, float]:
    """Return {model: total_cost_per_1M} from the matrix pricing section."""
    models = matrix.get("models", {})
    return {m: models[m].get("input_price", 0) + models[m].get("output_price", 0) for m in models}


def _benchmark_model_pool() -> ModelPool:
    """The shipped registry, restricted to the models the benchmark has enabled."""
    # Tiering still comes from the packaged models.yaml (production behaviour), but the
    # CANDIDATE set is the measured one. The unrestricted pool let the engine select a
    # registry model the benchmark never ran (disabled, research-estimated price); the
    # matrix has no such cell, so the task went to `unscorable` and vanished from scoring.
    enabled = config.enabled_models()
    if not enabled:
        # restrict_to_live([]) is a documented no-op — silently reinstating the whole
        # registry is the exact failure this function exists to prevent.
        raise ValueError("benchmark.yaml enables no models; kNN has no candidate set.")
    pool = ModelPool(_BUNDLED_MODEL_CONFIG)
    pool.restrict_to_live(enabled)
    return pool


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

    def effective_labeled(self) -> float:
        # Matrix tasks are labeled at confidence 1.0, so nₑ == raw count (uniform weights).
        return float(len(self._task_ids))

    def effective_tier2(self) -> float:
        return float(len(self._task_ids))

    def model_priors(self) -> dict[str, tuple[float, float]]:
        # Flat prior in the benchmark: the leave-one-out neighborhood already supplies the
        # evidence, and seeding from a global matrix aggregate would change eval numbers —
        # out of scope here (the offline-prior seeding is evaluated on the live loop).
        return {}

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

    def __init__(
        self,
        lookup: dict[str, np.ndarray],
        embed_texts: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        self._lookup = lookup
        self._embed_texts = embed_texts

    def embed(self, text: str) -> np.ndarray:
        cached = self._lookup.get(text)
        if cached is None:
            # Resolved at CALL time, never bound at construction: the module-level `_embed_texts`
            # is what tests monkeypatch, and a default captured in the signature would silently
            # ignore that and reach for the real 600MB ONNX load.
            embed = _embed_texts if self._embed_texts is None else self._embed_texts
            cached = np.asarray(embed([text]), dtype=np.float32)[0]
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
        embed_texts: Callable[[list[str]], np.ndarray] | None = None,
    ):
        self._k = k
        self._success_rate_threshold = success_rate_threshold
        self._min_samples = min_samples
        # None means the shipped embedder, resolved at CALL time so a monkeypatched module-level
        # `_embed_texts` still wins. The seam exists for `instrument_control`, which builds one
        # strategy per permutation draw over a corpus whose TEXT never changes — without it the
        # validity control would run hundreds of real ONNX passes over the same strings. Nothing
        # downstream of the vectors differs.
        self._embed_texts = embed_texts

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
            prompt_text=routing_text(task_id, task_meta),
        )
        return model_name

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _build(self, matrix: dict) -> None:
        task_ids = sorted(matrix.get("results", {}).keys())
        self._task_ids = task_ids
        self._pricing = _model_pricing(matrix)

        # Embed each task's routing text (problem statement, else the short description)
        texts = [routing_text(tid, matrix["tasks"].get(tid, {})) for tid in task_ids]
        embed = _embed_texts if self._embed_texts is None else self._embed_texts
        self._embeddings = np.asarray(embed(texts), dtype=np.float32)

        # Build HNSW index over ALL embeddings (index holds everything;
        # self-exclusion is handled at query time)
        index = _build_index(self._embeddings)
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
        # Pinned to the packaged shunt/config/models.yaml so an ambient SHUNT_CONFIG_DIR /
        # ~/.config/shunt override can't silently change the benchmark's tiering, then
        # narrowed to the enabled (measured) models so no pick can be unscorable.
        model_pool = _benchmark_model_pool()

        selection_rule = SelectionRule(
            min_success_rate=self._success_rate_threshold,
            min_samples=self._min_samples,
        )

        # Precomputed embedding lookup so the engine never re-embeds
        lookup = dict(zip(texts, list(self._embeddings), strict=True))
        embedder = _LookupEmbedder(lookup, self._embed_texts)

        self._engine = RouterEngine(
            model_pool=model_pool,
            session_manager=_DummySessionManager(),
            outcome_index=outcome_index,
            embedder=embedder,
            selection_rule=selection_rule,
            cold_start_strategy=ColdStartStrategy(threshold_tier2=0, threshold_tier1=0),
            cold_start_threshold=0,
            # The engine's neighbourhood size was left at its default 20 while `self._k` (also
            # 20 by default, but a real knob) was stored and never wired through — a benchmark
            # run with `strategies.knn.k != 20` silently measured k=20. Pass the knob through.
            neighbor_k=self._k,
        )

        self._ready = True
