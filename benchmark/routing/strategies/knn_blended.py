"""Blended kNN: our verified runs + external SWE-bench as down-weighted neighbours."""

from __future__ import annotations

# Item-2 proper — "use the swebench values in our kNN alongside our challenge runs." Plain
# kNN (knn.py) indexes only our own verified outcomes; while live coverage is a small
# partial subset the escalate-here signal isn't learnable from those alone. Here we embed
# our run descriptions AND the *other* Verified problem statements (self-excluded to avoid
# leakage) into one HNSW index and retrieve neighbours from the union. Weighting (owner's
# "smaller weight for their data"): our neighbours give
# a real per-model pass/fail at weight 1.0; each external neighbour gives its tier rate
# (cheap ← p_solve, since leaderboard p_cheap is degenerate; mid/frontier ← p_frontier) at
# external_weight < 1.0. Cost is never taken from external data. External evidence is
# fractional (a rate), which the shipped bool-only SelectionRule can't represent, so this
# strategy aggregates weighted evidence locally rather than via RouterEngine.
import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark import config

from . import Strategy
from .fixed import _model_pricing

_PRIOR_PATH = Path(__file__).resolve().parents[1] / "data" / "external_swebench.csv"
_CACHE_DIR = Path(__file__).resolve().parents[1] / "artifacts"
# Cap statement length before embedding: some Verified problem statements are tens
# of KB, and the embedder does not truncate — a single long text pads the whole
# batch and blows the ONNX attention matrix (seq² memory) to tens of GB. The router
# embeds a bounded prompt anyway, so the head of the statement is the right signal.
# 4000 chars (~1k tokens) keeps ~all difficulty-bearing content — median Verified
# statement is ~1.2k chars, p90 ~3.2k — while a small embed batch (below) bounds the
# seq² memory. An earlier 600-char cap discarded ~79% of statements and *confounded*
# the held-out difficulty-clustering finding (k=20 corr 0.068 → 0.113 at 4000 chars);
# see docs/benchmark-knn-analysis-2026-07.md.
_MAX_STATEMENT_CHARS = 4000
# Embed batch size: small enough that the ONNX attention seq² matrix stays bounded at
# the 4000-char cap (a 64-wide batch of ~1k-token texts OOMs a CPU box).
_EMBED_BATCH = 8


@dataclass(frozen=True)
class ExternalData:
    """External neighbours: aligned instance ids, embeddings, and tier rates."""

    iids: list[str]
    embeddings: np.ndarray  # (n, dim) float32
    p_cheap: dict[str, float]  # cheap-tier difficulty := p_solve (p_cheap is degenerate)
    p_frontier: dict[str, float]  # frontier-tier rate := p_frontier (else p_solve)


def load_external_priors(path: Path = _PRIOR_PATH) -> dict[str, tuple[float, float]]:
    """{iid: (cheap_rate=p_solve, frontier_rate=p_frontier-else-p_solve)}."""
    # cheap_rate := p_solve, NOT p_cheap: the cheap cohort reports only resolved instances
    # so p_cheap is a degenerate 1.0. p_solve (median 0.86, real hard tail) is the signal.
    out: dict[str, tuple[float, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            solve = row.get("p_solve", "")
            if solve in ("", None):
                continue
            base = float(solve)
            pf = float(row["p_frontier"]) if row.get("p_frontier") else base
            out[row["instance_id"]] = (base, pf)
    return out


def _tier_of(model: str) -> str:
    info = config.load_pricing().get(model)
    return info.get("tier", "cheap") if isinstance(info, dict) else "cheap"


# ---------------------------------------------------------------------------
# Pure selection core (embedding-free — unit-testable on its own)
# ---------------------------------------------------------------------------
def blend_scores(
    our_neighbors: list[dict],
    ext_neighbors: list[tuple[float, float]],
    models: list[str],
    tiers: dict[str, str],
    external_weight: float,
) -> dict[str, tuple[float, int]]:
    """Weighted success-rate + neighbour count per model."""
    # our_neighbors: matrix['results'][nid] dicts (bool pass, weight 1.0). ext_neighbors:
    # (cheap_rate, frontier_rate) tuples — the external tier signal at external_weight.
    scores: dict[str, tuple[float, int]] = {}
    for model in models:
        wsum = 0.0
        wsucc = 0.0
        n = 0
        for res in our_neighbors:
            outcome = res.get(model)
            if outcome is not None:
                wsum += 1.0
                wsucc += 1.0 if outcome.get("pass") else 0.0
                n += 1
        rate_idx = 0 if tiers.get(model, "cheap") == "cheap" else 1
        for pair in ext_neighbors:
            wsum += external_weight
            wsucc += external_weight * pair[rate_idx]
            n += 1
        if wsum > 0:
            scores[model] = (wsucc / wsum, n)
    return scores


def select_model(
    scores: dict[str, tuple[float, int]],
    by_cost: list[str],
    threshold: float,
    min_samples: int,
) -> str:
    """Cheapest model clearing the success threshold (with enough samples); else
    cheapest clearing it at any sample count; else the cheapest model overall.
    """
    eligible = [m for m in by_cost if _clears(scores, m, threshold, min_samples)]
    if eligible:
        return eligible[0]
    relaxed = [m for m in by_cost if scores.get(m, (0.0, 0))[0] >= threshold]
    if relaxed:
        return relaxed[0]
    return by_cost[0]


def _clears(
    scores: dict[str, tuple[float, int]], m: str, threshold: float, min_samples: int
) -> bool:
    rate, n = scores.get(m, (0.0, 0))
    return rate >= threshold and n >= min_samples


# ---------------------------------------------------------------------------
# Default external provider: CSV + HF problem statements + cached embeddings
# ---------------------------------------------------------------------------
def _default_external_provider(exclude: set[str]) -> ExternalData:
    """Load the ~490 external instances (minus ``exclude``), embed offline, cache."""
    from benchmark.routing.strategies.knn import _embed_texts

    priors = load_external_priors()
    iids = sorted(i for i in priors if i not in exclude)
    statements = _load_problem_statements(iids)
    keep = [i for i in iids if i in statements]
    embeddings = _cached_embeddings(keep, statements, _embed_texts)
    return ExternalData(
        iids=keep,
        embeddings=embeddings,
        p_cheap={i: priors[i][0] for i in keep},
        p_frontier={i: priors[i][1] for i in keep},
    )


def _load_problem_statements(iids: list[str]) -> dict[str, str]:
    """Problem statements for ``iids`` from the HF Verified dataset (offline cache)."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    from benchmark.runner import swebench_specs

    ds = load_dataset(swebench_specs.DATASET_NAME, split=swebench_specs.DATASET_SPLIT)
    # Columnar access — never materialise full rows (the patch/test_patch blobs are
    # large and would balloon memory for all 500 instances).
    want = set(iids)
    ids = ds["instance_id"]
    statements = ds["problem_statement"]
    return {
        str(i): str(s)[:_MAX_STATEMENT_CHARS]
        for i, s in zip(ids, statements, strict=True)
        if str(i) in want
    }


def _cached_embeddings(
    iids: list[str], statements: dict[str, str], embed: Callable[[list[str]], np.ndarray]
) -> np.ndarray:
    """Embed ``iids`` (in order), caching to a gitignored npz keyed by (ids, cap)."""
    import hashlib

    # Key on the id set AND the truncation cap: a cap change alters every embedding,
    # so it MUST bust the cache (keying on ids alone silently served stale 600-char
    # vectors after the cap was raised).
    key = hashlib.sha256(f"{_MAX_STATEMENT_CHARS}\n".join(iids).encode()).hexdigest()[:16]
    cache = _CACHE_DIR / f"external_emb_{key}.npz"
    if cache.exists():
        return np.load(cache)["emb"].astype(np.float32)
    # Chunk the embed so peak memory stays flat regardless of instance count.
    step = _EMBED_BATCH
    chunks = [embed([statements[i] for i in iids[j : j + step]]) for j in range(0, len(iids), step)]
    emb = np.vstack(chunks).astype(np.float32) if chunks else np.empty((0, 0), np.float32)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb)
    return emb


class kNNBlended(Strategy):  # noqa: N801 (kNN is the established algorithm name)
    """kNN over our verified runs ∪ external Verified priors, external neighbours down-weighted."""

    def __init__(
        self,
        k: int = 20,
        success_rate_threshold: float = 0.5,
        min_samples: int = 1,
        external_weight: float = 0.25,
        external_provider: Callable[[set[str]], ExternalData] | None = None,
    ):
        self._k = k
        self._threshold = success_rate_threshold
        self._min_samples = min_samples
        self._external_weight = external_weight
        self._provider = external_provider or _default_external_provider
        self._ready = False
        self._index = None
        self._nodes: list[tuple[str, bool]] = []  # (id, is_external) aligned to index labels
        self._external: ExternalData | None = None

    @property
    def name(self) -> str:
        return "kNN-Blended"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        pricing = _model_pricing(matrix)
        if not pricing:
            return "deepseek-v4-flash"
        by_cost = sorted(pricing, key=lambda m: pricing[m])
        if not matrix.get("results"):
            return by_cost[0]
        if not self._ready:
            self._build(matrix)
        return self._route(task_id, task_meta, matrix, by_cost)

    # ------------------------------------------------------------------
    def _build(self, matrix: dict) -> None:
        from benchmark.routing.strategies.knn import _embed_texts

        our_ids = sorted(matrix.get("results", {}).keys())
        # Truncate to the SAME cap as external statements so both sources occupy the
        # same embedding sub-region — otherwise a length/style shift lets kNN prefer
        # same-source neighbours and quietly weakens the cross-source blend.
        descriptions = [
            str(matrix["tasks"].get(t, {}).get("description", t))[:_MAX_STATEMENT_CHARS]
            for t in our_ids
        ]
        our_emb = _embed_texts(descriptions)
        self._our_lookup = dict(zip(our_ids, list(our_emb), strict=True))

        external = self._provider(set(our_ids))
        self._external = external

        all_emb = np.vstack([our_emb, external.embeddings]) if len(external.iids) else our_emb
        self._nodes = [(i, False) for i in our_ids] + [(i, True) for i in external.iids]
        self._index = _build_hnsw(all_emb)
        self._ready = True

    def _route(self, task_id: str, task_meta: dict, matrix: dict, by_cost: list[str]) -> str:
        emb = self._our_lookup.get(task_id)
        if emb is None:
            from benchmark.routing.strategies.knn import _embed_texts

            emb = _embed_texts([str(task_meta.get("description", task_id))[:_MAX_STATEMENT_CHARS]])[
                0
            ]
        our_nb, ext_nb = self._neighbors(emb, task_id, matrix)
        tiers = {m: _tier_of(m) for m in by_cost}
        scores = blend_scores(our_nb, ext_nb, by_cost, tiers, self._external_weight)
        return select_model(scores, by_cost, self._threshold, self._min_samples)

    def _neighbors(
        self, emb: np.ndarray, task_id: str, matrix: dict
    ) -> tuple[list[dict], list[tuple[float, float]]]:
        assert self._index is not None and self._external is not None
        n_total = len(self._nodes)
        labels, distances = self._index.knn_query(emb.reshape(1, -1), min(self._k + 1, n_total))
        our_nb: list[dict] = []
        ext_nb: list[tuple[float, float]] = []
        for label, dist in zip(labels[0], distances[0], strict=True):
            if len(our_nb) + len(ext_nb) >= self._k:
                break
            nid, is_external = self._nodes[label]
            if float(dist) < 0.001 or nid == task_id:  # self-exclusion (leave-one-out)
                continue
            if is_external:
                ext_nb.append((self._external.p_cheap[nid], self._external.p_frontier[nid]))
            else:
                our_nb.append(matrix["results"].get(nid, {}))
        return our_nb, ext_nb


def _build_hnsw(embeddings: np.ndarray):
    import hnswlib

    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
    # num_threads=1: multi-threaded add_items builds a non-deterministic neighbour
    # graph (results wobble at ~1e-4 run to run). Single-threaded pins the index so
    # every regeneration — metrics and plots — is bit-reproducible.
    index.add_items(embeddings, np.arange(len(embeddings)), num_threads=1)
    index.set_ef(max(50, len(embeddings) // 4))
    return index
