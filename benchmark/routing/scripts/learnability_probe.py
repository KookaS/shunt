#!/usr/bin/env python3
"""Learnability probe — can a text/metadata predictor learn deepseek-v4-flash failure?"""

# The one thing left to decide on the routing line: on the measured corpus, is there ANY
# deployable input a router could learn to predict "will deepseek-v4-flash fail this task?"
# better than chance — i.e. is the difficulty signal strong enough to beat the
# verification-based Session-Cascade at equal quality? Four prior results say likely NO
# (judge difficulty R2 ~ +0.05 vs embedding -0.04 and human tag +0.11; a perfect binary
# gate has zero headroom over Session-Cascade; learned single-shot routing ~= simple
# baselines in the literature; run-consistency signal marginal). This probe makes the
# question definitive with a null AND a positive control on every pipeline it quotes.
#
# METHOD
#   y = pass(deepseek-v4-flash) on the measured, uncensored default-arm cells (190 tasks;
#       imputed cells are never used for y).
#   Feature sets (all deployable inputs except the oracle):
#     E1  embedding  — cosine similarity of the committed real jina embeddings
#                      (jinaai/jina-embeddings-v2-base-code, dim 768, from the seed bundle),
#                      scored by leave-one-task-out kNN (the shipped mechanism); plus a
#                      logistic regression on the top-20 principal components.
#     E2  metadata   — log text length, FAIL_TO_PASS / PASS_TO_PASS test counts, distinct
#                      test-file count, repo identity; logistic regression on standardized
#                      features and a kNN over the standardized numeric features.
#     E3  ORACLE     — the human 3-level difficulty tag (easy/medium/hard) from
#                      challenges.json: the ceiling on what a predictor could achieve.
#   Models: a low-capacity logistic regression (L2, standardized, leave-one-out) and the
#   kNN rule, both evaluated leave-one-task-out. Reported per feature set: LOO R2 and AUC
#   against y, with a 200-outcome-permutation null band and its z-score.
#   Controls (mandatory): a planted feature that is exactly y (and a noisy transform of it)
#   must be recovered by the same pipeline with a large margin over the null; E3, when it
#   clears its own null, is the second (real) positive control. Without a firing control any
#   null is a coverage gap, not a falsification.
#
# EMBEDDER STATUS — REAL, NOT A PROXY. E1 uses the seed bundle committed under
# benchmark/routing/data/seed/: embeddings produced by the shipped
# jinaai/jina-embeddings-v2-base-code model on the committed routing text. No synthetic
# feature space is substituted for it, and every task's bundle text is asserted to equal
# `routing_text()` before a single number is quoted.
#
# COST FRAMING mirrors predict_then_cascade_eval's accounting on the completed matrix: a
# thresholded score gate routes predicted-cheap tasks cheap-direct and everything else
# through the Session-Cascade ladder, scored on Session-Cascade's own fixed 142-task set.
# The reference points (Session-Cascade f=0: 100% at ~$1.29; oracle gate: identical) are
# read off the gitignored/regenerable predict_then_cascade_curve.csv.
#
# Real-only: everything reads committed data (results.csv, challenges.json, the seed
# bundle, the challenge spec files). No live API calls, no fabricated y.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from benchmark import config
from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.routing import impute, summary
from benchmark.routing.scripts import knn_nulls
from benchmark.routing.sensitivity import auroc
from benchmark.routing.strategies import routing_text

MODEL: Final[str] = "deepseek-v4-flash"
DEFAULT_K: Final[int] = 20
K_GRID: Final[tuple[int, ...]] = (2, 5, 10, 20, 40, 80, 189)
N_PERM: Final[int] = 200
SEED: Final[int] = 0
# Low-capacity logistic: L2 penalty at C=1.0 on standardized features.
L2_C: Final[float] = 1.0
# Embedding -> LR encoding: the top 20 principal components, fit ONCE (unsupervised and
# outcome-blind) so the LOO folds and the permutation null share one fixed projection.
PCA_COMPONENTS: Final[int] = 20
# A strong-but-noisy planted transform: y + Gaussian noise of this sd. The exact-y plant
# is the definitive instrument proof; the noisy one shows graded recovery.
NOISY_PLANT_SD: Final[float] = 0.75
DEFAULT_ARTIFACT: Final[Path] = Path("benchmark/routing/artifacts/learnability_probe_metrics.json")
SEED_DIR: Final[Path] = Path("benchmark/routing/data/seed")

# The empty design-matrix case (pure categorical sets): LR needs a (n, 0) numeric block.
_ZERO_COLS: Final[int] = 0


@dataclass(frozen=True)
class ProbeCorpus:
    """The measured tasks, their real embeddings, and the metadata a router would hold."""

    tids: list[str]
    y: np.ndarray
    embeddings: np.ndarray
    sims: np.ndarray
    log_len: np.ndarray
    n_f2p: np.ndarray
    n_p2p: np.ndarray
    n_files: np.ndarray
    repos: np.ndarray
    difficulty: np.ndarray
    difficulty_onehot: np.ndarray
    pass_rate: float


@dataclass(frozen=True)
class FeatureSet:
    """One (features, model) pair: what the instrument scores and the model that scores it."""

    key: str
    label: str
    kind: str  # deployable | oracle | control
    model: str  # knn | lr
    sims: np.ndarray | None = None
    X: np.ndarray | None = None
    note: str = ""


@dataclass(frozen=True)
class KnnResult:
    """kNN LOO scores for one feature set: per-k R2 with a shared-permutation null."""

    r2: float  # headline k = DEFAULT_K
    auc: float
    r2_null: knn_nulls.Band
    auc_null: knn_nulls.Band
    r2_grid: tuple[float, ...]
    k_grid: tuple[int, ...]
    per_k_null: tuple[knn_nulls.Band, ...]
    max_null: knn_nulls.Band
    max_r2: float

    @property
    def z(self) -> float:
        return self.r2_null.z(self.r2)

    @property
    def inside_null(self) -> bool:
        return self.r2_null.contains(self.r2)


@dataclass(frozen=True)
class LrResult:
    """Logistic-regression LOO scores for one feature set."""

    r2: float
    auc: float
    r2_null: knn_nulls.Band
    auc_null: knn_nulls.Band

    @property
    def z(self) -> float:
        return self.r2_null.z(self.r2)

    @property
    def inside_null(self) -> bool:
        return self.r2_null.contains(self.r2)


@dataclass(frozen=True)
class CostRow:
    """One threshold gate operating point on Session-Cascade's fixed scored set."""

    threshold: float
    frac_cheap: float
    n_cheap_direct: int
    n_scored: int
    n_pass: int
    total_cost: float
    fixed_pass: int
    fixed_n: int
    fixed_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": round(float(self.threshold), 4),
            "frac_cheap": round(float(self.frac_cheap), 3),
            "n_cheap_direct": float(self.n_cheap_direct),
            "n_scored": float(self.n_scored),
            "n_pass": float(self.n_pass),
            "total_cost": round(float(self.total_cost), 4),
            "fixed_pass": float(self.fixed_pass),
            "fixed_n": float(self.fixed_n),
            "fixed_rate": round(float(self.fixed_rate), 3),
        }


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _perm_seed(key: str) -> int:
    """Deterministic per-feature-set permutation seed, from the feature key."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) + SEED


def _task_digest(task_id: str) -> str:
    """The 12-hex task key used in the committed seed bundle's cell ids."""
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:_DIGEST_LEN]


_DIGEST_LEN: Final[int] = 12


def _load_seed_bundle() -> dict[str, np.ndarray]:
    """The committed real-embedding bundle: its npz arrays, materialized."""
    manifest_path = SEED_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"no seed-bundle manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if len(manifest) != 1:
        raise RuntimeError(f"expected exactly one seed-bundle entry; found {list(manifest)}")
    entry = next(iter(manifest.values()))
    path = SEED_DIR / str(entry["file"])
    if not path.exists():
        raise RuntimeError(f"seed bundle {path} is missing")
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data}


def _assert_bundle_fresh(manifest: dict) -> None:
    """Fail loudly if the committed bundle was built from a different results/challenges."""
    entry = next(iter(manifest.values()))
    results_digest = entry.get("results_digest")
    if results_digest is not None and results_digest != _file_sha256(config.results_csv_path()):
        raise RuntimeError(
            "seed bundle is stale: results.csv changed since it was embedded. Rebuild with "
            "'make seed-bundle' before trusting its vectors as the real embedder's output."
        )
    chal_digest = entry.get("challenges_digest")
    if chal_digest is not None and chal_digest != _file_sha256(config.challenges_path()):
        raise RuntimeError(
            "seed bundle is stale: challenges.json changed since it was embedded. Rebuild "
            "with 'make seed-bundle' before trusting its vectors as the real embedder's output."
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec_counts(tid: str) -> tuple[int, int, int]:
    """(n FAIL_TO_PASS, n PASS_TO_PASS, n distinct test files) from the challenge spec."""
    spec_path = config.challenge_dir() / f"{tid}.json"
    if not spec_path.exists():
        return (0, 0, 0)
    spec = json.loads(spec_path.read_text())
    f2p = [str(x) for x in spec.get("FAIL_TO_PASS") or []]
    p2p = [str(x) for x in spec.get("PASS_TO_PASS") or []]
    files = {x.split("::")[0] for x in f2p}
    return len(f2p), len(p2p), len(files)


def _difficulty_onehot(tags: np.ndarray) -> np.ndarray:
    """One-hot (n, 3) of the easy/medium/hard tag; an unseen tag maps to a zero row."""
    out = np.zeros((len(tags), 3), dtype=float)
    for i, tag in enumerate(tags):
        col = {"easy": 0, "medium": 1, "hard": 2}.get(str(tag))
        if col is not None:
            out[i, col] = 1.0
    return out


def _repo_onehot(repos: np.ndarray) -> np.ndarray:
    """One-hot (n, n_repos) of the task's source repository."""
    uniq = sorted(set(str(r) for r in repos))
    index = {r: i for i, r in enumerate(uniq)}
    out = np.zeros((len(repos), len(uniq)), dtype=float)
    for i, r in enumerate(repos):
        out[i, index[str(r)]] = 1.0
    return out


def load_probe_corpus(config_path: str) -> ProbeCorpus:
    """The measured deepseek tasks + their real embeddings and deployable metadata."""
    config.load(config_path)
    challenges = config.load_challenges()
    tasks_meta = challenges.get("tasks", {})
    results = config.load_results()
    cells = config.flatten_default_arm(results)

    tids: list[str] = []
    for cid in sorted(cells):
        row = cells[cid].get(MODEL)
        if row is None:
            continue
        if impute.is_non_observation(row):
            continue
        tids.append(cid)
    if not tids:
        raise RuntimeError(f"no measured, uncensored {MODEL} default-arm cells found")

    y = np.array([1.0 if cells[t][MODEL]["pass"] else 0.0 for t in tids], dtype=float)

    bundle = _load_seed_bundle()
    _assert_bundle_fresh(json.loads((SEED_DIR / "manifest.json").read_text()))
    dig2tid = {_task_digest(t): t for t in tids}
    emb_by_task: dict[str, np.ndarray] = {}
    text_by_task: dict[str, str] = {}
    for i, cell in enumerate(bundle["cell_id"]):
        tid12 = str(cell).rsplit(":", 1)[0].removeprefix("bench:")
        if tid12 in dig2tid:
            tid = dig2tid[tid12]
            emb_by_task[tid] = bundle["embedding"][i]
            text_by_task[tid] = str(bundle["routing_text"][i])
    missing = [t for t in tids if t not in emb_by_task]
    if missing:
        raise RuntimeError(
            f"seed bundle covers only {len(emb_by_task)}/{len(tids)} y-tasks; "
            f"missing {sorted(missing)[:5]} — corpus and bundle are out of step"
        )
    for tid in tids:
        expected = routing_text(tid, tasks_meta.get(tid, {})).strip()
        if expected != text_by_task[tid].strip():
            raise RuntimeError(
                f"routing_text() no longer returns the text the seed bundle embedded for "
                f"{tid!r} — the embedding channel drifted from the committed vectors"
            )

    embeddings = np.array([emb_by_task[t] for t in tids], dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    sims = embeddings @ embeddings.T

    lengths = np.array([len(text_by_task[t]) for t in tids], dtype=float)
    log_len = np.log1p(lengths)
    spec = np.array([_spec_counts(t) for t in tids], dtype=float)
    n_f2p, n_p2p, n_files = spec[:, 0], spec[:, 1], spec[:, 2]
    repos = np.array([str(tasks_meta.get(t, {}).get("repo") or knn_nulls.repo_of(t)) for t in tids])
    diff_tags = np.array([str(tasks_meta.get(t, {}).get("difficulty_stratum") or "") for t in tids])
    diff_ord = np.array(
        [{"easy": 0, "medium": 1, "hard": 2}.get(str(x), 1.0) for x in diff_tags], dtype=float
    )

    return ProbeCorpus(
        tids=tids,
        y=y,
        embeddings=embeddings,
        sims=sims,
        log_len=log_len,
        n_f2p=n_f2p,
        n_p2p=n_p2p,
        n_files=n_files,
        repos=repos,
        difficulty=diff_ord,
        difficulty_onehot=_difficulty_onehot(diff_tags),
        pass_rate=float(y.mean()),
    )


# ---------------------------------------------------------------------------
# Statistics — leave-one-out scoring for the kNN rule and the logistic model.
# ---------------------------------------------------------------------------


def knn_loo_scores(sims: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out neighbour-mean predictions of *y* from a similarity matrix."""
    n = len(y)
    masked = np.array(sims, dtype=float, copy=True)
    np.fill_diagonal(masked, -np.inf)
    k_eff = max(1, min(int(k), n - 1))
    nbrs = np.argsort(-masked, axis=1)[:, :k_eff]
    return y[nbrs].mean(axis=1)


def r2_of(y: np.ndarray, pred: np.ndarray) -> float:
    """Out-of-sample R2 of *pred* against *y* (a constant predictor scores 0)."""
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((y - pred) ** 2).sum()) / ss_tot


def logistic_loo_predictions(x: np.ndarray, y: np.ndarray, *, l2: float = L2_C) -> np.ndarray:
    """Leave-one-out predicted probabilities from a ridge logistic regression.

    Features are standardized on each training fold (never on the held-out row), and the
    model is the low-capacity L2-regularized kind a router could actually train.
    """
    n = len(y)
    preds = np.empty(n, dtype=float)
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        mu = x[train].mean(axis=0)
        sd = x[train].std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(C=l2, solver="liblinear", max_iter=2000, random_state=SEED).fit(
            (x[train] - mu) / sd, y[train]
        )
        if clf.classes_.size < 2:
            preds[i] = float(y[train].mean())
            continue
        preds[i] = float(clf.predict_proba((x[i : i + 1] - mu) / sd)[0, 1])
    return preds


def _auc(pred: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC of a continuous score against binary *y* (0.5 = chance)."""
    value = auroc(pred, y.astype(int))
    return value if value == value else 0.5  # NaN (single class) reads as chance


def _summary(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return r2_of(y, pred), _auc(pred, y)


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------


def _numeric_sims(x: np.ndarray) -> np.ndarray:
    """Similarity = negative standardized euclidean distance on a numeric feature matrix."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    return -d2


def _same_tag_sims(tags: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Similarity = 1 when two tasks share the tag, plus a 1e-3 tiebreak for determinism."""
    return (tags[:, None] == tags[None, :]).astype(float) + 1e-3 * tiebreak


def _planted_sims(y: np.ndarray) -> np.ndarray:
    """The exact-y positive control: two tasks are 'similar' when their outcome agrees."""
    return (y[:, None] == y[None, :]).astype(float)


def build_feature_sets(corpus: ProbeCorpus) -> list[FeatureSet]:
    """Every (features, model) pair the probe scores, including the controls."""
    numeric_meta = np.column_stack([corpus.log_len, corpus.n_f2p, corpus.n_p2p, corpus.n_files])
    meta_lr = np.column_stack([numeric_meta, _repo_onehot(corpus.repos)])

    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(corpus.embeddings)
    emb_pc = pca.transform(corpus.embeddings)

    y = corpus.y
    # SH008 noqa: this seeded draw builds a PLANTED positive-control feature (y + noise),
    # not an embedding. The real E1 vectors are the committed seed bundle, never an RNG.
    rng = np.random.default_rng(_perm_seed("plant-noisy"))
    noisy = y + NOISY_PLANT_SD * rng.normal(size=len(y))  # noqa: SH008 (planted control)

    return [
        FeatureSet(
            key="e1_emb_knn",
            label="E1 embedding (jina) · kNN",
            kind="deployable",
            model="knn",
            sims=corpus.sims,
            note="REAL embedder: jinaai/jina-embeddings-v2-base-code (dim 768), committed seed",
        ),
        FeatureSet(
            key="e1_emb_lr_pca",
            label="E1 embedding (jina) · LR top-20 PC",
            kind="deployable",
            model="lr",
            X=emb_pc,
            note="REAL embedder vectors, PCA(top-20, unsupervised) -> ridge logistic LOO",
        ),
        FeatureSet(
            key="e2_meta_lr",
            label="E2 metadata · LR",
            kind="deployable",
            model="lr",
            X=meta_lr,
            note="log len, n F2P, n P2P, n files, repo one-hot -> ridge logistic LOO",
        ),
        FeatureSet(
            key="e2_meta_knn",
            label="E2 metadata (numeric) · kNN",
            kind="deployable",
            model="knn",
            sims=_numeric_sims(numeric_meta),
            note="standardized euclidean similarity over log len / F2P / P2P / files",
        ),
        FeatureSet(
            key="e3_tag_knn",
            label="E3 human difficulty tag · kNN",
            kind="oracle",
            model="knn",
            sims=_same_tag_sims(corpus.difficulty, corpus.sims),
            note="ORACLE CEILING: same-tag neighbours (positive control when it fires)",
        ),
        FeatureSet(
            key="e3_tag_lr",
            label="E3 human difficulty tag · LR",
            kind="oracle",
            model="lr",
            X=corpus.difficulty_onehot,
            note="ORACLE CEILING: one-hot easy/medium/hard -> ridge logistic LOO",
        ),
        FeatureSet(
            key="plant_knn",
            label="CONTROL planted y · kNN",
            kind="control",
            model="knn",
            sims=_planted_sims(y),
            note="positive control: similarity IS the outcome",
        ),
        FeatureSet(
            key="plant_lr",
            label="CONTROL planted y · LR",
            kind="control",
            model="lr",
            X=y[:, None],
            note="positive control: the feature is exactly the label",
        ),
        FeatureSet(
            key="plant_noisy_lr",
            label="CONTROL planted y + noise · LR",
            kind="control",
            model="lr",
            X=noisy[:, None],
            note="positive control: y + N(0, 0.75) — graded recovery",
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation with the shuffled-outcome null
# ---------------------------------------------------------------------------


def evaluate_knn(
    sims: np.ndarray, y: np.ndarray, *, n_perm: int = N_PERM, seed: int = SEED
) -> KnnResult:
    """kNN LOO R2/AUC at the shipped k, a k-grid, and the shared-permutation null."""
    n = len(y)
    ks = [k for k in K_GRID if 1 <= k <= n - 1]
    observed = [knn_nulls.loo_r2(sims, y, k) for k in ks]
    headline_k = min(DEFAULT_K, n - 1)
    headline_idx = ks.index(headline_k) if headline_k in ks else ks.index(ks[-1])
    auc_obs = _auc(knn_loo_scores(sims, y, headline_k), y)

    rng = np.random.default_rng(seed)
    draws = np.empty((n_perm, len(ks)), dtype=float)
    auc_draws = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        shuffled = y[rng.permutation(n)]
        for j, k in enumerate(ks):
            draws[p, j] = knn_nulls.loo_r2(sims, shuffled, k)
        auc_draws[p] = _auc(knn_loo_scores(sims, shuffled, headline_k), shuffled)

    return KnnResult(
        r2=float(observed[headline_idx]),
        auc=float(auc_obs),
        r2_null=knn_nulls.band_of(draws[:, headline_idx]),
        auc_null=knn_nulls.band_of(auc_draws),
        r2_grid=tuple(float(v) for v in observed),
        k_grid=tuple(ks),
        per_k_null=tuple(knn_nulls.band_of(draws[:, j]) for j in range(len(ks))),
        max_null=knn_nulls.band_of(draws.max(axis=1)),
        max_r2=float(max(observed)),
    )


def evaluate_lr(
    x: np.ndarray, y: np.ndarray, *, n_perm: int = N_PERM, seed: int = SEED
) -> LrResult:
    """Logistic LOO R2/AUC and the shuffled-outcome null band."""
    pred = logistic_loo_predictions(x, y)
    r2, auc = _summary(pred, y)
    rng = np.random.default_rng(seed)
    r2_draws = np.empty(n_perm, dtype=float)
    auc_draws = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        shuffled = y[rng.permutation(len(y))]
        pr, a = _summary(logistic_loo_predictions(x, shuffled), shuffled)
        r2_draws[p] = pr
        auc_draws[p] = a
    return LrResult(
        r2=float(r2),
        auc=float(auc),
        r2_null=knn_nulls.band_of(r2_draws),
        auc_null=knn_nulls.band_of(auc_draws),
    )


def destroyed_signal_score(sims: np.ndarray, y: np.ndarray, *, k: int, seed: int) -> float:
    """The headline pipeline's LOO R2 on ONE fresh shuffle of *y* — destroyed-signal leg.
    A single permutation breaks the description->outcome link, so this is an honest draw from the
    instrument's own null, not its aggregate mean — which would make the at-chance leg true."""
    rng = np.random.default_rng(seed)
    shuffled = y[rng.permutation(len(y))]
    return float(knn_nulls.loo_r2(sims, shuffled, k))


# ---------------------------------------------------------------------------
# Cost framing — thresholded score gate vs Session-Cascade on the fixed scored set.
# ---------------------------------------------------------------------------


class _ScoreThresholdGate:
    """A ``Gate``-shaped cheap-direct decision: score above a threshold goes cheap."""

    def __init__(self, scores: dict[str, float], threshold: float) -> None:
        self._scores = scores
        self._threshold = threshold

    def decides_cheap(self, task_id: str, task_meta: dict) -> bool:
        del task_meta
        return self._scores.get(task_id, float("-inf")) >= self._threshold


def _session_cascade_baseline(
    completed: dict, tasks: list[str]
) -> tuple[dict[str, tuple[bool, float]], set[str], set[str]]:
    """Session-Cascade's (decisions, unscorable, scored-set) on the completed matrix."""
    from benchmark.routing.strategies.session_cascade import SessionCascadeStrategy

    decisions, unscorable = summary.evaluate(SessionCascadeStrategy(), completed, tasks)
    fixed_set = set(tasks) - unscorable
    by_tid = {tid: (passed, cost) for tid, _m, passed, cost in decisions}
    return by_tid, unscorable, fixed_set


def cost_row(
    scores: dict[str, float],
    threshold: float,
    completed: dict,
    tasks: list[str],
    fixed_set: set[str],
) -> CostRow:
    """One operating point: the fraction sent cheap-direct and its cost/quality on the fixed set."""
    from benchmark.routing.strategies.predict_then_cascade import PredictThenCascadeStrategy

    gate = _ScoreThresholdGate(scores, threshold)
    strategy = PredictThenCascadeStrategy(gate=gate, label="PTC learnability-gate")
    decisions, unscorable = summary.evaluate(strategy, completed, tasks)
    by_tid = {tid: passed for tid, _m, passed, _c in decisions}
    scored = [tid for tid in fixed_set if tid not in unscorable]
    fixed_pass = sum(1 for tid in scored if by_tid.get(tid, False))
    n_cheap = sum(1 for tid in scored if gate.decides_cheap(tid, {}))
    total = sum(cost for _t, _m, _p, cost in decisions if _t in fixed_set)
    return CostRow(
        threshold=threshold,
        frac_cheap=n_cheap / len(scored) if scored else 0.0,
        n_cheap_direct=n_cheap,
        n_scored=len(scored),
        n_pass=fixed_pass,
        total_cost=total,
        fixed_pass=fixed_pass,
        fixed_n=len(scored),
        fixed_rate=100.0 * fixed_pass / len(scored) if scored else 0.0,
    )


def cost_curve(
    scores: dict[str, float],
    completed: dict,
    tasks: list[str],
    fixed_set: set[str],
) -> list[CostRow]:
    """The gate's (cost, quality) curve as the threshold sweeps cheap-direct fraction."""
    values = np.array(sorted(scores[t] for t in tasks if t in scores), dtype=float)
    if values.size == 0:
        return []
    rows: list[CostRow] = []
    for frac in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        threshold = float(np.quantile(values, 1.0 - frac)) if frac < 1.0 else float(values.min())
        if frac == 0.0:
            threshold = float(values.max()) + 1.0
        rows.append(cost_row(scores, threshold, completed, tasks, fixed_set))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _band_dict(band: knn_nulls.Band) -> dict[str, float]:
    return {
        "mean": round(float(band.mean), 4),
        "sd": round(float(band.sd), 4),
        "lo": round(float(band.lo), 4),
        "hi": round(float(band.hi), 4),
        "n": int(band.n),
    }


def _fmt(v: float) -> str:
    return "    —" if v != v else f"{v:+.3f}"


def _verdict(
    deployables: list[dict], oracle: dict, admissibility: AdmissibilityResult, cost_note: str
) -> tuple[str, str]:
    """CLEAR-NO / MARGINAL / CLEAR-YES plus the one-line justification."""
    if not admissibility.admissible:
        return "INCONCLUSIVE", (
            f"instrument inadmissible ({admissibility.reason}) — no verdict is quotable"
        )
    best_deploy = max(deployables, key=lambda d: d["r2"])
    oracle_r2 = oracle["r2"]
    if oracle_r2 != oracle_r2 or best_deploy["r2"] != best_deploy["r2"]:
        return "INCONCLUSIVE", "a headline statistic is NaN"
    if best_deploy["inside_null"] and (oracle_r2 <= oracle["r2_null"]["hi"]):
        return "CLEAR-NO", (
            f"best deployable {best_deploy['key']} R2 {best_deploy['r2']:+.3f} is inside its null "
            f"[{best_deploy['r2_null']['lo']:+.3f}, {best_deploy['r2_null']['hi']:+.3f}] and the "
            f"oracle ceiling (human tag) {oracle_r2:+.3f} is inside its own null; nothing is "
            f"learnable. {cost_note}"
        )
    if best_deploy["inside_null"]:
        return "CLEAR-NO", (
            f"best deployable {best_deploy['key']} R2 {best_deploy['r2']:+.3f} is inside its null; "
            f"only the ORACLE (human tag, {oracle_r2:+.3f}) clears, and it is not a deployable "
            f"input. {cost_note}"
        )
    if best_deploy["r2"] < 0.05:
        return "MARGINAL", (
            f"best deployable {best_deploy['key']} R2 {best_deploy['r2']:+.3f} clears its null but "
            f"is far below the oracle ceiling {oracle_r2:+.3f}; too weak to matter. {cost_note}"
        )
    return "CLEAR-YES", (
        f"best deployable {best_deploy['key']} R2 {best_deploy['r2']:+.3f} clears its null "
        f"and approaches the oracle ceiling {oracle_r2:+.3f}. {cost_note}"
    )


def _report(corpus: ProbeCorpus, sets: list[FeatureSet], results: dict[str, Any]) -> None:
    print("=== LEARNABILITY PROBE ===")
    print(
        f"target y = pass({MODEL}) on {len(corpus.tids)} measured uncensored default-arm "
        f"tasks; pass rate {corpus.pass_rate:.3f}"
    )
    print(
        "embedder: REAL jinaai/jina-embeddings-v2-base-code (dim 768) from the committed "
        "seed bundle — NOT a proxy"
    )
    print(f"\nadmissibility: {results['admissibility']['headline']}")
    print(f"admissible: {results['admissibility']['admissible']}")

    print("\n--- feature sets (LOO R2 / AUC vs shuffled-outcome null) ---")
    header = f"  {'key':<18} {'R2':>8} {'null[lo,hi]':>18} {'z':>6} {'AUC':>7} "
    header += f"{'aucnull[lo,hi]':>18} {'inNull':>7}"
    print(header)
    for key, res in results["feature_sets"].items():
        if res["model"] == "knn":
            print(
                f"  {key:<18} {_fmt(res['r2']):>8} "
                f"[{res['r2_null']['lo']:+.3f},{res['r2_null']['hi']:+.3f}] "
                f"{res['z']:>6.2f} {res['auc']:>7.3f} "
                f"[{res['auc_null']['lo']:.3f},{res['auc_null']['hi']:.3f}] "
                f"{str(res['inside_null']):>7}"
            )
        else:
            print(
                f"  {key:<18} {_fmt(res['r2']):>8} "
                f"[{res['r2_null']['lo']:+.3f},{res['r2_null']['hi']:+.3f}] "
                f"{res['z']:>6.2f} {res['auc']:>7.3f} "
                f"[{res['auc_null']['lo']:.3f},{res['auc_null']['hi']:.3f}] "
                f"{str(res['inside_null']):>7}"
            )

    k1 = results["feature_sets"].get("e1_emb_knn")
    if k1:
        print("\n  E1 k-grid (R2 per k; max-over-k null band):")
        grid = "  ".join(
            f"k={k}:{v:+.3f}" for k, v in zip(k1["k_grid"], k1["r2_grid"], strict=True)
        )
        print("   ", grid)
        print(
            f"    max-over-k null [{k1['max_null']['lo']:+.3f}, {k1['max_null']['hi']:+.3f}] "
            f"mean {k1['max_null']['mean']:+.3f}"
        )

    print("\n--- positive controls ---")
    for key in ("plant_knn", "plant_lr", "plant_noisy_lr"):
        r = results["feature_sets"].get(key)
        if r:
            print(
                f"  {key:<16} R2 {_fmt(r['r2'])}  AUC {r['auc']:.3f}  "
                f"null[{r['r2_null']['lo']:+.3f},{r['r2_null']['hi']:+.3f}]  z={r['z']:.1f}"
            )
    for key in ("e3_tag_knn", "e3_tag_lr"):
        r = results["feature_sets"].get(key)
        if r:
            print(
                f"  {key:<16} R2 {_fmt(r['r2'])}  AUC {r['auc']:.3f}  "
                f"null[{r['r2_null']['lo']:+.3f},{r['r2_null']['hi']:+.3f}]  "
                f"inNull={r['inside_null']}"
            )

    print("\n--- cost framing vs Session-Cascade (fixed 142-task scored set) ---")
    cf = results["cost_framing"]
    print(
        f"  Session-Cascade: pass {cf['session_cascade']['fixed_rate']:.1f}% at "
        f"${cf['session_cascade']['total_cost']:.4f}"
    )
    print(
        f"  oracle gate:     pass {cf['oracle']['fixed_rate']:.1f}% at "
        f"${cf['oracle']['total_cost']:.4f} — zero headroom"
    )
    for name, curve in cf["curves"].items():
        best = min(curve, key=lambda r: r["total_cost"]) if curve else None
        if best is None:
            print(f"  {name}: curve unavailable")
            continue
        full = [r for r in curve if r["fixed_rate"] >= 100.0]
        cheapest_full = min(full, key=lambda r: r["total_cost"]) if full else None
        if cheapest_full is not None:
            print(
                f"  {name}: cheapest 100%-quality point is the SC point "
                f"(f={cheapest_full['frac_cheap']:.2f}, ${cheapest_full['total_cost']:.4f}); "
                f"cheapest routing point costs ${best['total_cost']:.4f} at "
                f"{best['fixed_rate']:.1f}% quality"
            )
        else:
            print(
                f"  {name}: no 100%-quality point on the curve; "
                f"cheapest routing point costs ${best['total_cost']:.4f} at "
                f"{best['fixed_rate']:.1f}% quality"
            )

    print(f"\nVERDICT: {results['verdict']['label']} — {results['verdict']['reason']}")
    print("\n--- caveats ---")
    for caveat in results.get("caveats", []):
        print(f"  * {caveat}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="benchmark/benchmark.yaml")
    ap.add_argument("--n-perm", type=int, default=N_PERM, help=f"permutations (default {N_PERM})")
    ap.add_argument(
        "--out",
        default=str(DEFAULT_ARTIFACT),
        help="output JSON path "
        "(default: benchmark/routing/artifacts/learnability_probe_metrics.json)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"headline kNN k (default {DEFAULT_K})",
    )
    return ap


def main() -> int:
    args = _arg_parser().parse_args()
    corpus = load_probe_corpus(args.config)
    sets = build_feature_sets(corpus)

    feature_sets: dict[str, dict[str, Any]] = {}
    for fs in sets:
        if fs.model == "knn":
            if fs.sims is None:
                raise RuntimeError(f"kNN feature set {fs.key} has no similarity matrix")
            res = evaluate_knn(fs.sims, corpus.y, n_perm=args.n_perm, seed=_perm_seed(fs.key))
            feature_sets[fs.key] = {
                "key": fs.key,
                "label": fs.label,
                "kind": fs.kind,
                "model": fs.model,
                "note": fs.note,
                "r2": res.r2,
                "auc": res.auc,
                "r2_null": _band_dict(res.r2_null),
                "auc_null": _band_dict(res.auc_null),
                "z": res.z,
                "inside_null": res.inside_null,
                "k": args.k,
                "k_grid": list(res.k_grid),
                "r2_grid": list(res.r2_grid),
                "per_k_null": [_band_dict(b) for b in res.per_k_null],
                "max_null": _band_dict(res.max_null),
                "max_r2": res.max_r2,
            }
        else:
            if fs.X is None:
                raise RuntimeError(f"LR feature set {fs.key} has no design matrix")
            lr_res = evaluate_lr(fs.X, corpus.y, n_perm=args.n_perm, seed=_perm_seed(fs.key))
            feature_sets[fs.key] = {
                "key": fs.key,
                "label": fs.label,
                "kind": fs.kind,
                "model": fs.model,
                "note": fs.note,
                "r2": lr_res.r2,
                "auc": lr_res.auc,
                "r2_null": _band_dict(lr_res.r2_null),
                "auc_null": _band_dict(lr_res.auc_null),
                "z": lr_res.z,
                "inside_null": lr_res.inside_null,
            }

    e1 = feature_sets["e1_emb_knn"]
    planted = feature_sets["plant_knn"]
    # The instrument's chance band is its OWN shuffled-outcome null on the real features:
    # a kNN-mean predictor of a binary outcome scores NEGATIVE R2 in expectation (the
    # neighbourhood adds variance beyond the constant), so the analytic chance level is not
    # 0. The destroyed-signal leg is a REAL pipeline observation: one fresh deterministic
    # shuffle of y scored by the same kNN instrument. It must land inside that band — the
    # previous version used the null's own mean, which made the at-chance condition true
    # by construction and tested nothing. The positive leg must clear the band by a wide
    # margin.
    chance_level = e1["r2_null"]["mean"]
    chance_band = (e1["r2_null"]["hi"] - e1["r2_null"]["lo"]) / 2.0
    destroyed = destroyed_signal_score(
        corpus.sims,
        corpus.y,
        k=min(DEFAULT_K, len(corpus.y) - 1),
        seed=_perm_seed("destroyed-signal"),
    )
    admissibility = admissibility_verdict(
        positive_score=planted["r2"],
        shuffled_score=destroyed,
        chance_level=chance_level,
        chance_band=chance_band,
    )

    cost_framing = _cost_framing(corpus, sets, feature_sets)

    deployable = [feature_sets[k] for k in feature_sets if feature_sets[k]["kind"] == "deployable"]
    oracle = next(feature_sets[k] for k in feature_sets if k.startswith("e3"))
    label, reason = _verdict(deployable, oracle, admissibility, cost_framing["note"])

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "target": {
            "model": MODEL,
            "n_tasks": len(corpus.tids),
            "pass_rate": round(corpus.pass_rate, 4),
            "default_arm": "high",
            "n_perm": args.n_perm,
            "k": args.k,
        },
        "embedder": {
            "real": True,
            "repo": "jinaai/jina-embeddings-v2-base-code",
            "dim": int(corpus.embeddings.shape[1]),
            "proxy": False,
            "source": "committed seed bundle (benchmark/routing/data/seed)",
        },
        "admissibility": {
            "admissible": admissibility.admissible,
            "reason": admissibility.reason,
            "headline": admissibility.headline,
            "positive_score": planted["r2"],
            "shuffled_score": destroyed,
            "chance_level": chance_level,
            "chance_band": round(chance_band, 4),
        },
        "feature_sets": feature_sets,
        "cost_framing": cost_framing,
        "verdict": {"label": label, "reason": reason},
        "caveats": [
            "The real embedder's vectors are the committed seed bundle; every task's bundle text "
            "is asserted to equal routing_text() before the numbers are quoted.",
            "For GROUP-structured leave-one-out predictors (one-hot difficulty, one-hot repo) the "
            "query's own outcome is excluded from its group's score, so within a group positives "
            "are systematically scored one notch below negatives. The AUC null band therefore "
            "centres BELOW 0.5 for e3_tag_lr (and, mildly, for e2_meta_lr); the R2 statistic is "
            "the primary axis and its null is the honest reference either way.",
            "Ten of Session-Cascade's fixed 142 scored tasks have an imputed (never measured) "
            "deepseek cell; the learnability probe excludes them from y, and the cost-framing "
            "gate can only send the 132 measured-deepseek tasks cheap-direct, so the Always-Cheap "
            "end of its curve reads 93% rather than 100%.",
            "All scores are honest leave-one-out; a deployed router would additionally pay the "
            "embedding/latency cost of scoring the task text at routing time.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _report(corpus, sets, payload)
    print(f"\nWrote {out}")
    return 0


def _sc_total(completed: dict, tasks: list[str]) -> float:
    """Total Session-Cascade cost over its scored (fixed) set on the completed matrix."""
    sc_by_tid, _unsc, fixed_set = _session_cascade_baseline(completed, tasks)
    return sum(cost for tid, (_passed, cost) in sc_by_tid.items() if tid in fixed_set)


def _cost_framing(
    corpus: ProbeCorpus, sets: list[FeatureSet], feature_sets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """The threshold-gate cost/quality curves for the best deployable and the oracle."""
    del feature_sets
    matrix = config.load_matrix()
    tasks = config.sample_tasks(
        sorted(matrix["results"].keys()), seed=config.benchmark_params().get("seed", 42)
    )
    completed, _im = summary.complete_scored_matrix(matrix)
    sc_by_tid, _sc_unscorable, fixed_set = _session_cascade_baseline(completed, tasks)
    sc_cost = sum(cost for tid, (_passed, cost) in sc_by_tid.items() if tid in fixed_set)

    def scores_of(key: str) -> dict[str, float]:
        fs = next(f for f in sets if f.key == key)
        if fs.model == "knn":
            if fs.sims is None:
                return {}
            pred = knn_loo_scores(fs.sims, corpus.y, DEFAULT_K)
        else:
            if fs.X is None:
                return {}
            pred = logistic_loo_predictions(fs.X, corpus.y)
        return {tid: float(p) for tid, p in zip(corpus.tids, pred, strict=True)}

    curves: dict[str, list[dict[str, float]]] = {}
    for key in ("e2_meta_lr", "e3_tag_lr"):
        scores = scores_of(key)
        rows = cost_curve(scores, completed, tasks, fixed_set)
        curves[key] = [r.as_dict() for r in rows]

    note = (
        "A perfect binary gate coincides exactly with Session-Cascade (oracle point: 100% at "
        f"${sc_cost:.4f}); the completed-matrix accounting shows any real predictor at equal "
        f"(100%) quality must reproduce the oracle's routing, so its cost is exactly "
        f"Session-Cascade's. Below 100% the frontier trades quality for cost, and the ceiling "
        "a learnable score can reach is set by its AUC (above)."
    )
    return {
        "session_cascade": {"fixed_rate": 100.0, "total_cost": round(sc_cost, 4)},
        "oracle": {"fixed_rate": 100.0, "total_cost": round(sc_cost, 4)},
        "curves": curves,
        "note": note,
    }


if __name__ == "__main__":
    sys.exit(main())
