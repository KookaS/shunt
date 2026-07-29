#!/usr/bin/env python3
"""Null models for the kNN router: does it learn, or does it memorise the base rate?"""

# Everything a degenerate router CAN fake — allocation shape, neighbourhood purity, a
# cost bar — a constant function fakes identically. The only statistics that separate a
# learner from a constant are the ones in this module: an outcome-permutation band, the
# true chance level, and a leave-one-task-out transfer curve whose index can be
# restricted (to another repo, or to everything-but-the-query).
#
# EVERY null here permutes the OUTCOME matrix and re-derives the router's decisions. It
# never permutes the decisions themselves: those are endogenous (neighbouring tasks share
# neighbourhoods, so their picks are correlated with or without signal), and shuffling
# them yields a band that is far too narrow and manufactures significance from nothing.
#
# The selection rule lives here ONCE (`select_from_rates`) and viz_knn.knn_select
# delegates to it, so a null and the figure it nulls can never drift apart.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# A permutation band needs enough draws that its 2.5/97.5 percentiles are stable;
# 200 is the project floor for any permutation null quoted on a figure.
DEFAULT_PERMUTATIONS: Final[int] = 200

# Below this a permutation spread is float noise, not a distribution (see band_of).
_SD_EPSILON: Final[float] = 1e-12

# A diagonal-vs-off-diagonal contrast needs at least two repos to have an off-diagonal.
_MIN_REPOS_FOR_TRANSFER: Final[int] = 2


@dataclass(frozen=True)
class Band:
    """A permutation null: the distribution a statistic takes when outcomes carry no signal."""

    mean: float
    sd: float
    lo: float
    hi: float
    n: int

    def z(self, observed: float) -> float:
        """How many null standard deviations `observed` sits above the null mean."""
        return (observed - self.mean) / self.sd if self.sd > 0 else 0.0

    def contains(self, observed: float) -> bool:
        """True when `observed` falls inside the null's central 95% — i.e. no signal."""
        return self.lo <= observed <= self.hi


def band_of(samples: np.ndarray) -> Band:
    """Summarise permutation draws as mean / sd / central-95% band."""
    arr = np.asarray(samples, dtype=float)
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    # A degenerate null (every draw identical — e.g. k so large that the neighbourhood is
    # the whole corpus) leaves float noise around 1e-16 in the sd. Dividing by that would
    # report a z in the billions for a difference of nothing.
    if sd < _SD_EPSILON:
        sd = 0.0
    return Band(
        mean=float(np.mean(arr)),
        sd=sd,
        lo=float(np.percentile(arr, 2.5)),
        hi=float(np.percentile(arr, 97.5)),
        n=int(arr.size),
    )


# ---------------------------------------------------------------------------
# Base rates — what a constant router scores for free.
# ---------------------------------------------------------------------------


def majority_share(labels: list[str]) -> float:
    """Share of tasks carrying the single most common label (a constant router's purity)."""
    if not labels:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels))


def chance_purity(labels: list[str]) -> float:
    """Purity a random neighbour ordering yields: sum of squared label shares."""
    # THE reference line for a purity bar. Drawing 0.5 instead (a hand-set constant)
    # understates chance by ~0.47 on a lopsided allocation and turns an at-chance result
    # into an apparent triumph.
    if not labels:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    shares = counts / len(labels)
    return float(np.sum(shares**2))


# ---------------------------------------------------------------------------
# Neighbourhood purity + its outcome-permutation null.
# ---------------------------------------------------------------------------


def neighbour_rank(sims: np.ndarray) -> np.ndarray:
    """(n, n-1) neighbour indices per task, most similar first, query always excluded."""
    n = sims.shape[0]
    masked = np.array(sims, dtype=float, copy=True)
    np.fill_diagonal(masked, -np.inf)
    return np.argsort(-masked, axis=1)[:, : max(0, n - 1)]


def mean_purity(rank: np.ndarray, label_ids: np.ndarray, k: int) -> float:
    """Mean share of each task's k nearest neighbours carrying the task's own label."""
    k_eff = min(int(k), rank.shape[1])
    if k_eff <= 0:
        return 0.0
    nbrs = rank[:, :k_eff]
    return float(np.mean(label_ids[nbrs] == label_ids[:, None]))


def purity_null_band(  # noqa: PLR0913
    rank: np.ndarray,
    sims: np.ndarray,
    pass_mat: np.ndarray,
    k: int,
    threshold: float,
    n_perm: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> Band:
    """Purity's null: permute OUTCOME rows, then RE-DERIVE the router's selections."""
    # Permuting the LABEL array instead would be invalid, because these labels are
    # ENDOGENOUS: they are the router's own picks, and neighbouring tasks share
    # neighbourhoods, so their picks are autocorrelated BY CONSTRUCTION whether or not any
    # signal exists. Shuffling labels destroys that induced autocorrelation too, leaving a
    # band far too narrow — measured at 24/30 false positives (mean z=+3.6) on pure-null
    # data. Permuting outcomes and re-running the selection keeps the mechanism intact and
    # breaks only the description->outcome link, which is the thing being tested.
    rng = np.random.default_rng(seed)
    all_idx = np.arange(pass_mat.shape[0])
    draws = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = pass_mat[rng.permutation(pass_mat.shape[0])]
        rates = neighbourhood_rates(sims, shuffled, all_idx, all_idx, k)
        draws[i] = mean_purity(rank, select_from_rates(rates, threshold), k)
    return band_of(draws)


# ---------------------------------------------------------------------------
# The null-analysis selection rule. NOT the shipped product rule — see below.
# ---------------------------------------------------------------------------


def select_from_rates(rates: np.ndarray, threshold: float) -> np.ndarray:
    """Column index of the chosen model per row; `rates` columns MUST be price-ascending.

    An ANALYSIS rule for the permutation nulls, deliberately NOT `router.SelectionRule`.
    """
    # This is a vectorized approximation of the shipped rule, not the shipped rule. It shares
    # the core — cheapest column whose rate clears `threshold` — and differs in three named
    # ways, each forced by the fact that `rates` is a LOSSY aggregate: `neighbourhood_rates`
    # has already collapsed the neighbourhood to one number per (row, model), discarding the
    # per-neighbour distance, cost and confidence that the shipped rule needs.
    #
    #   1. WEIGHTING — `rates` is an unweighted mean over the top-k; `SelectionRule` weights
    #      each neighbour by `confidence * (1 - distance)`.
    #   2. min_samples — no floor here; `SelectionRule` rejects a model with fewer than
    #      `min_samples` neighbours regardless of its rate.
    #   3. FALLBACK — when nothing clears the threshold this returns `argmax` (the
    #      best-scoring model, ties to the cheaper); `SelectionRule._escalate` instead returns
    #      the cheapest UNTESTED model, so on identical evidence the two can disagree — the
    #      shipped rule escalating to a dearer model where this one picks a cheap tested one.
    #
    # Reconstructing per-neighbour objects inside the permutation loop to delegate would cost
    # `n_perm x n_tasks` object builds and would still need distances this matrix no longer
    # carries. The divergence is therefore pinned by tests rather than removed; the shared
    # core and each of the three differences are asserted in `tests/test_knn_nulls.py`.
    qualifies = rates >= threshold
    return np.where(qualifies.any(axis=1), qualifies.argmax(axis=1), rates.argmax(axis=1))


def neighbourhood_rates(
    sims: np.ndarray,
    pass_mat: np.ndarray,
    query_idx: np.ndarray,
    index_idx: np.ndarray,
    k: int,
) -> np.ndarray:
    """(len(query), n_models) neighbourhood pass-rates, voting ONLY over `index_idx` tasks."""
    # Restricting the index is what turns one routing rule into a transfer experiment: pass
    # every other task and it is leave-one-out; pass another repo's tasks and it is
    # cross-repo transfer; include the query itself and it is the memorisation ceiling.
    sub = sims[np.ix_(query_idx, index_idx)]
    self_cell = query_idx[:, None] == index_idx[None, :]
    holds_self = bool(self_cell.any())
    if holds_self:
        sub = np.where(self_cell, -np.inf, sub)
    k_eff = min(int(k), sub.shape[1] - (1 if holds_self else 0))
    if k_eff <= 0:
        return np.zeros((len(query_idx), pass_mat.shape[1]), dtype=float)
    nbrs = index_idx[np.argsort(-sub, axis=1)[:, :k_eff]]
    return pass_mat[nbrs].mean(axis=1)


def routed_pass_rate(
    sims: np.ndarray,
    pass_mat: np.ndarray,
    query_idx: np.ndarray,
    index_idx: np.ndarray,
    k: int,
    threshold: float,
) -> float:
    """Share of query tasks whose ROUTED model actually passed — the deployable score."""
    if len(query_idx) == 0 or len(index_idx) == 0:
        return float("nan")
    rates = neighbourhood_rates(sims, pass_mat, query_idx, index_idx, k)
    chosen = select_from_rates(rates, threshold)
    return float(np.mean(pass_mat[query_idx, chosen]))


# ---------------------------------------------------------------------------
# Leave-one-task-out transfer curve — accuracy vs k against three references.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferCurve:
    """Routing pass-rate vs k, with the three lines that make it falsifiable."""

    ks: tuple[int, ...]
    loo: tuple[float, ...]
    memorisation: tuple[float, ...]
    null_lo: tuple[float, ...]
    null_hi: tuple[float, ...]
    null_mean: tuple[float, ...]
    null_sd: tuple[float, ...]
    # The null of max-over-k, not of one k. The blue line's best point is SELECTED across
    # ~7 values of k, so testing it against its own per-k band double-counts the search and
    # roughly doubles the false-positive rate. Every permutation is shared across k (one
    # shuffle scored at every k), so this costs nothing extra to compute.
    max_null: Band
    best_constant: float
    best_constant_model: str
    n_tasks: int
    n_perm: int

    def band_at(self, i: int) -> Band:
        """The per-k null band at index ``i``, with its exact (not reconstructed) sd."""
        return Band(
            mean=self.null_mean[i],
            sd=self.null_sd[i],
            lo=self.null_lo[i],
            hi=self.null_hi[i],
            n=self.n_perm,
        )


def _permuted_pass_rates(  # noqa: PLR0913
    sims: np.ndarray,
    pass_mat: np.ndarray,
    all_idx: np.ndarray,
    ks: list[int],
    threshold: float,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    """``(n_perm, len(ks))`` routed pass-rates under task-wise shuffles of the outcomes."""
    # Permuting whole ROWS keeps each model's marginal pass rate and the outcome matrix's
    # correlation structure intact, and breaks only the description->outcome link — so the
    # band answers "what does this router score when the embedding says nothing?".
    # ONE shuffle is scored at every k, so a row is directly comparable across k and its
    # row-max is a valid null for the observed max-over-k.
    rng = np.random.default_rng(seed)
    draws = np.empty((n_perm, len(ks)), dtype=float)
    for i in range(n_perm):
        shuffled = pass_mat[rng.permutation(pass_mat.shape[0])]
        for j, k in enumerate(ks):
            draws[i, j] = routed_pass_rate(sims, shuffled, all_idx, all_idx, k, threshold)
    return draws


def transfer_curve(
    sims: np.ndarray,
    pass_mat: np.ndarray,
    models_by_price: list[str],
    ks: list[int],
    threshold: float,
    n_perm: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> TransferCurve:
    """Leave-one-task-out routing pass-rate across k, plus ceiling / constant / null."""
    n = pass_mat.shape[0]
    all_idx = np.arange(n)
    loo = [routed_pass_rate(sims, pass_mat, all_idx, all_idx, k, threshold) for k in ks]
    # Memorisation reference: the query is allowed into its own neighbourhood, so a router
    # that only looks itself up scores its own outcome. The gap between this line and the
    # LOO line is everything the router transfers to unseen tasks.
    ceiling = [_memorisation_rate(sims, pass_mat, k, threshold) for k in ks]
    draws = _permuted_pass_rates(sims, pass_mat, all_idx, ks, threshold, n_perm, seed)
    bands = [band_of(draws[:, j]) for j in range(len(ks))]
    const_rates = pass_mat.mean(axis=0)
    best_i = int(np.argmax(const_rates))
    return TransferCurve(
        ks=tuple(ks),
        loo=tuple(loo),
        memorisation=tuple(ceiling),
        null_lo=tuple(b.lo for b in bands),
        null_hi=tuple(b.hi for b in bands),
        null_mean=tuple(b.mean for b in bands),
        null_sd=tuple(b.sd for b in bands),
        max_null=band_of(draws.max(axis=1)),
        best_constant=float(const_rates[best_i]),
        best_constant_model=models_by_price[best_i],
        n_tasks=n,
        n_perm=n_perm,
    )


def _memorisation_rate(sims: np.ndarray, pass_mat: np.ndarray, k: int, threshold: float) -> float:
    """Routing pass-rate when each task IS in its own index — the memorisation ceiling."""
    n = pass_mat.shape[0]
    k_eff = max(1, min(int(k), n))
    nbrs = np.argsort(-sims, axis=1)[:, :k_eff]
    rates = pass_mat[nbrs].mean(axis=1)
    chosen = select_from_rates(rates, threshold)
    return float(np.mean(pass_mat[np.arange(n), chosen]))


# ---------------------------------------------------------------------------
# Cross-repo transfer — memorisation shows up as a bright diagonal.
# ---------------------------------------------------------------------------


def repo_of(task_id: str) -> str:
    """Source repository carried in a SWE-bench task id (`org__repo-1234` -> `org/repo`)."""
    # Splits on the LAST hyphen, not the first: repo names contain hyphens themselves
    # (`scikit-learn__scikit-learn-10297`, `pytest-dev__pytest-5495`), and splitting on the
    # first one silently collapses them into a bogus `scikit` bucket.
    head, sep, tail = task_id.rpartition("-")
    if not sep or not tail.isdigit():
        head = task_id
    return head.replace("__", "/") if "__" in head else head


@dataclass(frozen=True)
class CrossRepo:
    """Routing pass-rate for every (index repo, query repo) pair, plus its null."""

    repos: tuple[str, ...]
    counts: tuple[int, ...]
    grid: np.ndarray
    diagonal_mean: float
    off_diagonal_mean: float
    null: Band

    @property
    def advantage(self) -> float:
        """Diagonal minus off-diagonal — how much routing gains from a same-repo index."""
        return self.diagonal_mean - self.off_diagonal_mean


def cross_repo_transfer(
    sims: np.ndarray,
    pass_mat: np.ndarray,
    task_ids: list[str],
    k: int,
    threshold: float,
    min_tasks: int = 8,
    n_perm: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> CrossRepo:
    """Route each repo's tasks off each repo's index; a bright diagonal means memorisation."""
    labels = [repo_of(t) for t in task_ids]
    uniq, counts = np.unique(labels, return_counts=True)
    keep = [(r, int(c)) for r, c in zip(uniq, counts, strict=True) if c >= min_tasks]
    repos = [r for r, _ in keep]
    idx_of = {r: np.flatnonzero(np.array(labels) == r) for r in repos}
    if len(repos) < _MIN_REPOS_FOR_TRANSFER:
        # With fewer than two repos there is no off-diagonal, so "diagonal advantage" is
        # undefined; returning NaN would render as a spurious "BELOW the null" verdict.
        raise ValueError(
            f"cross-repo transfer needs at least {_MIN_REPOS_FOR_TRANSFER} repos with "
            f">= {min_tasks} tasks; found {len(repos)}"
        )
    grid = np.full((len(repos), len(repos)), np.nan)
    for qi, q_repo in enumerate(repos):
        for ii, i_repo in enumerate(repos):
            grid[qi, ii] = routed_pass_rate(
                sims, pass_mat, idx_of[q_repo], idx_of[i_repo], k, threshold
            )
    diag = np.diag(grid)
    off = grid[~np.eye(len(repos), dtype=bool)]
    # The null must be built on the SAME statistic the figure reports — the diagonal's
    # advantage over the off-diagonal. Nulling the diagonal against a whole-corpus routing
    # null instead compares two different quantities and can manufacture a z from nothing.
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = pass_mat[rng.permutation(pass_mat.shape[0])]
        null_grid = np.array(
            [
                [
                    routed_pass_rate(sims, shuffled, idx_of[q], idx_of[j], k, threshold)
                    for j in repos
                ]
                for q in repos
            ]
        )
        n_diag = np.diag(null_grid)
        n_off = null_grid[~np.eye(len(repos), dtype=bool)]
        draws[i] = float(np.nanmean(n_diag)) - float(np.nanmean(n_off))
    return CrossRepo(
        repos=tuple(repos),
        counts=tuple(c for _, c in keep),
        grid=grid,
        diagonal_mean=float(np.nanmean(diag)),
        off_diagonal_mean=float(np.nanmean(off)) if off.size else float("nan"),
        null=band_of(draws),
    )
