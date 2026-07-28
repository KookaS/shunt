"""The causal escalation eval: predict the TASK outcome from a prefix, conditional on the prior."""

# WHAT THIS REPLACES. The old harness graded a binary {0,1} policy flag against a POSITIONAL label
# ("this step is one of the last H of a run that failed"). Three audits proved that metric cannot
# see a detector: a perfect task-level oracle capped at AUROC 0.757, a content-free clock scored
# 0.970, and AUROC was pinned at ~0.503 for every H in {1,3,5,10}.
#
# WHAT THIS DOES INSTEAD. One row per trajectory. Label = "this attempt ultimately failed" — the
# question escalation exists to answer. Score = a calibrated probability from prefix-only features
# at a fixed decision depth (see features.py for the two structural anti-leak rules). Skill is
# reported THREE ways because only the third is honest:
#
#   auroc_prior     — the router's own t=0 knowledge (leave-one-out per-challenge failure rate).
#                     Task identity alone is a strong predictor, so any unconditional number is
#                     mostly a rediscovery of task difficulty.
#   auroc_prefix    — grouped-CV out-of-fold discrimination from the prefix alone. Grouped by
#                     challenge, so a challenge never appears in both train and test.
#   incremental     — auroc(prior + prefix) - auroc(prior). THE number. Everything else overstates.
#
# Every headline carries a label-permutation null (>= 200 shuffles, the whole pipeline re-run per
# shuffle) and a challenge-level bootstrap CI. "Beats chance" means "above the null's 97.5th
# percentile", never "above the point estimate".

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark.escalation import features, metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.escalation.features import EvalRow
    from benchmark.escalation.schema import Trajectory

N_SPLITS: Final[int] = 5
# Below this many rows or groups the grouped CV is not estimable and the depth is skipped.
MIN_ROWS: Final[int] = 40
_MIN_CLASSES: Final[int] = 2


@dataclass(frozen=True)
class DepthReport:
    """One decision depth's full result: skill, prior-conditional skill, null, and intervals."""

    depth: int
    n_rows: int
    n_excluded_short: int
    n_groups: int
    base_rate: float
    auroc_prior: float
    auroc_prefix: float
    auroc_combined: float
    incremental_auroc: float
    auprc_prefix: float
    null_prefix: metrics.NullResult
    null_incremental: metrics.NullResult
    ci_prefix: tuple[float, float]
    ci_incremental: tuple[float, float]
    scores: tuple[float, ...]
    labels: tuple[bool, ...]

    @property
    def has_skill(self) -> bool:
        """Skill means BOTH raw discrimination and incremental value clear their null bands."""
        # A constant score is excluded up front. Grouped CV under a permuted label is biased BELOW
        # 0.5 (a fold's model is fit without its own test challenge, so its errors anti-correlate),
        # which puts the empirical null mean near 0.40 — and a degenerate constant score, whose
        # AUROC is the 0.5 tie average by definition, would clear that band while discriminating
        # nothing. Non-constancy is the precondition for the comparison to mean anything.
        if len(set(self.scores)) <= 1:
            return False
        return self.null_prefix.beats_null and self.null_incremental.beats_null

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "n_rows": self.n_rows,
            "n_excluded_short": self.n_excluded_short,
            "n_groups": self.n_groups,
            "base_rate": round(self.base_rate, 4),
            "auroc_prior": round(self.auroc_prior, 4),
            "auroc_prefix": round(self.auroc_prefix, 4),
            "auroc_combined": round(self.auroc_combined, 4),
            "incremental_auroc": round(self.incremental_auroc, 4),
            "auprc_prefix": round(self.auprc_prefix, 4),
            "ci95_auroc_prefix": [round(v, 4) for v in self.ci_prefix],
            "ci95_incremental": [round(v, 4) for v in self.ci_incremental],
            "null_auroc_prefix": self.null_prefix.to_dict(),
            "null_incremental": self.null_incremental.to_dict(),
            "has_skill": self.has_skill,
        }


def task_prior(rows: Sequence[EvalRow], labels: Sequence[bool]) -> list[float]:
    """Leave-one-out per-challenge failure rate — what the router already knows at t=0."""
    # LOO (not in-sample) so a row never scores itself; a singleton challenge falls back to the
    # LOO global base rate. This prior is deliberately given same-challenge information the
    # grouped-CV model is denied, which makes the incremental number conservative.
    total = len(labels)
    n_failed = sum(labels)
    by_group: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_group.setdefault(row.group, []).append(index)
    out = [0.0] * total
    for indices in by_group.values():
        group_failed = sum(labels[i] for i in indices)
        for i in indices:
            if len(indices) > 1:
                out[i] = (group_failed - labels[i]) / (len(indices) - 1)
            elif total > 1:
                out[i] = (n_failed - labels[i]) / (total - 1)
    return out


def oof_scores(matrix: np.ndarray, labels: Sequence[bool], groups: Sequence[str]) -> list[float]:
    """Out-of-fold P(fail) from a grouped 5-fold logistic model — the continuous risk score.

    Logistic regression on standardized features is already a calibrated-probability model
    (its loss IS the log score), so no post-hoc Platt layer is stacked on top of it.
    """
    y = np.asarray(labels, dtype=int)
    out = np.full(len(y), float(y.mean()))
    n_splits = min(N_SPLITS, len(set(groups)))
    if n_splits < _MIN_CLASSES:
        return out.tolist()
    for train, test in GroupKFold(n_splits=n_splits).split(matrix, y, list(groups)):
        if len(np.unique(y[train])) < _MIN_CLASSES:
            out[test] = float(y[train].mean())
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(matrix[train], y[train])
        out[test] = model.predict_proba(matrix[test])[:, 1]
    return out.tolist()


@dataclass(frozen=True)
class _Fit:
    """One end-to-end pass of the pipeline over a single label vector."""

    prior: list[float]
    prefix: list[float]
    combined: list[float]
    auroc_prior: float
    auroc_prefix: float
    auroc_combined: float

    @property
    def incremental(self) -> float:
        return self.auroc_combined - self.auroc_prior


def _fit_once(
    rows: Sequence[EvalRow], base: np.ndarray, labels: Sequence[bool], groups: Sequence[str]
) -> _Fit:
    """Fit the prior, the prefix model and the combined model once for this label vector."""
    prior = task_prior(rows, labels)
    prefix = oof_scores(base, labels, groups)
    combined = oof_scores(np.column_stack([base, np.asarray(prior)]), labels, groups)
    return _Fit(
        prior=prior,
        prefix=prefix,
        combined=combined,
        auroc_prior=metrics.auroc(prior, labels),
        auroc_prefix=metrics.auroc(prefix, labels),
        auroc_combined=metrics.auroc(combined, labels),
    )


def evaluate_depth(
    trajectories: Sequence[Trajectory],
    depth: int,
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
) -> DepthReport | None:
    """Full grouped-CV evaluation at one decision depth, or None when the data cannot support it."""
    rows = features.build_rows(trajectories, depth)
    labels = [r.failed for r in rows]
    if len(rows) < MIN_ROWS or len(set(labels)) < _MIN_CLASSES:
        return None
    groups = [r.group for r in rows]
    base = np.asarray([r.features for r in rows], dtype=float)
    fit = _fit_once(rows, base, labels, groups)
    draws = _null_draws(rows, base, labels, groups, n_permutations=n_permutations, seed=seed)
    return DepthReport(
        depth=depth,
        n_rows=len(rows),
        n_excluded_short=len(trajectories) - len(rows),
        n_groups=len(set(groups)),
        base_rate=metrics.prevalence(labels),
        auroc_prior=fit.auroc_prior,
        auroc_prefix=fit.auroc_prefix,
        auroc_combined=fit.auroc_combined,
        incremental_auroc=fit.incremental,
        auprc_prefix=metrics.auprc(fit.prefix, labels),
        null_prefix=metrics.permutation_null(fit.auroc_prefix, [d[0] for d in draws]),
        null_incremental=metrics.permutation_null(fit.incremental, [d[1] for d in draws]),
        ci_prefix=metrics.grouped_bootstrap_ci(
            fit.prefix, labels, groups, metrics.auroc, seed=seed
        ),
        ci_incremental=_incremental_ci(fit, labels, groups, seed),
        scores=tuple(fit.prefix),
        labels=tuple(labels),
    )


def _null_draws(
    rows: Sequence[EvalRow],
    base: np.ndarray,
    labels: Sequence[bool],
    groups: Sequence[str],
    *,
    n_permutations: int,
    seed: int,
) -> list[tuple[float, float]]:
    """(prefix AUROC, incremental AUROC) under `n_permutations` label shuffles, pipeline re-run.

    The WHOLE pipeline is re-fit per shuffle — prior included — so the null absorbs every
    optimism the fitting itself introduces, not just the ranking step.
    """
    rng = random.Random(seed)
    shuffled = list(labels)
    draws: list[tuple[float, float]] = []
    while len(draws) < n_permutations:
        rng.shuffle(shuffled)
        if len(set(shuffled)) < _MIN_CLASSES:
            continue
        fit = _fit_once(rows, base, list(shuffled), groups)
        draws.append((fit.auroc_prefix, fit.incremental))
    return draws


def _incremental_ci(
    fit: _Fit, labels: Sequence[bool], groups: Sequence[str], seed: int
) -> tuple[float, float]:
    """Challenge-level PAIRED bootstrap CI for the incremental AUROC on the fitted scores."""
    # Paired matters: both arms are scored on the SAME resampled rows, so the interval reflects the
    # variance of the DIFFERENCE rather than the sum of two independent variances. It must also
    # bracket the same quantity the point estimate reports — combined minus prior, not prefix minus
    # prior — or the interval and the headline would be describing different statistics.
    draws: list[float] = []
    for picked in metrics.grouped_resamples(groups, seed=seed):
        sample_labels = [labels[i] for i in picked]
        if not any(sample_labels) or all(sample_labels):
            continue
        combined = [fit.combined[i] for i in picked]
        prior = [fit.prior[i] for i in picked]
        draws.append(metrics.auroc(combined, sample_labels) - metrics.auroc(prior, sample_labels))
    return metrics.bootstrap_ci(draws)


def evaluate(
    trajectories: Sequence[Trajectory],
    depths: Sequence[int] = features.DEFAULT_DEPTHS,
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
) -> list[DepthReport]:
    """Every requested depth that the corpus can actually support, in depth order."""
    out = []
    for depth in depths:
        report = evaluate_depth(trajectories, depth, n_permutations=n_permutations, seed=seed)
        if report is not None:
            out.append(report)
    return out


__all__ = [
    "MIN_ROWS",
    "N_SPLITS",
    "DepthReport",
    "evaluate",
    "evaluate_depth",
    "oof_scores",
    "task_prior",
]
