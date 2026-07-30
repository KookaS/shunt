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
#   auroc_prior     — the router's own t=0 knowledge, estimated the way a DEPLOYED router must:
#                     fold-wise, from TRAIN folds only. Under GroupKFold a test challenge is by
#                     construction absent from training, so this prior falls back to the train
#                     base rate — which is exactly the point, and is what a router facing an
#                     unseen instance actually has.
#   auroc_prefix    — grouped-CV out-of-fold discrimination from the prefix alone. Grouped by
#                     challenge, so a challenge never appears in both train and test.
#   incremental     — auroc(prior + prefix) - auroc(prior). THE number. Everything else overstates.
#
# WHY THE PRIOR IS NOT LEAVE-ONE-OUT. It used to be: each row's prior was the mean of its OWN
# instance's other labels, over the whole corpus. Grouped by instance, those siblings sit in the
# row's own TEST fold, so the baseline had read the test labels — it scored AUROC ~0.885 where the
# honest estimate is ~0.43, and the headline `incremental` was asking the detector to beat a
# leaked oracle. `leaked_task_prior` is retained purely as a DIAGNOSTIC (reported alongside as
# `auroc_prior_leaked`) because the gap between the two is itself the finding.
#
# Every headline carries a label-permutation null (>= 200 shuffles, the whole pipeline re-run per
# shuffle) and a challenge-level bootstrap CI. "Beats chance" means "above the null's 97.5th
# percentile", never "above the point estimate".
#
# THE NULL PERMUTES WITHIN A CHALLENGE, NOT GLOBALLY. A global shuffle destroys the challenge-level
# clustering of outcomes, which puts the observation and the null in DIFFERENT headroom regimes: the
# prior collapses to chance under the shuffle (headroom 0.5) while the real prior leaves the
# observed incremental capped near +0.117 — against a measured null ci_high of +0.117. The gate was
# arithmetically unpassable, i.e. it had zero power for any detector however good. Permuting inside
# each challenge preserves every challenge's outcome multiset, so the fold base rates — and with
# them the deployable prior — are IDENTICAL in both regimes and only the prefix's contribution is
# nulled. It also matches the paired grouped bootstrap, which is the independent check.
#
# THE PRICE, stated rather than hidden: a challenge whose runs all share an outcome is invariant
# under a within-challenge permutation, so its rows never move. On this corpus 40-46% of challenges
# are heterogeneous and 56-58% of rows can move, which is why the null's spread is far narrower
# than the global shuffle's (sd 0.003 against 0.034 at depth 5). Narrower is not weaker here — the
# global null's extra spread came from the PRIOR collapsing, a quantity the observation never had
# collapse — but it does mean this null is estimated off roughly half the corpus.
#
# THE COMPARATOR IS FLOORED AT CHANCE. The honest prior scores BELOW 0.5 on this corpus, and
# `combined - prior` against an anti-predictive baseline manufactures positive skill out of a broken
# comparator. `incremental` therefore measures against `max(prior, 0.5)`: beat the better of the
# router's prior and no-information, or report nothing.

from __future__ import annotations

import math
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
# No-information AUROC. The prior comparator is floored here so an anti-predictive baseline can
# never be "beaten" into an apparent finding.
CHANCE: Final[float] = 0.5


@dataclass(frozen=True)
class DepthReport:
    """One decision depth's full result: skill, prior-conditional skill, null, and intervals."""

    depth: int
    n_rows: int
    n_excluded_short: int
    n_groups: int
    base_rate: float
    auroc_prior: float
    # The old leaked leave-one-out prior, kept as a diagnostic contrast only — never the baseline
    # `incremental_auroc` is measured against.
    auroc_prior_leaked: float
    auroc_prefix: float
    # Diagnostic: the n-weighted mean of the PER-FOLD AUROCs. `auroc_prefix` pools every fold's
    # probabilities into one ranking, which carries a between-fold calibration offset unrelated to
    # discrimination. The pooled number stays the headline because it is what the drawn ROC
    # integrates to (a title that disagrees with its own curve is a defect this harness already
    # fixed once); the folded number says how much of it is that offset.
    auroc_prefix_folded: float
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
    def prior_comparator(self) -> float:
        """The baseline `incremental_auroc` is measured against: the prior, floored at chance."""
        return max(self.auroc_prior, CHANCE)

    @property
    def mde_auroc(self) -> float:
        """Minimum detectable prefix AUROC: chance plus this corpus's own 95% half-width.

        Clustering by challenge inflates the variance, so a point estimate below this is
        unresolvable here — "not detected" must not be read as "not there".
        """
        low, high = self.ci_prefix
        if math.isnan(low) or math.isnan(high):
            return float("nan")
        return CHANCE + (high - low) / 2.0

    @property
    def has_skill(self) -> bool:
        """Skill means the nulls AND the paired bootstrap all exclude no-information."""
        # A constant score is excluded up front: its AUROC is the 0.5 tie average by definition, so
        # it would clear a band while discriminating nothing. Non-constancy is the precondition for
        # the comparison to mean anything. The bootstrap term is not redundant with the null — the
        # null asks "could shuffled labels do this", the interval asks "would another sample of
        # challenges do this" — and it was computed and then ignored until now.
        if len(set(self.scores)) <= 1:
            return False
        return (
            self.null_prefix.beats_null
            and self.null_incremental.beats_null
            and self.ci_incremental[0] > 0.0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "n_rows": self.n_rows,
            "n_excluded_short": self.n_excluded_short,
            "n_groups": self.n_groups,
            "base_rate": round(self.base_rate, 4),
            "auroc_prior": round(self.auroc_prior, 4),
            "auroc_prior_leaked": round(self.auroc_prior_leaked, 4),
            "auroc_prefix": round(self.auroc_prefix, 4),
            # Fold-honest contrast to the pooled headline: the gap between them is the
            # between-fold calibration offset, not discrimination. Computed since this harness
            # was written and reported here so a reader can see both.
            "auroc_prefix_folded": round(self.auroc_prefix_folded, 4),
            "auroc_combined": round(self.auroc_combined, 4),
            # The floored baseline `incremental_auroc` is actually measured against, so a reader
            # can tell when the raw prior was anti-predictive and the floor bound instead.
            "prior_comparator": round(self.prior_comparator, 4),
            "incremental_auroc": round(self.incremental_auroc, 4),
            "auprc_prefix": round(self.auprc_prefix, 4),
            # The published "this eval can only resolve a detector at AUROC >= X" figure. It was
            # computed and never reported, so the number in the docs had no machine-readable
            # source — the same defect class as `auroc_prefix_folded` above.
            "mde_auroc": round(self.mde_auroc, 4),
            "ci95_auroc_prefix": [round(v, 4) for v in self.ci_prefix],
            "ci95_incremental": [round(v, 4) for v in self.ci_incremental],
            "null_auroc_prefix": self.null_prefix.to_dict(),
            "null_incremental": self.null_incremental.to_dict(),
            "has_skill": self.has_skill,
        }


def leaked_task_prior(rows: Sequence[EvalRow], labels: Sequence[bool]) -> list[float]:
    """DIAGNOSTIC ONLY: leave-one-out per-challenge failure rate over the WHOLE corpus."""
    # Leaked, not conservative: grouped CV puts a row's same-challenge siblings in its own TEST
    # fold, so this reads labels no deployed router can see. Reported only as the contrast that
    # shows how much of the old headline was leakage. `grouped_task_prior` is the headline prior.
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


def grouped_splits(
    labels: Sequence[bool], groups: Sequence[str]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ONE GroupKFold partition both the prior and the risk model are estimated on."""
    # Shared so the prior can never be fitted on rows the model treats as test: a prior estimated
    # on a different partition would re-introduce exactly the leak this module now walls off.
    y = np.asarray(labels, dtype=int)
    n_splits = min(N_SPLITS, len(set(groups)))
    if n_splits < _MIN_CLASSES:
        return []
    placeholder = np.zeros((len(y), 1))
    return list(GroupKFold(n_splits=n_splits).split(placeholder, y, list(groups)))


def prior_from_splits(
    labels: Sequence[bool],
    groups: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    """Per-challenge failure rate estimated from TRAIN folds only, fold by fold."""
    # A challenge unseen in training gets the train-fold base rate. Under GroupKFold that is EVERY
    # test row, which is the honest point: a router meeting a new instance has no history on it.
    y = np.asarray(labels, dtype=int)
    out = np.full(len(y), float(y.mean()))
    for train, test in splits:
        base = float(y[train].mean())
        seen: dict[str, list[int]] = {}
        for i in train:
            seen.setdefault(groups[i], []).append(int(y[i]))
        for i in test:
            observed = seen.get(groups[i])
            out[i] = base if observed is None else sum(observed) / len(observed)
    return out.tolist()


def grouped_task_prior(labels: Sequence[bool], groups: Sequence[str]) -> list[float]:
    """The deployable t=0 prior: `prior_from_splits` over the shared GroupKFold partition."""
    return prior_from_splits(labels, groups, grouped_splits(labels, groups))


def oof_scores(matrix: np.ndarray, labels: Sequence[bool], groups: Sequence[str]) -> list[float]:
    """Out-of-fold P(fail) from a grouped 5-fold logistic model — the continuous risk score.

    Logistic regression on standardized features is already a calibrated-probability model
    (its loss IS the log score), so no post-hoc Platt layer is stacked on top of it.
    """
    y = np.asarray(labels, dtype=int)
    out = np.full(len(y), float(y.mean()))
    for train, test in grouped_splits(labels, groups):
        if len(np.unique(y[train])) < _MIN_CLASSES:
            out[test] = float(y[train].mean())
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(matrix[train], y[train])
        out[test] = model.predict_proba(matrix[test])[:, 1]
    return out.tolist()


def folded_auroc(scores: Sequence[float], labels: Sequence[bool], groups: Sequence[str]) -> float:
    """n-weighted mean of the PER-FOLD AUROCs over the same grouped split `oof_scores` uses."""
    # Pooling every fold's probabilities into one ranking mixes in a between-fold calibration
    # offset: fold base rates here span 0.296-0.509, so a row can out-rank another purely for
    # sitting in a higher-prevalence fold. Scoring each fold on its own removes that offset, and
    # the gap between this and the pooled figure is how much of the pooled number was accounting.
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    total = 0.0
    weight = 0
    for _train, test in grouped_splits(labels, groups):
        if len(np.unique(y[test])) < _MIN_CLASSES:
            continue  # a single-class fold has no ROC to integrate; it is skipped, not scored 0.5
        total += metrics.auroc(s[test].tolist(), [bool(v) for v in y[test]]) * len(test)
        weight += len(test)
    return total / weight if weight else CHANCE


@dataclass(frozen=True)
class _Fit:
    """One end-to-end pass of the pipeline over a single label vector."""

    prior: list[float]
    prefix: list[float]
    combined: list[float]
    auroc_prior: float
    auroc_prior_leaked: float
    auroc_prefix: float
    auroc_combined: float

    @property
    def prior_comparator(self) -> float:
        """The baseline the increment is measured against: the prior, floored at chance."""
        return max(self.auroc_prior, CHANCE)

    @property
    def incremental(self) -> float:
        # Floored, per the module header: an anti-predictive prior (this corpus's is ~0.42) would
        # otherwise donate `0.5 - auroc_prior` of manufactured skill to every increment.
        return self.auroc_combined - self.prior_comparator


def _fit_once(
    rows: Sequence[EvalRow], base: np.ndarray, labels: Sequence[bool], groups: Sequence[str]
) -> _Fit:
    """Fit the prior, the prefix model and the combined model once for this label vector."""
    prior = grouped_task_prior(labels, groups)
    prefix = oof_scores(base, labels, groups)
    combined = oof_scores(np.column_stack([base, np.asarray(prior)]), labels, groups)
    return _Fit(
        prior=prior,
        prefix=prefix,
        combined=combined,
        auroc_prior=metrics.auroc(prior, labels),
        auroc_prior_leaked=metrics.auroc(leaked_task_prior(rows, labels), labels),
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
        auroc_prior_leaked=fit.auroc_prior_leaked,
        auroc_prefix=fit.auroc_prefix,
        auroc_prefix_folded=folded_auroc(fit.prefix, labels, groups),
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
    draws: list[tuple[float, float]] = []
    while len(draws) < n_permutations:
        shuffled = permute_within_groups(labels, groups, rng)
        if len(set(shuffled)) < _MIN_CLASSES:
            continue
        fit = _fit_once(rows, base, shuffled, groups)
        draws.append((fit.auroc_prefix, fit.incremental))
    return draws


def permute_within_groups(
    labels: Sequence[bool], groups: Sequence[str], rng: random.Random
) -> list[bool]:
    """Shuffle labels INSIDE each challenge, so every group's outcome multiset is preserved."""
    # A global shuffle destroys the challenge-level clustering of outcomes and so collapses the
    # prior to chance under the null while the observation keeps a real (anti-predictive) prior —
    # the two arms then sit in different headroom regimes and the gate has no power (module header).
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    out = list(labels)
    for indices in by_group.values():
        block = [labels[i] for i in indices]
        rng.shuffle(block)
        for index, value in zip(indices, block, strict=True):
            out[index] = value
    return out


def _incremental_ci(
    fit: _Fit, labels: Sequence[bool], groups: Sequence[str], seed: int
) -> tuple[float, float]:
    """Challenge-level PAIRED bootstrap CI for the incremental AUROC on the fitted scores."""
    # Paired matters: both arms are scored on the SAME resampled rows, so the interval reflects the
    # variance of the DIFFERENCE rather than the sum of two independent variances. It must also
    # bracket the same quantity the point estimate reports — combined minus the FLOORED prior, not
    # prefix minus prior — or the interval and the headline would be describing different
    # statistics.
    draws: list[float] = []
    for picked in metrics.grouped_resamples(groups, seed=seed):
        sample_labels = [labels[i] for i in picked]
        if not any(sample_labels) or all(sample_labels):
            continue
        combined = [fit.combined[i] for i in picked]
        prior = [fit.prior[i] for i in picked]
        # Floored per resample, exactly as the point estimate is: an interval built on the
        # unfloored difference would bracket a different statistic than the headline.
        comparator = max(metrics.auroc(prior, sample_labels), CHANCE)
        draws.append(metrics.auroc(combined, sample_labels) - comparator)
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
    "CHANCE",
    "MIN_ROWS",
    "N_SPLITS",
    "DepthReport",
    "evaluate",
    "evaluate_depth",
    "grouped_splits",
    "grouped_task_prior",
    "leaked_task_prior",
    "oof_scores",
    "permute_within_groups",
    "prior_from_splits",
]
