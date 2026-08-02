"""Detector metrics: ranking statistics, permutation nulls, and interval estimates."""

# AUPRC-vs-prevalence is primary on this imbalanced problem — ROC hides prevalence — so AUROC is
# reported AUXILIARY. `auroc`/`auprc` are differentially verified against sklearn (see
# tests/escalation/test_metrics.py) and are NOT the defect the R0 harness rework addresses: the
# label was. There is deliberately no positional `label_prefixes` here any more; the target is the
# TASK-LEVEL outcome scored from a prefix (see `benchmark/escalation/prefix_eval.py`), because a
# "last H steps of a failed run" label is won by a content-free clock and caps a perfect
# task-oracle at AUROC 0.757.

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from benchmark.calibration.labeler_metrics import ConfusionMatrix, LabelerMetrics, compute_metrics

# A permutation null needs enough draws that the 97.5th percentile is not itself noise.
MIN_PERMUTATIONS = 200

Statistic = Callable[[Sequence[float], Sequence[bool]], float]


def _require_finite(scores: Sequence[float]) -> None:
    """Reject NaN/inf scores, naming the first offending index so the caller can trace it."""
    for index, score in enumerate(scores):
        if not math.isfinite(score):
            raise ValueError(f"score at index {index} is not finite ({score!r}); cannot rank it")


def prevalence(labels: Sequence[bool]) -> float:
    """The positive rate — the no-skill AUPRC baseline drawn on every PR figure."""
    return sum(labels) / len(labels) if labels else 0.0


def auprc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the precision-recall curve (average precision), sklearn's step definition:
    sum of (R_n - R_{n-1}) * P_n over decreasing score thresholds, ties resolved by block.
    """
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    ap = 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    index = 0
    total = len(ranked)
    while index < total:
        threshold = ranked[index][0]
        while index < total and ranked[index][0] == threshold:
            if ranked[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic — AUXILIARY only.

    Returns 0.5 (chance) when one class is absent, matching the no-information point.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    # A NaN pivot used to HANG this function, not mis-score it: the tie scan below advances while
    # `scores[order[run]] == scores[order[index]]`, and `nan == nan` is False, so `run` never leaves
    # `index`, `index = run` is a no-op, and the outer loop spins forever. A non-finite score means
    # an upstream model emitted garbage; failing loudly is the only honest answer, and it is checked
    # here rather than at every call site because this is the one place the invariant is required.
    _require_finite(scores)
    # Rank-sum (average ranks for ties) → AUC = (rank_sum_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        run = index
        while run < len(order) and scores[order[run]] == scores[order[index]]:
            run += 1
        avg_rank = (index + run + 1) / 2.0  # 1-based average rank across the tie block
        for k in range(index, run):
            ranks[order[k]] = avg_rank
        index = run
    rank_sum_pos = sum(ranks[i] for i, lab in enumerate(labels) if lab)
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def flag_budget(
    scores: Sequence[float], labels: Sequence[bool], *, flag_rate: float | None = None
) -> int:
    """How many runs the operating threshold is MEANT to flag: `flag_rate` x n (base rate)."""
    # The INTENDED budget, which is not the same as what the cut actually admits.
    # `operating_threshold` returns a score and every consumer flags `score >= cut`, so a block of
    # rows tied with the cut is taken whole and the realised flag count can exceed this number. The
    # confusion figure reports that gap, and it can only do so honestly if the intended count has
    # ONE source — here — rather than being re-derived beside the threshold it came from.
    if not scores:
        return 0
    rate = prevalence(labels) if flag_rate is None else flag_rate
    if rate <= 0.0:
        return 0
    return max(1, min(len(scores), round(min(rate, 1.0) * len(scores))))


def operating_threshold(
    scores: Sequence[float], labels: Sequence[bool], *, flag_rate: float | None = None
) -> float:
    """The score cut that flags `flag_rate` of runs — the corpus base rate by default."""
    # DERIVED, never hardcoded. A fixed 0.5 used to stand here and was unreachable on the corpus of
    # the day: the score is a calibrated probability, so its centre tracks the corpus base rate and
    # a cut placed away from that rate degenerates (measured then: 2 true positives against 203
    # false negatives). Spending a flag budget equal to prevalence is the break-even point: at that
    # budget a no-skill flagger reaches exactly base-rate precision, so any excess is ranking skill.
    if not scores:
        return 0.0
    budget = flag_budget(scores, labels, flag_rate=flag_rate)
    if budget == 0:
        return math.inf
    return sorted(scores, reverse=True)[budget - 1]


def detection_metrics(
    scores: Sequence[float], labels: Sequence[bool], *, threshold: float | None = None
) -> LabelerMetrics:
    """Confusion + precision/recall/F1/FPR/Cohen-kappa at the operating threshold."""
    cut = operating_threshold(scores, labels) if threshold is None else threshold
    owner = {str(i): ("good" if lab else "bad") for i, lab in enumerate(labels)}
    auto = {str(i): ("good" if s >= cut else "bad") for i, s in enumerate(scores)}
    return compute_metrics(owner, auto)


def roc_operating_points(
    scores: Sequence[float], labels: Sequence[bool]
) -> list[tuple[float, float]]:
    """(fpr, tpr) at each DISTINCT score threshold, ties collapsed — the real operating points."""
    # The previous implementation admitted one row at a time in tie order, so on a binary score the
    # drawn polyline traced the corpus's row order (drawn area 0.554 against a titled 0.450). One
    # vertex per distinct threshold makes the drawn area equal the reported statistic.
    return _threshold_sweep(scores, labels, _roc_vertex, seed=(0.0, 0.0), tail=(1.0, 1.0))


def pr_operating_points(
    scores: Sequence[float], labels: Sequence[bool]
) -> list[tuple[float, float]]:
    """(recall, precision) at each DISTINCT score threshold, ties collapsed."""
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return [(0.0, 1.0), (1.0, prevalence(labels))]
    points = _threshold_sweep(scores, labels, _pr_vertex, seed=None, tail=None)
    return [(0.0, points[0][1]), *points]


def _threshold_sweep(
    scores: Sequence[float],
    labels: Sequence[bool],
    vertex: Callable[[int, int, int, int], tuple[float, float]],
    *,
    seed: tuple[float, float] | None,
    tail: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    """Walk descending distinct thresholds, emitting one `vertex(tp, fp, pos, neg)` per block."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return [p for p in (seed, tail) if p is not None] or [(0.0, 0.0)]
    ranked = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    out: list[tuple[float, float]] = [] if seed is None else [seed]
    tp = 0
    fp = 0
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == threshold:
            if ranked[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        out.append(vertex(tp, fp, positives, negatives))
    if tail is not None and out[-1] != tail:
        out.append(tail)
    return out


def _roc_vertex(tp: int, fp: int, positives: int, negatives: int) -> tuple[float, float]:
    return (fp / negatives, tp / positives)


def _pr_vertex(tp: int, fp: int, positives: int, _negatives: int) -> tuple[float, float]:
    return (tp / positives, tp / (tp + fp))


@dataclass(frozen=True)
class NullResult:
    """An observed statistic placed against its label-permutation null distribution."""

    observed: float
    mean: float
    sd: float
    ci_low: float
    ci_high: float
    p_value: float
    n_permutations: int
    draws: tuple[float, ...]

    @property
    def beats_null(self) -> bool:
        """True iff the observation sits ABOVE the null's 97.5th percentile — the skill gate."""
        return self.observed > self.ci_high

    def to_dict(self) -> dict[str, object]:
        return {
            "observed": round(self.observed, 4),
            "null_mean": round(self.mean, 4),
            "null_sd": round(self.sd, 4),
            "null_ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "p_value": round(self.p_value, 4),
            "n_permutations": self.n_permutations,
            "beats_null": self.beats_null,
        }


def permutation_null(
    observed: float,
    draws: Sequence[float],
) -> NullResult:
    """Summarise `draws` (statistics under permuted labels) around the `observed` value."""
    n = len(draws)
    if n < MIN_PERMUTATIONS:
        # Actionable because the usual way to hit this is a CLI `--permutations` below the floor,
        # several frames above here: name the knob and the value it has to clear.
        raise ValueError(
            f"permutation null needs >= {MIN_PERMUTATIONS} draws "
            f"(metrics.MIN_PERMUTATIONS), got {n} — "
            f"raise --permutations to at least {MIN_PERMUTATIONS}"
        )
    mean = sum(draws) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in draws) / (n - 1))
    ordered = sorted(draws)
    # +1 correction on both numerator and denominator: an exact permutation p can never be 0.
    p_value = (sum(1 for d in draws if d >= observed) + 1) / (n + 1)
    return NullResult(
        observed=observed,
        mean=mean,
        sd=sd,
        ci_low=_percentile(ordered, 2.5),
        ci_high=_percentile(ordered, 97.5),
        p_value=p_value,
        n_permutations=n,
        draws=tuple(draws),
    )


def permute_statistic(
    scores: Sequence[float],
    labels: Sequence[bool],
    statistic: Statistic,
    *,
    n_permutations: int = MIN_PERMUTATIONS,
    seed: int = 0,
) -> NullResult:
    """Null for a FIXED score vector: shuffle the labels `n_permutations` times, restat each."""
    rng = random.Random(seed)
    shuffled = list(labels)
    draws: list[float] = []
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        draws.append(statistic(scores, shuffled))
    return permutation_null(statistic(scores, labels), draws)


def _percentile(ordered: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * pct / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — honest at small n, unlike normal-approx."""
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def grouped_resamples(
    groups: Sequence[str], *, n_resamples: int = 1000, seed: int = 0
) -> Iterator[list[int]]:
    """Row-index samples drawn by resampling whole GROUPS (challenges) with replacement."""
    # Yielding indices rather than values is what makes a PAIRED statistic possible: two score
    # vectors can be compared on exactly the same resampled rows. Rows inside one challenge are
    # correlated, so a row-level bootstrap would understate every interval built on this.
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    keys = list(by_group)
    rng = random.Random(seed)
    for _ in range(n_resamples):
        yield [i for _ in keys for i in by_group[rng.choice(keys)]]


def bootstrap_ci(draws: Sequence[float]) -> tuple[float, float]:
    """The central 95% of a bootstrap draw set, or (nan, nan) when nothing was estimable."""
    if not draws:
        return (float("nan"), float("nan"))
    ordered = sorted(draws)
    return (_percentile(ordered, 2.5), _percentile(ordered, 97.5))


def grouped_bootstrap_ci(
    values: Sequence[float],
    labels: Sequence[bool],
    groups: Sequence[str],
    statistic: Statistic,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI for `statistic` under resampling whole GROUPS (challenges), not rows."""
    draws: list[float] = []
    for picked in grouped_resamples(groups, n_resamples=n_resamples, seed=seed):
        sample_labels = [labels[i] for i in picked]
        if not any(sample_labels) or all(sample_labels):
            continue
        draws.append(statistic([values[i] for i in picked], sample_labels))
    return bootstrap_ci(draws)


__all__ = [
    "MIN_PERMUTATIONS",
    "ConfusionMatrix",
    "NullResult",
    "Statistic",
    "auprc",
    "auroc",
    "bootstrap_ci",
    "detection_metrics",
    "flag_budget",
    "grouped_bootstrap_ci",
    "grouped_resamples",
    "operating_threshold",
    "permutation_null",
    "permute_statistic",
    "pr_operating_points",
    "prevalence",
    "roc_operating_points",
    "wilson_interval",
]
