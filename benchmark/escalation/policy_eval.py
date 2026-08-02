"""The SHIPPED policy, graded at the unit it actually acts on: one trajectory, one decision."""

# The old harness pooled 29 422 prefixes from 799 runs, which inflates n by ~37x (prefixes inside a
# run are near-perfectly correlated) and summed `n_escalated` across all 12 sweep cells, printing
# 4980 escalations for 799 trajectories. Here one trajectory is one row, `n_escalated` is per cell,
# and the headline is the question the product asks: given the policy fired, is this run more
# likely to fail than a run picked at random?
#
# ONE TRAJECTORY IS NOT ONE INDEPENDENT OBSERVATION, AND THIS HALF USED TO PRETEND IT WAS. The
# corpus is 791 stamped runs drawn from only 160 challenges (every challenge is attempted by
# several model/effort arms), so outcomes cluster hard by challenge. Until this was fixed the
# policy half took a GLOBAL label shuffle for its null and a Wilson interval over rows, while the
# prefix half (`prefix_eval.py`) already permuted WITHIN a challenge and bootstrapped over whole
# challenges. Measured over 2000 draws at escalate_after_n=10, the two nulls disagree by 2.2x in
# width — global 95% [0.4654, 0.5341] (p=0.0105) against challenge-clustered [0.4450, 0.5570]
# (p=0.0885) — so a cell that "beat chance" under the shipped harness sits comfortably inside the
# band once the clustering is respected. No cell on today's corpus tripped that (all p > 0.6), but
# it is a live false-positive generator and every printed interval was too narrow by the same
# factor. Both the null and the interval are now clustered by challenge; `_clustered_null` says
# why this half blocks whole challenges where the prefix half shuffles inside them.
#
# THE FIX WAS APPLIED TO ONE ARM AND LEFT OFF THE OTHER. `precision_ci` became a challenge
# bootstrap while P(fail | NOT fired) kept a Wilson interval over rows, and `plots._outcome_limits`
# then read a DIRECTION off the two — an "INVERTED" verdict from `fired_hi < quiet_lo`, silence
# from `quiet_hi < fired_lo`. Comparing an interval to one built under an assumption this module
# rejects is not a comparison. An artificially narrow quiet arm raises `quiet_lo` and lowers
# `quiet_hi`, so it makes BOTH directional branches easier to trip: it can manufacture a spurious
# inversion or a spurious silent endorsement. Both arms are now `_arm_intervals`, estimated from
# the SAME challenge resamples, so the figure compares like with like and the two are paired
# draw-for-draw. The fired arm is unchanged to the digit — one loop over the resamples reproduces
# what two separate calls drew — so this widens the quiet arm and moves nothing else. How much
# wider, and whether any cell's verdict moves with it, is a property of whatever corpus is loaded
# and is deliberately not restated here as a finding.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.escalation import features, metrics, replay

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.escalation.schema import Trajectory


@dataclass(frozen=True)
class PolicyCell:
    """One swept configuration, scored per trajectory against the terminal outcome."""

    escalate_after_n: int
    stale_window: int
    ladder: str
    n_trajectories: int
    n_escalated: int
    tp: int
    fp: int
    fn: int
    tn: int
    base_failure_rate: float
    precision_ci: tuple[float, float]
    # The quiet arm's interval, on the SAME footing as `precision_ci` (same estimator, same
    # resamples) because the outcome figure compares the two to call a direction. It is required
    # rather than defaulted: a default would let a caller ship the comparison against an interval
    # nobody computed, which is the defect this field exists to close.
    quiet_ci: tuple[float, float]
    null_auroc: metrics.NullResult
    # NO ESCALATION-TIMING ARRAYS. `lead_times_*` and `first_fire_*` existed solely to feed
    # `plots.lead_time_by_outcome`, which was deleted: on this corpus the policy fires in the first
    # two decisions of nearly every run it flags, so a lead time is the run LENGTH minus a constant
    # and the figure could not support a timing claim in either direction. Nothing else read them —
    # they never reached `to_dict` — so they are gone rather than left as data with no consumer.

    @property
    def precision(self) -> float | None:
        """P(run failed | policy fired), or None when the cell never fired — undefined, not 0.0."""
        # A never-firing cell used to score 0.0 here, which then RANKED below every real cell in
        # `max(cells, key=precision)`. It was safe only by luck: 0.0 happens to lose that argmax.
        # An undefined precision must be excluded from a ranking, not given the worst finite value.
        fired = self.tp + self.fp
        return self.tp / fired if fired else None

    @property
    def recall(self) -> float:
        failures = self.tp + self.fn
        return self.tp / failures if failures else 0.0

    @property
    def p_fail_given_quiet(self) -> float | None:
        """P(run failed | policy did NOT fire), or None when it fired on everything."""
        quiet = self.fn + self.tn
        return self.fn / quiet if quiet else None

    @property
    def lift(self) -> float | None:
        """precision / base rate. Below 1.0 means firing predicts SUCCESS, not failure."""
        if self.precision is None or not self.base_failure_rate:
            return None
        return self.precision / self.base_failure_rate

    def to_dict(self) -> dict[str, object]:
        return {
            "escalate_after_n": self.escalate_after_n,
            "stale_window": self.stale_window,
            "ladder": self.ladder,
            "n_trajectories": self.n_trajectories,
            "n_escalated": self.n_escalated,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "precision": _rounded(self.precision),
            "precision_ci95": [round(v, 4) for v in self.precision_ci],
            "recall": round(self.recall, 4),
            "p_fail_given_fired": _rounded(self.precision),
            "p_fail_given_not_fired": _rounded(self.p_fail_given_quiet),
            "p_fail_given_not_fired_ci95": [round(v, 4) for v in self.quiet_ci],
            "base_failure_rate": round(self.base_failure_rate, 4),
            "lift": _rounded(self.lift),
            "null_auroc": self.null_auroc.to_dict(),
        }


def _rounded(value: float | None) -> float | None:
    """Round for the JSON report, keeping an undefined quantity null rather than coercing it."""
    return None if value is None else round(value, 4)


@dataclass(frozen=True)
class _Scored:
    """One configuration's per-trajectory replay outcome, aligned index-for-index."""

    fired: Sequence[bool]
    failed: Sequence[bool]
    groups: Sequence[str]


def evaluate_cell(
    trajectories: Sequence[Trajectory],
    point: replay.GridPoint,
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
) -> PolicyCell:
    """Replay one configuration over every trajectory and score it at the trajectory level."""
    fired: list[bool] = []
    failed: list[bool] = []
    groups: list[str] = []
    for traj in trajectories:
        decision = replay.replay_config(traj, point.to_config())
        fired.append(decision.escalated)
        failed.append(not traj.header.terminal_resolved)
        # The SAME grouping key the prefix half's grouped CV uses (challenge, i.e. instance id):
        # the two halves must not disagree about what an independent observation is.
        groups.append(features.group_of(traj))
    return _cell(point, _Scored(fired, failed, groups), n_permutations, seed)


def _cell(point: replay.GridPoint, s: _Scored, n_permutations: int, seed: int) -> PolicyCell:
    """Assemble one cell's 2x2, its clustered arm intervals, and its clustered permutation null."""
    fired, failed = s.fired, s.failed
    tp = sum(f and y for f, y in zip(fired, failed, strict=True))
    scores = [1.0 if f else 0.0 for f in fired]
    fired_ci, quiet_ci = _arm_intervals(fired, failed, s.groups, seed=seed)
    return PolicyCell(
        escalate_after_n=point.escalate_after_n,
        stale_window=point.stale_window,
        ladder=point.ladder,
        n_trajectories=len(fired),
        n_escalated=sum(fired),
        tp=tp,
        fp=sum(f and not y for f, y in zip(fired, failed, strict=True)),
        fn=sum(not f and y for f, y in zip(fired, failed, strict=True)),
        tn=sum(not f and not y for f, y in zip(fired, failed, strict=True)),
        base_failure_rate=metrics.prevalence(failed),
        precision_ci=fired_ci,
        quiet_ci=quiet_ci,
        null_auroc=_clustered_null(
            scores, failed, s.groups, n_permutations=n_permutations, seed=seed
        ),
    )


def _permute_clusters(
    labels: Sequence[bool], groups: Sequence[str], rng: random.Random
) -> list[bool]:
    """Shuffle whole CHALLENGE blocks of outcomes, so the clustering survives the shuffle."""
    # Block permutation: read every challenge's outcomes out in a shuffled challenge order, then
    # deal that sequence back to the rows in their original block order. The global outcome
    # multiset is preserved exactly (as a row shuffle preserves it) but a row's outcome now
    # arrives in a correlated chunk, so the null carries the between-challenge variance the
    # observation carries. Unequal block sizes mean a block can straddle two source challenges;
    # that is the standard approximation and it does not bias the multiset.
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    blocks = list(by_group.values())
    order = list(range(len(blocks)))
    rng.shuffle(order)
    pool = [labels[i] for k in order for i in blocks[k]]
    out = list(labels)
    position = 0
    for block in blocks:
        for index in block:
            out[index] = pool[position]
            position += 1
    return out


def _clustered_null(
    scores: Sequence[float],
    labels: Sequence[bool],
    groups: Sequence[str],
    *,
    n_permutations: int,
    seed: int,
) -> metrics.NullResult:
    """The AUROC null under CHALLENGE-block shuffles — the unit this claim is actually made at."""
    # A GLOBAL shuffle (what `metrics.permute_statistic` does, and what this cell used to call)
    # treats 791 clustered runs as 791 exchangeable draws; the measured cost is in the module
    # header — a 2.2x too-narrow band and a p of 0.0115 where the honest figure is 0.0865.
    #
    # WHY BLOCK PERMUTATION HERE AND WITHIN-CHALLENGE PERMUTATION IN `prefix_eval`. They test
    # different nulls, and each half asks the question its headline claims. This half's claim is
    # UNCONDITIONAL — "given the policy fired, is this run likelier to fail than a run picked at
    # random" — so challenge identity is part of what may be driving it and the exchangeable unit
    # is the whole challenge. The prefix half's claim is conditional on the router's t=0 prior,
    # which is estimated from challenge base rates and must be IDENTICAL under the null, forcing a
    # within-challenge shuffle there. Measured on this corpus at n=10: a within-challenge shuffle
    # gives sd 0.0036 against the block shuffle's 0.0282, because it removes exactly the
    # between-challenge variation the unconditional statistic contains. Using it here would have
    # been the same too-narrow-band defect wearing a different name.
    rng = random.Random(seed)
    draws = [
        metrics.auroc(scores, _permute_clusters(labels, groups, rng)) for _ in range(n_permutations)
    ]
    return metrics.permutation_null(metrics.auroc(scores, labels), draws)


def _arm_intervals(
    fired: Sequence[bool], failed: Sequence[bool], groups: Sequence[str], *, seed: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% intervals for P(fail|fired) and P(fail|quiet), from ONE set of challenge resamples."""
    # Wilson over rows treats 791 correlated runs as 791 draws and is too narrow by the same
    # clustering factor as the old null. This resamples challenges with replacement, exactly as
    # `prefix_eval` does for its AUROC, so an interval reflects "another sample of challenges"
    # rather than "another sample of rows from these challenges". BOTH arms are walked in the same
    # loop rather than in two calls: the outcome figure subtracts these intervals to call a
    # direction, so they must come from the same draws — two estimators, or even the same estimator
    # on two independent resample sets, put a comparison on footing the numbers do not have. A
    # resample in which an arm is empty has no rate to contribute and is dropped for THAT arm only;
    # with no draws at all the interval is (nan, nan) — the honest reading of an arm that never
    # exists (a cell that never fires, or one that fires on everything).
    draws: dict[bool, list[float]] = {True: [], False: []}
    for picked in metrics.grouped_resamples(groups, seed=seed):
        for want, collected in draws.items():
            rate = _arm_rate(picked, fired, failed, want=want)
            if rate is not None:
                collected.append(rate)
    return metrics.bootstrap_ci(draws[True]), metrics.bootstrap_ci(draws[False])


def _arm_rate(
    picked: Sequence[int], fired: Sequence[bool], failed: Sequence[bool], *, want: bool
) -> float | None:
    """Failure rate among the resampled rows on one side of the fired/quiet split, or None."""
    arm = [i for i in picked if fired[i] == want]
    return sum(1 for i in arm if failed[i]) / len(arm) if arm else None


def evaluate(
    trajectories: Sequence[Trajectory],
    grid: Sequence[replay.GridPoint],
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
) -> list[PolicyCell]:
    """Every swept configuration, scored per trajectory."""
    return [
        evaluate_cell(trajectories, point, n_permutations=n_permutations, seed=seed)
        for point in grid
    ]


__all__ = ["PolicyCell", "evaluate", "evaluate_cell"]
