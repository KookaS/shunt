"""The SHIPPED policy, graded at the unit it actually acts on: one trajectory, one decision."""

# The old harness pooled 29 422 prefixes from 799 runs, which inflates n by ~37x (prefixes inside a
# run are near-perfectly correlated) and summed `n_escalated` across all 12 sweep cells, printing
# 4980 escalations for 799 trajectories. Here one trajectory is one row, `n_escalated` is per cell,
# and the headline is the question the product asks: given the policy fired, is this run more
# likely to fail than a run picked at random?
#
# ONE TRAJECTORY IS NOT ONE INDEPENDENT OBSERVATION, AND THIS HALF USED TO PRETEND IT WAS. The
# corpus is 727 stamped runs drawn from 166 challenges (152 among stamped; every challenge is
# attempted by several model/effort arms), so outcomes cluster hard by challenge. Until this was
# fixed the
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

import numpy as np

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
    # This cell's OWN marginal challenge bootstrap. Cells used to carry a FAMILY-WISE
    # max-over-cells reference when scored inside a swept family, so every cell reported the SAME
    # interval — the argmax cell's — even when it excluded the cell's own point estimate (the
    # shipped cell read `0.421 [0.606, 0.788]`). A CI that excludes its estimate is not an
    # interval for it: each cell reports its marginal, and the family-wise correction applies to
    # the AUROC null only (`null_auroc_family`), never to the precision interval.
    precision_ci: tuple[float, float]
    # The quiet arm's interval, on the SAME footing as `precision_ci` (same estimator, same
    # resamples) because the outcome figure compares the two to call a direction. It is required
    # rather than defaulted: a default would let a caller ship the comparison against an interval
    # nobody computed, which is the defect this field exists to close.
    quiet_ci: tuple[float, float]
    null_auroc: metrics.NullResult
    # THE GATE'S NULL: the max-over-cells family-wise null, because `run_eval._status` reads the
    # sweep as a family (OK if ANY cell clears). Each observed AUROC is gated against the
    # distribution of the MAX AUROC over all swept cells under one shared shuffle — the same maxT
    # construction the prefix half uses across depths. None means the cell was hand-built or scored
    # alone (a family of one IS the marginal null above, which the gate then reads instead).
    null_auroc_family: metrics.NullResult | None = None
    # ── THE RUN-LENGTH REFERENCE (disclosure, never a gate) ──
    # `null_auroc_length` is a MARGINAL length-stratified null: failure labels permuted within
    # equal-count length bins, so the length→failure association is preserved while the
    # fired→failure link is destroyed. It answers "is firing predictive BEYOND what the lengths of
    # the fired runs alone predict". `length_baseline_auroc` is the AUROC of the pure "run length
    # >= t" predictor at THIS cell's flag count — the number a reader would get if run length were
    # the whole story. Both exist because the challenge-block gate deliberately removes the
    # length→failure association along with everything else (labels move across challenges), so a
    # cell can clear that gate while most of its excess over chance is length selection; on this
    # corpus that is exactly what happens (n=30: observed 0.662, length-only 0.565). The
    # family-wise gate stays on the challenge-block null; these two are the honest disclosure that
    # separates the recurrence-specific share.
    length_baseline_auroc: float | None = None
    null_auroc_length: metrics.NullResult | None = None

    @property
    def gate_null(self) -> metrics.NullResult:
        """The null the admissibility gate reads — family-wise within a swept family."""
        return self.null_auroc if self.null_auroc_family is None else self.null_auroc_family

    @property
    def has_skill(self) -> bool:
        """Skill means the family-wise null AND the precision interval both clear their bars."""
        if self.precision is None:
            return False
        return self.gate_null.beats_null and self.precision_ci[0] > self.base_failure_rate

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
            # The FAMILY-WISE null the admissibility gate reads — the max-over-cells reference.
            # None when the cell was hand-built or scored alone; the gate reads the marginal then.
            "null_auroc_familywise": (
                None if self.null_auroc_family is None else self.null_auroc_family.to_dict()
            ),
            # The run-length reference: what a pure length>=t predictor scores at this flag count,
            # and the length-stratified null. Disclosure, never a gate — the gate stays on the
            # challenge-block null above. Read `null_auroc` against BOTH: an observed AUROC mostly
            # explained by the length baseline is length selection, not recurrence.
            "length_baseline_auroc": _rounded(self.length_baseline_auroc),
            "null_auroc_length_stratified": (
                None if self.null_auroc_length is None else self.null_auroc_length.to_dict()
            ),
            "has_skill": self.has_skill,
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
    # Total steps per trajectory. This is the run-length axis the high-threshold cells trade on:
    # firing at escalate_after_n=N needs >= N same-key failing steps, so a cell's fired vector is
    # mechanically correlated with length, and run length is outcome-correlated on this corpus.
    lengths: Sequence[int]


def _scored_for(
    trajectories: Sequence[Trajectory],
    point: replay.GridPoint,
    *,
    count_from_first_edit: bool = False,
) -> _Scored:
    """Replay one configuration over every trajectory — one alignment the cells share."""
    fired: list[bool] = []
    failed: list[bool] = []
    groups: list[str] = []
    lengths: list[int] = []
    for traj in trajectories:
        decision = replay.replay_config(
            traj, point.to_config(), count_from_first_edit=count_from_first_edit
        )
        fired.append(decision.escalated)
        failed.append(not traj.header.terminal_resolved)
        lengths.append(traj.header.n_steps)
        # The SAME grouping key the prefix half's grouped CV uses (challenge, i.e. instance id):
        # the two halves must not disagree about what an independent observation is.
        groups.append(features.group_of(traj))
    return _Scored(fired, failed, groups, lengths)


def evaluate_cell(
    trajectories: Sequence[Trajectory],
    point: replay.GridPoint,
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
    count_from_first_edit: bool = False,
) -> PolicyCell:
    """Replay one configuration over every trajectory and score it at the trajectory level."""
    return _cell(
        point,
        _scored_for(trajectories, point, count_from_first_edit=count_from_first_edit),
        n_permutations,
        seed,
    )


def _cell(
    point: replay.GridPoint,
    s: _Scored,
    n_permutations: int,
    seed: int,
    *,
    family_draws: Sequence[float] | None = None,
) -> PolicyCell:
    """Assemble one cell's 2x2, its clustered arm intervals, and its clustered permutation null."""
    fired, failed = s.fired, s.failed
    tp = sum(f and y for f, y in zip(fired, failed, strict=True))
    scores = [1.0 if f else 0.0 for f in fired]
    fired_ci, quiet_ci = _arm_intervals(fired, failed, s.groups, seed=seed)
    null_auroc = _clustered_null(scores, failed, s.groups, n_permutations=n_permutations, seed=seed)
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
        null_auroc=null_auroc,
        null_auroc_family=(
            None
            if family_draws is None
            else metrics.permutation_null(metrics.auroc(scores, failed), family_draws)
        ),
        length_baseline_auroc=_length_only_auroc(fired, failed, s.lengths, fire_count=sum(fired)),
        null_auroc_length=_length_stratified_null(
            scores, failed, s.lengths, n_permutations=n_permutations, seed=seed
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


def _length_bins(lengths: Sequence[int], n_bins: int = 10) -> list[int]:
    """Equal-count length bins: the axis a length-stratified permutation must not break."""
    # Equal-count rather than fixed-width: every bin keeps a usable row count AND the corpus's
    # monotone length→failure association, which is the association the null must preserve. Fixed
    # width would leave the tail bins (a handful of 200-step runs) too small to permute.
    out = [0] * len(lengths)
    ranked = sorted(range(len(lengths)), key=lambda i: lengths[i])
    for rank, index in enumerate(ranked):
        out[index] = min(rank * n_bins // max(len(ranked), 1), n_bins - 1)
    return out


def _permute_within_length_bins(
    labels: Sequence[bool], bins: Sequence[int], rng: random.Random
) -> list[bool]:
    """Shuffle failure labels INSIDE each length bin, so every bin's failure rate is fixed."""
    by_bin: dict[int, list[int]] = {}
    for index, bin_id in enumerate(bins):
        by_bin.setdefault(bin_id, []).append(index)
    out = list(labels)
    for indices in by_bin.values():
        block = [labels[i] for i in indices]
        rng.shuffle(block)
        for index, value in zip(indices, block, strict=True):
            out[index] = value
    return out


def _length_stratified_null(
    scores: Sequence[float],
    labels: Sequence[bool],
    lengths: Sequence[int],
    *,
    n_permutations: int,
    seed: int,
) -> metrics.NullResult:
    """The AUROC null under within-length-bin label shuffles — recurrence BEYOND run length."""
    # The challenge-block null (`_clustered_null`) permutes labels across challenges, which breaks
    # the length→failure association along with everything else — so a cell whose firing is really
    # run-length selection can clear it. Here failures stay inside their length bin (every bin's
    # failure rate is preserved), so the null keeps the length→failure link the fired vector rides
    # on, and a cell only clears it if firing predicts failure for runs of comparable length. It is
    # deliberately MARGINAL (one cell's own draws, no max-over-cells correction): it is a disclosure
    # reference, not the gate.
    bins = _length_bins(lengths)
    rng = random.Random(seed)
    draws = [
        metrics.auroc(scores, _permute_within_length_bins(labels, bins, rng))
        for _ in range(n_permutations)
    ]
    return metrics.permutation_null(metrics.auroc(scores, labels), draws)


def _length_only_auroc(
    fired: Sequence[bool], failed: Sequence[bool], lengths: Sequence[int], *, fire_count: int
) -> float | None:
    """The AUROC a pure 'run length >= t' predictor reaches at THIS cell's flag count, or None."""
    # Fires no rows => the length predictor is constant (all quiet), AUROC undefined at a nonzero
    # flag budget. Otherwise: find the length threshold whose `length >= t` flag set is closest to
    # the cell's fire count, and score it. That is the number a reader would get if run length
    # were the whole story — the honest share of the observed AUROC that is NOT recurrence.
    if fire_count <= 0 or not lengths:
        return None
    ordered = sorted(lengths)
    threshold = ordered[-fire_count]
    flag = [length >= threshold for length in lengths]
    return metrics.auroc([1.0 if f else 0.0 for f in flag], failed)


def family_null_draws(
    score_vectors: Sequence[Sequence[float]],
    labels: Sequence[bool],
    groups: Sequence[str],
    *,
    n_permutations: int,
    seed: int,
) -> list[float]:
    """One shared challenge-block shuffle scored at EVERY score vector, keeping the max AUROC."""
    # The same maxT construction as `_family_null_draws`, generalized to arbitrary score vectors
    # (there the vectors are binary fired masks; the recurrence ROC passes continuous
    # `max_recurrence` scores). Challenge identity is the exchangeable unit, so the shuffle is
    # block-permutation — a global shuffle would understate the band exactly as `_clustered_null`
    # documents for the policy cells.
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_permutations):
        shuffled = _permute_clusters(labels, groups, rng)
        draws.append(max(metrics.auroc(scores, shuffled) for scores in score_vectors))
    return draws


def null_roc_band(
    score_vectors: Sequence[Sequence[float]],
    labels: Sequence[bool],
    groups: Sequence[str],
    *,
    n_permutations: int,
    seed: int,
    n_fpr: int = 101,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """The 2.5-97.5% TPR envelope of the null ROC curves, on a shared FPR grid."""
    # The recurrence ROC's detection floor is drawn as a BAND, not just a scalar CI: on ROC axes
    # the chance region is where permuted-label curves live, so one grid of FPR values collects
    # every permutation's TPR at each point and the band is the central 95% of those columns.
    # Reuses the SAME challenge-block shuffle as `family_null_draws` so the scalar AUROC null and
    # the drawn band are one construction, not two.
    rng = random.Random(seed)
    grid = np.linspace(0.0, 1.0, n_fpr)
    columns: list[list[float]] = [[] for _ in grid]
    for _ in range(n_permutations):
        shuffled = _permute_clusters(labels, groups, rng)
        for scores in score_vectors:
            fpr_tpr = metrics.roc_operating_points(scores, list(shuffled))
            xs = [p[0] for p in fpr_tpr]
            ys = [p[1] for p in fpr_tpr]
            tpr = np.interp(grid, xs, ys)
            for index, value in enumerate(tpr):
                columns[index].append(float(value))
    lo = tuple(float(np.percentile(col, 2.5)) for col in columns)
    hi = tuple(float(np.percentile(col, 97.5)) for col in columns)
    return tuple(float(x) for x in grid), lo, hi


def _family_null_draws(scored: Sequence[_Scored], *, n_permutations: int, seed: int) -> list[float]:
    """One shared challenge-block shuffle scored at EVERY cell, keeping the max AUROC."""
    # The sweep's cells share every trajectory (only the knobs differ), so their fired vectors are
    # strongly dependent — the same way the prefix depths' row sets nest. A gate that read each
    # cell at a nominal 2.5% would let the best of N cells clear by luck; the maxT construction
    # (one shuffle, max over cells) is the exact family-wise reference.
    labels = list(scored[0].failed)
    groups = list(scored[0].groups)
    score_vectors = [[1.0 if f else 0.0 for f in s.fired] for s in scored]
    return family_null_draws(
        score_vectors, labels, groups, n_permutations=n_permutations, seed=seed
    )


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
    count_from_first_edit: bool = False,
) -> list[PolicyCell]:
    """Every swept configuration, scored per trajectory, gated as ONE family."""
    # The cells share every trajectory (only the knobs differ), so their statistics are dependent
    # and the admissibility gate reads the max over them — each null is the maxT family-wise null
    # built from the SAME shuffles (`_family_null_draws`). Each cell's PRECISION interval stays
    # its own marginal challenge bootstrap: the family-wise correction is applied to the AUROC
    # null only, because a CI that excludes a cell's own point estimate is not an interval for it.
    # A single-cell call (or a hand-built grid of one) has a family of one, where the family null
    # IS the marginal null above.
    if not grid:
        raise ValueError("evaluate requires at least one grid point (the shipped cell is default)")
    scored = [
        _scored_for(trajectories, point, count_from_first_edit=count_from_first_edit)
        for point in grid
    ]
    family_draws = _family_null_draws(scored, n_permutations=n_permutations, seed=seed)
    return [
        _cell(
            point,
            s,
            n_permutations,
            seed,
            family_draws=family_draws,
        )
        for point, s in zip(grid, scored, strict=True)
    ]


__all__ = ["PolicyCell", "evaluate", "evaluate_cell", "family_null_draws", "null_roc_band"]
