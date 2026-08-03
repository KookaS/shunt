"""The shipped policy graded per trajectory: counts are per cell, direction is stated honestly."""

from __future__ import annotations

import dataclasses
import math
import random

import pytest

from benchmark.escalation import datasets, metrics, policy_eval, replay
from benchmark.escalation.schema import Trajectory
from tests.escalation.factories import make_step, make_trajectory

_PERMUTATIONS = 200


def _thrashing(tid: str, *, resolved: bool, n: int = 8):  # type: ignore[no-untyped-def]
    """A run that repeats one failing check id, so the recurrence trigger fires."""
    steps = [
        make_step(step_index=i, decision_index=i, success=False, failing_check_id="pkg::t")
        for i in range(n)
    ]
    return make_trajectory(steps, trajectory_id=tid, terminal_resolved=resolved)


def _quiet(tid: str, *, resolved: bool, n: int = 8):  # type: ignore[no-untyped-def]
    """A run with no failing check id at all: the trigger cannot fire."""
    steps = [make_step(step_index=i, decision_index=i) for i in range(n)]
    return make_trajectory(steps, trajectory_id=tid, terminal_resolved=resolved)


def test_n_escalated_counts_trajectories_not_sweep_cells() -> None:
    # The old report summed n_escalated across all 12 grid cells and printed 4980 next to
    # n_trajectories=799 — 415 real escalations shown as 4980. Each cell now owns its own count.
    corpus = [_thrashing(f"t{i}", resolved=False) for i in range(5)]
    corpus += [_quiet(f"q{i}", resolved=True) for i in range(5)]
    cells = policy_eval.evaluate(corpus, datasets.DEFAULT_GRID, n_permutations=_PERMUTATIONS)
    assert len(cells) == len(datasets.DEFAULT_GRID)
    for cell in cells:
        assert cell.n_escalated <= cell.n_trajectories == len(corpus)
        # Every thrashing run repeats one key `n` times, so the recurrence trigger fires on the 5
        # thrashing runs whenever escalate_after_n <= the run's failure count (8). Cells past the
        # fixture's count structurally cannot fire — the grid's high-n tail is there to probe the
        # corpus's own edge, not this synthetic fixture's, so the invariant is the per-cell bound
        # above plus the fired-count identity below, never a fixed 5 for every cell.
        assert cell.n_escalated == 5 or cell.n_escalated == 0
        if cell.n_escalated == 5:
            assert cell.tp + cell.fp == 5


def test_confusion_partitions_the_corpus_exactly_once() -> None:
    corpus = [_thrashing(f"t{i}", resolved=i % 2 == 0) for i in range(6)]
    corpus += [_quiet(f"q{i}", resolved=i % 3 == 0) for i in range(6)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    assert cell.tp + cell.fp + cell.fn + cell.tn == len(corpus)
    assert cell.tp + cell.fp == cell.n_escalated


def test_an_inverted_policy_reports_lift_below_one() -> None:
    # This is the direction check. When the policy fires only on runs that RESOLVE, precision must
    # come out below the base rate and lift below 1.0 — the honest reading, not a suppressed one.
    corpus = [_thrashing(f"good{i}", resolved=True) for i in range(10)]
    corpus += [_quiet(f"bad{i}", resolved=False) for i in range(10)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    # Asserted against the Optional explicitly: `precision`/`lift` are `float | None`, and
    # `None == 0.0` is False while `None < 1.0` raises — so a bare comparison here would pass only
    # because this fixture happens to fire, and would report the wrong failure when it stopped.
    assert cell.precision is not None
    assert cell.lift is not None
    assert cell.precision == pytest.approx(0.0)
    assert cell.p_fail_given_quiet == pytest.approx(1.0)
    assert cell.lift < 1.0


def test_a_correct_policy_reports_lift_above_one() -> None:
    corpus = [_thrashing(f"bad{i}", resolved=False) for i in range(10)]
    corpus += [_quiet(f"good{i}", resolved=True) for i in range(10)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    assert cell.precision == pytest.approx(1.0)
    assert cell.lift > 1.0
    assert cell.null_auroc.beats_null


def test_the_cell_carries_no_escalation_timing_arrays() -> None:
    # `lead_times_*` / `first_fire_*` fed exactly one figure, `lead_time_by_outcome`, and that
    # figure was deleted: the policy fires in the first two decisions of nearly every run it flags,
    # so a lead time is the run length minus a constant and no timing claim survives. Data with no
    # consumer is the thing that quietly grows back, so the absence is pinned here.
    corpus = [_thrashing("bad", resolved=False), _thrashing("good", resolved=True)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    for gone in ("lead_times_failed", "lead_times_resolved", "first_fire_failed"):
        assert not hasattr(cell, gone)


def test_precision_interval_is_reported_and_contains_the_estimate() -> None:
    corpus = [_thrashing(f"t{i}", resolved=i < 3) for i in range(10)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    assert cell.precision is not None
    low, high = cell.precision_ci
    assert low <= cell.precision <= high


def test_a_single_cell_family_keeps_the_marginal_precision_interval() -> None:
    # The family-wise max-over-cells correction only kicks in for a family of MORE than one cell:
    # a single swept cell IS its own reference, so it must carry the exact marginal bootstrap.
    corpus = [_thrashing(f"t{i}", resolved=i < 3) for i in range(10)]
    corpus += [_quiet(f"q{i}", resolved=True) for i in range(10)]
    point = replay.GridPoint(2, 5)
    swept = policy_eval.evaluate(corpus, [point], n_permutations=_PERMUTATIONS)
    assert len(swept) == 1
    alone = policy_eval.evaluate_cell(corpus, point, n_permutations=_PERMUTATIONS)
    assert swept[0].precision_ci == alone.precision_ci


def test_every_fired_cell_reports_its_own_marginal_precision_interval() -> None:
    # The bug this replaces put the SAME family-wise max-over-cells reference on every cell of a
    # swept family — the argmax cell's interval — so a cell whose estimate sat outside it (the
    # shipped cell read `0.421 [0.606, 0.788]`) reported a CI that excluded its own point
    # estimate. Each cell now carries its own marginal bootstrap: the CI contains the estimate it
    # is an interval for, and equals what a lone `evaluate_cell` of the same point would produce.
    corpus = [_thrashing(f"f{i}", resolved=False, n=30) for i in range(10)]
    corpus += [_thrashing(f"r{i}", resolved=True, n=5) for i in range(10)]
    corpus += [_quiet(f"q{i}", resolved=True) for i in range(10)]
    low, high = replay.GridPoint(2, 1000), replay.GridPoint(30, 1000)
    cells = policy_eval.evaluate(corpus, [low, high], n_permutations=_PERMUTATIONS)
    assert len(cells) == 2
    best = max(cells, key=lambda c: c.null_auroc.observed)
    for cell in cells:
        if cell.precision is None:
            assert all(math.isnan(v) for v in cell.precision_ci)
            continue
        low, high_ci = cell.precision_ci
        assert low <= cell.precision <= high_ci
        marginal = policy_eval.evaluate_cell(
            corpus,
            replay.GridPoint(cell.escalate_after_n, cell.stale_window),
            n_permutations=_PERMUTATIONS,
        ).precision_ci
        assert cell.precision_ci == pytest.approx(marginal)
    # The AUROC family-wise null is untouched: `gate_null` IS the max-over-cells reference, and
    # the best cell still clears it (only the precision interval went marginal).
    assert best.gate_null is best.null_auroc_family
    assert best.gate_null.beats_null


def test_a_never_firing_configuration_has_no_precision_at_all() -> None:
    # The None contract lived only in the PLOT module's tests, so the module that owns the type
    # never pinned it. A cell that cannot fire has no P(fail | fired) to report: precision, lift
    # and the interval are undefined, and `to_dict` must carry that through as null rather than 0.
    corpus = [_quiet(f"q{i}", resolved=i % 2 == 0) for i in range(10)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    assert cell.n_escalated == 0
    assert cell.precision is None
    assert cell.lift is None
    assert all(math.isnan(v) for v in cell.precision_ci)
    assert cell.p_fail_given_quiet == pytest.approx(0.5)
    payload = cell.to_dict()
    assert payload["precision"] is None
    assert payload["p_fail_given_fired"] is None
    assert payload["lift"] is None


def test_a_configuration_that_fires_on_everything_has_no_quiet_arm() -> None:
    # The mirror of the case above, and the one the outcome figure used to read a clean bill of
    # health off: with no quiet rows there is no P(fail | not fired), so its interval is undefined
    # rather than the (0.0, 0.0) a row-level formula returns for zero trials.
    corpus = [_thrashing(f"t{i}", resolved=i < 4) for i in range(10)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    assert cell.fn + cell.tn == 0
    assert cell.p_fail_given_quiet is None
    assert all(math.isnan(v) for v in cell.quiet_ci)


def _clustered(trajectories: list[Trajectory], challenge: str) -> list[Trajectory]:
    """Re-key whole runs onto one challenge, so `group_of` sees them as a single cluster."""
    return [
        dataclasses.replace(t, header=dataclasses.replace(t.header, instance_id=challenge))
        for t in trajectories
    ]


def test_the_quiet_arm_interval_is_clustered_by_challenge_not_wilson_over_rows() -> None:
    # The clustering fix was applied to `precision_ci` and left off the quiet arm, and the outcome
    # figure then compared the two. Here 10 challenges each contribute 6 runs that share an
    # outcome, so a row-level interval sees 60 independent draws where there are 10. The bootstrap
    # must be materially wider than Wilson; equal widths would mean the row assumption came back.
    corpus: list[Trajectory] = []
    for c in range(10):
        runs = [_quiet(f"c{c}r{r}", resolved=c % 2 == 0) for r in range(6)]
        corpus += _clustered(runs, f"challenge{c}")
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    assert cell.fn + cell.tn == 60
    low, high = cell.quiet_ci
    assert low <= cell.p_fail_given_quiet <= high
    wilson_lo, wilson_hi = metrics.wilson_interval(cell.fn, cell.fn + cell.tn)
    assert high - low > 2 * (wilson_hi - wilson_lo)


def test_default_grid_sweeps_both_knobs_and_guarantees_the_shipped_cell() -> None:
    # stale_window and ladder were measured inert (12 cells -> 2 distinct score vectors) in the
    # OLD single-axis grid, which stopped at escalate_after_n=3 and never reached the regime where
    # the recurrence edge exists. The 2026-08-02 audit re-measured: the edge lives at HIGH
    # thresholds (n>=10 clears the family-wise null), and `_in_window` admits at most
    # `stale_window` events — so reaching n recurrences needs a window at least that wide, and the
    # grid sweeps both knobs. `ladder` stays pinned (the detection metric reads whether the policy
    # fired, not which rung it climbed).
    assert sorted({p.escalate_after_n for p in datasets.DEFAULT_GRID}) == [2, 5, 8, 10, 15, 20, 30]
    assert sorted({p.stale_window for p in datasets.DEFAULT_GRID}) == [10, 1000]
    assert len({p.ladder for p in datasets.DEFAULT_GRID}) == 1
    # The shipped configuration (escalate_after_n=2, stale_window=10) must be in-grid so the
    # report measures what ships, never only what scores better.
    assert any(p.escalate_after_n == 2 and p.stale_window == 10 for p in datasets.DEFAULT_GRID)


def test_policy_null_uses_block_permutation_not_within_challenge_shuffles() -> None:
    # The policy half's claim is UNCONDITIONAL ("given the policy fired, is this run likelier to
    # fail than one picked at random"), so the exchangeable unit is the whole CHALLENGE and the
    # null must be a BLOCK permutation: whole challenge blocks are shuffled and outcome labels
    # MOVE between challenges, with only the global multiset preserved. A within-challenge shuffle
    # removes exactly the between-challenge variation the unconditional statistic contains, giving
    # a too-narrow null (measured sd 0.0036 vs 0.0282 for the block shuffle at n=10). This test
    # pins the mechanism so the docs cannot drift from it again (2026-08-02: three shipped
    # documents described the policy null as a within-challenge shuffle while the code permuted
    # whole blocks; no test caught it).
    labels = [True, False, False, True, True, False, False, False, True, False]
    groups = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"]
    # A within-challenge permutation preserves every group's multiset in EVERY draw. A block
    # permutation cannot guarantee that, so over many draws at least one group's multiset must
    # change (a block swap of unequal composition is near-certain to land). We assert the two
    # procedures are distinguishable: the policy null must behave like block permutation.
    changed_any = False
    for seed in range(25):
        shuffled = policy_eval._permute_clusters(labels, groups, random.Random(seed))
        assert sorted(shuffled) == sorted(labels)  # global multiset preserved
        for g in {"a", "b", "c"}:
            orig = sorted(labels[i] for i, x in enumerate(groups) if x == g)
            perm = sorted(shuffled[i] for i, x in enumerate(groups) if x == g)
            if orig != perm:
                changed_any = True
                break
    assert changed_any
