"""The shipped policy graded per trajectory: counts are per cell, direction is stated honestly."""

from __future__ import annotations

import dataclasses
import math

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
        assert cell.n_escalated == 5


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


def test_default_grid_pins_the_inert_knobs() -> None:
    # stale_window and ladder were measured inert (12 cells -> 2 distinct score vectors), so they
    # are pinned rather than swept; escalate_after_n=1 is included because it is the one level
    # where precision separates and the report must show it next to the shipped default.
    assert len(datasets.DEFAULT_GRID) == 3
    assert sorted(p.escalate_after_n for p in datasets.DEFAULT_GRID) == [1, 2, 3]
    assert len({p.stale_window for p in datasets.DEFAULT_GRID}) == 1
    assert len({p.ladder for p in datasets.DEFAULT_GRID}) == 1
