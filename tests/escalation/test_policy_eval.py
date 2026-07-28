"""The shipped policy graded per trajectory: counts are per cell, direction is stated honestly."""

from __future__ import annotations

import pytest

from benchmark.escalation import datasets, policy_eval, replay
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
    assert cell.precision == 0.0
    assert cell.p_fail_given_quiet == pytest.approx(1.0)
    assert cell.lift < 1.0


def test_a_correct_policy_reports_lift_above_one() -> None:
    corpus = [_thrashing(f"bad{i}", resolved=False) for i in range(10)]
    corpus += [_quiet(f"good{i}", resolved=True) for i in range(10)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    assert cell.precision == pytest.approx(1.0)
    assert cell.lift > 1.0
    assert cell.null_auroc.beats_null


def test_lead_times_are_split_by_terminal_outcome() -> None:
    corpus = [_thrashing("bad", resolved=False), _thrashing("good", resolved=True)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    # Both fired, so each outcome class contributes exactly one lead time — the old figure pooled
    # them across grid cells and drew each event six times over.
    assert len(cell.lead_times_failed) == 1
    assert len(cell.lead_times_resolved) == 1


def test_precision_interval_is_reported_and_contains_the_estimate() -> None:
    corpus = [_thrashing(f"t{i}", resolved=i < 3) for i in range(10)]
    cell = policy_eval.evaluate_cell(corpus, replay.GridPoint(2, 5), n_permutations=_PERMUTATIONS)
    low, high = cell.precision_ci
    assert low <= cell.precision <= high


def test_default_grid_pins_the_inert_knobs() -> None:
    # stale_window and ladder were measured inert (12 cells -> 2 distinct score vectors), so they
    # are pinned rather than swept; escalate_after_n=1 is included because it is the one level
    # where precision separates and the report must show it next to the shipped default.
    assert len(datasets.DEFAULT_GRID) == 3
    assert sorted(p.escalate_after_n for p in datasets.DEFAULT_GRID) == [1, 2, 3]
    assert len({p.stale_window for p in datasets.DEFAULT_GRID}) == 1
    assert len({p.ladder for p in datasets.DEFAULT_GRID}) == 1
