from __future__ import annotations

import pytest

from shunt.router.budget import ConservativeGate, ExplorationBudget


def test_bootstrap_allows_exploration_before_any_spend() -> None:
    assert ExplorationBudget(0.4).can_explore() is True


def test_zero_frac_never_explores_even_during_bootstrap() -> None:
    # Regression: exploratory spend never raises exploit_cost, so a frac=0 budget whose
    # bootstrap branch ran first would stay open forever and permit unbounded exploration.
    b = ExplorationBudget(0.0)
    assert b.can_explore() is False
    b.record(baseline_cost=0.0, actual_cost=1.0, is_exploratory=True)
    assert b.can_explore() is False


def test_negative_frac_rejected() -> None:
    with pytest.raises(ValueError, match="explore_budget_frac"):
        ExplorationBudget(-0.1)


def test_explore_capped_at_fraction_of_exploit() -> None:
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)  # exploit spend = 10
    # explore = 3, ratio 0.3 <= 0.4
    b.record(baseline_cost=0.0, actual_cost=3.0, is_exploratory=True)
    assert b.can_explore() is True
    # explore = 5, ratio 0.5 > 0.4
    b.record(baseline_cost=0.0, actual_cost=2.0, is_exploratory=True)
    assert b.can_explore() is False


def test_explore_ratio_reported() -> None:
    b = ExplorationBudget(0.5)
    b.record(baseline_cost=8.0, actual_cost=8.0, is_exploratory=False)
    b.record(baseline_cost=0.0, actual_cost=2.0, is_exploratory=True)
    assert b.explore_ratio == pytest.approx(0.25)


def test_exploit_spend_never_ages_out_of_the_cap() -> None:
    # Regression: a cumulative cap must not let a burst of exploration re-open unbounded
    # exploration by aging exploit spend out of a window.
    b = ExplorationBudget(explore_budget_frac=0.4)
    # exploit baseline, forever counted
    b.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)
    for _ in range(100):
        # 100 exploratory pulls
        b.record(baseline_cost=0.0, actual_cost=1.0, is_exploratory=True)
    assert b.can_explore() is False  # explore 100 >> 0.4*10; stays bound
    assert b.explore_ratio == pytest.approx(10.0)


def test_conservative_gate_alpha_zero_never_downshifts() -> None:
    g = ConservativeGate(alpha=0.0)
    assert g.allows_downshift() is False
    g.record_outcome(downshift=True, success=True)
    assert g.allows_downshift() is False  # alpha=0 never permits, regardless of slack


def test_conservative_gate_banks_slack_from_downshift_successes() -> None:
    g = ConservativeGate(alpha=0.1)
    assert g.allows_downshift() is False
    g.record_outcome(downshift=True, success=True)
    assert g.slack == pytest.approx(1.0)
    assert g.allows_downshift() is True
    g.record_outcome(downshift=True, success=False)
    assert g.slack == pytest.approx(0.0)


def test_conservative_gate_ignores_non_downshift_outcomes() -> None:
    # An upward-exploration success must NOT unlock a downshift.
    g = ConservativeGate(alpha=0.5)
    g.record_outcome(downshift=False, success=True)
    assert g.slack == pytest.approx(0.0)
    assert g.allows_downshift() is False


def test_conservative_gate_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ConservativeGate(alpha=1.5)


def test_cap_binds_in_an_all_exploratory_run() -> None:
    # The defect: crediting the baseline only on the exploit branch left the denominator at
    # 0 in a run where every decision explored, so the bootstrap allowance never closed and
    # the fraction never bound (measured: 2000 exploratory decisions, cap never applied).
    b = ExplorationBudget(0.4)
    allowed = 0
    for _ in range(2000):
        if not b.can_explore():
            break
        b.record(baseline_cost=1.0, actual_cost=2.0, is_exploratory=True)
        allowed += 1

    assert not b.can_explore()
    assert allowed < 2000


def test_a_cheaper_exploration_consumes_no_budget() -> None:
    # Exploring *down* costs less than the baseline; the extra is clamped at 0, so a
    # downshift must never eat the exploration budget.
    b = ExplorationBudget(0.4)
    for _ in range(50):
        b.record(baseline_cost=1.0, actual_cost=0.2, is_exploratory=True)

    assert b.explore_ratio == 0.0
    assert b.can_explore()


def test_exploration_extra_is_measured_against_the_baseline_not_the_full_cost() -> None:
    b = ExplorationBudget(1.0)
    b.record(baseline_cost=10.0, actual_cost=13.0, is_exploratory=True)

    assert b.explore_ratio == pytest.approx(0.3)  # 3 extra over a 10 baseline, not 13
