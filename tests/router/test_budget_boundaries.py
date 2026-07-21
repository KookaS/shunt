"""Exact-boundary behaviour of exploration guardrails — check >= / <= semantics."""

from __future__ import annotations

import pytest

from shunt.router.budget import ConservativeGate, ExplorationBudget

# ── ConservativeGate: the alpha=1.0 "no conservatism" extreme ────────────────


def test_gate_alpha_one_allows_downshift_with_zero_slack() -> None:
    # Docstring contract: alpha=1 → slack >= (1 - 1) = 0, true from the start.
    # A strict `>` would make the extreme silently as conservative as alpha=0.
    gate = ConservativeGate(alpha=1.0)
    assert gate.slack == 0.0
    assert gate.allows_downshift() is True


def test_gate_alpha_one_still_blocks_after_a_verified_failure() -> None:
    # "No conservatism" is not "no evidence": a failed downshift drives slack to -1,
    # which is below the 0 threshold even at alpha=1.
    gate = ConservativeGate(alpha=1.0)
    gate.record_outcome(downshift=True, success=False)
    assert gate.slack == -1.0
    assert gate.allows_downshift() is False
    gate.record_outcome(downshift=True, success=True)
    assert gate.allows_downshift() is True  # back to exactly 0.0 → permitted


def test_gate_alpha_one_ignores_non_downshift_outcomes() -> None:
    gate = ConservativeGate(alpha=1.0)
    for _ in range(5):
        gate.record_outcome(downshift=False, success=False)
    assert gate.slack == 0.0
    assert gate.allows_downshift() is True


def test_gate_alpha_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ConservativeGate(alpha=1.0001)


def test_gate_slack_exactly_at_threshold_is_permitted() -> None:
    # alpha=0.5 → threshold 0.5; one banked success (slack 1.0) clears it, and the
    # inclusive comparison means landing exactly on the threshold permits a downshift.
    gate = ConservativeGate(alpha=0.5)
    assert gate.allows_downshift() is False
    gate.record_outcome(downshift=True, success=True)
    assert gate.allows_downshift() is True


def test_gate_alpha_zero_is_the_opposite_extreme_of_alpha_one() -> None:
    banked = ConservativeGate(alpha=0.0)
    for _ in range(10):
        banked.record_outcome(downshift=True, success=True)
    assert banked.slack == 10.0
    assert banked.allows_downshift() is False  # alpha=0 never permits, whatever the slack
    assert ConservativeGate(alpha=1.0).allows_downshift() is True


# ── ExplorationBudget: exactly at the cap ───────────────────────────────────


def test_budget_exactly_at_cap_still_allows_exploration() -> None:
    # extra == frac * baseline is INCLUSIVE (`<=`). A flip to `<` would close the budget
    # one exploration early and silently shrink the learning rate.
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)  # baseline 10
    b.record(baseline_cost=0.0, actual_cost=4.0, is_exploratory=True)  # extra 4 == 0.4*10
    assert b.explore_ratio == pytest.approx(0.4)
    assert b.can_explore() is True


def test_budget_one_cent_past_the_cap_closes() -> None:
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)
    b.record(baseline_cost=0.0, actual_cost=4.01, is_exploratory=True)
    assert b.can_explore() is False


def test_budget_boundary_holds_when_the_exploration_pays_its_own_baseline() -> None:
    # The realistic shape: an exploratory decision records its greedy baseline too, and
    # only the EXTRA over that baseline consumes budget.
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.record(baseline_cost=5.0, actual_cost=7.0, is_exploratory=True)  # extra 2, baseline 5
    b.record(baseline_cost=5.0, actual_cost=5.0, is_exploratory=False)  # baseline 10 total
    assert b.explore_ratio == pytest.approx(0.2)
    b.record(baseline_cost=0.0, actual_cost=2.0, is_exploratory=True)  # extra 4 == 0.4*10
    assert b.explore_ratio == pytest.approx(0.4)
    assert b.can_explore() is True  # exactly at the cap: still open


def test_budget_downshift_exploration_consumes_nothing() -> None:
    # actual < baseline → the extra is clamped at 0: exploring cheap is free, and it
    # still credits the baseline, so it makes the cap *more* permissive, never less.
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.record(baseline_cost=10.0, actual_cost=1.0, is_exploratory=True)
    assert b.explore_ratio == 0.0
    assert b.can_explore() is True


def test_budget_reopens_when_baseline_spend_grows_past_the_boundary() -> None:
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)
    b.record(baseline_cost=0.0, actual_cost=5.0, is_exploratory=True)  # extra 5 > 0.4*10
    assert b.can_explore() is False
    b.record(baseline_cost=2.5, actual_cost=2.5, is_exploratory=False)  # baseline 12.5
    assert b.explore_ratio == pytest.approx(0.4)  # 5 == 0.4*12.5 — exactly at the cap
    assert b.can_explore() is True


def test_cap_binds_when_no_cost_is_ever_reported() -> None:
    """The cost-blind regime: unknown pricing left the cap permanently un-enforceable."""
    # A provider that omits `usage.cost` leaves every baseline at 0.0, so the spend
    # ratio has no denominator. Before the rate bound, this loop explored 2000/2000.
    budget = ExplorationBudget(explore_budget_frac=0.4)
    explored = 0
    for _ in range(2000):
        if budget.can_explore():
            explored += 1
            budget.record(baseline_cost=0.0, actual_cost=5.0, is_exploratory=True)
        else:
            budget.record(baseline_cost=0.0, actual_cost=0.0, is_exploratory=False)
    assert explored < 2000
    assert explored / 2000 <= 0.4 + 1e-9


def test_cost_blind_regime_still_honours_frac_zero() -> None:
    budget = ExplorationBudget(explore_budget_frac=0.0)
    budget.record(baseline_cost=0.0, actual_cost=0.0, is_exploratory=False)
    assert budget.can_explore() is False


def test_first_decision_is_still_allowed_to_explore() -> None:
    assert ExplorationBudget(explore_budget_frac=0.4).can_explore() is True
