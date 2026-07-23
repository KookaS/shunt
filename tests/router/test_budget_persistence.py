from __future__ import annotations

import logging

import pytest

from shunt.router.budget import ConservativeGate, ExplorationBudget


def test_budget_cap_survives_restart() -> None:
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)  # exploit = 10
    b.record(baseline_cost=0.0, actual_cost=6.0, is_exploratory=True)  # explore = 6 > 0.4*10
    assert b.can_explore() is False
    snap = b.snapshot()

    restored = ExplorationBudget(explore_budget_frac=0.4)
    restored.restore(snap)
    assert restored.can_explore() is False
    assert restored.snapshot() == snap


def test_gate_slack_survives_restart() -> None:
    g = ConservativeGate(alpha=1.0)  # alpha=1 → any slack >= 0 unlocks
    g.record_outcome(downshift=True, success=True)
    g.record_outcome(downshift=True, success=True)
    assert g.allows_downshift() is True
    snap = g.snapshot()

    restored = ConservativeGate(alpha=1.0)
    restored.restore(snap)
    assert restored.allows_downshift() is True
    assert restored.slack == g.slack


def test_gate_negative_slack_survives_restart() -> None:
    g = ConservativeGate(alpha=0.1)
    g.record_outcome(downshift=True, success=False)  # slack = -1
    assert g.allows_downshift() is False
    restored = ConservativeGate(alpha=0.1)
    restored.restore(g.snapshot())
    assert restored.slack == -1.0
    assert restored.allows_downshift() is False


def test_config_frac_read_fresh_not_restored() -> None:
    # A snapshot taken under one budget_frac must not resurrect that frac on restore:
    # config comes from router.yaml and may legitimately change between runs.
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.record(baseline_cost=10.0, actual_cost=10.0, is_exploratory=False)
    b.record(baseline_cost=0.0, actual_cost=5.0, is_exploratory=True)  # ratio 0.5 > 0.4
    assert b.can_explore() is False

    loosened = ExplorationBudget(explore_budget_frac=0.6)  # new run raised the cap
    loosened.restore(b.snapshot())
    # 0.5 <= 0.6 now — the fresh config governs, not the snapshot's implied 0.4.
    assert loosened.can_explore() is True


def test_gate_alpha_read_fresh_not_restored() -> None:
    g = ConservativeGate(alpha=0.1)
    g.record_outcome(downshift=True, success=True)  # slack = 1
    reconfigured = ConservativeGate(alpha=0.0)  # alpha=0 → never downshift, regardless of slack
    reconfigured.restore(g.snapshot())
    assert reconfigured.allows_downshift() is False


def test_restore_none_keeps_zero_state() -> None:
    b = ExplorationBudget(explore_budget_frac=0.4)
    b.restore(None)
    assert b.snapshot() == {
        "explore_cost": 0.0,
        "exploit_cost": 0.0,
        "decisions": 0,
        "explorations": 0,
    }
    g = ConservativeGate(alpha=0.1)
    g.restore(None)
    assert g.slack == 0.0


def test_restore_partial_snapshot_falls_back_to_zero(caplog: pytest.LogCaptureFixture) -> None:
    b = ExplorationBudget(explore_budget_frac=0.4)
    with caplog.at_level(logging.WARNING):
        b.restore({"explore_cost": 1.0})  # missing keys
    # Partial → all-zero, never half-applied.
    assert b.snapshot()["decisions"] == 0
    assert b.snapshot()["explore_cost"] == 0.0
    assert caplog.records  # logged once, did not crash


def test_restore_corrupt_snapshot_falls_back_to_zero_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    b = ExplorationBudget(explore_budget_frac=0.4)
    with caplog.at_level(logging.WARNING):
        b.restore(
            {
                "explore_cost": "not-a-number",
                "exploit_cost": 1,
                "decisions": 1,
                "explorations": 0,
            }
        )
    assert b.snapshot()["exploit_cost"] == 0.0
    assert caplog.records

    g = ConservativeGate(alpha=0.1)
    with caplog.at_level(logging.WARNING):
        g.restore({"slack": None})
    assert g.slack == 0.0


def test_restore_infinite_cost_falls_back_to_zero_not_uncapped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A tampered/corrupt row can carry Infinity (json round-trips it by default). If accepted,
    # exploit_cost=inf makes can_explore() True forever — the money cap silently disabled.
    b = ExplorationBudget(explore_budget_frac=0.4)
    with caplog.at_level(logging.WARNING):
        b.restore(
            {
                "explore_cost": 0.0,
                "exploit_cost": float("inf"),
                "decisions": 5,
                "explorations": 5,
            }
        )
    assert b.snapshot() == {
        "explore_cost": 0.0,
        "exploit_cost": 0.0,
        "decisions": 0,
        "explorations": 0,
    }
    assert b.can_explore() is True  # fresh bootstrap, NOT an unconditionally-uncapped budget
    assert caplog.records


def test_restore_nan_counter_falls_back_to_zero_without_crash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # int(nan) raises ValueError and int(inf) raises OverflowError — a non-finite counter must
    # degrade to zero, never abort startup (the never-crash contract).
    b = ExplorationBudget(explore_budget_frac=0.4)
    with caplog.at_level(logging.WARNING):
        b.restore(
            {
                "explore_cost": 0.0,
                "exploit_cost": 1.0,
                "decisions": float("nan"),
                "explorations": 0,
            }
        )
    assert b.snapshot()["decisions"] == 0
    assert b.snapshot()["exploit_cost"] == 0.0  # all-or-nothing: nothing half-applied
    assert caplog.records


def test_gate_restore_infinite_slack_falls_back_to_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    g = ConservativeGate(alpha=0.1)
    with caplog.at_level(logging.WARNING):
        g.restore({"slack": float("inf")})
    assert g.slack == 0.0
    assert g.allows_downshift() is False
    assert caplog.records
