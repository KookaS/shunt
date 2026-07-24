"""Per-step verified-outcome stamping — the change that lets escalation recurrence fire."""

# Proves the full chain end-to-end: mocked per-step VerifierResults (from an offline replay) →
# stamped StepViews carrying a REAL recurring dedup_key → `run_eval.evaluate` reports
# ``n_escalated > 0``. Real machinery, mocked only at the container boundary (unit mocking).

from __future__ import annotations

from benchmark.escalation import run_eval
from benchmark.escalation.normalize.mini_swe_agent import (
    MiniSweAgentParser,
    restamp_trajectory,
    stamp_step,
)
from benchmark.escalation.replay import GridPoint
from shunt.verifiers.parse import parse_test_outcome
from tests.escalation.factories import make_step, make_trajectory

_KEY = "tests/t.py::test_widget"


def _fail(node_id: str) -> object:
    return parse_test_outcome(f"{node_id} FAILED\nAssertionError", 1)


def _pass() -> object:
    return parse_test_outcome("2 passed", 0)


def _infra() -> object:
    return parse_test_outcome("ERROR collecting tests/t.py\nImportError", 2)


def _messages(n: int) -> list[dict[str, object]]:
    """A mini-swe-agent-shaped message list: n assistant turns each followed by a tool result."""
    msgs: list[dict[str, object]] = [{"role": "user", "content": "task"}]
    for i in range(n):
        msgs.append(
            {
                "role": "assistant",
                "content": f"edit {i}",
                "extra": {"actions": [{"command": f"echo {i}"}]},
            }
        )
        msgs.append({"role": "tool", "content": "ok", "extra": {"returncode": 0}})
    return msgs


# ── stamp_step: the shared per-step field-set ─────────────────────────────────


def test_stamp_step_failure_carries_dedup_key() -> None:
    stamped = stamp_step(make_step(step_index=1), _fail(_KEY))
    assert stamped.success is False
    assert stamped.failing_check_id == _KEY
    assert stamped.dedup_key == _KEY
    assert stamped.blocking is True
    assert stamped.confirmed is True


def test_stamp_step_success_is_non_failure() -> None:
    stamped = stamp_step(make_step(step_index=1, success=False, failing_check_id="x"), _pass())
    assert stamped.success is True
    assert stamped.failing_check_id is None
    assert stamped.dedup_key is None
    assert stamped.blocking is False


def test_stamp_step_infra_is_never_a_capability_failure() -> None:
    stamped = stamp_step(make_step(step_index=1), _infra())
    assert stamped.is_infra_failure is True
    assert stamped.failing_check_id is None
    assert stamped.blocking is False  # infra never counts toward escalation


# ── parser side-channel map (ADR-PSV-4) ───────────────────────────────────────


def test_parser_stamps_steps_from_the_outcome_map() -> None:
    outcomes = {1: _fail(_KEY), 2: _fail(_KEY)}
    traj = MiniSweAgentParser().parse(_messages(3), {"trajectory_id": "t"}, outcomes)
    assert traj.steps[1].failing_check_id == _KEY
    assert traj.steps[2].dedup_key == _KEY
    assert traj.steps[0].failing_check_id is None  # unmapped step keeps the default


def test_terminal_stamp_wins_over_a_per_step_outcome() -> None:
    # The last step is both mapped AND terminal-stamped; the harness resolved label wins.
    outcomes = {2: _fail(_KEY)}
    traj = MiniSweAgentParser().parse(
        _messages(3), {"trajectory_id": "t", "terminal_resolved": True}, outcomes
    )
    assert traj.steps[-1].success is True  # terminal resolved wins
    assert traj.steps[-1].failing_check_id is None


# ── restamp_trajectory (offline path) ─────────────────────────────────────────


def test_restamp_preserves_terminal_and_updates_hash() -> None:
    base = make_trajectory([make_step(step_index=i) for i in range(3)], terminal_resolved=True)
    restamped = restamp_trajectory(base, {0: _fail(_KEY), 1: _fail(_KEY)})
    assert restamped.steps[0].dedup_key == _KEY
    assert restamped.steps[-1].success is True  # terminal authority
    assert restamped.header.content_sha256 != base.header.content_sha256  # hash re-derived


# ── the Phase-1 goal: a recurring dedup_key makes the trigger fire ────────────


def test_recurring_dedup_key_yields_n_escalated_gt_zero() -> None:
    # Steps 1 and 2 fail on the SAME node id within the window → escalate_after_n=2 fires.
    outcomes = {1: _fail(_KEY), 2: _fail(_KEY)}
    traj = MiniSweAgentParser().parse(
        _messages(4), {"trajectory_id": "t", "terminal_resolved": False}, outcomes
    )
    report = run_eval.evaluate([traj], [GridPoint(2, 10)])
    assert report.n_escalated > 0


def test_two_distinct_single_failures_do_not_escalate() -> None:
    # Different keys never aggregate — the recurrence trigger stays silent (guards over-firing).
    outcomes = {1: _fail("tests/t.py::test_a"), 2: _fail("tests/t.py::test_b")}
    traj = MiniSweAgentParser().parse(
        _messages(4), {"trajectory_id": "t", "terminal_resolved": False}, outcomes
    )
    report = run_eval.evaluate([traj], [GridPoint(2, 10)])
    assert report.n_escalated == 0
