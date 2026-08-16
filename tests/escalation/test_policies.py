"""The session-cadence escalation-policy registry: registered policies, their decide
outputs, and the KeyError-style failure naming unknown policies.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from benchmark.escalation import policies, schema
from benchmark.escalation.policies import (
    ARM_ALWAYS_CHEAP,
    ARM_ALWAYS_FRONTIER,
    ARM_ESCALATE,
    ARM_RANDOM,
    ARM_RETRY,
    build_policy,
)
from tests.escalation.factories import make_step, make_trajectory


def _session(model: str, *, resolved: bool, n_steps: int = 5):
    steps = [make_step(step_index=i, decision_index=i) for i in range(n_steps)]
    traj = make_trajectory(
        steps,
        trajectory_id=f"i1__{model}__default",
        terminal_resolved=resolved,
    )
    header = replace(traj.header, instance_id="i1")
    return schema.Trajectory(header=header, steps=steps)


def _cheap(*resolved: bool, n_steps: int | tuple[int, ...] = 5) -> list:
    lengths = [n_steps] * len(resolved) if isinstance(n_steps, int) else list(n_steps)
    return [
        _session("deepseek-v4-flash", resolved=r, n_steps=lengths[i])
        for i, r in enumerate(resolved)
    ]


def _frontier(*resolved: bool) -> list:
    return [_session("kimi-k3", resolved=r) for r in resolved]


def _outcomes(decided) -> list[bool]:
    return [s.resolved for s in decided]


def test_every_registered_policy_is_buildable_and_names_itself() -> None:
    for name in policies.REGISTERED_POLICIES:
        assert build_policy(name).name == name


def test_the_registry_covers_every_per_instance_arm_except_the_random_null() -> None:
    # `random_escalate` is the eval's cross-instance null (seeded over the whole instance set at
    # the headline fire rate), not a per-instance decision — so it is deliberately not registered.
    assert set(policies.REGISTERED_POLICIES) == {
        ARM_ESCALATE,
        ARM_RETRY,
        ARM_ALWAYS_FRONTIER,
        ARM_ALWAYS_CHEAP,
    }
    assert ARM_RANDOM not in policies.REGISTERED_POLICIES


def test_an_unknown_policy_fails_keyerror_style_naming_the_allowed_ones() -> None:
    with pytest.raises(ValueError, match="unknown session escalation policy"):
        build_policy("bogus")
    with pytest.raises(ValueError, match="allowed"):
        build_policy("random_escalate")  # a real eval arm, but not a registered policy


def test_escalate_decides_to_escalate_only_after_a_cheap_failure() -> None:
    # The shipped decision: a cheap session that failed escalates the NEXT session to frontier;
    # a resolved cheap session stays put (no frontier session is bought).
    failed = _cheap(False, True)
    resolved = _cheap(True, True)
    frontier = _frontier(True)
    fired = build_policy(ARM_ESCALATE).decide(failed, frontier)
    quiet = build_policy(ARM_ESCALATE).decide(resolved, frontier)
    assert _outcomes(fired) == [True]
    assert len(fired[0].attempts) == 2  # the failed cheap session is billed in front
    assert quiet == ()


def test_always_cheap_is_the_never_escalate_hold_policy() -> None:
    # The hold comparator: the first cheap session's outcome, whatever the frontier did. This is
    # the policy the eval's always-cheap arm is, and the second real consumer of the registry.
    cheap = _cheap(False, True, n_steps=(3, 5))
    held = build_policy(ARM_ALWAYS_CHEAP).decide(cheap, _frontier(True))
    assert _outcomes(held) == [False]
    assert len(held[0].attempts) == 1  # nothing was escalated, nothing extra was billed


def test_always_frontier_never_being_cheap() -> None:
    frontier = _frontier(False, True)
    decided = build_policy(ARM_ALWAYS_FRONTIER).decide(_cheap(True), frontier)
    assert _outcomes(decided) == [False, True]


def test_retry_runs_a_second_cheap_session_only_after_a_failure() -> None:
    # The same-cost incumbent: retry cheap after a cheap failure. A resolved first cheap session
    # buys no retry.
    decided = build_policy(ARM_RETRY).decide(_cheap(False, True), _frontier(True))
    assert _outcomes(decided) == [True]
    assert len(decided[0].attempts) == 2  # the failed first cheap session is billed in front
    assert build_policy(ARM_RETRY).decide(_cheap(True, False), _frontier(True)) == ()


def test_every_registered_policy_is_in_the_evals_arm_set() -> None:
    # The seam's structural invariant: session_eval's interval/cost computation must cover every
    # registered policy, or a policy added to the registry passes `--policy` validation and then
    # KeyErrors deep inside `_instance_intervals`. `_ALL_ARMS` is derived from the registry, so
    # this pins the derivation rather than trusting it.
    from benchmark.escalation.session_eval import _ALL_ARMS

    assert set(policies.REGISTERED_POLICIES) <= set(_ALL_ARMS)
    assert ARM_RANDOM in _ALL_ARMS  # the eval's own cross-instance null rides along
