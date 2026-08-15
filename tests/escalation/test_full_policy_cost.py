"""The full-policy cost read: one denominator for every arm, and a point estimate inside its CI."""

# THE DEFECT THIS FILE PINS. `_arm_costs` reads each arm on the instances that arm covers, so the
# escalate arm's ratio is computed on the fired subset while always-frontier's is computed on all
# of them — two ratios against different floors, never comparable. The full-policy read gives a
# conditional arm a defined cost at EVERY instance (it stays cheap where it does not fire), so all
# arms share one denominator.
#
# It also pins the shape of an earlier, separately-fixed bootstrap bug: a floor drawn over one
# instance set while the point estimate used another, which put a point estimate outside its own
# interval. Any denominator mismatch reintroduced here trips `test_every_point_estimate_sits_...`.

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from benchmark.escalation import cost_join, schema, session_eval
from tests.escalation.factories import make_step, make_trajectory

if TYPE_CHECKING:
    from pathlib import Path

_HEADER = "challenge_id,model,reasoning,real_cost\n"
_CHEAP = "deepseek-v4-flash"
_FRONTIER = "kimi-k3"


def _session(instance: str, model: str, effort: str, *, resolved: bool, n_steps: int = 5):
    steps = [make_step(step_index=i, decision_index=i) for i in range(n_steps)]
    traj = make_trajectory(
        steps, trajectory_id=f"{instance}__{model}__{effort}", terminal_resolved=resolved
    )
    return schema.Trajectory(header=replace(traj.header, instance_id=instance), steps=steps)


def _results(tmp_path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    path = tmp_path / "results.csv"
    path.write_text(_HEADER + "".join(",".join(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _mixed_corpus(tmp_path: Path):
    """Two instances: the escalate arm fires on i1 (cheap failed) and stays quiet on i2."""
    trajs = [
        _session("i1", _CHEAP, "high", resolved=False, n_steps=3),
        _session("i1", _CHEAP, "high", resolved=False, n_steps=5),
        _session("i1", _FRONTIER, "max", resolved=True),
        _session("i2", _CHEAP, "high", resolved=True, n_steps=3),
        _session("i2", _CHEAP, "high", resolved=True, n_steps=5),
        _session("i2", _FRONTIER, "max", resolved=True),
    ]
    csv_path = _results(
        tmp_path,
        [
            ("i1", _CHEAP, "high", "0.10"),
            ("i1", _FRONTIER, "max", "1.00"),
            ("i2", _CHEAP, "high", "0.10"),
            ("i2", _FRONTIER, "max", "1.00"),
        ],
    )
    costs, _summary = cost_join.session_costs(trajs, csv_path)
    report = session_eval.session_cadence(trajs, costs)
    assert report is not None
    return report


def _by_arm(report, currency: str) -> dict[str, session_eval.FullPolicyCost]:
    return {c.name: c for c in report.full_policy_costs if c.currency == currency}


def test_every_arm_is_read_on_every_instance_not_on_its_own_coverage(tmp_path: Path) -> None:
    report = _mixed_corpus(tmp_path)
    naive = _by_arm(report, cost_join.NAIVE)
    assert set(naive) == {
        "escalate",
        "cheap_retry",
        "always_frontier",
        "always_cheap",
        "random_escalate",
    }
    for arm in naive.values():
        assert arm.n_instances == report.n_overlap_instances == 2
    # The escalate arm fires on exactly one of the two, and that is REPORTED rather than being
    # silently baked into the denominator.
    assert naive["escalate"].n_fired == 1
    assert naive["always_frontier"].n_fired == 2


def test_the_escalate_arm_pays_the_failed_cheap_session_where_it_fires(tmp_path: Path) -> None:
    naive = _by_arm(_mixed_corpus(tmp_path), cost_join.NAIVE)
    # i1: cheap 0.10 failed, then frontier 1.00 -> 1.10.  i2: quiet, so it pays cheap only -> 0.10.
    assert naive["escalate"].cost_per_instance == pytest.approx((1.10 + 0.10) / 2)
    # always_frontier pays frontier on BOTH -> 1.00, and always_cheap pays 0.10 on both.
    assert naive["always_frontier"].cost_per_instance == pytest.approx(1.00)
    assert naive["always_cheap"].cost_per_instance == pytest.approx(0.10)
    # Both arms resolve everything here, so escalating buys nothing above the frontier arm and
    # costs less than it only because it stays cheap on the task cheap already solved.
    assert naive["escalate"].cost_diff_vs_always_frontier == pytest.approx(0.60 - 1.00)


def test_the_marginal_ratio_divides_by_the_same_instances_it_spent_on(tmp_path: Path) -> None:
    naive = _by_arm(_mixed_corpus(tmp_path), cost_join.NAIVE)
    # Floor resolves 1 of 2; escalate resolves 2 of 2. Gain 0.5, extra spend 0.60-0.10 = 0.50.
    assert naive["escalate"].resolve_rate == pytest.approx(1.0)
    assert naive["always_cheap"].resolve_rate == pytest.approx(0.5)
    assert naive["escalate"].marginal_cost_per_resolve == pytest.approx(0.50 / 0.5)
    # The floor has no marginal to price against itself.
    assert naive["always_cheap"].marginal_cost_per_resolve is None


def test_every_point_estimate_sits_inside_its_own_resampled_interval(tmp_path: Path) -> None:
    # THE REGRESSION GUARD. The earlier bug drew the floor over a different instance set from the
    # point estimate, which can put an estimate outside the interval that is supposed to cover it.
    # A denominator mismatch of any shape shows up here.
    report = _mixed_corpus(tmp_path)
    assert report.full_policy_costs
    for arm in report.full_policy_costs:
        low, high = arm.cost_ci
        assert low <= arm.cost_per_instance <= high, f"{arm.currency}/{arm.name} cost outside CI"
        if arm.marginal_cost_per_resolve is not None and arm.marginal_undefined_share == 0.0:
            m_low, m_high = arm.marginal_ci
            assert m_low <= arm.marginal_cost_per_resolve <= m_high, (
                f"{arm.currency}/{arm.name} marginal outside CI"
            )
        d_low, d_high = arm.cost_diff_ci
        assert d_low <= arm.cost_diff_vs_always_frontier <= d_high


def _ratio_invariant_corpus(tmp_path: Path):
    """Four instances where escalate and the floor differ on ONE — so the ratio is 1.00 exactly.

    Cheap prices differ 50x across instances, which is what makes a floor drawn over the wrong
    instance set visible: the true ratio is invariant across resamples, a mismatched one is not.
    """
    prices = {"i1": "0.10", "i2": "0.10", "i3": "5.00", "i4": "5.00"}
    trajs = []
    for inst in prices:
        resolved = inst != "i1"
        trajs.append(_session(inst, _CHEAP, "high", resolved=resolved, n_steps=3))
        trajs.append(_session(inst, _CHEAP, "high", resolved=resolved, n_steps=5))
        trajs.append(_session(inst, _FRONTIER, "max", resolved=True))
    rows = [(inst, _CHEAP, "high", price) for inst, price in prices.items()]
    rows += [(inst, _FRONTIER, "max", "1.00") for inst in prices]
    costs, _summary = cost_join.session_costs(trajs, _results(tmp_path, rows))
    report = session_eval.session_cadence(trajs, costs)
    assert report is not None
    return report


def test_the_floor_is_drawn_over_the_same_instances_as_the_arm(tmp_path: Path) -> None:
    # THE BOOTSTRAP-SIDE GUARD, and the exact shape of the earlier fixed bug: a floor read on a
    # different instance set from the arm. Here escalate differs from the floor on exactly one
    # instance (+$1.00, +1 resolve), so every resample that can price a marginal must price it at
    # 1.00 — the interval is a point. A floor drawn over any other set makes it spread by ~20x.
    naive = _by_arm(_ratio_invariant_corpus(tmp_path), cost_join.NAIVE)
    escalate = naive["escalate"]
    assert escalate.cost_per_instance == pytest.approx((1.10 + 0.10 + 5.00 + 5.00) / 4)
    assert naive["always_cheap"].cost_per_instance == pytest.approx((0.10 + 0.10 + 5.00 + 5.00) / 4)
    assert escalate.marginal_cost_per_resolve == pytest.approx(1.00)
    assert escalate.marginal_ci == pytest.approx((1.00, 1.00))
    # Draws that never pick the one instance escalate acts on buy no resolve, so the ratio is
    # undefined there — reported as a share, never silently dropped.
    assert 0.0 < escalate.marginal_undefined_share < 1.0


def test_both_currencies_are_reported_and_the_payload_keeps_them_apart(tmp_path: Path) -> None:
    report = _mixed_corpus(tmp_path)
    payload = report.to_dict()["cost_full_policy"]
    assert set(payload) == {cost_join.NAIVE, cost_join.CACHE_AWARE}
    row = payload[cost_join.NAIVE]["escalate"]
    assert row["n_instances"] == 2
    assert row["n_instances_fired"] == 1
    assert "paired_cost_difference_vs_always_frontier" in row
    # The naive block is a DIFFERENT quantity and must still be present and distinct.
    assert "cost" in report.to_dict()
    assert report.to_dict()["cost"][cost_join.NAIVE]["escalate"]["n_tasks_acted_on"] == 1


def test_an_arm_that_holds_the_same_sessions_reports_a_zero_information_quality_axis(
    tmp_path: Path,
) -> None:
    # THE GUARD AGAINST "CHEAPER AT EQUAL QUALITY". On this fixture the frontier resolves both
    # instances and so does the escalate arm (it fires on i1 and stays cheap on i2, which cheap
    # already solved), so the two arms differ in outcome on ZERO instances. The equality is an
    # identity of construction, and the payload must say so rather than implying a measurement.
    naive = _by_arm(_mixed_corpus(tmp_path), cost_join.NAIVE)
    escalate = naive["escalate"]
    assert escalate.resolve_diff_vs_always_frontier == pytest.approx(0.0)
    assert escalate.n_outcome_differs == 0
    # The cost side, on the same instances, is a real difference.
    assert escalate.cost_diff_vs_always_frontier < 0.0


def test_the_paired_cost_difference_is_zero_for_the_arm_it_is_taken_against(
    tmp_path: Path,
) -> None:
    naive = _by_arm(_mixed_corpus(tmp_path), cost_join.NAIVE)
    assert naive["always_frontier"].cost_diff_vs_always_frontier == pytest.approx(0.0)
    assert naive["always_frontier"].cost_diff_ci == pytest.approx((0.0, 0.0))
    assert naive["always_frontier"].diff_excludes_zero is False
