"""The escalation corpus carries no per-step cost, so the cost axis is a join — and a join that
is anything less than total publishes a number for a subset while implying the whole corpus."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from benchmark.escalation import cost_join, schema, session_eval
from tests.escalation.factories import make_step, make_trajectory

if TYPE_CHECKING:
    from pathlib import Path

_HEADER = "challenge_id,model,reasoning,real_cost\n"


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


def test_the_join_key_comes_off_the_header_not_the_steps(tmp_path: Path) -> None:
    # Nine StepView fields — `real_cost` and `model` among them — are populated on no step in this
    # corpus, so a step-level join would price nothing. The arm is on the header: `instance_id`
    # plus the `<instance>__<model>__<effort>` trajectory id.
    traj = _session("astropy__astropy-1", "deepseek-v4-flash", "high", resolved=False)
    assert cost_join.arm_of(traj) == ("astropy__astropy-1", "deepseek-v4-flash", "high")
    csv_path = _results(tmp_path, [("astropy__astropy-1", "deepseek-v4-flash", "high", "0.25")])
    joined, summary = cost_join.join([traj], csv_path)
    assert joined[traj.header.trajectory_id] == ("deepseek-v4-flash", 0.25)
    assert summary.join_rate == 1.0
    assert summary.n_joined == summary.n_sessions == 1


def test_a_single_unpriced_session_raises_instead_of_pricing_the_rest(tmp_path: Path) -> None:
    # THE RULE THIS PINS. A partial join is worse than no join: the cost it reports is real, and
    # the corpus it implies is not. One miss and nothing is priced.
    priced = _session("i1", "deepseek-v4-flash", "high", resolved=False)
    unpriced = _session("i2", "kimi-k3", "max", resolved=True)
    csv_path = _results(tmp_path, [("i1", "deepseek-v4-flash", "high", "0.25")])
    with pytest.raises(cost_join.CostJoinError) as caught:
        cost_join.join([priced, unpriced], csv_path)
    assert "i2/kimi-k3/max" in str(caught.value)
    assert "1 of 2" in str(caught.value)


def test_a_row_with_no_real_cost_is_a_miss_not_a_zero(tmp_path: Path) -> None:
    # An empty `real_cost` cell means the run was never billed-out, not that it was free. Treating
    # it as 0.0 would quietly make an arm look cheaper than every arm it is compared against.
    traj = _session("i1", "deepseek-v4-flash", "high", resolved=False)
    csv_path = _results(tmp_path, [("i1", "deepseek-v4-flash", "high", "")])
    with pytest.raises(cost_join.CostJoinError):
        cost_join.join([traj], csv_path)


def test_the_reasoning_arm_is_part_of_the_key(tmp_path: Path) -> None:
    # The same model at a different reasoning effort is a different arm and a different bill;
    # joining on (challenge, model) alone would price one arm with the other's cost.
    traj = _session("i1", "deepseek-v4-flash", "high", resolved=False)
    csv_path = _results(tmp_path, [("i1", "deepseek-v4-flash", "nothink", "0.25")])
    with pytest.raises(cost_join.CostJoinError):
        cost_join.join([traj], csv_path)


def test_a_missing_results_file_raises_rather_than_reporting_a_free_corpus(tmp_path: Path) -> None:
    traj = _session("i1", "deepseek-v4-flash", "high", resolved=False)
    with pytest.raises(cost_join.CostJoinError):
        cost_join.join([traj], tmp_path / "absent.csv")


def test_both_currencies_are_built_and_the_cache_aware_one_discounts_a_repeat(
    tmp_path: Path,
) -> None:
    # Naive charges a second session on the same model at full price; the cache-aware currency
    # banks that model's registry discount. They are different quantities, which is why both are
    # published and neither is published alone.
    traj = _session("i1", "deepseek-v4-flash", "high", resolved=False)
    csv_path = _results(tmp_path, [("i1", "deepseek-v4-flash", "high", "1.0")])
    costs, _summary = cost_join.session_costs([traj], csv_path)
    assert set(costs.currencies) == {cost_join.NAIVE, cost_join.CACHE_AWARE}
    repeat = [("deepseek-v4-flash", 1.0), ("deepseek-v4-flash", 1.0)]
    assert costs.currencies[cost_join.NAIVE](repeat) == 2.0
    assert costs.currencies[cost_join.CACHE_AWARE](repeat) < 2.0
    # A switch is not a repeat: two different models bank nothing.
    switch = [("deepseek-v4-flash", 1.0), ("kimi-k3", 1.0)]
    assert costs.currencies[cost_join.CACHE_AWARE](switch) == 2.0


def test_the_committed_corpus_joins_completely() -> None:
    # The whole cost axis rests on this being total on the REAL corpus, not on a fixture.
    from benchmark.escalation import corpus  # noqa: PLC0415

    trajectories = [schema.load_jsonl(p) for p in sorted(corpus.LIVE_DIR.glob("*.jsonl"))]
    _joined, summary = cost_join.join(trajectories)
    assert summary.n_sessions > 0
    assert summary.join_rate == 1.0


def test_an_arm_pays_for_the_sessions_it_had_to_run_first(tmp_path: Path) -> None:
    # The escalate arm's bill is the failed cheap session PLUS the frontier one — a policy does
    # not get the cheap attempt for free just because it was not the attempt that resolved.
    trajs = [
        _session("i1", "deepseek-v4-flash", "high", resolved=False, n_steps=3),
        _session("i1", "deepseek-v4-flash", "high", resolved=False, n_steps=5),
        _session("i1", "kimi-k3", "max", resolved=True),
    ]
    csv_path = _results(
        tmp_path,
        [
            ("i1", "deepseek-v4-flash", "high", "0.10"),
            ("i1", "kimi-k3", "max", "1.00"),
        ],
    )
    costs, _summary = cost_join.session_costs(trajs, csv_path)
    report = session_eval.session_cadence(trajs, costs)
    assert report is not None
    naive = {c.name: c for c in report.costs if c.currency == cost_join.NAIVE}
    assert naive["always_frontier"].cost_per_instance == pytest.approx(1.00)
    assert naive["always_cheap"].cost_per_instance == pytest.approx(0.10)
    # escalate = the cheap session it ran first, then the frontier one.
    assert naive["escalate"].cost_per_instance == pytest.approx(1.10)
    # $ per marginal resolve is against the always-cheap floor: 0.10 -> 1.10 buys 0 -> 1 resolve.
    assert naive["escalate"].marginal_cost_per_resolve == pytest.approx(1.00)
    # The floor itself has no marginal to price.
    assert naive["always_cheap"].marginal_cost_per_resolve is None


def test_a_session_with_no_price_raises_when_an_arm_is_costed(tmp_path: Path) -> None:
    # The second wall behind the join's totality: even if a caller hand-builds a partial cost map,
    # the arm cost raises rather than treating the unpriced session as free.
    trajs = [
        _session("i1", "deepseek-v4-flash", "high", resolved=False, n_steps=3),
        _session("i1", "deepseek-v4-flash", "high", resolved=False, n_steps=5),
        _session("i1", "kimi-k3", "max", resolved=True),
    ]
    partial = session_eval.SessionCosts(
        session={trajs[0].header.trajectory_id: ("deepseek-v4-flash", 0.1)},
        currencies={"naive": lambda attempts: sum(c for _m, c in attempts)},
    )
    with pytest.raises(KeyError):
        session_eval.session_cadence(trajs, partial)
