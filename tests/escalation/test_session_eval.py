"""Session-cadence escalation value: escalate-to-frontier vs same-cost retry, same instances."""

from __future__ import annotations

from dataclasses import replace

from benchmark.escalation import metrics, schema, session_eval
from tests.escalation.factories import make_step, make_trajectory


def _session(instance: str, model: str, *, resolved: bool, n_steps: int = 5):
    steps = [make_step(step_index=i, decision_index=i) for i in range(n_steps)]
    traj = make_trajectory(
        steps,
        trajectory_id=f"{instance}__{model}__default",
        terminal_resolved=resolved,
    )
    # The factory keys the instance by the trajectory id; a session sequence needs several
    # sessions sharing ONE instance id (repeated attempts at the same task).
    header = replace(traj.header, instance_id=instance)
    return schema.Trajectory(header=header, steps=steps)


def test_escalating_to_frontier_beats_a_cheap_retry_on_the_same_instances() -> None:
    # Three tasks, each with two CHEAP (deepseek) sessions and one FRONTIER (kimi-k3) session.
    # Cheap fails; the frontier resolves. A same-cost cheap retry keeps failing. The contrast
    # must read 1.0 vs 0.0 on the overlap subset, with the context rates reported.
    trajs = []
    for inst in ("i1", "i2", "i3"):
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False))
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False))
        trajs.append(_session(inst, "kimi-k3", resolved=True))
    report = session_eval.session_cadence(trajs)
    assert report.n_overlap_instances == 3
    assert report.escalate_rate == 1.0
    assert report.n_escalated == 3 and report.n_escalated_resolved == 3
    assert report.retry_rate == 0.0
    assert report.lift is None  # retry rate 0 -> lift undefined, not infinity
    assert report.cheap_base_rate == 0.0


def test_escalation_is_reported_as_context_when_cheap_retry_also_helps() -> None:
    # One task where the cheap retry also resolves, and the frontier too: both rates read their
    # real values, and the lift is finite and above 1.
    trajs = [
        _session("i1", "deepseek-v4-flash", resolved=False),
        _session("i1", "deepseek-v4-flash", resolved=True),  # retry resolves
        _session("i1", "kimi-k3", resolved=True),
    ]
    report = session_eval.session_cadence(trajs)
    assert report.escalate_rate == 1.0
    assert report.retry_rate == 1.0
    assert report.lift == 1.0


def test_models_absent_from_the_registry_do_not_crash_the_ranking() -> None:
    # A model name the registry does not price (a synthetic/unknown arm) must be excluded from the
    # price ranking rather than raising KeyError — the ranking is over PRESENT-and-priced models.
    # The unknown model is neither a cheap base nor a frontier target, so it takes no part in the
    # contrast: with only ONE cheap session the instance is not in the overlap subset at all, and
    # the estimate is "not supported" (None) rather than NaN bars.
    trajs = [
        _session("i1", "deepseek-v4-flash", resolved=False),
        _session("i1", "kimi-k3", resolved=True),
        _session("i1", "some-future-model", resolved=True),
    ]
    assert session_eval.session_cadence(trajs) is None  # no overlap subset -> no estimate


def test_the_arms_report_wilson_intervals_and_serialize_to_dict() -> None:
    # The bars' error arms are Wilson intervals, and to_dict must carry the SAME numbers (rounded)
    # plus the full-corpus context — the figure and the JSON cannot disagree about a rate.
    trajs = []
    for inst in ("i1", "i2"):
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False))
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False))
        trajs.append(_session(inst, "kimi-k3", resolved=True))
    report = session_eval.session_cadence(trajs)
    assert report is not None
    assert report.escalate_ci == metrics.wilson_interval(
        report.n_escalated_resolved, report.n_escalated
    )
    assert report.retry_ci == metrics.wilson_interval(report.n_retried_resolved, report.n_retried)
    # Full-corpus context: every cheap failure was followed by a frontier session, so the
    # context's frontier-after-fail arm is the escalate arm's superset — not a separate figure.
    assert report.n_frontier_after_fail == report.n_escalated
    assert report.frontier_after_fail_rate == report.escalate_rate
    payload = report.to_dict()
    assert payload["escalate"] == {
        "n": 2,
        "resolved": 2,
        "rate": 1.0,
        "ci95": [round(v, 4) for v in report.escalate_ci],
    }
    assert payload["context"]["cheap_base_rate"] == round(report.cheap_base_rate, 4)
    # retry_rate is 0.0 here, so lift is undefined and must serialize as null, never 0.0.
    assert report.lift is None
    assert payload["lift"] is None


def test_session_cadence_is_none_on_an_empty_corpus() -> None:
    # None means "not supported", not NaN bars: with no tasks there is no overlap subset and the
    # contrast is undefined. The caller skips the figure rather than drawing NaNs.
    assert session_eval.session_cadence([]) is None
