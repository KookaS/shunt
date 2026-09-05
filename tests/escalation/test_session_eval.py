"""Session-cadence escalation value: escalate-to-frontier vs same-cost retry, same instances."""

from __future__ import annotations

from dataclasses import replace

import pytest

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


def test_the_arms_resample_instances_not_sessions_and_serialize_to_dict() -> None:
    # THE FIX THIS PINS. The arms carried Wilson intervals over SESSIONS, which treats several
    # frontier sessions on one task as several independent draws. They are one task, one repo, one
    # target test — the instance is the exchangeable unit, so both arms resample whole instances
    # and the paired difference comes off the same resamples. `to_dict` must carry the SAME
    # numbers (rounded) plus the full-corpus context — the figure and the JSON cannot disagree.
    trajs = []
    for inst in ("i1", "i2"):
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False))
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False))
        trajs.append(_session(inst, "kimi-k3", resolved=True))
    report = session_eval.session_cadence(trajs)
    assert report is not None
    # Every escalate draw resolves here, so the instance bootstrap is degenerate at 1.0 — and
    # that is the point: Wilson over 2 sessions would have manufactured a spread of [0.34, 1.0]
    # from a corpus holding exactly two instances, both of which resolved.
    assert report.escalate_ci == (1.0, 1.0)
    assert report.escalate_ci != metrics.wilson_interval(
        report.n_escalated_resolved, report.n_escalated
    )
    assert report.n_instances_resampled == 2
    # The paired difference is estimable on the same instances and is published beside the arms.
    assert report.to_dict()["paired_difference"]["n_instances"] == 2  # type: ignore[index]
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


def test_the_baseline_arms_are_read_on_the_same_instances_as_the_escalate_arm() -> None:
    # THE ARMS THAT COULD KILL THE CLAIM. Two tasks: on i1 the cheap sessions fail and the frontier
    # resolves; on i2 the FIRST cheap session already resolved, so the escalate arm never fires
    # there but always_frontier and always_cheap still cover it. Their denominator is therefore the
    # overlap subset, not the fired subset — which is exactly what makes them a competitor.
    trajs = [
        _session("i1", "deepseek-v4-flash", resolved=False, n_steps=3),
        _session("i1", "deepseek-v4-flash", resolved=False, n_steps=5),
        _session("i1", "kimi-k3", resolved=True),
        _session("i2", "deepseek-v4-flash", resolved=True, n_steps=3),
        _session("i2", "deepseek-v4-flash", resolved=True, n_steps=5),
        _session("i2", "kimi-k3", resolved=False),
    ]
    report = session_eval.session_cadence(trajs)
    assert report is not None
    arms = report.comparison
    assert set(arms) == set(session_eval.BASELINE_ARMS)
    # always_frontier: BOTH frontier sessions, whatever the cheap arm did -> 1 of 2.
    assert (arms["always_frontier"].n, arms["always_frontier"].resolved) == (2, 1)
    # always_cheap: the first cheap session on each task -> i1 failed, i2 resolved.
    assert (arms["always_cheap"].n, arms["always_cheap"].resolved) == (2, 1)
    # The escalate arm fired on one of the two tasks, so the random arm matches that fire rate.
    assert report.random_fire_rate == 0.5
    assert arms["random_escalate"].n_instances == report.n_overlap_instances == 2
    # Every arm's difference is escalate MINUS the arm, paired on the same instance resamples.
    assert arms["always_frontier"].diff_estimate == 1.0 - 0.5


def test_a_baseline_arm_that_beats_the_escalate_arm_is_reported_as_a_negative_difference() -> None:
    # Escalation must be able to LOSE here. Three tasks where the frontier resolves everywhere but
    # the escalate arm only ever fires on the one task whose cheap sessions failed: always_frontier
    # covers all three, so the paired difference is negative and serializes as such.
    trajs = [
        _session("i1", "deepseek-v4-flash", resolved=False, n_steps=3),
        _session("i1", "deepseek-v4-flash", resolved=False, n_steps=5),
        _session("i1", "kimi-k3", resolved=False),
    ]
    for inst in ("i2", "i3"):
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=True, n_steps=3))
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=True, n_steps=5))
        trajs.append(_session(inst, "kimi-k3", resolved=True))
    report = session_eval.session_cadence(trajs)
    assert report is not None
    frontier = report.comparison["always_frontier"]
    assert frontier.rate > report.escalate_rate
    assert frontier.diff_estimate < 0.0
    payload = report.to_dict()["comparisons"]["always_frontier"]  # type: ignore[index]
    assert payload["rate"] == round(frontier.rate, 4)
    assert payload["paired_difference_vs_escalate"]["estimate"] < 0.0  # type: ignore[index]


def test_the_random_arm_is_seeded_so_the_published_number_is_reproducible() -> None:
    # A random arm whose value moved run to run would be unciteable. The assignment is seeded and
    # exact-count, so two calls on the same corpus agree bit for bit.
    trajs = []
    for inst in ("i1", "i2", "i3", "i4"):
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False, n_steps=3))
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=inst == "i4", n_steps=5))
        trajs.append(_session(inst, "kimi-k3", resolved=True))
    first = session_eval.session_cadence(trajs)
    second = session_eval.session_cadence(trajs)
    assert first is not None and second is not None
    assert first.comparison["random_escalate"].to_dict() == (
        second.comparison["random_escalate"].to_dict()
    )


def test_session_cadence_is_none_on_an_empty_corpus() -> None:
    # None means "not supported", not NaN bars: with no tasks there is no overlap subset and the
    # contrast is undefined. The caller skips the figure rather than drawing NaNs.
    assert session_eval.session_cadence([]) is None


def test_the_arm_membership_and_the_shipped_ladder_are_recorded_not_implied() -> None:
    # The figure must be able to say WHICH models the escalate arm is, and which of them the
    # shipped ladder would actually step to. Both are derived — from the registry's price order
    # and from the packaged rank_shortlist — so a config change moves them instead of leaving a
    # stale sentence on the canvas.
    trajs = []
    for inst in ("i1", "i2", "i3"):
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False, n_steps=3))
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False, n_steps=5))
        trajs.append(_session(inst, "qwen3.7-plus", resolved=False))
        trajs.append(_session(inst, "gpt-5-mini", resolved=False))
        trajs.append(_session(inst, "kimi-k2.5", resolved=False))
        trajs.append(_session(inst, "zai-glm-5.2", resolved=True))
        trajs.append(_session(inst, "kimi-k3", resolved=True))
    sc = session_eval.session_cadence(trajs)
    assert sc is not None
    # Six priced models present: cheapest is the base, the top two are the escalate arm.
    assert sc.cheap_models == ("deepseek-v4-flash",)
    assert sc.frontier_models == ("zai-glm-5.2", "kimi-k3")
    # The shipped shortlist walks the SHIPPED live pool (not the corpus's models): since
    # deepseek-v4-pro joined the pool on 2026-09-04 it is the FIRST rung, then zai-glm-5.2 —
    # the first arm member — then the jump to the top live rank, so it still never reaches
    # kimi-k3, the other half of the arm the figure scores.
    assert sc.rank_shortlist > 0
    assert sc.ladder_visits == ("deepseek-v4-pro", "zai-glm-5.2", "claude-fable-5")
    assert "zai-glm-5.2" in sc.ladder_visits
    assert "kimi-k3" not in sc.ladder_visits
    assert sc.to_dict()["context"]["shipped_ladder_visits"] == list(sc.ladder_visits)


def test_session_cadence_headlines_the_selected_policy() -> None:
    # `policy="always_cheap"` headlines the never-escalate hold policy: the report's escalate
    # read becomes the always-cheap arm (its rate, its paired difference), the selected policy is
    # not its own comparator, and the policy name rides on the serialized payload so the numbers
    # can never be misattributed to the shipped escalate decision.
    trajs = []
    for inst in ("i1", "i2", "i3"):
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False))
        trajs.append(_session(inst, "deepseek-v4-flash", resolved=False))
        trajs.append(_session(inst, "kimi-k3", resolved=True))
    default = session_eval.session_cadence(trajs)
    held = session_eval.session_cadence(trajs, policy=session_eval.ARM_ALWAYS_CHEAP)
    assert default is not None and held is not None
    # The headline always-cheap rate IS the default run's always-cheap comparator rate.
    assert held.escalate_rate == default.comparison["always_cheap"].rate
    assert held.diff_estimate == held.escalate_rate - held.retry_rate
    assert "always_cheap" not in held.comparison
    assert "always_frontier" in held.comparison
    # Provenance: the default run serializes no policy key (bit-identical metrics.json); a
    # non-default selection carries its own name.
    assert "policy" not in default.to_dict()
    assert held.to_dict()["policy"] == "always_cheap"


def test_an_unknown_policy_fails_keyerror_style_naming_the_allowed_ones() -> None:
    trajs = [
        _session("i1", "deepseek-v4-flash", resolved=False),
        _session("i1", "deepseek-v4-flash", resolved=True),
        _session("i1", "kimi-k3", resolved=True),
    ]
    with pytest.raises(ValueError, match="allowed"):
        session_eval.session_cadence(trajs, policy="nope")
