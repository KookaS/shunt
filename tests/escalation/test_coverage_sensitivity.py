"""The coverage-restricted re-score: nested strata, own nulls, and a stated power cost."""

from __future__ import annotations

from dataclasses import replace

from benchmark.escalation import coverage_sensitivity, replay, schema
from tests.escalation.factories import make_step, make_trajectory

_PERMUTATIONS = 200
_GRID = (replay.GridPoint(2, 10, "effort_then_rank"), replay.GridPoint(5, 10, "effort_then_rank"))
_POINT = replay.GridPoint(2, 10, "effort_then_rank")
# A command the edit-gate recognises, so the reproduction phase ends on step 0 and the recurrence
# counter is allowed to count. A bare "edit" is NOT an edit action to the gate.
_EDIT = "sed -i s/a/b/ f.py"


def _run(
    instance: str,
    model: str,
    *,
    resolved: bool,
    recurring: bool,
    stamped: bool = True,
) -> schema.Trajectory:
    """One trajectory; `recurring` repeats a single failing check id after the first edit."""
    steps = [
        make_step(
            step_index=i,
            decision_index=i,
            success=not (recurring and i >= 1),
            failing_check_id="check-a" if recurring and i >= 1 else None,
            action=_EDIT if i == 0 else "pytest",
            confirmed=stamped,
        )
        for i in range(6)
    ]
    traj = make_trajectory(
        steps, trajectory_id=f"{instance}__{model}__default", terminal_resolved=resolved
    )
    return schema.Trajectory(header=replace(traj.header, instance_id=instance), steps=steps)


def _uneven_coverage_corpus() -> list[schema.Trajectory]:
    """Two models carrying the SAME recurrence->failure link, one of them half-unstamped."""
    trajs: list[schema.Trajectory] = []
    for i in range(14):
        failed = i % 2 == 0
        trajs.append(_run(f"c{i}", "thin-coverage", resolved=not failed, recurring=failed))
        trajs.append(
            _run(f"c{i}x", "thin-coverage", resolved=not failed, recurring=failed, stamped=False)
        )
        trajs.append(_run(f"r{i}", "full-coverage", resolved=not failed, recurring=failed))
    return trajs


def test_stamped_share_is_derived_from_the_corpus_not_hardcoded() -> None:
    # The stratum boundaries must move with the corpus rather than naming a model set.
    trajs = [_run(f"i{i}", "a", resolved=True, recurring=False) for i in range(4)]
    trajs += [_run(f"j{i}", "b", resolved=True, recurring=False, stamped=i < 2) for i in range(4)]
    shares = coverage_sensitivity.stamped_share_by_model(trajs)
    assert shares == {"a": 1.0, "b": 0.5}


def test_the_ladder_is_nested_loosest_first_and_states_the_power_it_costs() -> None:
    result = coverage_sensitivity.evaluate(
        _uneven_coverage_corpus(), _GRID, _POINT, n_permutations=_PERMUTATIONS
    )
    assert result is not None
    assert [s.min_stamped_share for s in result.strata] == [0.5, 1.0]
    # Loosest rung is every model and is the reference the deltas are taken against.
    assert result.strata[0].models == ("full-coverage", "thin-coverage")
    assert result.strata[0].retained_share == 1.0
    assert result.strata[0].delta_auroc == 0.0
    # Strictest rung drops the thin-coverage model, and says how much sample that cost.
    strictest = result.strictest
    assert strictest is not None
    assert strictest.models == ("full-coverage",)
    assert strictest.n_stamped < result.strata[0].n_stamped
    assert 0.0 < strictest.retained_share < 1.0


def test_a_restricted_rung_is_judged_inside_its_own_null_not_the_unrestricted_one() -> None:
    # The whole point of re-running the family per stratum: each rung's null is estimated on that
    # rung's rows, so `clears_null` is a statement about the restricted cell.
    result = coverage_sensitivity.evaluate(
        _uneven_coverage_corpus(), _GRID, _POINT, n_permutations=_PERMUTATIONS
    )
    assert result is not None
    for stratum in result.strata:
        assert stratum.null.n_permutations == _PERMUTATIONS
        assert stratum.null_familywise is not None
        assert stratum.null_familywise.observed == stratum.auroc
        assert stratum.null_length_stratified is not None
    # The planted link is identical in both models, so restriction must not destroy it.
    assert result.survives_restriction is True


def test_a_single_outcome_class_stratum_is_dropped_rather_than_scored_at_chance() -> None:
    # Every run resolved: AUROC is undefined there. Publishing 0.5 would read as a measured null.
    trajs = [_run(f"i{i}", "only", resolved=True, recurring=False) for i in range(6)]
    result = coverage_sensitivity.evaluate(trajs, _GRID, _POINT, n_permutations=_PERMUTATIONS)
    assert result is not None
    assert result.strata == ()
    assert result.survives_restriction is False


def test_the_payload_is_json_shaped_and_names_the_counting_mode_and_the_cell() -> None:
    result = coverage_sensitivity.evaluate(
        _uneven_coverage_corpus(), _GRID, _POINT, n_permutations=_PERMUTATIONS
    )
    assert result is not None
    payload = result.to_dict()
    assert payload["counting"] == "edit_gated"
    assert payload["cell"] == {
        "escalate_after_n": 2,
        "stale_window": 10,
        "ladder": "effort_then_rank",
    }
    rows = payload["strata"]
    assert isinstance(rows, list)
    assert rows[0]["models"] == ["full-coverage", "thin-coverage"]
    assert rows[0]["delta_auroc_vs_all_models"] == 0.0
    assert set(payload["stamped_share_by_model"]) == {"full-coverage", "thin-coverage"}
