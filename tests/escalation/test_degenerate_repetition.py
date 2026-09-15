from __future__ import annotations

from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.escalation import degenerate_repetition as dr
from tests.escalation.factories import make_step, make_trajectory


def _traj(*, actions: list[str], resolved: bool, challenge: str) -> object:
    steps = [
        make_step(step_index=i, decision_index=i, action=action, args=action)
        for i, action in enumerate(actions)
    ]
    traj = make_trajectory(steps, trajectory_id=challenge, terminal_resolved=resolved)
    from dataclasses import replace

    return replace(traj, header=replace(traj.header, instance_id=challenge))


def test_repetition_score_counts_the_longest_run() -> None:
    actions = ["a", "b", "b", "b", "c"]
    traj = _traj(actions=actions, resolved=False, challenge="c1")
    assert dr.repetition_score(traj) == 3


def test_a_run_of_n_minus_one_is_not_a_fire_at_n() -> None:
    traj = _traj(actions=["a", "b", "b", "c"], resolved=False, challenge="c1")
    assert dr.first_fire_depth(traj, 2) is not None
    assert dr.first_fire_depth(traj, 3) is None


def test_first_fire_depth_matches_the_depth_convention() -> None:
    # Fire at index 1 of a 10-step run -> depth (1 + 1) / 10 = 0.2 (the harness convention).
    traj = _traj(actions=["a", "a"] + ["x"] * 8, resolved=True, challenge="c1")
    depth = dr.first_fire_depth(traj, 2)
    assert depth is not None
    assert abs(depth - 0.2) < 1e-9


def test_operating_point_splits_shallow_from_deep() -> None:
    shallow = _traj(actions=["a", "a"] + ["x"] * 8, resolved=False, challenge="s")
    deep = _traj(actions=[f"d{j}" for j in range(8)] + ["a", "a"], resolved=True, challenge="d")
    point = dr.operating_point([shallow, deep], 2)
    assert point.fires == 2
    assert point.shallow_fires == 1
    assert point.shallow_volume == 0.5
    assert point.shallow_precision == 1.0


def test_target_met_requires_precision_and_volume() -> None:
    shallow_doomed = [
        _traj(actions=["a", "a"] + ["x"] * 8, resolved=False, challenge=f"s{i}") for i in range(10)
    ]
    points = dr.operating_points(shallow_doomed, (2,))
    assert dr.target_met(points) is not None  # 100% shallow precision, 100% volume


def test_target_not_met_when_volume_is_below_the_floor() -> None:
    one = _traj(actions=["a", "a"] + ["x"] * 8, resolved=False, challenge="s0")
    rest = [
        _traj(actions=[f"u{i}_{j}" for j in range(10)], resolved=True, challenge=f"r{i}")
        for i in range(30)
    ]
    points = dr.operating_points([one, *rest], (2,))
    assert points[0].shallow_volume < dr.VOLUME_FLOOR
    assert dr.target_met(points) is None


def test_repetition_census_evidences_an_untestable_n_grid() -> None:
    trajs = [
        _traj(actions=["a"], resolved=False, challenge="r1"),
        _traj(actions=["a", "a", "b"], resolved=True, challenge="r2"),
        _traj(actions=["a", "a", "a", "b"], resolved=False, challenge="r3"),
    ]
    census = dr.repetition_census(trajs, n_grid=(1, 2, 3))
    assert census.longest_run_histogram == {1: 1, 2: 1, 3: 1}
    assert census.reaching_n == {1: 3, 2: 2, 3: 1}
    assert census.max_longest_run == 3
    assert census.to_dict()["reaching_n"] == {"1": 3, "2": 2, "3": 1}


def test_auroc_result_carries_the_operating_point_census() -> None:
    trajs = [
        _traj(actions=["a", "a", "b"], resolved=index % 2 == 0, challenge=f"c{index}")
        for index in range(6)
    ]
    report = dr.score_auroc_and_null(trajs, n_permutations=200)
    census = report["census"]
    assert isinstance(census, dict)
    assert census["longest_run_histogram"] == {"2": 6}
    assert census["reaching_n"]["3"] == 0


def test_planted_positive_control_recovers_and_null_collapses() -> None:
    report = dr.control_report(dr.planted_corpus(), n_permutations=400, seed=0)
    chance_upper = float(report["chance_level"]) + float(report["chance_band"])
    # The assembled detector recovers the planted loop ...
    assert float(report["positive_score"]) > chance_upper
    # ... and collapses to chance once the labels are destroyed.
    assert abs(float(report["shuffled_score"]) - 0.5) <= float(report["chance_band"])
    # The verdict is the shared adjudicator's, not a locally re-derived boolean.
    assert report["admissible"] is True
    assert report["positive_passed"] is True
    assert report["null_at_chance"] is True
    assert str(report["reason"]).startswith("ADMISSIBLE")


def test_control_report_routes_through_the_shared_adjudicator(monkeypatch) -> None:
    calls: list[tuple[float, float]] = []

    def spy(
        positive: float, shuffled: float, *, chance_level: float, chance_band: float
    ) -> AdmissibilityResult:
        calls.append((positive, shuffled))
        return admissibility_verdict(
            positive, shuffled, chance_level=chance_level, chance_band=chance_band
        )

    monkeypatch.setattr(dr, "admissibility_verdict", spy)
    report = dr.control_report(dr.planted_corpus(), n_permutations=200, seed=0)
    assert calls, "control_report must call the shared admissibility adjudicator"
    assert report["admissible"] is True


def test_shuffle_preserves_each_challenge_label_multiset() -> None:
    from collections import Counter

    corpus = dr.planted_corpus()
    shuffled = dr.shuffle_labels_within_challenge(corpus, seed=0)
    assert len(shuffled) == len(corpus)

    def census(trajs: list) -> dict[str, Counter]:
        out: dict[str, Counter] = {}
        for traj in trajs:
            group = traj.header.instance_id or ""
            out.setdefault(group, Counter())[traj.header.terminal_resolved] += 1
        return out

    assert census(corpus) == census(shuffled)
