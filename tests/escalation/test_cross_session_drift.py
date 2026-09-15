from __future__ import annotations

from dataclasses import replace

import pytest

from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.escalation import cross_session_drift as csd
from benchmark.escalation import metrics
from shunt.proxy.session_drift import SessionCounters
from tests.escalation.factories import make_step, make_trajectory


def _counters(drift: int) -> SessionCounters:
    return SessionCounters(
        is_reverts=drift,
        retry_total=drift,
        loop_signals=drift,
        wire_tool_errors=drift,
        n_steps=10,
    )


def _sample(repo: str, session_id: str, *, drift: int, failed: bool) -> csd.SessionSample:
    return csd.SessionSample(repo, session_id, _counters(drift), failed)


def test_repo_of_reads_the_instance_prefix() -> None:
    traj = make_trajectory([make_step()])
    traj = replace(traj, header=replace(traj.header, instance_id="astropy__astropy-12907"))
    assert csd.repo_of(traj) == "astropy"
    fallback = make_trajectory([make_step()], trajectory_id="sympy__sympy-1__flash__high")
    assert csd.repo_of(fallback) == "sympy"


def test_session_counters_read_the_whitelisted_fields_and_companion_status() -> None:
    clean = make_trajectory([make_step(success=True), make_step(success=False)])
    counters = csd.session_counters(clean)
    assert counters.counts() == (0, 0, 0, 0)
    assert counters.n_steps == 2
    companion = csd.session_counters(clean, companion=True)
    assert companion.wire_tool_errors == 1  # the failing step's status == "error"


def test_scored_windows_filter_to_same_repo_and_enforce_the_window() -> None:
    samples = [_sample("a", f"a-{i}", drift=1, failed=i >= 3) for i in range(5)] + [
        _sample("b", f"b-{i}", drift=1, failed=False) for i in range(3)
    ]
    windows = csd.scored_windows(samples, 3)
    assert {window.repo for window in windows} == {"a"}
    assert len(windows) == 2
    assert windows[0].window == 3
    assert windows[0].n_prior == 3
    assert windows[-1].session_id == "a-4"


def test_k_outside_the_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be in"):
        csd.scored_windows([], 2)


def _status_trajs(n_sessions: int) -> list:
    trajs = []
    for index in range(n_sessions):
        steps = [make_step(step_index=j, success=(j + index) % 2 == 0) for j in range(6)]
        trajs.append(make_trajectory(steps, trajectory_id=f"r__s{index:02d}__m__h"))
    return trajs


def test_companion_report_carries_the_status_error_auroc_and_null() -> None:
    trajs = _status_trajs(11)
    report = csd.companion_report(trajs, csd.PRIMARY_AXIS)
    assert str(report["substitution"]).startswith("wire_tool_error_count")
    per_k = report["per_k"]
    assert isinstance(per_k, list)
    assert [row["k"] for row in per_k] == list(csd.WINDOW_GRID)
    for row in per_k:
        assert "auroc" in row
        assert isinstance(row["null"], dict)
        assert "null_ci95" in row["null"]
    # The substituted field actually varies, so unlike the dead primary counters the
    # companion carries a non-zero window score.
    assert any(float(row["fire"]["cut"]) > 0 for row in per_k)


def test_planted_positive_control_recovers_and_null_collapses() -> None:
    report = csd.control_report(csd.planted_corpus(n_repos=40), n_permutations=400)
    assert float(report["positive_score"]) > 0.9
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

    monkeypatch.setattr(csd, "admissibility_verdict", spy)
    report = csd.control_report(csd.planted_corpus(n_repos=40), n_permutations=200)
    assert calls, "control_report must call the shared admissibility adjudicator"
    assert report["admissible"] is True


def test_shuffled_labels_destroy_the_planted_signal() -> None:
    planted = csd.planted_corpus(n_repos=40)
    windows = csd.scored_windows(planted, 5)
    real = metrics.auroc(
        [window.features.drift_level for window in windows],
        [window.failed for window in windows],
    )
    shuffled = csd.shuffle_labels_within_repo(planted, seed=7)
    shuffled_windows = csd.scored_windows(shuffled, 5)
    collapsed = metrics.auroc(
        [window.features.drift_level for window in shuffled_windows],
        [window.failed for window in shuffled_windows],
    )
    assert real > 0.9
    assert collapsed < 0.6


def test_constant_scores_sit_at_chance_and_do_not_beat_the_null() -> None:
    samples = [_sample("a", f"a-{i}", drift=0, failed=i % 2 == 0) for i in range(20)]
    report = csd.score_report(csd.scored_windows(samples, 3), csd.PRIMARY_AXIS)
    assert report["auroc"] == 0.5
    assert report["beats_null"] is False
    assert report["null"]["null_ci95"] == [0.5, 0.5]


def test_target_met_requires_every_pre_registered_bar() -> None:
    passing = {
        "auroc": 0.65,
        "beats_null": True,
        "fire": {"precision": 0.6, "volume": 0.1},
    }
    assert csd.target_met(passing) is True
    assert csd.target_met({**passing, "auroc": 0.55}) is False
    assert csd.target_met({**passing, "beats_null": False}) is False
    assert csd.target_met({**passing, "fire": {"precision": 0.4, "volume": 0.1}}) is False
    assert csd.target_met({**passing, "fire": {"precision": 0.6, "volume": 0.01}}) is False


def test_sessions_from_corpus_carry_the_terminal_label() -> None:
    failed = make_trajectory([make_step(success=False)], trajectory_id="r__a__m__h")
    resolved = make_trajectory([make_step()], trajectory_id="r__b__m__h", terminal_resolved=True)
    samples = csd.samples_from_corpus([failed, resolved])
    assert [sample.failed for sample in samples] == [True, False]
    assert {sample.repo for sample in samples} == {"r"}
