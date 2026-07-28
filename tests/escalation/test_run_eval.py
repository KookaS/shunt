"""run_eval end-to-end: the null gates the status, the shipped default is never silently moved,
and every figure is written through the annotated frame.
"""

from __future__ import annotations

import random
from typing import Final

import pytest

from benchmark.escalation import features, run_eval, schema
from tests.escalation.factories import make_step, make_trajectory

pytest.importorskip("sklearn")

_PERMUTATIONS = 200
_DEPTHS = (5,)
_FIGURES: Final[frozenset[str]] = frozenset(
    {
        "pr_curve.png",
        "roc_curve.png",
        "confusion_matrix.png",
        "permutation_null.png",
        "lead_time_by_outcome.png",
        "sweep_table.png",
        "trajectory_outcomes.png",
        "failure_capture_coverage.png",
    }
)


def _run(tid: str, *, resolved: bool, rng: random.Random, fail_rate: float, n: int = 10):  # type: ignore[no-untyped-def]
    steps = [
        make_step(
            step_index=i,
            decision_index=i,
            action=f"cmd{rng.randrange(4)}",
            success=rng.random() >= fail_rate,
            failing_check_id="pkg::t" if rng.random() < fail_rate else None,
        )
        for i in range(n - 1)
    ]
    steps.append(make_step(step_index=n - 1, decision_index=n - 1, success=resolved))
    return make_trajectory(steps, trajectory_id=tid, terminal_resolved=resolved)


def _null_corpus(n_pairs: int = 22):  # type: ignore[no-untyped-def]
    """Both classes behave identically: there is nothing to find, and the report must say so."""
    from dataclasses import replace

    rng = random.Random(4)
    out = []
    for i in range(n_pairs):
        for resolved, suffix in ((False, "a"), (True, "b")):
            traj = _run(f"c{i}__m__{suffix}", resolved=resolved, rng=rng, fail_rate=0.5)
            out.append(replace(traj, header=replace(traj.header, instance_id=f"c{i}")))
    return out


def _signal_corpus(n_pairs: int = 22):  # type: ignore[no-untyped-def]
    """Failed runs thrash, resolved runs do not — genuine prefix evidence at the same length."""
    from dataclasses import replace

    rng = random.Random(6)
    out = []
    for i in range(n_pairs):
        for resolved, suffix, rate in ((False, "a", 0.9), (True, "b", 0.1)):
            traj = _run(f"c{i}__m__{suffix}", resolved=resolved, rng=rng, fail_rate=rate)
            out.append(replace(traj, header=replace(traj.header, instance_id=f"c{i}")))
    return out


@pytest.fixture(scope="module")
def null_report() -> run_eval.EvalReport:
    """The pipeline is deterministic; refitting it per test only multiplies permutation cost."""
    return run_eval.evaluate(_null_corpus(), depths=_DEPTHS, n_permutations=_PERMUTATIONS)


@pytest.fixture(scope="module")
def signal_report() -> run_eval.EvalReport:
    return run_eval.evaluate(_signal_corpus(), depths=_DEPTHS, n_permutations=_PERMUTATIONS)


def test_a_pure_null_reports_no_skill_not_ok(null_report: run_eval.EvalReport) -> None:
    # The old gate was `auprc > prevalence` with no noise floor: a +0.0008 excess against a null sd
    # of 0.00055 printed `status: OK`. The gate is now the permutation band.
    report = null_report
    assert report.status == "NO_SKILL"
    assert "no usable signal" in report.reason
    assert not any(d.has_skill for d in report.depth_reports)


def test_real_signal_reports_ok(signal_report: run_eval.EvalReport) -> None:
    report = signal_report
    assert report.status == "OK"
    assert "clears its permutation null" in report.reason


def test_no_skill_reason_carries_the_number_not_just_the_verdict(
    null_report: run_eval.EvalReport,
) -> None:
    assert "incremental AUROC" in null_report.reason
    report = null_report
    assert "p=" in report.reason


def test_headline_cell_is_the_shipped_default_not_an_argmax(
    signal_report: run_eval.EvalReport,
) -> None:
    # `best_config` used to be an argmax over a grid that held 2 distinct results, so the reported
    # configuration was a coin flip dressed as an optimisation. The headline is now the shipped
    # default by definition; a better cell is FLAGGED in the notes, never applied.
    report = signal_report
    assert report.headline_cell is not None
    assert report.headline_cell.escalate_after_n == run_eval.SHIPPED_ESCALATE_AFTER_N


def test_a_better_configuration_is_flagged_rather_than_adopted(
    signal_report: run_eval.EvalReport,
) -> None:
    cells = signal_report.policy_cells
    best = max(cells, key=lambda c: c.precision)
    notes = run_eval._default_gap_note(cells)
    if best.escalate_after_n == run_eval.SHIPPED_ESCALATE_AFTER_N:
        assert notes == []
    else:
        assert any("SHIPPED default" in n and "unchanged" in n for n in notes)


def test_unstamped_trajectories_are_excluded_and_counted() -> None:
    corpus = _signal_corpus()
    blind = [
        make_trajectory(
            [make_step(step_index=j, decision_index=j, confirmed=False) for j in range(9)]
            + [make_step(step_index=9, decision_index=9, success=False)],
            trajectory_id=f"z{i}__blind__x",
            terminal_resolved=False,
        )
        for i in range(20)
    ]
    report = run_eval.evaluate([*corpus, *blind], depths=_DEPTHS, n_permutations=_PERMUTATIONS)
    assert report.n_trajectories == len(corpus) + 20
    assert report.n_stamped == len(corpus)
    assert any("no per-step verified outcomes" in n for n in report.notes)
    # …and the exclusion reaches every figure's footer, not just the JSON.
    limits = run_eval._run_annotations(report).limitations
    assert any("excluded from this figure" in lim for lim in limits)


def test_report_dict_has_no_summed_escalation_count(
    signal_report: run_eval.EvalReport,
) -> None:
    payload = signal_report.to_dict()
    # The top-level `n_escalated: 4980` next to `n_trajectories: 799` is gone: the count lives on
    # each cell, where it means "trajectories this configuration escalated".
    assert "n_escalated" not in payload
    for cell in payload["policy_cells"]:  # type: ignore[union-attr]
        assert cell["n_escalated"] <= cell["n_trajectories"]


def test_capture_coverage_is_reported_over_the_whole_corpus(
    signal_report: run_eval.EvalReport,
) -> None:
    assert sum(c.n_trajectories for c in signal_report.coverage) == len(_signal_corpus())


def test_main_writes_every_figure_through_the_frame(tmp_path, capsys) -> None:
    live = tmp_path / "live"
    live.mkdir()
    for traj in _signal_corpus(
        n_pairs=25
    ):  # >= MIN_ROWS at depth 5, else the prefix figures are (correctly) skipped
        schema.dump_jsonl(traj, live / f"{traj.header.trajectory_id}.jsonl")
    plots_dir = tmp_path / "reports"
    rc = run_eval.main(
        [
            "--live-dir",
            str(live),
            "--plots-dir",
            str(plots_dir),
            "--permutations",
            str(_PERMUTATIONS),
            "--depths",
            "5",
        ]
    )
    assert rc == 0
    assert {p.name for p in plots_dir.glob("*.png")} == _FIGURES
    # dpi 150 at (9, 5.5) plus the footer: a bare dpi=80 figure never reaches this size.
    assert all(p.stat().st_size > 20_000 for p in plots_dir.glob("*.png"))
    assert '"prefix_model"' in capsys.readouterr().out


def test_main_errors_when_the_live_directory_is_missing(tmp_path) -> None:
    assert run_eval.main(["--live-dir", str(tmp_path / "nope")]) == 1


def test_main_errors_on_an_empty_live_directory(tmp_path) -> None:
    empty = tmp_path / "live"
    empty.mkdir()
    assert run_eval.main(["--live-dir", str(empty)]) == 1


def test_status_caveats_reach_the_footer(
    null_report: run_eval.EvalReport, signal_report: run_eval.EvalReport
) -> None:
    annotations = run_eval._run_annotations(null_report)
    assert any("NO USABLE SIGNAL" in lim for lim in annotations.limitations)
    assert run_eval._run_annotations(signal_report).limitations == ()


def test_depths_are_absolute_decisions_not_fractions() -> None:
    # A fractional prefix needs the total length, which is future information. The CLI takes
    # integers, and a depth no run reaches yields no report rather than an imputed one.
    report = run_eval.evaluate(_signal_corpus(), depths=(5, 500), n_permutations=_PERMUTATIONS)
    assert [d.depth for d in report.depth_reports] == [5]
    assert features.DEFAULT_DEPTHS == (5, 10, 20)
