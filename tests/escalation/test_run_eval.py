"""run_eval runs end-to-end on the fixtures AND the results.csv bootstrap, produces metrics +
plots + an authenticity count, and reports the length-1 degeneracy explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.escalation import datasets, metrics, replay, run_eval
from shunt.router.escalation import EscalationAction
from tests.escalation.factories import make_step, make_trajectory

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).parent / "fixtures"
_RESULTS = _ROOT / "benchmark/routing/results.csv"


def test_evaluate_reports_degeneracy_on_bootstrap() -> None:
    trajectories = datasets.results_csv_bootstrap(_RESULTS)
    assert trajectories, "the real results.csv produced bootstrap trajectories"
    report = run_eval.evaluate(trajectories, datasets.DEFAULT_GRID)
    # every bootstrap row is length-1 → fully degenerate, stated explicitly
    assert report.n_degenerate == report.n_trajectories
    assert "CANNOT fire" in report.degeneracy_note
    assert report.n_escalated == 0  # the trigger cannot fire on length-1 streams
    assert report.authenticity_errors == 0


def test_main_runs_end_to_end_exit_zero(tmp_path, capsys) -> None:
    rc = run_eval.main(
        [
            "--fixtures",
            str(_FIXTURES),
            "--results-csv",
            str(_RESULTS),
            "--plots-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr()
    assert '"degeneracy_note"' in out.out
    assert "CANNOT fire" in out.err  # degeneracy surfaced, not implied
    assert (tmp_path / "reports" / "pr_curve.png").exists()  # plots produced


def test_default_grid_is_the_full_sweep() -> None:
    # 2 escalate_after_n x 3 stale_window x 2 ladder = 12 grid points
    assert len(datasets.DEFAULT_GRID) == 12


def _long_single_key_trajectory(n_failures: int):  # type: ignore[no-untyped-def]
    """A long single-dedup-key confirmed-failing trajectory (terminal-failed)."""
    steps = [
        make_step(step_index=i, decision_index=i, success=False, failing_check_id="pkg/x.py::t")
        for i in range(n_failures)
    ]
    return make_trajectory(steps, terminal_resolved=False)


def test_prefix_scores_are_cumulative_and_monotone() -> None:
    # The detector score is "has the policy flagged this failing task by prefix t": 0.0 before the
    # first non-HOLD directive, 1.0 at and after it. A long trajectory drives the abstract ladder
    # to its ceiling, so the RAW per-step directive stream saturates back to HOLD (non-monotone) —
    # the exact stream the live engine never emits. The cumulative score is invariant to that.
    traj = _long_single_key_trajectory(60)
    point = replay.GridPoint(2, 10)
    raw = [
        0.0 if a is EscalationAction.HOLD else 1.0
        for a in replay.replay_config(traj, point.to_config()).directives
    ]
    # The raw stream is non-monotone: it flags, then falls back to HOLD at the ceiling.
    assert raw != sorted(raw)

    scores = run_eval._prefix_scores(traj, point)
    assert len(scores) == len(traj.steps)
    assert scores == sorted(scores), "cumulative detection score must be non-decreasing"
    assert set(scores) <= {0.0, 1.0}
    first_flag = raw.index(1.0)
    assert scores[:first_flag] == [0.0] * first_flag
    assert scores[first_flag:] == [1.0] * (len(scores) - first_flag)


def test_bootstrap_reports_insufficient_data_status() -> None:
    # Audit P0: on data the trigger cannot fire on, the report is INSUFFICIENT_DATA with a reason,
    # NOT a misleading auprc==prevalence presented as signal.
    trajectories = datasets.results_csv_bootstrap(_RESULTS)
    report = run_eval.evaluate(trajectories, datasets.DEFAULT_GRID)
    d = report.to_dict()
    assert d["status"] == "INSUFFICIENT_DATA"
    assert "recurrence trigger never fired" in d["reason"]
    assert d["best_config"] is None  # no cell discriminates on degenerate data
    assert d["recall"] == 0.0  # the legible "detector never fires", not a weak-but-real number
    assert d["confusion"]["tp"] == 0
    assert len(d["cells"]) == 12  # full sweep computed, not grid[0] only


def _all_fail_single_key(n: int, key: str = "pkg/x.py::t"):  # type: ignore[no-untyped-def]
    steps = [
        make_step(step_index=i, decision_index=i, success=False, failing_check_id=key)
        for i in range(n)
    ]
    return make_trajectory(steps, terminal_resolved=False)


def _cell_at(auprc: float, prevalence: float) -> metrics.CellReport:
    from benchmark.calibration.labeler_metrics import ConfusionMatrix

    return metrics.CellReport(
        escalate_after_n=2,
        stale_window=5,
        ladder="effort_then_tier",
        confusion=ConfusionMatrix(tp=1, fp=9, fn=0, tn=90),
        precision=0.1,
        recall=1.0,
        f1=0.18,
        fpr=0.09,
        cohen_kappa=0.0,
        auprc=auprc,
        auroc=0.4,
        prevalence=prevalence,
        n_escalated=10,
        mean_steps_to_detection=2.0,
    )


def test_skill_gate_demotes_ok_to_no_skill_at_or_below_baseline() -> None:
    # A worse-than-no-skill detector (AUPRC ≤ prevalence) must NEVER present as OK — otherwise the
    # unannotated PR/ROC plots read as "works". This is the honesty gate the audit added.
    weak = run_eval._apply_skill_gate("OK", "sufficient", _cell_at(auprc=0.099, prevalence=0.111))
    assert weak[0] == "NO_SKILL"
    assert "no usable signal" in weak[1]
    # A real detector (AUPRC > prevalence) stays OK; a non-OK status is passed through untouched.
    assert run_eval._apply_skill_gate("OK", "s", _cell_at(0.5, 0.111))[0] == "OK"
    assert run_eval._apply_skill_gate("INSUFFICIENT_DATA", "x", _cell_at(0.0, 0.5))[0] == (
        "INSUFFICIENT_DATA"
    )


def test_best_config_selects_the_genuine_argmax_on_sufficient_data() -> None:
    # Proof the selection is REAL when data IS sufficient: a length-5 all-failing single-key
    # trajectory. escalate_after_n=2 flags at step 1 (a false positive on the pre-risky prefix,
    # F1=0.857); escalate_after_n=3 flags at step 2, exactly at the risky window (F1=1.0). The
    # cells genuinely differ, so best_config is a real choice, not an arbitrary pick.
    report = run_eval.evaluate([_all_fail_single_key(5)], datasets.DEFAULT_GRID)
    assert report.status == "OK"  # trigger fired, positive labels present
    assert report.best_config is not None
    # cells actually discriminate (not all identical) — the precondition for a meaningful choice
    f1s = {round(c.f1, 3) for c in report.cells}
    assert len(f1s) > 1, "the sweep must discriminate for the selection to be meaningful"
    # the selected cell is the independent argmax under the default objective
    best_recomputed = max(report.cells, key=metrics.objective_max_f1)
    assert report.best_config.escalate_after_n == best_recomputed.escalate_after_n
    assert report.best_config.stale_window == best_recomputed.stale_window
    assert report.best_config.escalate_after_n == 3  # the genuine winner (avoids early-prefix FP)
    assert report.best_config.f1 == pytest.approx(1.0)
    assert report.best_config.f1 == max(c.f1 for c in report.cells)


def test_main_includes_live_dir_trajectories(tmp_path, capsys) -> None:
    # Captured live trajectories (schema JSONL under --live-dir) are loaded into the eval.
    from benchmark.escalation import schema

    live = tmp_path / "live"
    live.mkdir()
    steps = [
        make_step(step_index=i, decision_index=i, failing_check_id="k", success=(i == 2))
        for i in range(3)
    ]
    schema.dump_jsonl(make_trajectory(steps, trajectory_id="live-1"), live / "live-1.jsonl")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    rc = run_eval.main(
        [
            "--fixtures",
            str(fixtures),
            "--results-csv",
            str(tmp_path / "none.csv"),
            "--live-dir",
            str(live),
        ]
    )
    assert rc == 0
    assert '"n_trajectories": 1' in capsys.readouterr().out
