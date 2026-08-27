"""Unit tests for the consistency-probe variance metrics (no live calls)."""

from __future__ import annotations

from benchmark.routing.scripts.consistency_probe import mean_pairwise_similarity
from benchmark.routing.scripts.consistency_probe_metrics import (
    auroc_fail,
    calibration,
    finish_reason_diversity,
    length_cv,
    measured_pass,
    per_task_similarity,
    shuffled_null,
)


def test_mean_pairwise_identical_texts() -> None:
    assert mean_pairwise_similarity(["same text here", "same text here"]) == 1.0


def test_mean_pairwise_disjoint_texts() -> None:
    sim = mean_pairwise_similarity(["alpha beta gamma", "delta epsilon zeta"])
    assert sim == 0.0


def test_mean_pairwise_requires_two() -> None:
    assert mean_pairwise_similarity(["only one"]) == 0.0


def test_mean_pairwise_case_and_whitespace_insensitive() -> None:
    assert mean_pairwise_similarity(["Hello  World", "hello world"]) == 1.0


def _record(text: str, run: int = 1, finish: str = "stop") -> dict:
    return {
        "task_id": "t1",
        "run": run,
        "output_text": text,
        "finish_reason": finish,
    }


def test_per_task_similarity_last_run_wins() -> None:
    rows = {
        "t1": {
            "1": [_record("aaa bbb", run=1)],
            "2": [_record("old", run=2), _record("aaa bbb", run=2)],
        }
    }
    out = per_task_similarity(rows)
    assert out["t1"] == 1.0


def test_length_cv_zero_for_constant_lengths() -> None:
    rows = {
        "t1": {
            "1": [_record("hello world")],
            "2": [_record("hello world")],
        }
    }
    assert length_cv(rows)["t1"] == 0.0


def test_length_cv_positive_for_spread_lengths() -> None:
    rows = {
        "t1": {
            "1": [_record("a")],
            "2": [_record("a " * 100)],
        }
    }
    assert length_cv(rows)["t1"] > 0.0


def test_finish_reason_diversity_counts_distinct() -> None:
    rows = {
        "t1": {
            "1": [_record("x", finish="stop")],
            "2": [_record("x", finish="stop")],
            "3": [_record("x", finish="length")],
        }
    }
    assert finish_reason_diversity(rows)["t1"] == 2


def test_calibration_bins_pass_rate() -> None:
    scores = {"a": 0.1, "b": 0.2, "c": 0.5, "d": 0.6, "e": 0.9, "f": 1.0}
    pass_by = {"a": True, "b": True, "c": True, "d": False, "e": False, "f": False}
    bins = calibration(scores, pass_by, bins=3)
    assert len(bins) == 3
    # Low-inconsistency bin passes more than high-inconsistency bin here.
    assert bins[0]["pass_rate"] > bins[2]["pass_rate"]


def test_shuffled_null_collapses_to_chance() -> None:
    # A PERFECT signal (higher score => fail) must sit above the null band.
    n = 20
    scores = {f"t{i}": i / n for i in range(n)}
    # High inconsistency (score) => FAIL, per the paper's direction.
    pass_by = {f"t{i}": i < n // 2 for i in range(n)}
    auc = auroc_fail(scores, pass_by)
    assert auc is not None and auc == 1.0
    lo, hi, _mean = shuffled_null(scores, pass_by, n_perm=500)
    assert lo == lo  # not NaN
    # The perfect observed AUROC must be outside the shuffled null band.
    assert auc > hi


def test_measured_pass_drops_censored(tmp_path) -> None:
    p = tmp_path / "results.csv"
    p.write_text(
        "challenge_id,model,reasoning,pass,stop_reason,timeout_flag\n"
        "t_pass,deepseek-v4-flash,high,True,solved,False\n"
        "t_censored,deepseek-v4-flash,high,False,step_limit,False\n"
    )
    out = measured_pass(["t_pass", "t_censored"], results_path=str(p))
    assert out == {"t_pass": True}


def test_measured_pass_no_row(tmp_path) -> None:
    p = tmp_path / "results.csv"
    p.write_text(
        "challenge_id,model,reasoning,pass,stop_reason,timeout_flag\n"
        "t1,gpt-5-mini,medium,True,solved,False\n"
    )
    assert measured_pass(["t1"], results_path=str(p)) == {}


def test_within_vs_cross_control_recovers_task_identity() -> None:
    # Two tasks whose runs are internally near-identical but mutually disjoint:
    # the planted signal (task identity) must be recovered by the pipeline.
    from benchmark.routing.scripts.consistency_probe_metrics import within_vs_cross_control

    rows = {}
    for idx, tid in enumerate(["t1", "t2", "t3"]):
        vocab = ["alpha", "beta", "gamma", "delta", "epsilon"]
        words = " ".join(f"{w}{idx}" for w in vocab)
        rows[tid] = {str(i): [{"output_text": words, "run": i}] for i in range(1, 6)}
    ctl = within_vs_cross_control(rows, n_perm=200)
    assert ctl["within_auroc"] == 1.0
    assert ctl["clears_null"]


def test_perm_p_value_perfect_signal_is_low() -> None:
    from benchmark.routing.scripts.consistency_probe_metrics import perm_p_value

    n = 20
    scores = {f"t{i}": i / n for i in range(n)}
    pass_by = {f"t{i}": i < n // 2 for i in range(n)}
    p = perm_p_value(1.0, scores, pass_by, n_perm=200)
    assert p < 0.05
