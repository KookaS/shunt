"""TDD suite for benchmark.routing.validate + benchmark.validate_results.

No network, no real registry — an in-test pricing dict and tmp_path CSVs only.
Each invariant: a violating row is caught (right code/severity) and a valid row passes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

import pytest

from benchmark import config
from benchmark import validate_results as vr
from benchmark.routing import validate
from benchmark.routing.validate import (
    ACCOUNTING_HOLE,
    BAD_STOP_REASON,
    MALFORMED_NUMERIC,
    MALFORMED_TIMESTAMP,
    MISSING_COLLECTION_FIELD,
    PASS_STOP_MISMATCH,
    SUSPICIOUS_ZERO,
    TIMEOUT_FLAG_MISMATCH,
    UNSOLVED_NOT_RUN,
    DataIntegrityError,
    Severity,
)

# A paid model and a genuinely free model, as load_pricing would key them.
PRICING: Final[dict[str, dict[str, float]]] = {
    "kimi-k3": {"input_cost_per_1m": 3.0, "output_cost_per_1m": 15.0},
    "free-local": {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0},
}

_COLUMNS: Final[list[str]] = [
    "challenge_id",
    "model",
    "reasoning",
    "pass",
    "cost",
    "in_tok",
    "out_tok",
    "calls",
    "version_hash",
    "model_version",
    "arm_hash",
    "real_cost",
    "estimated_cost",
    "timeout_flag",
    "image_digest",
    "computed_at",
    "stop_reason",
    "step_limit",
    "cost_limit",
    "scaffold_version",
    "sampling_hash",
    "prompt_hash",
]


def _row(**over: object) -> dict:
    """A VALID solved paid-model row; override any field to violate one invariant."""
    base: dict = {
        "challenge_id": "astropy__astropy-1",
        "model": "kimi-k3",
        "reasoning": "high",
        "pass": "True",
        "cost": "0.05",
        "in_tok": "100000",
        "out_tok": "5000",
        "calls": "12",
        "version_hash": "vh",
        "model_version": "kimi-k3",
        "arm_hash": "ah",
        "real_cost": "0.05",
        "estimated_cost": "0.06",
        "timeout_flag": "False",
        "image_digest": "sha256:x",
        "computed_at": "2026-07-25T18:50:14.557864+00:00",
        "stop_reason": "solved",
        "step_limit": "150",
        "cost_limit": "4.0",
        "scaffold_version": "2.4.5",
        "sampling_hash": "sh",
        "prompt_hash": "ph",
    }
    base.update(over)
    return base


def _codes(violations: list) -> set[str]:
    return {v.code for v in violations}


# ── valid rows pass ───────────────────────────────────────────────────────────


def test_valid_solved_row_has_no_violations() -> None:
    assert validate.validate_row(_row(), PRICING) == []


def test_valid_unsolved_row_that_ran_passes() -> None:
    row = _row(**{"pass": "False", "stop_reason": "unsolved", "real_cost": "0.04", "calls": "9"})
    assert validate.validate_row(row, PRICING) == []


def test_valid_censored_row_passes() -> None:
    row = _row(**{"pass": "False", "stop_reason": "wall_limit", "timeout_flag": "True"})
    assert validate.validate_row(row, PRICING) == []


# ── accounting hole (the $35 fingerprint) ─────────────────────────────────────


def test_paid_zero_cost_with_calls_is_accounting_hole_error() -> None:
    row = _row(**{"real_cost": "0", "calls": "12", "stop_reason": "unsolved", "pass": "False"})
    violations = validate.validate_row(row, PRICING)
    hole = [v for v in violations if v.code == ACCOUNTING_HOLE]
    assert len(hole) == 1
    assert hole[0].severity is Severity.ERROR


def test_free_model_zero_cost_with_calls_is_not_flagged() -> None:
    row = _row(
        **{
            "model": "free-local",
            "model_version": "free-local",
            "real_cost": "0",
            "calls": "8",
            "stop_reason": "unsolved",
            "pass": "False",
        }
    )
    assert ACCOUNTING_HOLE not in _codes(validate.validate_row(row, PRICING))


def test_censored_zero_cost_with_calls_is_exempt_from_accounting_hole() -> None:
    # A reaped (censored) paid cell that ran but never harvested cost is NOT a hole.
    row = _row(
        **{
            "real_cost": "0",
            "calls": "20",
            "pass": "False",
            "timeout_flag": "True",
            "stop_reason": "wall_limit",
        }
    )
    assert ACCOUNTING_HOLE not in _codes(validate.validate_row(row, PRICING))


def test_unpriced_model_zero_cost_with_calls_is_not_flagged() -> None:
    row = _row(**{"model": "mystery", "real_cost": "0", "stop_reason": "unsolved", "pass": "False"})
    assert ACCOUNTING_HOLE not in _codes(validate.validate_row(row, PRICING))


# ── ran-ness ──────────────────────────────────────────────────────────────────


def test_unsolved_with_zero_calls_is_error() -> None:
    row = _row(**{"pass": "False", "stop_reason": "unsolved", "calls": "0", "real_cost": "0"})
    violations = validate.validate_row(row, PRICING)
    assert UNSOLVED_NOT_RUN in _codes(violations)
    ranness = [v for v in violations if v.code == UNSOLVED_NOT_RUN]
    assert ranness[0].severity is Severity.ERROR


# ── schema ────────────────────────────────────────────────────────────────────


def test_bad_stop_reason_is_error() -> None:
    row = _row(**{"stop_reason": "exploded", "pass": "False"})
    assert BAD_STOP_REASON in _codes(validate.validate_row(row, PRICING))


def test_pass_true_but_not_solved_is_error() -> None:
    row = _row(**{"pass": "True", "stop_reason": "unsolved"})
    assert PASS_STOP_MISMATCH in _codes(validate.validate_row(row, PRICING))


def test_solved_but_pass_false_is_error() -> None:
    row = _row(**{"pass": "False", "stop_reason": "solved"})
    assert PASS_STOP_MISMATCH in _codes(validate.validate_row(row, PRICING))


def test_timeout_flag_inconsistent_with_stop_reason_is_error() -> None:
    # wall_limit ⇒ timeout_flag must be True; here it is False.
    row = _row(**{"pass": "False", "stop_reason": "wall_limit", "timeout_flag": "False"})
    assert TIMEOUT_FLAG_MISMATCH in _codes(validate.validate_row(row, PRICING))


def test_timeout_flag_true_on_step_limit_is_error() -> None:
    # step_limit is censored but NOT a timeout stop ⇒ timeout_flag must be False.
    row = _row(**{"pass": "False", "stop_reason": "step_limit", "timeout_flag": "True"})
    assert TIMEOUT_FLAG_MISMATCH in _codes(validate.validate_row(row, PRICING))


# ── collection-param provenance ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "field",
    ["step_limit", "cost_limit", "scaffold_version", "sampling_hash", "prompt_hash"],
)
def test_row_missing_collection_field_is_error(field: str) -> None:
    row = _row()
    del row[field]
    violations = validate.validate_row(row, PRICING)
    missing = [v for v in violations if v.code == MISSING_COLLECTION_FIELD]
    assert len(missing) == 1
    assert missing[0].severity is Severity.ERROR


def test_empty_collection_field_is_not_error() -> None:
    # The grandfather rule: a legacy row may carry an EMPTY anchor (never recollect it).
    # Only an ABSENT column is a schema error.
    row = _row(**{"sampling_hash": "", "prompt_hash": "", "scaffold_version": ""})
    assert validate.validate_row(row, PRICING) == []


def test_cli_gate_exits_nonzero_on_missing_collection_column(tmp_path, monkeypatch) -> None:
    # A CSV whose header lacks the collection-param columns: every parsed row misses them.
    results = tmp_path / "results.csv"
    header = ",".join(c for c in _COLUMNS if c != "step_limit")
    with results.open("w", newline="") as fh:
        fh.write(header + "\n")
        fh.write("astropy__astropy-1,kimi-k3,high,True,0.05,100000,5000,12\n")
    monkeypatch.setattr(config, "load_pricing", lambda *a, **k: PRICING)
    monkeypatch.setattr(config, "load", lambda *a, **k: None)
    assert vr.main(["--results", str(results)]) == 1


# ── well-formedness ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["", "nan", "inf", "-1", "abc"])
def test_malformed_real_cost_is_error(bad: str) -> None:
    row = _row(real_cost=bad)
    assert MALFORMED_NUMERIC in _codes(validate.validate_row(row, PRICING))


def test_negative_tokens_is_error() -> None:
    row = _row(in_tok="-5")
    assert MALFORMED_NUMERIC in _codes(validate.validate_row(row, PRICING))


@pytest.mark.parametrize("bad", ["", "not-a-date", "2026-07-25", "2026-07-25T00:00:00"])
def test_malformed_or_naive_timestamp_is_error(bad: str) -> None:
    # A naive (offset-less) timestamp is rejected: computed_at must be UTC-aware.
    row = _row(computed_at=bad)
    assert MALFORMED_TIMESTAMP in _codes(validate.validate_row(row, PRICING))


# ── suspicious (WARN) ─────────────────────────────────────────────────────────


def test_zero_cost_zero_calls_noncensored_is_warn() -> None:
    # Neither ran nor censored: solved with no cost/calls is odd → WARN, not ERROR.
    row = _row(**{"real_cost": "0", "calls": "0", "pass": "True", "stop_reason": "solved"})
    violations = validate.validate_row(row, PRICING)
    warns = [v for v in violations if v.code == SUSPICIOUS_ZERO]
    assert len(warns) == 1
    assert warns[0].severity is Severity.WARN


def test_zero_cost_zero_calls_censored_is_not_warn() -> None:
    row = _row(
        **{
            "real_cost": "0",
            "calls": "0",
            "pass": "False",
            "timeout_flag": "True",
            "stop_reason": "wall_limit",
        }
    )
    assert SUSPICIOUS_ZERO not in _codes(validate.validate_row(row, PRICING))


# ── report + blocking ─────────────────────────────────────────────────────────


def test_report_counts_and_blocking() -> None:
    hole = {"real_cost": "0", "calls": "5", "stop_reason": "unsolved", "pass": "False"}
    warn = {"real_cost": "0", "calls": "0", "pass": "True", "stop_reason": "solved"}
    rows = [_row(), _row(**hole), _row(**warn)]
    report = validate.validate_results(rows, PRICING)
    assert report.total_rows == 3
    assert len(report.offending) == 2
    assert report.count_by_code().get(ACCOUNTING_HOLE) == 1
    assert report.error_count == 1
    assert validate.has_blocking_violations(report) is True


def test_report_clean_is_not_blocking() -> None:
    report = validate.validate_results([_row(), _row()], PRICING)
    assert report.offending == []
    assert validate.has_blocking_violations(report) is False


def test_warn_only_report_is_not_blocking() -> None:
    row = _row(**{"real_cost": "0", "calls": "0", "pass": "True", "stop_reason": "solved"})
    report = validate.validate_results([row], PRICING)
    assert report.error_count == 0
    assert validate.has_blocking_violations(report) is False


# ── write-time enforcement ────────────────────────────────────────────────────


def test_enforce_row_raises_on_error() -> None:
    row = _row(**{"real_cost": "0", "calls": "5", "stop_reason": "unsolved", "pass": "False"})
    with pytest.raises(DataIntegrityError):
        validate.enforce_row(row, PRICING)


def test_enforce_row_does_not_raise_on_valid() -> None:
    validate.enforce_row(_row(), PRICING)  # no exception


def test_enforce_row_does_not_raise_on_warn_only() -> None:
    row = _row(**{"real_cost": "0", "calls": "0", "pass": "True", "stop_reason": "solved"})
    validate.enforce_row(row, PRICING)  # WARN never aborts


# ── build_row integration (the abort would fire on the $35 row) ────────────────


def test_build_row_raises_on_accounting_hole() -> None:
    from benchmark.runner import run_matrix

    outcome = {
        "pass": False,
        "real_cost": 0.0,  # paid model ran but no cost harvested — the $35 fingerprint
        "in_tok": 100000,
        "out_tok": 5000,
        "calls": 12,
        "stop_reason": "unsolved",
        "computed_at": "2026-07-25T18:50:14.557864+00:00",
    }
    with pytest.raises(DataIntegrityError):
        run_matrix._build_row(
            "astropy__astropy-1",
            "kimi-k3",
            outcome,
            {"astropy__astropy-1": "vh"},
            {"kimi-k3": "kimi-k3"},
            {"kimi-k3": {"input": 3.0, "output": 15.0}},
        )


def test_build_row_valid_outcome_succeeds() -> None:
    from benchmark.runner import run_matrix

    outcome = {
        "pass": True,
        "real_cost": 0.05,
        "in_tok": 100000,
        "out_tok": 5000,
        "calls": 12,
        "computed_at": "2026-07-25T18:50:14.557864+00:00",
    }
    built = run_matrix._build_row(
        "astropy__astropy-1",
        "kimi-k3",
        outcome,
        {"astropy__astropy-1": "vh"},
        {"kimi-k3": "kimi-k3"},
        {"kimi-k3": {"input": 3.0, "output": 15.0}},
    )
    assert built["real_cost"] == 0.05


# ── pre-analysis CLI gate ─────────────────────────────────────────────────────


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _COLUMNS})


def test_cli_gate_exits_nonzero_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = tmp_path / "results.csv"
    hole = {"real_cost": "0", "calls": "5", "stop_reason": "unsolved", "pass": "False"}
    _write_csv(results, [_row(), _row(**hole)])
    monkeypatch.setattr(config, "load_pricing", lambda *a, **k: PRICING)
    monkeypatch.setattr(config, "load", lambda *a, **k: None)
    code = vr.main(["--results", str(results)])
    assert code == 1


def test_cli_gate_exits_zero_on_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = tmp_path / "results.csv"
    _write_csv(results, [_row(), _row()])
    monkeypatch.setattr(config, "load_pricing", lambda *a, **k: PRICING)
    monkeypatch.setattr(config, "load", lambda *a, **k: None)
    code = vr.main(["--results", str(results)])
    assert code == 0


def test_cli_gate_zero_on_warn_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = tmp_path / "results.csv"
    _write_csv(
        results,
        [_row(**{"real_cost": "0", "calls": "0", "pass": "True", "stop_reason": "solved"})],
    )
    monkeypatch.setattr(config, "load_pricing", lambda *a, **k: PRICING)
    monkeypatch.setattr(config, "load", lambda *a, **k: None)
    code = vr.main(["--results", str(results)])
    assert code == 0
