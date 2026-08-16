"""TDD suite for benchmark.cost_reconcile — no network, no real registry, tmp_path only."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest

from benchmark import cost_reconcile as cr

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
]


def _row(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "challenge_id": "astropy__astropy-1",
        "model": "deepseek-v4-flash",
        "reasoning": "high",
        "pass": "True",
        "cost": "0.01",
        "in_tok": "100000",
        "out_tok": "5000",
        "calls": "10",
        "version_hash": "vh",
        "model_version": "deepseek-v4-flash",
        "arm_hash": "ah",
        "real_cost": "0.01",
        "estimated_cost": "0.02",
        "timeout_flag": "False",
        "image_digest": "sha256:x",
        "computed_at": "2026-07-25T12:00:00+00:00",
    }
    base.update(over)
    return base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pricing() -> dict[str, dict[str, object]]:
    return {
        "deepseek-v4-flash": {
            "input_cost_per_1m": 0.14,
            "output_cost_per_1m": 0.28,
        },
        "kimi-k3": {
            "input_cost_per_1m": 2.0,
            "output_cost_per_1m": 8.0,
        },
    }


# ── reconciliation ────────────────────────────────────────────────────────────


def test_reconcile_within_tolerance_no_alarm() -> None:
    rec = cr.reconcile(tracked_sum=1.05, billed_amount=1.0, tolerance=0.15)
    assert rec.ratio == pytest.approx(1.05)
    assert rec.abs_divergence == pytest.approx(0.05)
    assert rec.alarm is False


def test_reconcile_past_tolerance_alarms() -> None:
    # the real bug: tracked 1.25 vs billed 35.0 → ratio ~0.036, massive undercount
    rec = cr.reconcile(tracked_sum=1.25, billed_amount=35.0, tolerance=0.15)
    assert rec.ratio == pytest.approx(1.25 / 35.0)
    assert rec.alarm is True


def test_reconcile_billed_zero_but_tracked_positive_alarms() -> None:
    rec = cr.reconcile(tracked_sum=5.0, billed_amount=0.0, tolerance=0.15)
    assert rec.alarm is True


def test_reconcile_ratio_edge_just_over_tolerance() -> None:
    assert cr.reconcile(tracked_sum=1.16, billed_amount=1.0, tolerance=0.15).alarm is True
    assert cr.reconcile(tracked_sum=1.14, billed_amount=1.0, tolerance=0.15).alarm is False


# ── tracked summary + accounting holes ─────────────────────────────────────────


def test_zero_cost_with_calls_is_suspicious(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_csv(
        path,
        [
            _row(real_cost="0.02", calls="10"),
            _row(real_cost="0.0", calls="8"),  # HOLE: work happened, no cost recorded
            _row(real_cost="0.0", calls="0"),  # genuine non-run
        ],
    )
    rows = cr.load_rows(path)
    summ = cr.summarize_tracked(rows, _pricing())
    assert summ.tracked_sum == pytest.approx(0.02)
    assert summ.zero_cost_with_calls == 1
    assert summ.zero_cost_no_calls == 1
    assert summ.row_count == 3


def test_censored_zero_cost_row_is_not_a_hole() -> None:
    # A reaped CENSORED cell (calls>0, cost=0, step_limit) legitimately harvested no cost —
    # it must NOT be counted as an accounting hole (parity with validate._check_accounting).
    censored = cr.ResultRow(
        model="kimi-k3",  # a paid model
        in_tok=1000,
        out_tok=100,
        calls=5,
        real_cost=0.0,
        estimated_cost=0.0,
        computed_at="2026-07-25T12:00:00+00:00",
        passed=False,
        stop_reason="step_limit",
    )
    summ = cr.summarize_tracked([censored], _pricing())
    assert summ.zero_cost_with_calls == 0
    assert summ.zero_cost_no_calls == 0  # calls>0 so it is not a non-run either


def test_paid_noncensored_zero_cost_row_is_a_hole() -> None:
    # The $35-miss fingerprint: a PAID model ran (calls>0) but recorded $0 and is NOT censored.
    hole = cr.ResultRow(
        model="kimi-k3",
        in_tok=1000,
        out_tok=100,
        calls=5,
        real_cost=0.0,
        estimated_cost=0.0,
        computed_at="2026-07-25T12:00:00+00:00",
        passed=False,
        stop_reason="unsolved",
    )
    summ = cr.summarize_tracked([hole], _pricing())
    assert summ.zero_cost_with_calls == 1


def test_tracked_sum_uses_real_cost(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_csv(path, [_row(real_cost="0.10"), _row(real_cost="0.25")])
    summ = cr.summarize_tracked(cr.load_rows(path))
    assert summ.tracked_sum == pytest.approx(0.35)


# ── window filtering ────────────────────────────────────────────────────────────


def test_window_filters_by_computed_at(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_csv(
        path,
        [
            _row(real_cost="0.10", computed_at="2026-07-25T12:00:00+00:00"),
            _row(real_cost="0.20", computed_at="2026-07-26T12:00:00+00:00"),
            _row(real_cost="0.40", computed_at="2026-07-27T12:00:00+00:00"),
        ],
    )
    start = datetime(2026, 7, 26, tzinfo=UTC)
    end = datetime(2026, 7, 27, tzinfo=UTC)
    summ = cr.summarize_tracked(cr.load_rows(path, start=start, end=end))
    assert summ.tracked_sum == pytest.approx(0.20)


# ── independent cross-check ─────────────────────────────────────────────────────


def test_cross_check_flags_large_gap() -> None:
    # 1M in + 1M out on kimi-k3 list price = 2.0 + 8.0 = $10 token-estimate,
    # but usage.cost recorded as $0.50 → 20x gap → anomaly.
    rows = [
        cr.ResultRow(
            model="kimi-k3",
            in_tok=1_000_000,
            out_tok=1_000_000,
            calls=20,
            real_cost=0.50,
            estimated_cost=10.0,
            computed_at="2026-07-25T12:00:00+00:00",
        )
    ]
    checks = cr.cross_check(rows, _pricing(), gap_threshold=3.0)
    kimi = next(c for c in checks if c.model == "kimi-k3")
    assert kimi.token_estimate == pytest.approx(10.0)
    assert kimi.usage_cost == pytest.approx(0.50)
    assert kimi.gap_factor == pytest.approx(20.0)
    assert kimi.anomaly is True


def test_cross_check_small_gap_no_anomaly() -> None:
    # token-estimate $10, usage $4 (cache discount) → 2.5x < 3.0 threshold
    rows = [
        cr.ResultRow(
            model="kimi-k3",
            in_tok=1_000_000,
            out_tok=1_000_000,
            calls=20,
            real_cost=4.0,
            estimated_cost=10.0,
            computed_at="2026-07-25T12:00:00+00:00",
        )
    ]
    kimi = cr.cross_check(rows, _pricing(), gap_threshold=3.0)[0]
    assert kimi.anomaly is False


def test_cross_check_zero_usage_with_tokens_is_anomaly() -> None:
    rows = [
        cr.ResultRow(
            model="deepseek-v4-flash",
            in_tok=500_000,
            out_tok=10_000,
            calls=12,
            real_cost=0.0,
            estimated_cost=0.07,
            computed_at="2026-07-25T12:00:00+00:00",
        )
    ]
    check = cr.cross_check(rows, _pricing(), gap_threshold=3.0)[0]
    assert check.zero_usage_with_tokens is True
    assert check.anomaly is True


def test_cross_check_unpriced_model_with_spend_is_anomaly() -> None:
    rows = [
        cr.ResultRow(
            model="ghost-model",
            in_tok=100,
            out_tok=100,
            calls=1,
            real_cost=0.5,
            estimated_cost=0.5,
            computed_at="2026-07-25T12:00:00+00:00",
        )
    ]
    check = cr.cross_check(rows, _pricing(), gap_threshold=3.0)[0]
    assert check.priced is False
    assert check.anomaly is True


# ── ledger ──────────────────────────────────────────────────────────────────────


def test_ledger_append_and_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "artifacts" / "ledger.jsonl"
    entry = cr.build_ledger_entry(
        cr.reconcile(1.25, 35.0, 0.15),
        window_start=None,
        window_end=None,
        timestamp="2026-07-27T09:00:00+00:00",
    )
    assert cr.append_ledger(ledger, entry) is True
    assert cr.append_ledger(ledger, entry) is False  # identical → not re-appended
    lines = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(lines) == 1
    assert lines[0]["tracked_sum"] == pytest.approx(1.25)
    assert lines[0]["billed_amount"] == pytest.approx(35.0)
    assert lines[0]["alarm"] is True
    assert lines[0]["timestamp"] == "2026-07-27T09:00:00+00:00"


def test_ledger_distinct_entries_both_appended(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    e1 = cr.build_ledger_entry(
        cr.reconcile(1.0, 1.0, 0.15), None, None, "2026-07-27T09:00:00+00:00"
    )
    e2 = cr.build_ledger_entry(
        cr.reconcile(2.0, 1.0, 0.15), None, None, "2026-07-27T10:00:00+00:00"
    )
    assert cr.append_ledger(ledger, e1) is True
    assert cr.append_ledger(ledger, e2) is True
    lines = [x for x in ledger.read_text().splitlines() if x.strip()]
    assert len(lines) == 2


# ── CLI ──────────────────────────────────────────────────────────────────────────


def _clean_csv(path: Path) -> None:
    _write_csv(
        path,
        [
            _row(
                model="deepseek-v4-flash",
                real_cost="1.0",
                estimated_cost="1.2",
                in_tok="1000000",
                out_tok="1000000",
                calls="10",
            ),
        ],
    )


def test_cli_exits_nonzero_on_alarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "results.csv"
    _clean_csv(path)
    monkeypatch.setattr(cr, "_registry_pricing", lambda: _pricing())
    code = cr.main(
        [
            "--results",
            str(path),
            "--billed",
            "35.0",
            "--tolerance",
            "0.15",
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
            "--timestamp",
            "2026-07-27T09:00:00+00:00",
        ]
    )
    assert code != 0


def test_cli_exits_zero_on_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "results.csv"
    # token-estimate = 0.14 + 0.28 = 0.42; usage 1.0 → gap 2.38x < 3; billed 1.0 == tracked
    _clean_csv(path)
    monkeypatch.setattr(cr, "_registry_pricing", lambda: _pricing())
    code = cr.main(
        [
            "--results",
            str(path),
            "--billed",
            "1.0",
            "--tolerance",
            "0.15",
            "--gap-threshold",
            "3.0",
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
            "--timestamp",
            "2026-07-27T09:00:00+00:00",
        ]
    )
    assert code == 0


def test_cli_writes_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "results.csv"
    _clean_csv(path)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(cr, "_registry_pricing", lambda: _pricing())
    cr.main(
        [
            "--results",
            str(path),
            "--billed",
            "1.0",
            "--ledger",
            str(ledger),
            "--timestamp",
            "2026-07-27T09:00:00+00:00",
        ]
    )
    assert ledger.exists()
    assert len(ledger.read_text().splitlines()) == 1
