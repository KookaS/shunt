"""Tests for derive_defer_labels.py: exclusion semantics and the committed output."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from benchmark import config
from benchmark.routing.scripts import derive_defer_labels as ddl

_CHEAP = "deepseek-v4-flash"


def _cell(passed: bool, cost: float = 0.01, calls: int = 5, timeout: bool = False) -> dict:
    return {
        "pass": passed,
        "cost": cost,
        "real_cost": cost,
        "calls": calls,
        "reasoning": "high",
        "stop_reason": "" if not timeout else "step_limit",
        "timeout_flag": timeout,
    }


def test_defer_rows_excludes_missing_and_non_observations() -> None:
    raw = {
        "t1": {_CHEAP: {"high": _cell(True)}},  # -> defer row
        "t2": {_CHEAP: {"high": _cell(False)}},  # -> defer row (cheap fail)
        "t3": {},  # no cheap row at all
        "t4": {_CHEAP: {"nothink": _cell(True), "think": _cell(True)}},  # no default arm
        "t5": {_CHEAP: {"high": _cell(False, timeout=True)}},  # censored
    }
    flat = {
        "t1": {_CHEAP: raw["t1"][_CHEAP]["high"]},
        "t2": {_CHEAP: raw["t2"][_CHEAP]["high"]},
        "t5": {_CHEAP: raw["t5"][_CHEAP]["high"]},
    }
    rows, excluded, arms_per_task = ddl._defer_rows(_CHEAP, raw, flat)
    assert [r["challenge_id"] for r in rows] == ["t1", "t2"]
    assert [r["cheap_pass"] for r in rows] == [1, 0]
    reasons = [r for _c, r in excluded]
    assert any("no rung-0 row" in r for r in reasons)
    assert any("default-arm" in r for r in reasons)
    assert any("non-observation" in r for r in reasons)
    assert arms_per_task == {1: 2}


def test_main_writes_committed_defer_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "defer_labels.csv"
    monkeypatch.setattr(sys, "argv", ["derive", "--out", str(out)])
    assert ddl.main() == 0
    with out.open(newline="") as f:
        written = list(csv.DictReader(f))
    assert written, "committed results.csv should yield at least one defer label"
    assert list(written[0]) == list(ddl.COLUMNS)
    assert {r["cheap_model"] for r in written} == {config.enabled_models()[0]}
    assert all(r["cheap_pass"] in ("0", "1") for r in written)
