"""Tests for derive_judge_difficulty.py's label-source enforcement and glob handling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmark.routing.scripts import derive_judge_difficulty as djd


def _fake_records(judges: list[str]) -> list[dict]:
    return [
        {
            "task_id": f"t{i}",
            "judge": judge,
            "prompt_version": 2,
            "difficulty": 3.0,
            "raw_cost": 0.001,
            "parsed": True,
        }
        for i, judge in enumerate(judges)
    ]


@pytest.fixture
def probe_file(tmp_path: Path) -> str:
    p = tmp_path / "fake_probe.jsonl"
    p.write_text(json.dumps(_fake_records(["gpt-5.6-terra"])[0]) + "\n")
    return str(p)


def _run(probe: str, out: str, monkeypatch: pytest.MonkeyPatch, records: list[dict]) -> int:
    monkeypatch.setattr(sys, "argv", ["derive", "--probe", probe, "--out", out])
    monkeypatch.setattr(djd.judge_probe_metrics, "load_records", lambda _paths: records)
    monkeypatch.setattr(djd, "_loo_r2", lambda *args: (0.027, 190))
    return djd.main()


def test_refuses_without_both_judges(
    tmp_path, probe_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = str(tmp_path / "out.json")
    rc = _run(probe_file, out, monkeypatch, _fake_records(["gpt-5.6-terra"]))
    assert rc == 1
    assert not (tmp_path / "out.json").exists()


def test_absolute_probe_glob_is_handled_not_a_crash(
    tmp_path, probe_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Path.glob raises NotImplementedError on an absolute pattern; the absolute path must
    # resolve (here to a terra-only set, which then REFUSES cleanly rather than crashing).
    out = str(tmp_path / "out.json")
    rc = _run(probe_file, out, monkeypatch, _fake_records(["gpt-5.6-terra"]))
    assert rc == 1


def test_enforcement_refuses_when_terra_diverges_from_the_anchor(
    tmp_path, probe_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = str(tmp_path / "out.json")
    monkeypatch.setattr(
        djd.judge_probe_metrics,
        "load_records",
        lambda _p: _fake_records(["gpt-5.6-terra", "claude-sonnet-5"]),
    )
    monkeypatch.setattr(
        djd, "_loo_r2", lambda j, _r, _t: (0.05, 190) if j == "gpt-5.6-terra" else (0.0, 190)
    )
    monkeypatch.setattr(sys, "argv", ["derive", "--probe", probe_file, "--out", out])
    rc = djd.main()
    assert rc == 1
    assert not (tmp_path / "out.json").exists()


def test_writes_the_table_when_terra_matches_the_anchor(
    tmp_path, probe_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = str(tmp_path / "out.json")
    monkeypatch.setattr(
        djd.judge_probe_metrics,
        "load_records",
        lambda _p: _fake_records(["gpt-5.6-terra", "claude-sonnet-5"]),
    )
    monkeypatch.setattr(
        djd, "_loo_r2", lambda j, _r, _t: (0.029, 190) if j == "gpt-5.6-terra" else (0.027, 190)
    )
    monkeypatch.setattr(sys, "argv", ["derive", "--probe", probe_file, "--out", out])
    rc = djd.main()
    assert rc == 0
    payload = json.loads((tmp_path / "out.json").read_text())
    assert "adoption_rule" in payload
    assert payload["tasks"]["t0"]["difficulty"] == 3.0
