"""Pipeline orchestration: stage order, gating flags, failure isolation, and the summary.

Every stage module is stubbed (no live/Docker/paid path) by patching `pipeline.run_module`.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from benchmark import config, pipeline
from benchmark.escalation import schema


def _args(**over: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "strategy": "cost_optimal",
        "config": "benchmark/benchmark.yaml",
        "live": False,
        "max_cost": None,
        "max_cost_overshoot": 0.0,
        "workers": 1,
        "timeout": 600,
        "max_start_failures": 5,
        "check_images": False,
        "no_report": False,
        "start_from": pipeline.COLLECT,
        "replay_timeout": 5.0,
    }
    base.update(over)
    return argparse.Namespace(**base)


class _Recorder:
    """Stub for pipeline.run_module: records (module, argv), returns canned output / raises."""

    def __init__(
        self,
        stdout_map: dict[str, str] | None = None,
        raise_for: dict[str, Exception] | None = None,
    ) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._stdout = stdout_map or {}
        self._raise = raise_for or {}

    def __call__(
        self, module: str, argv: list[str], *, timeout: float | None = None, capture: bool = False
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((module, list(argv)))
        if module in self._raise:
            raise self._raise[module]
        return subprocess.CompletedProcess(
            [module], 0, stdout=self._stdout.get(module, ""), stderr=""
        )

    @property
    def modules(self) -> list[str]:
        return [m for m, _ in self.calls]


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Stub run_module and neutralise the file-reading summary helpers (no real artifacts)."""
    rec = _Recorder(
        stdout_map={
            pipeline.ESCALATION_EVAL: '{\n  "status": "OK",\n  "reason": "beats no-skill"\n}',
            pipeline.ROUTING_EVAL: "  Paired contrast (kNN vs Always-Frontier) [all tasks]: +3.0pp",
        }
    )
    monkeypatch.setattr(pipeline, "run_module", rec)
    monkeypatch.setattr(config, "load", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_capability_lines", lambda: ("m1 < m2", 2, "OK"))
    monkeypatch.setattr(pipeline, "_real_cost", lambda: 1.2345)
    return rec


def test_stages_dispatch_in_order_when_live(
    stub: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_unstamped_trajectories",
        lambda *a, **k: [("traj1", "inst1", Path("x/traj1.jsonl"))],
    )
    result = pipeline.run_pipeline(_args(live=True))
    mods = stub.modules
    assert mods.index(pipeline.RUN_MATRIX) < mods.index(pipeline.OFFLINE_REPLAY)
    assert mods.index(pipeline.OFFLINE_REPLAY) < mods.index(pipeline.ESCALATION_EVAL)
    assert mods.index(pipeline.ESCALATION_EVAL) < mods.index(pipeline.ROUTING_REPORT)
    assert result.outcomes == {
        pipeline.COLLECT: "ran",
        pipeline.STAMP: "ran",
        pipeline.EVALUATE: "ran",
        pipeline.REPORT: "ran",
    }


def test_no_report_runs_only_collect(stub: _Recorder, capsys: pytest.CaptureFixture[str]) -> None:
    result = pipeline.run_pipeline(_args(live=True, no_report=True))
    assert stub.modules == [pipeline.RUN_MATRIX]
    assert result.outcomes[pipeline.STAMP] == "skipped"
    assert "Consolidated benchmark summary" not in capsys.readouterr().out


def test_from_report_skips_collect_stamp_evaluate(stub: _Recorder) -> None:
    result = pipeline.run_pipeline(_args(start_from=pipeline.REPORT))
    assert result.outcomes[pipeline.COLLECT] == "skipped"
    assert result.outcomes[pipeline.STAMP] == "skipped"
    assert result.outcomes[pipeline.EVALUATE] == "skipped"
    assert result.outcomes[pipeline.REPORT] == "ran"
    assert pipeline.RUN_MATRIX not in stub.modules
    assert pipeline.OFFLINE_REPLAY not in stub.modules
    assert pipeline.ROUTING_REPORT in stub.modules


def test_simulated_run_skips_stamp(stub: _Recorder) -> None:
    result = pipeline.run_pipeline(_args(live=False))
    assert pipeline.OFFLINE_REPLAY not in stub.modules
    assert result.outcomes[pipeline.STAMP] == "skipped"
    assert result.outcomes[pipeline.COLLECT] == "ran"
    assert result.outcomes[pipeline.REPORT] == "ran"


def test_stamp_timeout_is_caught_and_pipeline_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder(raise_for={pipeline.OFFLINE_REPLAY: subprocess.TimeoutExpired("cmd", 5.0)})
    monkeypatch.setattr(pipeline, "run_module", rec)
    monkeypatch.setattr(config, "load", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_capability_lines", lambda: ("m1", 1, "OK"))
    monkeypatch.setattr(pipeline, "_real_cost", lambda: 0.0)
    monkeypatch.setattr(
        pipeline,
        "_unstamped_trajectories",
        lambda *a, **k: [("traj1", "inst1", Path("x/traj1.jsonl"))],
    )
    result = pipeline.run_pipeline(_args(live=True))
    # The straggler is skipped inside stamp, so the stage itself still "ran" and report follows.
    assert result.outcomes[pipeline.STAMP] == "ran"
    assert pipeline.ROUTING_REPORT in rec.modules


def test_report_stage_failure_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder()

    def failing(module: str, argv: list[str], **k: object) -> subprocess.CompletedProcess[str]:
        rec.calls.append((module, list(argv)))
        rc = 1 if module == pipeline.ROUTING_REPORT else 0
        return subprocess.CompletedProcess([module], rc, stdout="", stderr="")

    monkeypatch.setattr(pipeline, "run_module", failing)
    monkeypatch.setattr(config, "load", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_capability_lines", lambda: ("m1", 1, "OK"))
    monkeypatch.setattr(pipeline, "_real_cost", lambda: 0.0)
    result = pipeline.run_pipeline(_args(live=False))
    assert result.outcomes[pipeline.REPORT] == "failed"
    assert result.returncode == 1


def test_consolidated_summary_is_emitted(
    stub: _Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline.run_pipeline(_args(live=False))
    out = capsys.readouterr().out
    assert "Consolidated benchmark summary" in out
    assert "routing kill-gate : " in out
    assert "Paired contrast" in out
    assert "escalation        : OK (SKILL)" in out
    assert "capability rank   : m1 < m2" in out
    assert "capability bands  : 2" in out
    assert "real cost         : $1.2345" in out
    assert "=== [pipeline] stage: summary ===" in out


def test_collect_argv_passes_through_flags() -> None:
    argv = pipeline._collect_argv(
        _args(live=True, max_cost=2.0, workers=3, check_images=True, strategy="ladder")
    )
    assert "--live" in argv
    assert argv[argv.index("--max-cost") + 1] == "2.0"
    assert argv[argv.index("--workers") + 1] == "3"
    assert "--check-images" in argv
    assert argv[argv.index("--strategy") + 1] == "ladder"


def test_unstamped_skips_already_stamped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Step:
        def __init__(self, fid: str | None) -> None:
            self.failing_check_id = fid

    class _Header:
        trajectory_id = "t"
        instance_id = "inst"

    class _Traj:
        header = _Header()

        def __init__(self, steps: list[_Step]) -> None:
            self.steps = steps

    stamped = tmp_path / "stamped.jsonl"
    fresh = tmp_path / "fresh.jsonl"
    stamped.write_text("{}")
    fresh.write_text("{}")

    def fake_load(path: Path) -> _Traj:
        return _Traj([_Step("check-x")]) if path.name == "stamped.jsonl" else _Traj([_Step(None)])

    monkeypatch.setattr(schema, "load_jsonl", fake_load)
    pending = pipeline._unstamped_trajectories(tmp_path)
    names = [p.name for _, _, p in pending]
    assert names == ["fresh.jsonl"]
