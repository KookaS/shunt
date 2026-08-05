"""Pipeline orchestration: stage order, gating flags, failure isolation, and the summary.

Every stage module is stubbed (no live/Docker/paid path) by patching `pipeline.run_module`.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import pytest

from benchmark import config, pipeline
from benchmark.escalation import schema
from benchmark.runner import replay_admissibility

# Captured before the autouse fixture below stubs the module attribute out.
_REAL_WRITE_MANIFEST = pipeline.write_figure_manifest


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
        "restamp": False,
        "stamp_workers": 1,
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


@pytest.fixture(autouse=True)
def _never_rebaseline_the_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the stubbed figures stage from re-baselining the committed manifest."""
    # Every test here stubs run_module, so nothing is ever actually redrawn — letting the
    # write through would re-baseline the staleness gate from the test suite, which is the
    # one thing that gate must never do.
    monkeypatch.setattr(pipeline, "write_figure_manifest", lambda *a, **k: pipeline.FIGURE_MANIFEST)


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
    stub: _Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        pipeline,
        "pending_trajectories",
        lambda *a, **k: [("traj1", "inst1", tmp_path / "traj1.jsonl")],
    )
    result = pipeline.run_pipeline(_args(live=True))
    mods = stub.modules
    assert mods.index(pipeline.RUN_MATRIX) < mods.index(pipeline.OFFLINE_REPLAY)
    assert mods.index(pipeline.OFFLINE_REPLAY) < mods.index(pipeline.ESCALATION_EVAL)
    assert mods.index(pipeline.ESCALATION_EVAL) < mods.index(pipeline.ROUTING_REPORT)
    assert mods.index(pipeline.ROUTING_REPORT) < mods.index(pipeline.STANDALONE_FIGURES[0].module)
    assert result.outcomes == {
        pipeline.COLLECT: "ran",
        pipeline.STAMP: "ran",
        pipeline.EVALUATE: "ran",
        pipeline.REPORT: "ran",
        pipeline.FIGURES: "ran",
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


def test_stamp_timeout_fails_the_stage_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    # A timed-out replay leaves the trajectory carrying whatever it was stamped with BEFORE, so
    # swallowing it (the old `skipping straggler` warning) is how fabricated outcomes survive a
    # rebuild that reports success. The stage now fails; isolation still lets report/figures run.
    rec = _Recorder(raise_for={pipeline.OFFLINE_REPLAY: subprocess.TimeoutExpired("cmd", 5.0)})
    monkeypatch.setattr(pipeline, "run_module", rec)
    monkeypatch.setattr(config, "load", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_capability_lines", lambda: ("m1", 1, "OK"))
    monkeypatch.setattr(pipeline, "_real_cost", lambda: 0.0)
    reaped: list[str] = []
    monkeypatch.setattr(pipeline, "_reap_replay_container", reaped.append)
    monkeypatch.setattr(
        pipeline,
        "pending_trajectories",
        lambda *a, **k: [("traj1", "inst1", Path("x/traj1.jsonl"))],
    )
    result = pipeline.run_pipeline(_args(live=True))
    assert result.outcomes[pipeline.STAMP] == "failed"
    assert result.returncode == 1
    assert reaped == ["traj1"]  # the timed-out replay's container is not left running
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


class _Step:
    def __init__(self, fid: str | None, *, confirmed: bool) -> None:
        self.failing_check_id = fid
        self.confirmed = confirmed


class _Traj:
    def __init__(self, name: str, steps: list[_Step]) -> None:
        self.header = SimpleNamespace(trajectory_id=name, instance_id="inst")
        self.steps = steps


_BODIES: Final[dict[str, list[_Step]]] = {
    "stamped.jsonl": [_Step("check-x", confirmed=True), _Step(None, confirmed=True)],
    "clean.jsonl": [_Step(None, confirmed=True), _Step(None, confirmed=True)],
    # Only the terminal step is confirmed: the replay never ran on the prefix.
    "fresh.jsonl": [_Step(None, confirmed=False), _Step(None, confirmed=True)],
}


def _live_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in _BODIES:
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(schema, "load_jsonl", lambda path: _Traj(path.name, _BODIES[path.name]))
    return tmp_path


def test_pending_skips_already_stamped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # ONE predicate, shared with the eval (`features.is_stamped`). The old test here was
    # `any(step.failing_check_id)`, which re-queued a fully replayed run forever whenever no step
    # in it had failed — while the eval scored that same run as stamped. `clean.jsonl` is that run.
    live = _live_dir(monkeypatch, tmp_path)
    pending = pipeline.pending_trajectories(live)
    assert [p.name for _, _, p in pending] == ["fresh.jsonl"]


def test_restamp_queues_every_trajectory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # `--from stamp` alone cannot drive a rebuild: every committed trajectory is already stamped,
    # so the default predicate makes the stage a no-op on the corpus it is meant to regenerate.
    live = _live_dir(monkeypatch, tmp_path)
    pending = pipeline.pending_trajectories(live, restamp=True)
    assert sorted(p.name for _, _, p in pending) == sorted(_BODIES)


def test_ledger_entry_under_the_current_instrument_is_not_requeued(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The resume point, and the ONE thing that tells a gate-CLEARED trajectory apart from one the
    # stamping stage never reached: both are unstamped on disk, by design.
    live = _live_dir(monkeypatch, tmp_path)
    pipeline.record_stamped(live, "fresh.jsonl", replay_admissibility.instrument_digest())
    assert pipeline.pending_trajectories(live) == []
    assert [p.name for _, _, p in pipeline.pending_trajectories(live, restamp=True)] == [
        "clean.jsonl",
        "stamped.jsonl",
    ]


def test_default_replay_timeout_clears_the_slowest_measured_trajectory() -> None:
    # One measured sympy trajectory took 1 271 s. At the old 120 s default the stage logged
    # `skipping straggler` and dropped exactly the biggest trajectories, silently.
    assert pipeline.DEFAULT_REPLAY_TIMEOUT >= 1271.0


def test_a_ledger_from_a_different_instrument_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A stale ledger must never mark work done: the digest changes with the replay source, so an
    # instrument edit re-queues everything instead of silently keeping the old outcomes.
    live = _live_dir(monkeypatch, tmp_path)
    pipeline.record_stamped(live, "fresh.jsonl", "an-older-instrument")
    assert [p.name for _, _, p in pipeline.pending_trajectories(live)] == ["fresh.jsonl"]


class TestParallelStamping:
    """The stamp stage fans out over instances. These pin what the fan-out must not change."""

    def test_default_workers_leave_the_host_headroom_and_stay_under_the_memory_cap(self) -> None:
        # Each replay is a container running a test file — roughly one core — and the measured
        # penalty for oversubscribing CPU is ~2x, so a too-high default is actively harmful.
        workers = pipeline.default_stamp_workers()
        assert 1 <= workers <= pipeline._MAX_STAMP_WORKERS
        assert workers <= max(1, (os.cpu_count() or 1) - pipeline._RESERVED_CORES)

    def test_an_instance_is_never_split_across_workers(self) -> None:
        # THE CORRECTNESS REASON FOR GROUPING. The admissibility gate is per INSTANCE and costs
        # two container test runs; its verdict is cached. Two workers on two trajectories of one
        # instance both miss that cache, both run the gate, and — the legs being real test runs —
        # CAN disagree, leaving some of the instance's trajectories stamped and the rest cleared.
        pending = [(f"t{i}", f"inst-{i % 3}", Path(f"t{i}.jsonl")) for i in range(9)]
        groups = pipeline.group_by_instance(pending)
        assert sorted(len(g) for g in groups) == [3, 3, 3]
        for group in groups:
            assert len({instance for _, instance, _ in group}) == 1
        assert sum(len(g) for g in groups) == len(pending)

    def test_the_longest_instance_is_scheduled_first(self) -> None:
        # An 11-trajectory unit picked up last is a tail that idles every other worker.
        pending = [("a1", "a", Path("a1"))]
        pending += [(f"b{i}", "b", Path(f"b{i}")) for i in range(4)]
        assert [len(g) for g in pipeline.group_by_instance(pending)] == [4, 1]

    def test_every_trajectory_is_replayed_exactly_once_and_only_successes_are_ledgered(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: list[str] = []
        seen_lock = threading.Lock()

        def fake(module: str, argv: list[str], **_k: object) -> subprocess.CompletedProcess[str]:
            with seen_lock:
                seen.append(argv[0])
            rc = 1 if argv[0] == "t-bad" else 0
            return subprocess.CompletedProcess([module], rc, stdout="done", stderr="")

        monkeypatch.setattr(pipeline, "run_module", fake)
        pending = [(f"t{i}", f"inst-{i % 4}", tmp_path / f"t{i}.jsonl") for i in range(12)]
        pending.append(("t-bad", "inst-bad", tmp_path / "t-bad.jsonl"))
        failures = pipeline.run_stamp_stage(
            pending, live_dir=tmp_path, workers=4, replay_timeout=5.0, heartbeat_s=1000.0
        )
        assert sorted(seen) == sorted(t for t, _, _ in pending)
        assert len(failures) == 1 and "t-bad" in failures[0]
        # A replay that exited non-zero must NOT be ledgered: the ledger is the resume point, and
        # marking a failed trajectory done is how it keeps its pre-rebuild stamps for ever.
        ledger = pipeline.load_stamp_ledger(tmp_path)
        assert "t-bad" not in ledger
        assert len(ledger) == 12

    def test_the_workers_actually_overlap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Without this the whole change could be a no-op that still passes every other test here.
        peak, live = 0, 0
        counter_lock = threading.Lock()

        def fake(module: str, argv: list[str], **_k: object) -> subprocess.CompletedProcess[str]:
            nonlocal peak, live
            with counter_lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.15)
            with counter_lock:
                live -= 1
            return subprocess.CompletedProcess([module], 0, stdout="", stderr="")

        monkeypatch.setattr(pipeline, "run_module", fake)
        pending = [(f"t{i}", f"inst-{i}", tmp_path / f"t{i}.jsonl") for i in range(8)]
        pipeline.run_stamp_stage(
            pending, live_dir=tmp_path, workers=4, replay_timeout=5.0, heartbeat_s=1000.0
        )
        assert peak >= 2

    def test_a_resumed_run_redoes_only_what_the_ledger_does_not_have(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Resume under parallelism: the ledger written by N workers must still let the next run
        # skip completed trajectories and redo only the unfinished ones.
        digest = replay_admissibility.instrument_digest()
        live = _live_dir(monkeypatch, tmp_path)
        replayed: list[str] = []

        def fake(module: str, argv: list[str], **_k: object) -> subprocess.CompletedProcess[str]:
            replayed.append(argv[0])
            return subprocess.CompletedProcess([module], 0, stdout="", stderr="")

        monkeypatch.setattr(pipeline, "run_module", fake)
        pipeline.record_stamped(live, "clean.jsonl", digest)
        pending = pipeline.pending_trajectories(live, restamp=True)
        pipeline.run_stamp_stage(
            pending, live_dir=live, workers=4, replay_timeout=5.0, heartbeat_s=1000.0
        )
        assert sorted(replayed) == ["fresh.jsonl", "stamped.jsonl"]
        assert sorted(pipeline.load_stamp_ledger(live)) == sorted(_BODIES)
        assert pipeline.pending_trajectories(live, restamp=True) == []

    def test_the_heartbeat_makes_a_stall_distinguishable_from_slow_progress(
        self, tmp_path: Path
    ) -> None:
        # A supervisor must not have to infer "hung" from the absence of output: the in-flight
        # trajectories are named with their age, so a 40-minute replay is visible as one.
        progress = pipeline._StampProgress(total=3, live_dir=tmp_path)
        progress.begin("t-slow")
        progress.finish("t-done", None)
        line = progress.heartbeat()
        assert "1/3 done" in line and "2 left" in line
        assert "1 in flight" in line and "t-slow" in line
        assert "instances rejected" in line and "last failure: none" in line

    def test_a_childs_output_is_tagged_with_the_trajectory_it_came_from(self) -> None:
        # N children share one pipe; untagged lines cannot be attributed, and a gate verdict that
        # cannot be attributed to a trajectory makes a multi-hour rebuild undiagnosable.
        block = pipeline._child_block("traj-9", "restamped x", b"INFO admissibility: ADMISSIBLE")
        assert block.splitlines() == [
            "  [traj-9] INFO admissibility: ADMISSIBLE",
            "  [traj-9] restamped x",
        ]


class TestStandaloneFigureFreshness:
    """The 12 standalone figures sat on NO refresh path; timing_comparison.png shipped
    stale because of it. The gate below is what makes that state visible."""

    def test_every_committed_standalone_figure_is_declared(self) -> None:
        declared = {out for job in pipeline.STANDALONE_FIGURES for out in job.outputs}
        # report.py owns the rest of the directory; these are the ones no stage touched.
        assert "timing_comparison.png" in declared
        assert "strategy_comparison.png" in declared
        assert len(declared) == 12

    def test_declared_outputs_all_exist(self) -> None:
        assert pipeline.missing_figures() == []

    def test_committed_figures_are_not_stale(self) -> None:
        """FAILS when a figure's inputs moved without the figure being redrawn.

        Fix by running `make benchmark-figures` (it redraws and re-records the manifest) —
        never by hand-editing benchmark/routing/figure_inputs.json.
        """
        assert pipeline.stale_figures() == []

    def test_a_changed_input_is_detected(self, tmp_path: Path) -> None:
        manifest = tmp_path / "figure_inputs.json"
        _REAL_WRITE_MANIFEST(manifest)
        assert pipeline.stale_figures(manifest) == []
        digests = pipeline.figure_digests()
        digests["plot_timing"] = "0" * 64
        manifest.write_text(json.dumps(digests))
        assert pipeline.stale_figures(manifest) == ["plot_timing"]

    def test_an_absent_manifest_is_stale_not_silently_ok(self, tmp_path: Path) -> None:
        assert pipeline.stale_figures(tmp_path / "absent.json") == [
            job.name for job in pipeline.STANDALONE_FIGURES
        ]

    def test_the_strategy_set_is_part_of_the_digest(self) -> None:
        """Adding a strategy must mark the strategy/timing figures stale — the exact
        drift that let Price-Cascade never reach timing_comparison.png."""
        jobs = {job.name: job for job in pipeline.STANDALONE_FIGURES}
        for name in ("plot_timing", "plot_strategies"):
            assert any(p.name == "strategies" for p in jobs[name].inputs)

    def test_the_model_registry_is_part_of_the_data_digest(self) -> None:
        """A `default_arm` change in models.yaml re-picks each model's canonical routing
        row, which moved every strategy number without any results.csv edit. It must be
        fingerprinted or the committed figures silently outlive the registry."""
        inputs = pipeline._data_inputs()
        assert any(p.name == "models.yaml" and "config" in str(p) for p in inputs)


def _module_origin(name: str) -> Path | None:
    """Source file for an IN-REPO module name, or None for anything else."""
    if not (name.startswith("benchmark") or name.startswith("shunt")):
        return None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or spec.origin is None or spec.origin == "namespace":
        return None
    path = Path(spec.origin).resolve()
    return path if path.exists() else None


def _import_deps(name: str) -> set[str]:
    """The in-repo modules a module's source imports (absolute and relative, ast-resolved)."""
    origin = _module_origin(name)
    if origin is None:
        return set()
    tree = ast.parse(origin.read_text(encoding="utf-8"))
    pkg = name.rpartition(".")[0]
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            deps.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".") if pkg else []
                cut = node.level - 1
                base = ".".join(parts[: len(parts) - cut]) if cut < len(parts) else ""
                full = f"{base}.{node.module}" if base and node.module else (base or node.module)
                if full:
                    deps.add(full)
            elif node.module:
                deps.add(node.module)
                for alias in node.names:
                    sub = f"{node.module}.{alias.name}"
                    if _module_origin(sub):
                        deps.add(sub)
    return deps


def _figure_closure(module: str) -> set[str]:
    """The transitive in-repo import closure of a figure producer module."""
    seen: set[str] = set()
    queue = [module]
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        for dep in sorted(_import_deps(name)):
            if _module_origin(dep) is not None:
                queue.append(dep)
    return seen


def _input_origins(job: pipeline.FigureJob) -> set[Path]:
    """Every source file a figure job's inputs hash (files plus directories' *.py)."""
    origins: set[Path] = set()
    for item in job.inputs:
        item = item.resolve()
        if item.is_dir():
            origins.update(p.resolve() for p in item.rglob("*.py"))
        else:
            origins.add(item)
    return origins


class TestFigureAnalysisClosure:
    """The figure digest must cover the analysis modules each figure's imports reach."""

    # viz_knn imported summary/config/impute/metrics while its `inputs` tuple omitted them, so
    # stale_figures()==[] could certify a figure drawn from changed analysis code. The digest now
    # includes the routing analysis layer; this pins the closure so the tuple cannot silently
    # shrink again. Collection/eval machinery is exempt: its outputs are already fingerprinted
    # as data.

    def test_every_figure_analysis_dependency_is_fingerprinted(self) -> None:
        for job in pipeline.STANDALONE_FIGURES:
            origins = _input_origins(job)
            missing = sorted(
                name
                for name in _figure_closure(job.module)
                if (
                    name.startswith("benchmark.routing")
                    or name
                    in (
                        "benchmark.config",
                        "benchmark.plot_frame",
                        "benchmark.admissibility",
                    )
                )
                and not name.startswith(("benchmark.runner", "benchmark.escalation"))
                and name != "benchmark.corpus_lock"
                and (_module_origin(name) not in origins)
            )
            assert not missing, (
                f"{job.name} reaches analysis module(s) its inputs tuple omits — add them to "
                f"`inputs` (a change there can alter the figure without stale_figures noticing): "
                + ", ".join(missing)
            )
