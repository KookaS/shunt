"""StepSnapshotRecorder + the per-step snapshot store: observe-only capture via an injected
exec callable (no container), swallowing errors, and a write/read round-trip."""

from __future__ import annotations

from pathlib import Path

from benchmark.runner.step_snapshots import (
    DIFF_COMMAND,
    StepSnapshotRecorder,
    read_snapshots,
    write_snapshots,
)


def test_recorder_captures_the_diff_keyed_by_step() -> None:
    calls: list[str] = []

    def exec_fn(command: str) -> str:
        calls.append(command)
        return "diff --git a/x b/x\n+changed"

    rec = StepSnapshotRecorder(exec_fn)
    rec.capture(0)
    rec.capture(1)
    assert calls == [DIFF_COMMAND, DIFF_COMMAND]
    assert rec.snapshots == {0: "diff --git a/x b/x\n+changed", 1: "diff --git a/x b/x\n+changed"}


def test_recorder_swallows_exec_errors() -> None:
    def exec_fn(command: str) -> str:
        raise RuntimeError("container gone")

    rec = StepSnapshotRecorder(exec_fn)
    rec.capture(0)  # must not raise — observe-only
    assert rec.snapshots == {}


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    snapshots = {0: "diff-zero", 3: "diff-three"}
    write_snapshots("traj__a__b", snapshots, root=tmp_path)
    assert read_snapshots("traj__a__b", root=tmp_path) == snapshots


def test_read_missing_trajectory_is_empty(tmp_path: Path) -> None:
    assert read_snapshots("nope", root=tmp_path) == {}


def test_attach_recorder_passes_action_dict_to_env_execute() -> None:
    # Regression: mini-swe-agent v2 env.execute takes an action DICT ({"command": ...}), not a
    # bare string — passing a string raised AttributeError and silently captured zero snapshots.
    from benchmark.runner.infer import _attach_snapshot_recorder

    calls: list[object] = []

    class _Env:
        def execute(self, action: object, cwd: str = "", *, timeout: int | None = None) -> dict:
            calls.append(action)
            return {"output": "diff-text", "returncode": 0}

    class _Agent:
        def step(self, *a: object, **k: object) -> None:
            return None

    recorder = _attach_snapshot_recorder(_Agent(), _Env())
    assert recorder is not None
    recorder.capture(0)
    assert calls and isinstance(calls[0], dict) and calls[0].get("command") == DIFF_COMMAND
    assert recorder.snapshots.get(0) == "diff-text"
