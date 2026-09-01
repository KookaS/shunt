"""StepSnapshotRecorder + the per-step snapshot store: observe-only capture via an injected
exec callable (no container), swallowing errors, and a write/read round-trip. Plus the full
message-list dump that pairs with the snapshots and the trajectory jsonl."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from benchmark.runner.step_snapshots import (
    DIFF_COMMAND,
    MESSAGE_LIST_ROOT,
    TESTBED,
    StepSnapshotRecorder,
    message_list_path,
    read_snapshots,
    snapshot_dir,
    write_snapshots,
)

_ROOT = Path(__file__).resolve().parents[1]
_LIVE_DIR = _ROOT / "benchmark" / "escalation" / "data" / "live"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, check=False)


def _check_ignore(path: str) -> bool:
    """True iff git ignores `path` (check-ignore exits 0 when the path is ignored)."""
    return _git("check-ignore", "-q", path).returncode == 0


# ---------------------------------------------------------------------------
# The full message-list dump pairs with the trajectory jsonl and the snapshots.
# ---------------------------------------------------------------------------


def test_message_list_path_is_keyed_like_the_snapshot_dir(tmp_path: Path) -> None:
    trajectory_id = "psf__requests-1142__kimi-k3__default"
    assert message_list_path(trajectory_id, root=tmp_path).name == f"{trajectory_id}.json"
    assert message_list_path(trajectory_id, root=tmp_path).parent == tmp_path
    assert snapshot_dir(trajectory_id, root=tmp_path).name == trajectory_id


def test_three_artifact_keys_pair_by_one_trajectory_id(tmp_path: Path) -> None:
    # One cell's three artifacts share the same trajectory-id key — the committed escalation
    # trajectory jsonl, the per-step snapshot scratch dir, and the full message-list dump. All
    # three names derive from the SAME `make_trajectory_id` value, so a consumer can pair them.
    from benchmark.escalation.live_capture import make_trajectory_id

    trajectory_id = make_trajectory_id("psf__requests-1142", "kimi-k3", "default")
    live_dir = tmp_path / "live"
    snap_root = tmp_path / "snapshots"
    msg_root = tmp_path / "message_lists"

    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / f"{trajectory_id}.jsonl").write_text("{}")
    write_snapshots(trajectory_id, {0: "diff-zero", 3: "diff-three"}, root=snap_root)
    dump = message_list_path(trajectory_id, root=msg_root)
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text("{}")

    assert dump.stem == trajectory_id
    assert snapshot_dir(trajectory_id, root=snap_root).name == trajectory_id
    assert (live_dir / f"{trajectory_id}.jsonl").exists()
    assert read_snapshots(trajectory_id, root=snap_root) == {0: "diff-zero", 3: "diff-three"}


@pytest.mark.skipif(not (_ROOT / ".git").exists(), reason="not a git checkout")
def test_message_list_scratch_is_gitignored() -> None:
    # `git status` stays clean after a run — the dump path is ignored, so an accidental
    # `git add -A` can never sweep a raw transcript into the public repo.
    assert _check_ignore(str(MESSAGE_LIST_ROOT / "traj__m__d.json"))


@pytest.mark.skipif(not (_ROOT / ".git").exists(), reason="not a git checkout")
def test_git_status_stays_clean_after_a_captured_dump() -> None:
    # End-to-end: write a message-list dump under the real scratch and prove `git status
    # --porcelain` reports nothing for it (the path is ignored, tracked files untouched).
    from benchmark.escalation.live_capture import make_trajectory_id

    trajectory_id = make_trajectory_id("psf__requests-1142", "kimi-k3", "default")
    dump = message_list_path(trajectory_id)
    dump.parent.mkdir(parents=True, exist_ok=True)
    existed = dump.exists()
    dump.write_text('{"messages": []}')
    try:
        porcelain = _git("status", "--porcelain")
        assert porcelain.stdout == "" or str(dump.relative_to(_ROOT)) not in porcelain.stdout
    finally:
        if not existed:
            dump.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# No provider credential appears in a captured dump.
# ---------------------------------------------------------------------------

# Self-identifying provider-key shapes (the prefixes the shipped providers and gitleaks'
# default ruleset key off). Assembled from halves in the positive control below so the literal
# forms never appear in this file and trip the gitleaks gate.
_PROVIDER_SECRET_PATTERN = re.compile(
    r"sk-ant-api03-[A-Za-z0-9_-]{20,}"
    r"|\bsk-[A-Za-z0-9]{20,}"
    r"|\brq-[A-Za-z0-9]{20,}"
    r"|\bxai-[A-Za-z0-9]{20,}"
    r"|\bAIza[0-9A-Za-z_-]{20,}"
    r"|\bAKIA[0-9A-Z]{16}"
    r"|\bgh[pousr]_[0-9A-Za-z]{20,}"
)


def _plant(prefix: str, body_len: int = 24) -> str:
    return f"{prefix}{'A' * body_len}"


def _credential_hits(text: str) -> list[str]:
    return _PROVIDER_SECRET_PATTERN.findall(text)


def test_credential_scan_fires_on_planted_provider_keys() -> None:
    # Positive control for the absence assertions below — a scanner that never fires would
    # make "no credential found" vacuous. Assembled at runtime so no literal key-shaped string
    # lives in the repo (same construction as tests/router/test_escalation_redaction.py).
    assert _credential_hits(_plant("sk-ant-api03-"))
    assert _credential_hits(_plant("sk-"))
    assert _credential_hits(_plant("rq-"))
    assert _credential_hits(_plant("xai-"))
    assert _credential_hits(_plant("ghp_"))
    assert _credential_hits(_plant("AIza"))
    assert _credential_hits(_plant("AKIA", body_len=16))


# The committed escalation corpus is LFS-tracked; an un-hydrated checkout holds pointer files.
_CORPUS_PRESENT = _LIVE_DIR.is_dir() and any(
    p.read_bytes()[:1] == b"{" for p in list(_LIVE_DIR.glob("*.jsonl"))[:1]
)


@pytest.mark.skipif(not _CORPUS_PRESENT, reason="live corpus absent or LFS pointers not hydrated")
def test_captured_files_carry_no_provider_credential() -> None:
    # No provider credential appears in a captured dump. The message-list dump is an
    # UNREDACTED raw transcript (it passes through `redact_secrets` nowhere), so its defences
    # are the gitignore above (never committed ⇒ gitleaks never sees it) plus this explicit
    # scan over the real captured files. This scan is FORENSIC — it only sees what a past run
    # happened to write; the structural half is `benchmark.runner.scaffold_model`, asserted by
    # tests/test_scaffold_credential_isolation.py, which keeps the credential out of the dump
    # in the first place — the committed escalation corpus (scrubbed on its own
    # write path via `schema.dump_jsonl`) and any message-list dump already in the scratch.
    files = sorted(_LIVE_DIR.glob("*.jsonl"))
    if MESSAGE_LIST_ROOT.is_dir():
        files += sorted(MESSAGE_LIST_ROOT.rglob("*.json"))
    assert files, "no captured files to scan"
    for path in files:
        hits = _credential_hits(path.read_text(encoding="utf-8"))
        assert not hits, f"{path} carries {len(hits)} credential-shaped hit(s): {hits}"


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


def test_the_capture_command_sees_staged_work_not_just_unstaged(tmp_path: Path) -> None:
    """Positive control for the capture command, against a REAL git repo (no container)."""
    # A bare ``git diff`` is worktree-vs-index, so an agent's ``git add`` erases its own work
    # from the capture: the offline replay then rebuilds a tree the agent never had and the
    # FAIL_TO_PASS tests fail for a reason the agent did not cause. That defect cost the
    # 2026-07 corpus 1 897 empty-capture steps and 62 partial ones, so this is asserted on
    # behaviour, not on the string.
    # REVERT PROBE: drop ``HEAD`` from ``DIFF_COMMAND`` and ``alpha.py`` vanishes from the capture.
    import subprocess  # noqa: PLC0415

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)  # noqa: S603, S607

    git("init", "-q")
    git("config", "user.email", "t@e.st")
    git("config", "user.name", "t")
    # Distinct stems on purpose: "staged.py" is a SUBSTRING of "unstaged.py", so naming them that
    # way makes the assertion below pass even when the staged half is missing from the capture.
    for name in ("alpha.py", "beta.py"):
        (tmp_path / name).write_text("original\n")
    git("add", "-A")
    git("commit", "-qm", "base")
    for name in ("alpha.py", "beta.py"):
        (tmp_path / name).write_text("agent edit\n")
    git("add", "alpha.py")  # the exact move that used to blind the instrument

    def exec_fn(command: str) -> str:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)  # noqa: S602
        return proc.stdout

    rec = StepSnapshotRecorder(exec_fn, DIFF_COMMAND.replace(TESTBED, str(tmp_path)))
    rec.capture(0)
    captured = rec.snapshots[0]
    assert "alpha.py" in captured, "STAGED work must still be visible to the capture"
    assert "beta.py" in captured, "unstaged work must still be visible to the capture"


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
