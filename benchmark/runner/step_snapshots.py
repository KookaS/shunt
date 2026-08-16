"""Per-step code snapshots for offline verified-outcome replay (Mechanism B)."""

# During a live agent run we capture a cheap per-step ``git diff HEAD`` of the agent's checkout
# (~1s/step, observe-only) and persist it to a local, gitignored scratch keyed by trajectory +
# step. An offline pass later replays each snapshot in a rebuilt container to derive the real
# per-step outcome — zero added model spend, re-runnable.

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

_LOG = logging.getLogger(__name__)

# The agent works in /testbed (a git checkout at the instance base_commit). Reading a diff is
# non-mutating (observe-only): it never stages, so it cannot alter the agent's index or the
# patch it submits. New (untracked) files are not captured — a documented Mechanism-B limit.
TESTBED: Final[str] = "/testbed"

# `HEAD`, NOT a bare `git diff`. A bare `git diff` is worktree-vs-INDEX, so the moment an agent
# runs `git add` its work leaves the capture while staying on disk: the capture collapses to 0
# bytes (or, once the agent edits something else, to a NON-EMPTY but incomplete diff), the offline
# replay rebuilds a tree the agent never had, and the FAIL_TO_PASS tests fail for a reason the
# agent did not cause. That is not hypothetical — it is the defect `state_capture_audit` exists to
# clean up after, and it cost the 2026-07 corpus 1 897 empty-capture steps plus 62 partial ones.
# `git diff HEAD` is worktree-vs-HEAD, so staged and unstaged edits are both captured.
#
# WHAT THIS STILL DOES NOT CATCH, deliberately: an agent that COMMITS moves HEAD, and the delta
# ends up behind it. Capturing that needs the instance's base_commit, which this recorder is not
# given — see the module note in `state_capture_audit`. Untracked files remain uncaptured too;
# `git add -N` would fix that but mutates the index, which observe-only forbids.
DIFF_COMMAND: Final[str] = f"git -C {TESTBED} diff HEAD"

# Local, gitignored scratch (mirrors benchmark/routing/artifacts/): snapshots are ephemeral
# session output, never committed data.
SNAPSHOT_ROOT: Final[Path] = Path(__file__).resolve().parent / "artifacts" / "step_snapshots"
_STEP_GLOB: Final[str] = "step_*.diff"

# Local, gitignored scratch holding each live cell's FULL agent message list (mini-swe-agent's
# `output_path` dump, written by DefaultAgent.run() in its finally block). Keyed by trajectory id
# EXACTLY like the per-step snapshot dir and the committed trajectory jsonl
# (`escalation/data/live/<trajectory_id>.jsonl`), so the three artifacts pair one-to-one. The dump
# is a RAW, UNREDACTED transcript — the message list passes through `redact_secrets` nowhere — so
# treat it as untrusted output: never commit it, never publish it, keep it on the collection host.
MESSAGE_LIST_ROOT: Final[Path] = Path(__file__).resolve().parent / "artifacts" / "message_lists"


class StepSnapshotRecorder:
    """Capture a per-step ``git diff HEAD`` via an injected exec callable (observe-only)."""

    def __init__(self, exec_fn: Callable[[str], str], diff_command: str = DIFF_COMMAND) -> None:
        # exec_fn runs a shell command inside the agent's container and returns its stdout. For
        # mini-swe-agent this wraps ``env.execute(cmd)["output"]``.
        self._exec_fn = exec_fn
        self._diff_command = diff_command
        self.snapshots: dict[int, str] = {}

    def capture(self, step_index: int) -> None:
        """Record the current diff for *step_index*; any error is swallowed (never breaks a run)."""
        try:
            diff = self._exec_fn(self._diff_command)
        except Exception:  # noqa: BLE001 (observe-only: capture must never break a paid run)
            _LOG.exception("step snapshot capture failed at step %d", step_index)
            return
        self.snapshots[step_index] = diff


def snapshot_dir(trajectory_id: str, root: Path = SNAPSHOT_ROOT) -> Path:
    """The scratch directory holding one trajectory's per-step diffs."""
    return root / trajectory_id


def write_snapshots(
    trajectory_id: str, snapshots: dict[int, str], root: Path = SNAPSHOT_ROOT
) -> Path:
    """Persist per-step diffs as ``step_NNNN.diff`` under the trajectory's scratch dir."""
    out = snapshot_dir(trajectory_id, root)
    out.mkdir(parents=True, exist_ok=True)
    for step_index, diff in snapshots.items():
        (out / f"step_{step_index:04d}.diff").write_text(diff, encoding="utf-8")
    return out


def read_snapshots(trajectory_id: str, root: Path = SNAPSHOT_ROOT) -> dict[int, str]:
    """Load a trajectory's per-step diffs keyed by step index (empty if none captured)."""
    out = snapshot_dir(trajectory_id, root)
    if not out.is_dir():
        return {}
    snapshots: dict[int, str] = {}
    for path in out.glob(_STEP_GLOB):
        index = int(path.stem.removeprefix("step_"))
        snapshots[index] = path.read_text(encoding="utf-8")
    return snapshots


def message_list_path(trajectory_id: str, root: Path = MESSAGE_LIST_ROOT) -> Path:
    """The gitignored scratch path holding one trajectory's full agent message list."""
    return root / f"{trajectory_id}.json"


__all__ = [
    "DIFF_COMMAND",
    "MESSAGE_LIST_ROOT",
    "SNAPSHOT_ROOT",
    "StepSnapshotRecorder",
    "message_list_path",
    "read_snapshots",
    "snapshot_dir",
    "write_snapshots",
]
