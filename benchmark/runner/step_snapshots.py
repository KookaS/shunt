"""Per-step code snapshots for offline verified-outcome replay (Mechanism B)."""

# During a live agent run we capture a cheap per-step ``git diff`` of the agent's checkout
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

# The agent works in /testbed (a git checkout at the instance base_commit). A plain diff is
# non-mutating (observe-only): it never stages, so it cannot alter the agent's index or the
# patch it submits. New (untracked) files are not captured — a documented Mechanism-B limit.
TESTBED: Final[str] = "/testbed"
DIFF_COMMAND: Final[str] = f"git -C {TESTBED} diff"

# Local, gitignored scratch (mirrors benchmark/routing/artifacts/): snapshots are ephemeral
# session output, never committed data.
SNAPSHOT_ROOT: Final[Path] = Path(__file__).resolve().parent / "artifacts" / "step_snapshots"
_STEP_GLOB: Final[str] = "step_*.diff"


class StepSnapshotRecorder:
    """Capture a per-step ``git diff`` via an injected in-container exec callable (observe-only)."""

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


__all__ = [
    "DIFF_COMMAND",
    "SNAPSHOT_ROOT",
    "StepSnapshotRecorder",
    "read_snapshots",
    "snapshot_dir",
    "write_snapshots",
]
