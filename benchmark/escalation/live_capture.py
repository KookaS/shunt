"""Persist a REAL mini-swe-agent benchmark trajectory into the committable escalation plane.

Real-only: only a live run with real ``agent.messages`` writes anything (a classify-only run
passes empty messages and writes nothing) — a trajectory is never fabricated.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.escalation import authenticity, schema
from benchmark.escalation.normalize.mini_swe_agent import MiniSweAgentParser

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from shunt.verifiers.base import VerifierResult

_LOG = logging.getLogger(__name__)

DATASET = "swebench_verified"
# The committable plane's live-run subdir (LFS-tracked; secrets scrubbed on the write path).
LIVE_DIR = Path(__file__).resolve().parent / "data" / "live"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _UNSAFE.sub("_", text).strip("_") or "x"


def make_trajectory_id(instance_id: str, model: str, arm: str) -> str:
    """The stable id keying a trajectory's jsonl AND its per-step snapshot scratch dir."""
    return f"{_slug(instance_id)}__{_slug(model)}__{_slug(arm)}"


def capture_live_trajectory(
    messages: Sequence[dict[str, object]],
    *,
    instance_id: str,
    model: str,
    arm: str,
    resolved: bool,
    out_dir: Path = LIVE_DIR,
    step_outcomes: Mapping[int, VerifierResult] | None = None,
    snapshot_steps: int | None = None,
) -> Path | None:
    """Persist the agent's per-step trajectory + terminal outcome; None (no write) if not real."""
    if not messages:
        _LOG.info("escalation capture skipped for %s/%s: no agent messages", instance_id, model)
        return None
    trajectory_id = make_trajectory_id(instance_id, model, arm)
    meta = {
        "trajectory_id": trajectory_id,
        "dataset": DATASET,
        "plane": "committable",
        "framework": "mini_swe_agent",
        "instance_id": instance_id,
        "terminal_resolved": resolved,
        # How many per-step diffs this run captured. Committed here because the diffs are not:
        # without it, an offline rebuild cannot tell an unreplayable trajectory from a checkout
        # that merely lacks the gitignored scratch. Zero means "never replayable, anywhere".
        "snapshot_steps": snapshot_steps,
    }
    traj = MiniSweAgentParser().parse(list(messages), meta, step_outcomes)
    if not traj.steps:
        _LOG.info("escalation capture skipped for %s: no assistant turns to persist", instance_id)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{trajectory_id}.jsonl"
    schema.dump_jsonl(traj, path)  # scrubs free-text through redact_secrets on the way out
    _write_manifest(out_dir)
    _LOG.info("escalation trajectory captured: %s (%d steps)", path, traj.header.n_steps)
    return path


def _write_manifest(out_dir: Path) -> None:
    """Rebuild the Layer-1 manifest from every trajectory in the plane (idempotent)."""
    payload = json.dumps(authenticity.manifest(out_dir), indent=2, sort_keys=True)
    (out_dir / authenticity.MANIFEST_NAME).write_text(payload + "\n", encoding="utf-8")


def record_snapshot_provenance(
    out_dir: Path = LIVE_DIR, root: Path | None = None
) -> dict[str, int]:
    """Backfill `header.snapshot_steps` from the scratch — RUN ON THE COLLECTION HOST ONLY."""
    # For a corpus captured before `snapshot_steps` existed. The count is read off the local,
    # gitignored scratch, so on any other host every count comes back 0 — and 0 is the one value
    # that authorises the offline replay to CLEAR a trajectory. Running this on a fresh clone
    # would therefore rewrite all 799 headers to 0 and hand the next rebuild a licence to wipe the
    # corpus. "Collection host only" has to be a wall, not a docstring, so the wrong-host
    # SIGNATURE is what is checked: no scratch root at all, or a scratch that holds diffs for not
    # one trajectory. A downgrade check alone cannot see this, because a not-yet-migrated header
    # reads `None` — which is exactly the corpus this command exists to migrate.
    #
    # Only the header changes: `dump_jsonl` recomputes `content_sha256` from the steps, which are
    # untouched, so a rewritten trajectory keeps its hash and the manifest keeps its entry. Every
    # count is collected BEFORE anything is written, so a refusal leaves the corpus untouched
    # rather than half-migrated.
    from benchmark.runner.step_snapshots import SNAPSHOT_ROOT, read_snapshots  # noqa: PLC0415

    scratch = SNAPSHOT_ROOT if root is None else root
    if not scratch.is_dir():
        raise RuntimeError(
            f"no per-step snapshot scratch at {scratch}: this is not the collection host. "
            "Recording 0 for every trajectory would tell the offline replay to clear the corpus."
        )
    counts: dict[str, int] = {}
    pending: list[tuple[Path, schema.Trajectory, int]] = []
    for path in sorted(out_dir.glob("*.jsonl")):
        traj = schema.load_jsonl(path)
        tid = traj.header.trajectory_id
        captured = len(read_snapshots(tid, root=scratch))
        if captured == 0 and traj.header.snapshot_steps:
            raise RuntimeError(
                f"{tid} already records snapshot_steps={traj.header.snapshot_steps} but this host "
                "has none of its per-step diffs. Recording 0 here would tell the offline replay to "
                "clear real stamps. Run this on the collection host, or restore the scratch."
            )
        counts[tid] = captured
        pending.append((path, traj, captured))
    if counts and not any(counts.values()):
        raise RuntimeError(
            f"{scratch} holds per-step diffs for none of the {len(counts)} trajectories in "
            f"{out_dir}. That is the wrong-host signature, not a corpus that captured nothing; "
            "recording 0 for all of them would authorise the next rebuild to clear every one."
        )
    for path, traj, captured in pending:
        header = replace(traj.header, snapshot_steps=captured)
        schema.dump_jsonl(schema.Trajectory(header=header, steps=traj.steps), path)
    _write_manifest(out_dir)
    return counts


def _main() -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=record_snapshot_provenance.__doc__)
    ap.add_argument("--out-dir", type=Path, default=LIVE_DIR, help="the committed trajectory plane")
    ap.add_argument("--root", type=Path, default=None, help="per-step snapshot scratch to read")
    args = ap.parse_args()
    counts = record_snapshot_provenance(args.out_dir, args.root)
    unreplayable = sorted(tid for tid, n in counts.items() if n == 0)
    print(f"recorded snapshot provenance for {len(counts)} trajectories")
    print(f"{len(unreplayable)} captured NO per-step snapshots and can never be replayed:")
    for tid in unreplayable:
        print(f"  {tid}")
    return 0


__all__ = [
    "DATASET",
    "LIVE_DIR",
    "capture_live_trajectory",
    "make_trajectory_id",
    "record_snapshot_provenance",
]


if __name__ == "__main__":
    raise SystemExit(_main())
