"""Layer-1 authenticity for committed trajectories: recompute every derivable field and"""

# cross-check the per-trajectory SHA-256 against a manifest, so corruption and naive
# fabrication fail CI. Reuses the routing Finding/severity primitives — never forks them.
#
# Layer-1 cannot catch a forger who reproduces every invariant; signing (Layer-2) and sampled
# re-execution (Layer-3) are future work. `manifest()` returns a plain structure a future
# signer can sign without touching this recompute logic.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmark.escalation import schema
from benchmark.routing.authenticity import ERROR, WARN, Finding, errors, warnings

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.escalation.schema import Trajectory

MANIFEST_NAME = "manifest.json"


def verify_trajectory(traj: Trajectory) -> list[Finding]:
    """Recompute the content hash and every derivable field for one trajectory."""
    out: list[Finding] = []
    tid = traj.header.trajectory_id
    recomputed = schema.content_sha256(traj.steps)
    if recomputed != traj.header.content_sha256:
        out.append(
            Finding(ERROR, "content_sha256.mismatch", tid, "header hash != recomputed payload hash")
        )
    if traj.header.n_steps != len(traj.steps):
        out.append(
            Finding(ERROR, "n_steps.mismatch", tid, f"{traj.header.n_steps} != {len(traj.steps)}")
        )
    for step in traj.steps:
        key = f"{tid}#{step.step_index}"
        if schema.recompute_dedup_key(step) != step.dedup_key:
            out.append(Finding(ERROR, "dedup_key.mismatch", key, "dedup_key != failing_check_id"))
        if schema.recompute_blocking(step) != step.blocking:
            out.append(Finding(ERROR, "blocking.mismatch", key, "blocking != (¬success ∧ ¬infra)"))
    return out


def manifest(data_dir: Path) -> dict[str, object]:
    """A future-signable manifest: trajectory_id -> {content_sha256, n_steps}. NOT signed."""
    entries: dict[str, dict[str, object]] = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        traj = schema.load_jsonl(path)
        entries[traj.header.trajectory_id] = {
            "content_sha256": traj.header.content_sha256,
            "n_steps": traj.header.n_steps,
        }
    return {"schema_version": schema.SCHEMA_VERSION, "trajectories": entries}


def verify_manifest(data_dir: Path) -> list[Finding]:
    """Cross-check every committed trajectory against the on-disk manifest.json."""
    out: list[Finding] = []
    manifest_path = data_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return [Finding(ERROR, "manifest.missing", str(data_dir), f"no {MANIFEST_NAME}")]
    recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("trajectories", {})
    seen: set[str] = set()
    for path in sorted(data_dir.glob("*.jsonl")):
        traj = schema.load_jsonl(path)
        tid = traj.header.trajectory_id
        seen.add(tid)
        out.extend(verify_trajectory(traj))
        entry = recorded.get(tid)
        if entry is None:
            out.append(Finding(ERROR, "manifest.unlisted", tid, "trajectory absent from manifest"))
        elif entry.get("content_sha256") != traj.header.content_sha256:
            out.append(Finding(ERROR, "manifest.hash_mismatch", tid, "manifest hash != header"))
    for tid in recorded.keys() - seen:
        out.append(Finding(WARN, "manifest.orphan", tid, "manifest lists a missing trajectory"))
    return out


__all__ = [
    "MANIFEST_NAME",
    "errors",
    "manifest",
    "verify_manifest",
    "verify_trajectory",
    "warnings",
]
