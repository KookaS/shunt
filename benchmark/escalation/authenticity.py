"""Layer-1 authenticity for committed trajectories: a CONSISTENCY check, not a tamper detector."""

# Read the next paragraph before citing this module as evidence that data is genuine.
#
# WHAT IT DETECTS. Exactly one thing: that a committed file is internally inconsistent with
# itself or with the manifest. Concretely — a header hash that does not match the steps as
# written, a declared `n_steps` that disagrees with the payload, a `blocking` flag that
# contradicts `derive_blocking(success, is_infra_failure)`, a trajectory the manifest does not
# list, and a manifest entry whose hash or terminal label disagrees with the file's header. That
# catches CORRUPTION and CARELESS editing, which is what it was built for.
#
# WHAT IT DOES NOT DETECT, measured by executing each attack on copies of real committed data:
#   - flip a step's `success` AND rehash AND regenerate the manifest -> 1 error (the derived
#     `blocking`), and zero once the forger also fixes that field;
#   - rewrite EVERY step to success, consistently -> 0 errors;
#   - append 500 fabricated failing steps and rehash -> 0 errors;
#   - flip `header.terminal_resolved`, THE EVAL'S LABEL -> 0 errors before the manifest bound it
#     (see below), and 0 again for anyone who regenerates the manifest.
# A forger who keeps the invariants passes completely. This is a ceiling of the design, not a
# gap in the implementation: every value here is recomputed from the same file that declares it,
# so nothing in Layer-1 can testify that the file describes a run that actually happened.
# Signing (Layer-2, a key the collector holds) and sampled re-execution (Layer-3) are what would.
#
# THE LABEL IS BOUND TO THE MANIFEST, WITH ITS CEILING STATED. `terminal_resolved` is the y of
# the whole eval and was covered by nothing: it lives on the header, outside the step payload the
# content hash commits to. `manifest()` now records it alongside the hash, so a post-hoc label
# flip contradicts the manifest — and so a future signature over `manifest()` covers the label,
# which is the point of putting it there. It does NOT defend against a flip that also rewrites
# the manifest, and both writers on the collection path (`live_capture._write_manifest`,
# `runner.offline_replay.commit_trajectory`) regenerate the manifest wholesale from whatever is
# on disk. Until the manifest is signed, treat the label binding as tamper-EVIDENT for edits made
# outside those paths, never as tamper-PROOF. `commit_trajectory` rebuilds it in the SAME
# `corpus_lock` transaction that writes the trajectory: with the stamping stage running N
# workers, an unpaired rebuild binds a `content_sha256` a sibling worker's file no longer has.
#
# `recompute_dedup_key` IS CURRENTLY A NO-OP LEG, DELIBERATELY KEPT. `schema.normalize_dedup_key`
# is the identity function today (the live verifier already emits the normalized id), so that
# check can never fire. It is retained because it pins the parser contract the moment a real
# normalizer lands — but it must not be counted as coverage, which is why it is named here
# rather than left to look like a fourth safeguard.
#
# Reuses the routing Finding/severity primitives — never forks them.

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
    """A future-signable manifest: id -> {content_sha256, n_steps, terminal_resolved}. UNSIGNED."""
    entries: dict[str, dict[str, object]] = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        traj = schema.load_jsonl(path)
        entries[traj.header.trajectory_id] = {
            "content_sha256": traj.header.content_sha256,
            "n_steps": traj.header.n_steps,
            # The eval's label. It rides the header, which the content hash does NOT commit to,
            # so before it was recorded here nothing in the repo could notice it being flipped.
            "terminal_resolved": traj.header.terminal_resolved,
            # Mirrored (the header stays the source of truth) so the trajectories that can never
            # carry a verified per-step outcome are greppable in one committed file: any entry
            # with `snapshot_steps: 0` is unreplayable by construction, not merely un-replayed.
            "snapshot_steps": traj.header.snapshot_steps,
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
            continue
        if entry.get("content_sha256") != traj.header.content_sha256:
            out.append(Finding(ERROR, "manifest.hash_mismatch", tid, "manifest hash != header"))
        out.extend(_label_finding(entry, traj, tid))
    for tid in recorded.keys() - seen:
        out.append(Finding(WARN, "manifest.orphan", tid, "manifest lists a missing trajectory"))
    return out


def _label_finding(entry: dict[str, object], traj: Trajectory, tid: str) -> list[Finding]:
    """Cross-check the eval label against the manifest, or WARN that this manifest predates it."""
    # A manifest written before the label was bound carries no `terminal_resolved` key. That is a
    # COVERAGE GAP, not a pass: saying so out loud is the difference between "checked" and "not
    # checked but silent", and the silent version is what let the label go uncovered at all.
    if "terminal_resolved" not in entry:
        return [
            Finding(
                WARN,
                "manifest.unbound_label",
                tid,
                "manifest predates terminal_resolved binding; the eval label is UNVERIFIED here",
            )
        ]
    if entry.get("terminal_resolved") != traj.header.terminal_resolved:
        return [
            Finding(
                ERROR,
                "manifest.label_mismatch",
                tid,
                "manifest terminal_resolved != header — the eval's outcome label was changed",
            )
        ]
    return []


__all__ = [
    "MANIFEST_NAME",
    "errors",
    "manifest",
    "verify_manifest",
    "verify_trajectory",
    "warnings",
]
