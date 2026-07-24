"""Persist a REAL mini-swe-agent benchmark trajectory into the committable escalation plane.

Real-only: only a live run with real ``agent.messages`` writes anything (a classify-only run
passes empty messages and writes nothing) — a trajectory is never fabricated.
"""

from __future__ import annotations

import json
import logging
import re
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


__all__ = ["DATASET", "LIVE_DIR", "capture_live_trajectory", "make_trajectory_id"]
