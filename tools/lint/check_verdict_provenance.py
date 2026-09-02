#!/usr/bin/env python3
"""SH016: a tracked kill-gate verdict artifact must agree with the corpus it summarises."""

# THE FAILURE THIS EXISTS FOR. `benchmark/runner/kill_gate_verdict.json` and
# `benchmark/runner/multidim_kill_gate_verdict.json` are TRACKED files that no hook, CI job or
# pipeline stage regenerates, and the suite only ever asserted their headline word. Editing
# `provisional_verdict` to PASS and fabricating a cost win in the axis table left the entire
# suite green — and a verdict artifact is precisely the file a reader quotes. Prose cannot stop
# that; recomputation can.
#
# EXACT SCOPE — read this before quoting a green exit as coverage, because the two artifacts are
# covered to DIFFERENT depths:
#   multidim_kill_gate_verdict.json — FULLY re-derived. The record is a pure function of the
#       committed inputs (`strategy_summary.csv` plus the offline gate's coverage census), so
#       every field is recomputed and diffed. Any hand edit fails.
#   kill_gate_verdict.json — its COVERAGE CENSUS only. That artifact's cost ratios and bootstrap
#       CIs come from a live matrix this gate cannot replay offline, so they are NOT checked
#       here. What IS checked is the block the multi-dimensional gate reads as a precondition:
#       `tripped`, `control_coverage` and `router_coverage` must follow from `n`, the floor and
#       the measured-cell counts. A hand-flipped `tripped` — one character that moves the other
#       gate UNTESTED -> FAIL — fails.
#
# Whole-tree gate (the SH011/SH015 shape): `pass_filenames: false` + `always_run: true`. The
# failure it catches is an edit to a JSON no staged Python file mentions, and it also fires when
# the CORPUS moves under an unchanged artifact — which is the stale-verdict half of the story.

from __future__ import annotations

import json
import sys
from pathlib import Path

_CODE = "SH016"
_ROOT = Path(__file__).resolve().parents[2]


def _findings() -> list[tuple[str, str]]:
    """(path, message) for every artifact field that disagrees with the committed corpus."""
    if str(_ROOT) not in sys.path:  # noqa: TID251 (banned-api read; tooling bootstrap)
        sys.path.insert(0, str(_ROOT))  # noqa: TID251, SH003 (benchmark is not installed)
    from benchmark.routing import gate_dimensions as gd  # noqa: PLC0415 (needs the bootstrap)

    summary = _ROOT / gd.SUMMARY_PATH
    offline = _ROOT / gd.OFFLINE_VERDICT_PATH
    multidim = _ROOT / gd.VERDICT_PATH
    out: list[tuple[str, str]] = []
    if offline.exists():
        try:
            payload = json.loads(offline.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            out.append((str(gd.OFFLINE_VERDICT_PATH), f"not readable JSON: {exc}"))
            payload = None
        if isinstance(payload, dict):
            out += [(str(gd.OFFLINE_VERDICT_PATH), p) for p in gd.coverage_census(payload).problems]
    out += [
        (str(gd.VERDICT_PATH), p) for p in gd.verdict_integrity_problems(multidim, summary, offline)
    ]
    return out


def main() -> int:
    """Report every disagreement; exit 1 when a tracked verdict does not match its corpus."""
    findings = _findings()
    for path, message in findings:
        print(f"{path}:0:0: [{_CODE} ERROR] {message}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
