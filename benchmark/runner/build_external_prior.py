"""Reproducible producer for the external difficulty prior (external_swebench.csv)."""

from __future__ import annotations

# Regenerates ``routing/data/external_swebench.csv`` from a local clone of the public
# ``SWE-bench/experiments`` leaderboard (134 Verified submissions):
#     python -m benchmark.runner.build_external_prior /path/to/experiments-clone
# The clone is a gitignored artifact; this script + its committed output are the
# reproducible record. Deterministic (instances + submissions sorted → byte-identical).
#
# Denominator (differs from the first hand-made CSV): a submission counts for an
# instance iff it *attempted* it — the instance is in any eval bucket (resolved/
# applied/no_apply/test_errored/test_timeout/reset_failed/install_fail/with_logs/
# generated) and NOT in no_generation/no_logs; ``resolved`` is the numerator. The
# earlier CSV used the resolved∪no_generation universe, which DROPPED generated-but-
# failed attempts and inflated rates (astropy-12907: 0.99 → true 0.87).
#
# Tiers reuse select_swebench's substring cohorts; ~a third of submissions are agent-
# branded with no identifiable model and stay uncohorted (counted in the overall rate,
# excluded from p_cheap/p_frontier). Tiering is heuristic — reviewable, not ground truth.
import argparse
import csv
import json
from pathlib import Path
from typing import Final

from benchmark.runner.select_swebench import (
    DEFAULT_CHEAP_MARKERS,
    DEFAULT_FRONTIER_MARKERS,
    classify_submissions,
)

# Buckets that evidence an instance was actually run by a submission (any
# outcome, pass or fail). ``no_generation``/``no_logs`` are NOT attempts.
_EVAL_BUCKETS: Final[tuple[str, ...]] = (
    "resolved",
    "applied",
    "no_apply",
    "test_errored",
    "test_timeout",
    "reset_failed",
    "install_fail",
    "with_logs",
    "generated",
)
_NON_ATTEMPT: Final[tuple[str, ...]] = ("no_generation", "no_logs")

FIELDS: Final[tuple[str, ...]] = (
    "instance_id",
    "n_sub",
    "n_solved",
    "p_solve",
    "n_cheap",
    "cheap_solved",
    "p_cheap",
    "n_frontier",
    "frontier_solved",
    "p_frontier",
    "gap",
)

# instance_id -> {submission -> resolved(bool)} over *attempted* pairs only.
ResolveTable = dict[str, dict[str, bool]]


def _submission_attempts(results: dict[str, list[str]]) -> tuple[set[str], set[str]]:
    """(resolved, attempted) instance sets for one submission's results.json."""
    resolved = set(results.get("resolved", []))
    attempted: set[str] = set()
    for bucket in _EVAL_BUCKETS:
        attempted |= set(results.get(bucket, []))
    for bucket in _NON_ATTEMPT:
        attempted -= set(results.get(bucket, []))
    return resolved, attempted | resolved


def load_resolves(root: Path, subset: str = "verified") -> ResolveTable:
    """Read ``<root>/evaluation/<subset>/*/results/results.json`` into an
    attempted-only resolve table (True=resolved, False=attempted-but-failed).
    """
    base = root / "evaluation" / subset
    table: ResolveTable = {}
    for results_file in sorted(base.glob("*/results/results.json")):
        submission = results_file.parent.parent.name
        resolved, attempted = _submission_attempts(json.loads(results_file.read_text()))
        for iid in attempted:
            table.setdefault(iid, {})[submission] = iid in resolved
    return table


def _cohort_stats(row: dict[str, bool], cohort: set[str]) -> tuple[int, int, float | str]:
    """(n, solved, rate) over a tier cohort's attempts for one instance."""
    vals = [row[s] for s in row if s in cohort]
    n = len(vals)
    solved = sum(vals)
    return n, solved, (round(solved / n, 4) if n else "")


def _instance_row(iid: str, row: dict[str, bool], cheap: set[str], frontier: set[str]) -> dict:
    n_sub = len(row)
    n_solved = sum(row.values())
    nc, cs, pc = _cohort_stats(row, cheap)
    nf, fs, pf = _cohort_stats(row, frontier)
    gap = round(pf - pc, 4) if isinstance(pc, float) and isinstance(pf, float) else ""
    return {
        "instance_id": iid,
        "n_sub": n_sub,
        "n_solved": n_solved,
        "p_solve": round(n_solved / n_sub, 4) if n_sub else "",
        "n_cheap": nc,
        "cheap_solved": cs,
        "p_cheap": pc,
        "n_frontier": nf,
        "frontier_solved": fs,
        "p_frontier": pf,
        "gap": gap,
    }


def build_rows(table: ResolveTable) -> list[dict]:
    """Per-instance aggregate rows (sorted by instance_id) — deterministic."""
    cheap, frontier = classify_submissions(
        {s for row in table.values() for s in row},
        DEFAULT_CHEAP_MARKERS,
        DEFAULT_FRONTIER_MARKERS,
    )
    return [_instance_row(iid, table[iid], cheap, frontier) for iid in sorted(table)]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _default_out() -> Path:
    return Path(__file__).resolve().parent.parent / "routing" / "data" / "external_swebench.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate the external SWE-bench difficulty prior.")
    ap.add_argument("experiments_root", type=Path, help="Local clone of SWE-bench/experiments")
    ap.add_argument("--subset", default="verified", help="experiments subset dir")
    ap.add_argument(
        "--out", type=Path, default=None, help="Output CSV (default: data/external_swebench.csv)"
    )
    args = ap.parse_args()

    table = load_resolves(args.experiments_root, args.subset)
    rows = build_rows(table)
    out = args.out or _default_out()
    write_csv(rows, out)
    print(f"Wrote {len(rows)} instances -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
