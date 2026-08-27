#!/usr/bin/env python3
"""Derive the committed per-task judge-difficulty table from the gitignored probe artifacts."""

# WHY A COMMITTED DERIVED TABLE, AND NOT THE RAW JUDGE OUTPUT. The knn_difficulty
# strategies must score inside the committed figures, and the figure-certification
# machinery hashes every committed figure's data inputs. Raw judge responses stay
# gitignored (synthetic training-signal data, never benchmark measurements), but the
# small DERIVED projection they feed — one difficulty number and one measured judge
# cost per task — is committed so a clean checkout can reproduce every figure that
# plots a difficulty strategy. It is a projection, never a measurement: nothing here
# scores a model or writes results.csv.
#
# LABEL-SOURCE DECISION (pre-registered 2026-08-26). gpt-5.6-terra is the knn_difficulty
# label source over the claude-sonnet-5 anchor because its leave-one-out R^2 vs measured
# deepseek pass (+0.027) is within 0.01 of the anchor's (+0.029) at half the price
# ($1/$6 vs $2/$10) and from a different model family (decorrelation). The comparison is
# RE-COMPUTED and enforced here, not assumed: this script refuses to emit a table when
# terra's labels drift more than 0.01 from the anchor's on the available artifacts.

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing.scripts import judge_probe_metrics as judge_probe_metrics

JUDGE: Final[str] = "gpt-5.6-terra"
ANCHOR: Final[str] = "claude-sonnet-5"
ROUND2: Final[int] = 2
CHEAP_MODEL: Final[str] = "deepseek-v4-flash"
ADOPTION_TOLERANCE: Final[float] = 0.01


def _task_costs(records: list[dict]) -> dict[str, float]:
    """Per-task mean measured judge cost over its round-2 runs (one call at inference)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        groups[rec["task_id"]].append(float(rec.get("raw_cost", 0.0)))
    return {tid: sum(costs) / len(costs) for tid, costs in groups.items()}


def _loo_r2(judge: str, records: list[dict], tasks: list[str]) -> tuple[float, int]:
    """Leave-one-out R^2 of one judge's aggregated labels vs measured deepseek pass."""
    agg = judge_probe_metrics.aggregate(records)
    labels = {t: agg[(t, judge)]["mean"] for (t, j) in agg if j == judge}
    pass_by_task = judge_probe_metrics.measured_pass(tasks, CHEAP_MODEL)
    return judge_probe_metrics.label_r2_auc(labels, pass_by_task)["r2_loo"], len(labels)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(
        description="Derive benchmark/routing/data/judge_difficulty.json from probe artifacts"
    )
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to config YAML")
    ap.add_argument(
        "--probe",
        default="benchmark/routing/artifacts/judge_probe_*.jsonl",
        help="Glob of judge-probe JSONL artifacts to derive from",
    )
    ap.add_argument(
        "--out",
        default="benchmark/routing/data/judge_difficulty.json",
        help="Output path for the committed derived table",
    )
    args = ap.parse_args()

    config.load(args.config)
    paths = judge_probe_metrics.glob_probe_artifacts(args.probe)
    if not paths:
        print(f"No probe artifacts matched {args.probe!r}. Run judge_probe.py first.")
        return 1
    records = judge_probe_metrics.load_records(paths)
    round2 = [r for r in records if r.get("prompt_version") == ROUND2]
    terra = [r for r in round2 if r["judge"] == JUDGE]
    anchor = [r for r in round2 if r["judge"] == ANCHOR]
    if not terra or not anchor:
        print(
            f"Refusing to derive: need round-2 records for BOTH {JUDGE} and {ANCHOR} "
            f"(got terra={len(terra)}, anchor={len(anchor)}) — the adoption comparison "
            "cannot run without both."
        )
        return 1

    tasks = sorted(config.load_matrix().get("results", {}).keys())
    terra_loo, _n = _loo_r2(JUDGE, terra, tasks)
    anchor_loo, _m = _loo_r2(ANCHOR, anchor, tasks)
    if terra_loo is None or anchor_loo is None:
        print("Refusing to derive: could not compute both judges' LOO R^2 on the available data.")
        return 1
    gap = abs(terra_loo - anchor_loo)
    print(f"label-source check: {JUDGE} LOO R2 {terra_loo:+.3f} vs {ANCHOR} {anchor_loo:+.3f}")
    if gap > ADOPTION_TOLERANCE:
        print(
            f"REFUSE: {JUDGE} LOO R2 diverges from the anchor by {gap:.3f} > "
            f"{ADOPTION_TOLERANCE} — do NOT use it as the difficulty source. Keep "
            f"{ANCHOR}; investigate before re-deriving."
        )
        return 1

    agg = judge_probe_metrics.aggregate(terra)
    costs = _task_costs(terra)
    out_tasks = {
        tid: {
            "difficulty": round(agg[(tid, JUDGE)]["mean"], 3),
            "judge_cost_usd": round(costs[tid], 6),
        }
        for tid in sorted(costs)
    }
    payload = {
        "judge": JUDGE,
        "generated": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "source_artifacts": [str(p) for p in paths],
        "aggregation": (
            "difficulty = mean over round-2 runs; "
            "judge_cost_usd = mean raw_cost over that task's runs "
            "(one judge call per task at inference)"
        ),
        "adoption_rule": (
            f"{JUDGE} adopted because |LOO_R2_terra - LOO_R2_anchor| = {gap:.3f} <= "
            f"{ADOPTION_TOLERANCE} ({terra_loo:+.3f} vs {anchor_loo:+.3f}) at half the "
            "price and a different model family"
        ),
        "tasks": out_tasks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out} ({len(out_tasks)} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
