#!/usr/bin/env python3
"""Derive the committed per-(task, judge) difficulty CSV and the label-quality gate."""

# WHY A SECOND COMMITTED DERIVED TABLE, AND WHY THE GATE. judge_difficulty.json is
# terra-only and predates the label expansion; raw judge responses stay gitignored
# (synthetic training-signal data). This script persists the ADDITIVE per-(task,
# judge) projection — every paid terra and sol label aggregated to one mean
# difficulty and measured cost per task — so the 500-task expansion is committed
# once and never re-paid, and it records the pre-registered label-quality gate
# next to that table: terra<->sol agreement, each judge's difficulty->measured-
# cheap-fail association, each judge's concordance with the HUMAN difficulty
# stratum (SWE-bench Verified's gold-standard annotation), and a re-derivation of
# derive_judge_difficulty.py's terra-vs-anchor adoption comparison. terra's JSON
# remains the knn_difficulty source; this CSV is additive, never a replacement,
# and nothing here scores a model or writes results.csv.

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

from scipy.stats import spearmanr

from benchmark import config
from benchmark.routing.scripts import derive_judge_difficulty as judge_difficulty
from benchmark.routing.scripts import judge_probe_metrics

JUDGES: Final[tuple[str, str]] = ("gpt-5.6-terra", "gpt-5.6-sol")
TERRA: Final[str] = "gpt-5.6-terra"
SOL: Final[str] = "gpt-5.6-sol"
ANCHOR: Final[str] = "claude-sonnet-5"
ROUND2: Final[int] = 2
EXPECTED_RUNS: Final[dict[str, int]] = {TERRA: 2, SOL: 1}
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "challenge_id",
    "judge_model",
    "difficulty",
    "judge_cost_usd",
    "n_runs",
)


def _stats(records: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """Per-(task, judge) mean difficulty, mean measured cost and run count (round-2)."""
    agg = judge_probe_metrics.aggregate(records)
    costs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in records:
        costs[(rec["task_id"], rec["judge"])].append(float(rec.get("raw_cost", 0.0)))
    return {
        (tid, judge): {
            "difficulty": vals["mean"],
            "judge_cost_usd": sum(costs[(tid, judge)]) / len(costs[(tid, judge)]),
            "n_runs": float(vals["n_runs"]),
        }
        for (tid, judge), vals in agg.items()
    }


def _csv_rows(stats: dict[tuple[str, str], dict[str, float]]) -> list[dict]:
    """One CSV row per (task, judge) present, matching judge_difficulty.json rounding."""
    rows = [
        {
            "challenge_id": tid,
            "judge_model": judge,
            "difficulty": round(vals["difficulty"], 3),
            "judge_cost_usd": round(vals["judge_cost_usd"], 6),
            "n_runs": int(vals["n_runs"]),
        }
        for (tid, judge), vals in stats.items()
        if judge in JUDGES
    ]
    return sorted(rows, key=lambda r: (r["challenge_id"], r["judge_model"]))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _task_sets(stats: dict[tuple[str, str], dict[str, float]]) -> dict[str, set[str]]:
    """Per judge, the task ids carrying at least one parsed round-2 record."""
    out: dict[str, set[str]] = {judge: set() for judge in JUDGES}
    for tid, judge in stats:
        if judge in out:
            out[judge].add(tid)
    return out


def _spearman_rho(a: list[float], b: list[float]) -> tuple[float | None, int]:
    """Spearman rho over paired (a, b); None when n<10 or the ranking is degenerate."""
    n = len(a)
    if n < 10:
        return None, n
    rho, _ = spearmanr(a, b)
    return (float(rho) if rho == rho else None), n


def _outcome_validity(
    stats: dict[tuple[str, str], dict[str, float]], cheap_pass: dict[str, int]
) -> dict[str, dict]:
    """Per judge, Spearman(difficulty, 1-cheap_pass) over tasks in the labels AND defer set."""
    out: dict[str, dict] = {}
    for judge in JUDGES:
        labels = {tid: stats[(tid, judge)]["difficulty"] for (tid, j) in stats if j == judge}
        shared = sorted(set(labels) & set(cheap_pass))
        rho, n = _spearman_rho([labels[t] for t in shared], [1.0 - cheap_pass[t] for t in shared])
        out[judge] = {"spearman_vs_cheap_fail": rho, "n": n}
    return out


# SWE-bench Verified's HUMAN difficulty stratum (easy/medium/hard), the gold-standard
# reference the judge labels are measured against: an ordinal 1..3 by increasing
# human-estimated fix time. It is an annotation of the benchmark instance, never an
# output of the judge pipeline.
_STRATUM: Final[dict[str, int]] = {"easy": 1, "medium": 2, "hard": 3}


def _human_concordance(
    stats: dict[tuple[str, str], dict[str, float]], tasks: dict
) -> dict[str, dict]:
    """Per judge, Spearman(difficulty, human stratum) over every task the judge labelled."""
    out: dict[str, dict] = {}
    for judge in JUDGES:
        pairs = [
            (stats[(tid, judge)]["difficulty"], float(_STRATUM[tasks[tid]["difficulty_stratum"]]))
            for (tid, j) in stats
            if j == judge and tid in tasks and tasks[tid].get("difficulty_stratum") in _STRATUM
        ]
        rho, n = _spearman_rho([p[0] for p in pairs], [p[1] for p in pairs])
        out[judge] = {"spearman_vs_human_stratum": rho, "n": n}
    return out


def _anchor_consistency(
    round2: list[dict], measured_tasks: list[str]
) -> dict[str, float | bool | None]:
    """Re-derive derive_judge_difficulty.py's terra-vs-anchor LOO R2 adoption comparison."""
    terra_loo, _ = judge_difficulty._loo_r2(TERRA, round2, measured_tasks)
    anchor_loo, _ = judge_difficulty._loo_r2(ANCHOR, round2, measured_tasks)
    if terra_loo is None or anchor_loo is None:
        return {
            "terra_r2_loo_vs_cheap_pass": terra_loo,
            "anchor_r2_loo_vs_cheap_pass": anchor_loo,
            "abs_diff": None,
            "within_0.01": False,
        }
    abs_diff = abs(terra_loo - anchor_loo)
    return {
        "terra_r2_loo_vs_cheap_pass": terra_loo,
        "anchor_r2_loo_vs_cheap_pass": anchor_loo,
        "abs_diff": abs_diff,
        "within_0.01": abs_diff <= judge_difficulty.ADOPTION_TOLERANCE,
    }


def _anomalous_run_counts(
    stats: dict[tuple[str, str], dict[str, float]],
) -> dict[str, dict[str, int]]:
    """(task, judge) pairs whose run count is outside the round-2 protocol expectation."""
    out: dict[str, dict[str, int]] = {}
    for (tid, judge), vals in stats.items():
        if judge in EXPECTED_RUNS and int(vals["n_runs"]) != EXPECTED_RUNS[judge]:
            out.setdefault(tid, {})[judge] = int(vals["n_runs"])
    return out


def _cheap_pass(path: Path) -> dict[str, int]:
    """defer_labels.csv rows as challenge_id -> cheap_pass."""
    with path.open(newline="") as f:
        return {row["challenge_id"]: int(row["cheap_pass"]) for row in csv.DictReader(f)}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(
        description="Derive benchmark/routing/data/judge_difficulty.csv + the label-quality gate"
    )
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to config YAML")
    ap.add_argument(
        "--probe",
        default="benchmark/routing/artifacts/judge_probe_*.jsonl",
        help="Glob of judge-probe JSONL artifacts to aggregate",
    )
    ap.add_argument(
        "--out",
        default="benchmark/routing/data/judge_difficulty.csv",
        help="Output path for the committed per-(task, judge) difficulty CSV",
    )
    ap.add_argument(
        "--quality-out",
        default="benchmark/routing/data/judge_label_quality.json",
        help="Output path for the committed label-quality gate JSON",
    )
    args = ap.parse_args()

    config.load(args.config)
    paths = judge_probe_metrics.glob_probe_artifacts(args.probe)
    if not paths:
        print(f"No probe artifacts matched {args.probe!r}. Run judge_probe.py first.")
        return 1
    records = judge_probe_metrics.load_records(paths)
    round2 = [r for r in records if r.get("prompt_version") == ROUND2]
    if not round2:
        print("No parsed round-2 (prompt_version == 2) judge records in the matched artifacts.")
        return 1

    stats = _stats(round2)
    tasks_by_judge = _task_sets(stats)
    rows = _csv_rows(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out, rows)

    cheap_pass = _cheap_pass(Path(args.out).parent / "defer_labels.csv")
    measured_tasks = sorted(config.load_matrix().get("results", {}).keys())
    indexed_tasks = config.load_challenges().get("tasks", {})

    shared = sorted(tasks_by_judge[TERRA] & tasks_by_judge[SOL])
    terra_vals = [stats[(t, TERRA)]["difficulty"] for t in shared]
    sol_vals = [stats[(t, SOL)]["difficulty"] for t in shared]
    terra_sol_rho, terra_sol_n = _spearman_rho(terra_vals, sol_vals)

    quality = {
        "judges": {judge: len(tasks_by_judge[judge]) for judge in JUDGES},
        "shared_terra_sol_tasks": len(shared),
        "terra_sol_spearman_rho": terra_sol_rho,
        "terra_sol_spearman_n": terra_sol_n,
        "outcome_validity": _outcome_validity(stats, cheap_pass),
        "human_stratum_concordance": _human_concordance(stats, indexed_tasks),
        "anchor_consistency": _anchor_consistency(round2, measured_tasks),
        "coverage": {
            "terra_no_sol": sorted(tasks_by_judge[TERRA] - tasks_by_judge[SOL]),
            "sol_no_terra": sorted(tasks_by_judge[SOL] - tasks_by_judge[TERRA]),
        },
        "anomalous_run_counts": _anomalous_run_counts(stats),
    }
    quality_out = Path(args.quality_out)
    _write_json(quality_out, quality)

    per_judge_rows = {judge: len(tasks_by_judge[judge]) for judge in JUDGES}
    rho_s = f"{terra_sol_rho:.4f}" if terra_sol_rho is not None else "n/a (n<10)"
    print(
        f"Wrote {out} ({len(rows)} rows; per-judge task counts {per_judge_rows}); "
        f"terra<->sol Spearman rho {rho_s} n={terra_sol_n}"
    )
    print(f"Wrote {quality_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
