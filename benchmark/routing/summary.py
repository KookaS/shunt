"""Canonical per-strategy benchmark summary derived from results.csv."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Final

from benchmark.routing.metrics import (
    bootstrap_ci,
    compare_to_oracle,
    compute_metrics,
    compute_pareto,
)
from benchmark.routing.strategies.oracle import OracleRewardAware

SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "strategy",
    "n_tasks",
    "n_pass",
    "AvgPerf%",
    "AvgPerf_ci_lower",
    "AvgPerf_ci_upper",
    "TotalCost",
    "AvgCost",
    "Reward",
    "CumReg",
    "CumReg_ci_lower",
    "CumReg_ci_upper",
    "rAcc",
    "Pareto",
)
# Columns that are deterministic given the matrix (no bootstrap, no strategy-set
# dependency) — what the drift check compares.
DETERMINISTIC_FIELDS: Final[tuple[str, ...]] = (
    "n_pass",
    "AvgPerf%",
    "TotalCost",
    "AvgCost",
    "Reward",
    "CumReg",
    "rAcc",
)

Decision = tuple[str, str, bool, float]


def evaluate(strategy: object, matrix: dict, tasks: list[str]) -> list[Decision]:
    """Run one strategy over the tasks, returning (task, model, passed, cost) tuples."""
    decisions: list[Decision] = []
    for tid in tasks:
        task_meta = matrix["tasks"].get(tid, {})
        model = strategy.select(tid, task_meta, matrix)  # type: ignore[attr-defined]
        outcome = matrix["results"].get(tid, {}).get(model, {})
        passed = outcome.get("pass", False)
        cascade_cost = getattr(strategy, "cascade_total_cost", None)
        cost = cascade_cost if cascade_cost is not None else outcome.get("cost", 0.0)
        decisions.append((tid, model, passed, cost))
    return decisions


def compute_strategy_rows(
    matrix: dict,
    tasks: list[str],
    strategies: list[object],
    gamma: float = 0.1,
    bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Evaluate every strategy against the matrix and return summary rows (Reward-sorted)."""
    if not tasks:
        return []
    oracle_decisions = evaluate(OracleRewardAware(gamma=gamma), matrix, tasks)
    random.seed(seed)
    rows: list[dict] = []
    strat_metrics: dict[str, dict] = {}
    for strategy in strategies:
        rows.append(_strategy_row(strategy, matrix, tasks, oracle_decisions, gamma, bootstrap))
        strat_metrics[rows[-1]["strategy"]] = {
            "AvgPerf%": rows[-1]["AvgPerf%"],
            "TotalCost": rows[-1]["TotalCost"],
        }
    pareto = compute_pareto(strat_metrics)
    for r in rows:
        r["Pareto"] = pareto.get(r["strategy"], False)
    rows.sort(key=lambda r: r.get("Reward", 0), reverse=True)
    return rows


def _strategy_row(
    strategy: object,
    matrix: dict,
    tasks: list[str],
    oracle_decisions: list[Decision],
    gamma: float,
    bootstrap: int,
) -> dict:
    decisions = evaluate(strategy, matrix, tasks)
    metrics = compute_metrics(decisions, gamma=gamma)
    comparison = compare_to_oracle(decisions, oracle_decisions, gamma=gamma)
    avgperf_ci, cumreg_ci = bootstrap_ci(decisions, oracle_decisions, bootstrap, gamma=gamma)
    return {
        "strategy": strategy.name,  # type: ignore[attr-defined]
        **metrics,
        **comparison,
        "AvgPerf_ci_lower": avgperf_ci[0],
        "AvgPerf_ci_upper": avgperf_ci[1],
        "CumReg_ci_lower": cumreg_ci[0],
        "CumReg_ci_upper": cumreg_ci[1],
    }


def write_summary_csv(rows: list[dict], path: Path) -> None:
    """Write summary rows to ``path`` using the canonical field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
