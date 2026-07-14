#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random as rng
from pathlib import Path

from metrics import GAMMA, bootstrap_ci, compare_to_oracle, compute_metrics, compute_pareto
from strategies.fixed import AlwaysCheap, AlwaysFrontier, Random
from strategies.oracle import Oracle, OracleRewardAware

HERE = Path(__file__).resolve().parent
DEFAULT_MATRIX = HERE / "matrices" / "coderouterbench100.json"
RESULTS_FILE = HERE / "results.csv"

FIELDS = [
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
]


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text())


def evaluate(strategy, matrix: dict, tasks: list[str]) -> list[tuple[str, str, bool, float]]:
    decisions: list[tuple[str, str, bool, float]] = []
    for tid in tasks:
        task_meta = matrix["tasks"].get(tid, {})
        model = strategy.select(tid, task_meta, matrix)
        outcome = matrix["results"].get(tid, {}).get(model, {})
        passed = outcome.get("pass", False)
        cost = outcome.get("cost", 0.0)
        decisions.append((tid, model, passed, cost))
    return decisions


def get_strategies():
    return [
        Oracle(),
        OracleRewardAware(gamma=GAMMA),
        AlwaysCheap(),
        AlwaysFrontier(),
        Random(seed=42),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--gamma", type=float, default=GAMMA, help="cost weight")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap iterations for CI")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap reproducibility")
    args = ap.parse_args()

    matrix = load_matrix(Path(args.matrix))
    tasks = sorted(matrix["results"].keys())
    strategies = get_strategies()

    print(f"Evaluating {len(strategies)} strategies on {len(tasks)} tasks\n")

    if not tasks:
        print("  No tasks in matrix — nothing to evaluate.")
        return

    oracle_decisions = evaluate(OracleRewardAware(gamma=args.gamma), matrix, tasks)

    rng.seed(args.seed)

    all_metrics: list[dict] = []
    all_strategy_metrics: dict[str, dict] = {}

    for strategy in strategies:
        decisions = evaluate(strategy, matrix, tasks)
        metrics = compute_metrics(decisions, gamma=args.gamma)
        comparison = compare_to_oracle(decisions, oracle_decisions, gamma=args.gamma)
        avgperf_ci, cumreg_ci = bootstrap_ci(
            decisions, oracle_decisions, args.bootstrap, gamma=args.gamma
        )

        row = {
            "strategy": strategy.name,
            **metrics,
            **comparison,
            "AvgPerf_ci_lower": avgperf_ci[0],
            "AvgPerf_ci_upper": avgperf_ci[1],
            "CumReg_ci_lower": cumreg_ci[0],
            "CumReg_ci_upper": cumreg_ci[1],
        }
        all_metrics.append(row)
        all_strategy_metrics[strategy.name] = metrics

        ci_ap = f"[{avgperf_ci[0]:>5.2f},{avgperf_ci[1]:>5.2f}]"
        ci_cr = f"[{cumreg_ci[0]:>7.4f},{cumreg_ci[1]:>7.4f}]"
        print(
            f"  {strategy.name:25}  AvgPerf={metrics['AvgPerf%']:>5.2f}%  {ci_ap}  "
            f"TotalCost=${metrics['TotalCost']:<8.4f}  "
            f"CumReg={comparison['CumReg']:<8.4f}  {ci_cr}  "
            f"rAcc={comparison['rAcc']:<6.4f}"
        )

    pareto = compute_pareto(all_strategy_metrics)
    for r in all_metrics:
        r["Pareto"] = pareto.get(r["strategy"], False)

    all_metrics.sort(key=lambda r: r.get("Reward", 0), reverse=True)

    with RESULTS_FILE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in all_metrics:
            w.writerow({k: row.get(k, "") for k in FIELDS})

    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
