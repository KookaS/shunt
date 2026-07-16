#!/usr/bin/env python3
"""Compute per-model cost projections from the challenges matrix."""

from __future__ import annotations

from pathlib import Path

from benchmark import config


def main(config_path: str = "benchmark/config.yaml"):
    config.load(config_path)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    matrix = config.load_matrix(matrix_path)

    models_order = config.enabled_models()
    if not models_order:
        # Fallback if none enabled
        models_order = list(matrix.get("models", {}).keys())

    results = matrix["results"]
    n_tasks = len(results)

    if n_tasks == 0:
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    totals = {
        m: {"cost": 0.0, "in_tok": 0, "out_tok": 0, "calls": 0, "pass": 0} for m in models_order
    }

    for _tid, task_results in results.items():
        for model in models_order:
            r = task_results.get(model, {})
            totals[model]["cost"] += r.get("cost", 0.0)
            totals[model]["in_tok"] += r.get("in_tok", 0)
            totals[model]["out_tok"] += r.get("out_tok", 0)
            totals[model]["calls"] += r.get("calls", 0)
            if r.get("pass", False):
                totals[model]["pass"] += 1

    # Cost projections are derived on demand and printed, not persisted as a
    # committed CSV (regenerable any time from results.csv).
    print()
    print(
        f"{'Model':<22} {'Cost(100)':<12} {'Cost(10)':<12} {'Cost(1000)':<12} "
        f"{'Avg$/task':<12} {'Avg tok/task':<14} {'Avg calls':<10} {'Pass%':<8}"
    )
    print("-" * 104)
    for m in models_order:
        t = totals[m]
        cost_100 = round(t["cost"], 4)
        cost_10 = round(t["cost"] / 10, 4)
        cost_1000 = round(t["cost"] * 10, 4)
        avg_cost = round(t["cost"] / n_tasks, 6)
        avg_tok = round((t["in_tok"] + t["out_tok"]) / n_tasks, 1)
        avg_calls = round(t["calls"] / n_tasks, 2)
        pass_pct = round(t["pass"] / n_tasks * 100, 1)
        print(
            f"{m:<22} ${cost_100:<9} ${cost_10:<9} ${cost_1000:<9} "
            f"${avg_cost:<9} {avg_tok:<14} {avg_calls:<10} {pass_pct:<7}%"
        )

    # Compare cheapest vs frontier
    pricing = config.enabled_pricing()
    if pricing:
        cheap = min(pricing, key=lambda m: config.cost_per_1m(m, pricing))
        frontier = max(pricing, key=lambda m: config.cost_per_1m(m, pricing))
        print()
        print(f"=== Savings: cheapest model ({cheap}) vs frontier ({frontier}) ===")
        cheap_cost = totals[cheap]["cost"]
        frontier_cost = totals[frontier]["cost"]
        for scale, label in [(n_tasks, "100 tasks"), (10, "10 tasks"), (1000, "1000 tasks")]:
            cc = cheap_cost / n_tasks * scale
            fc = frontier_cost / n_tasks * scale
            savings = fc - cc
            pct = (1 - cc / fc) * 100 if fc > 0 else 0
            print(
                f"  {label:>12}: frontier=${fc:<8.2f} cheap=${cc:<8.2f} "
                f"save=${savings:<8.2f} ({pct:.1f}%)"
            )

        print()
        print(f"=== Frontier-only cost (always {frontier}) ===")
        for scale, label in [(n_tasks, f"{n_tasks} tasks"), (10, "10 tasks"), (1000, "1000 tasks")]:
            fc = frontier_cost / n_tasks * scale
            print(f"  {label:>12}: ${fc:.2f}")


if __name__ == "__main__":
    main()
