#!/usr/bin/env python3
"""Latency-proxy plot: API calls per task, per model and per routed strategy."""

# We do not record wall-clock latency, so the number of agent<->model round-trips
# (``calls`` in results.csv) is used as a coarse proxy: more calls means, roughly,
# more turns and more latency. The figure states that caveat on-canvas so the bars
# are never misread as measured seconds.

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import config  # noqa: E402
from benchmark.routing import plot_style, report  # noqa: E402

_STRATEGY_BAR = "#607D8B"  # a single neutral slate — strategies are not model-hued


def _model_calls(results_csv: Path) -> dict[str, list[int]]:
    """{model: [calls, ...]} over EVERY measured row (all arms) in results.csv."""
    per_model: dict[str, list[int]] = {}
    with results_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            calls = row.get("calls")
            if not calls:
                continue
            per_model.setdefault(row["model"], []).append(int(float(calls)))
    return per_model


def _strategy_calls(matrix: dict, tasks: list[str], gamma: float) -> dict[str, list[int]]:
    """{strategy: [calls of the chosen cell, ...]} by replaying each derived strategy.

    Data-driven: the strategy set comes from report's factory builder, so a new or
    removed strategy shows up automatically on the next regeneration.
    """
    factories = report._build_strategy_factories(gamma)
    evaluated = report._evaluate_strategies(factories, matrix, tasks)
    results = matrix.get("results", {})
    out: dict[str, list[int]] = {}
    # _evaluate_strategies yields (decisions, unscorable) per strategy; skip the
    # coverage-gap picks (chosen model unmeasured on that task) rather than count them.
    for name, (picks, unscorable) in evaluated.items():
        calls: list[int] = []
        for tid, model, _passed, _cost in picks:
            if tid in unscorable:
                continue
            cell = results.get(tid, {}).get(model, {})
            c = cell.get("calls")
            if c:
                calls.append(int(float(c)))
        if calls:
            out[name] = calls
    return out


def _sem(values: list[int]) -> float:
    """Standard error of the mean (0 for n<=1) — the error bar on each bar."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _draw_bars(
    ax,  # noqa: ANN001 (matplotlib Axes)
    labels: list[str],
    series: dict[str, list[int]],
    colors: list[str],
    ylabel: str,
    title: str,
) -> None:
    means = [float(np.mean(series[k])) if series.get(k) else 0.0 for k in labels]
    errs = [_sem(series.get(k, [])) for k in labels]
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, edgecolor="white", yerr=errs, capsize=4, ecolor="#333333")
    for xi, m, e, k in zip(x, means, errs, labels, strict=True):
        ax.annotate(
            f"{m:.1f}\nn={len(series.get(k, []))}",
            xy=(xi, m + e),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    top = max([m + e for m, e in zip(means, errs, strict=True)], default=1.0)
    ax.set_ylim(0, top * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def plot_timing(results_csv: Path, matrix: dict, tasks: list[str], out_path: Path) -> Path:
    """Two panels: avg calls/task per model (left) and per routed strategy (right)."""
    models = config.enabled_models() or sorted(_model_calls(results_csv))
    model_calls = _model_calls(results_csv)
    models = [m for m in models if m in model_calls]
    color_map = plot_style.model_color_map(config.enabled_models() or models)

    strat_calls = _strategy_calls(matrix, tasks, config.gamma())
    strat_labels = list(strat_calls.keys())

    fig, (ax_m, ax_s) = plt.subplots(1, 2, figsize=(14, 6.5))
    _draw_bars(
        ax_m,
        models,
        model_calls,
        [color_map.get(m, "#9E9E9E") for m in models],
        "avg API calls per task",
        "Per model — every measured run (all arms)",
    )
    _draw_bars(
        ax_s,
        strat_labels,
        strat_calls,
        [_STRATEGY_BAR] * len(strat_labels),
        "avg API calls per task",
        "Per routed strategy — the cell each router picks",
    )

    fig.suptitle(
        "API calls per task — a coarse latency proxy (we do not record wall-clock time)",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Calls = agent<->model round-trips (results.csv `calls`); more calls means, roughly, "
        "more turns and more latency,\nbut a call's duration also depends on its token size — "
        "read this as ordinal, not seconds. For a router, FEWER calls at\nequal quality is better "
        "(lower latency). Error bars = standard error of the mean; n = measured tasks per bar.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    config.load(config_path)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output", default="benchmark/routing/reports/timing_comparison.png")
    args = ap.parse_args()
    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    matrix = config.load_matrix(matrix_path)
    if not matrix.get("results"):
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    tasks = report.derive_tasks(matrix, config.benchmark_params().get("seed", 42))
    results_csv = config.results_csv_path()
    out = plot_timing(results_csv, matrix, tasks, Path(args.output))
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
