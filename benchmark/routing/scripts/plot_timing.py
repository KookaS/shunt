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

from benchmark import config, plot_frame  # noqa: E402
from benchmark.plot_frame import Annotations, FigureSpec  # noqa: E402
from benchmark.routing import censoring, plot_style, report, summary  # noqa: E402

_STRATEGY_BAR = "#607D8B"  # a single neutral slate — strategies are not model-hued

SPEC = FigureSpec(
    reading=(
        "Each bar is the mean number of agent<->model API round-trips needed to finish one "
        "task. Left: one bar per model, over every measured run on any arm. Right: one bar "
        "per routed strategy, counting only the cell that strategy actually picked. Error "
        "bars are the standard error of that mean; each bar is labelled with its mean and "
        "the number of tasks behind it. Expect models to differ several-fold and each routed "
        "strategy to land somewhere between the models it mixes."
    ),
    goal=(
        "For a router, look for FEWER calls at equal quality: a low right-hand bar sitting "
        "beside the taller model bars it routes to."
    ),
    definitions=(
        ("call", "one agent<->model round-trip, recorded as `calls` in results.csv"),
        ("SEM", "standard error of the mean — how precisely this bar's average is known"),
    ),
    notes=(
        "We do not record wall-clock latency; calls stand in as a coarse ordinal proxy.",
        "A strategy's coverage-gap picks (chosen model unmeasured on that task) are skipped in "
        "the right panel, and rows recording zero calls are dropped from the left panel — "
        "neither is ever counted as a real 0.",
    ),
    limitations=(
        "Calls are not seconds: a call's duration also depends on its token size, and real "
        "wall-clock time depends on host load and container startup as much as on the model. "
        "Read the bars as ordinal, never as measured latency.",
        "Each left-hand bar pools every reasoning arm that model ran, and the arm mixes differ "
        "sharply between models, so a cross-model comparison measures 'model x its arm mix', "
        "not the model alone.",
    ),
)


def _as_bool(value: object) -> bool:
    """Parse a CSV cell as a boolean — `bool("False")` is True, which is the whole trap."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_censored_row(row: dict[str, str]) -> bool:
    """True iff a raw results.csv row is a CENSORED stop, via the shared vocabulary.

    censoring.is_censored expects real booleans; a CSV hands it the STRINGS "True"/"False",
    both truthy, so it must be coerced first or every legacy row reads as solved.
    """
    return censoring.is_censored(
        {
            "pass": _as_bool(row.get("pass")),
            "timeout_flag": _as_bool(row.get("timeout_flag")),
            "stop_reason": row.get("stop_reason") or "",
        }
    )


def _model_calls(results_csv: Path) -> tuple[dict[str, list[int]], dict[str, int], int]:
    """``(per-model call counts, per-model dropped-row counts, censored rows kept)``."""
    # A row recording ZERO calls is a run that never happened (resource-limit stop before the
    # first round-trip): it carries no latency observation and must be dropped, not averaged
    # in as a genuine 0. The guard used to be `if not calls`, but `calls` is the STRING "0" —
    # truthy — so all 15 such rows survived as real zeros and pulled the frontier models'
    # means down by up to 8%, in the direction that flatters the router.
    per_model: dict[str, list[int]] = {}
    dropped: dict[str, int] = {}
    censored_kept = 0
    with results_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("calls")
            if raw is None or raw == "":
                continue
            calls = int(float(raw))
            censored = _is_censored_row(row)
            if calls == 0:
                dropped[row["model"]] = dropped.get(row["model"], 0) + 1
                continue
            censored_kept += 1 if censored else 0
            per_model.setdefault(row["model"], []).append(calls)
    return per_model, dropped, censored_kept


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


def _annotations(  # noqa: PLR0913
    models: list[str],
    model_calls: dict[str, list[int]],
    strat_calls: dict[str, list[int]],
    dropped: list[str],
    zero_call: dict[str, int],
    censored_kept: int,
) -> Annotations:
    """Footer content derived from the data: sample sizes, absent models, censoring."""
    sizes = [len(model_calls[m]) for m in models] + [len(v) for v in strat_calls.values()]
    notes: list[str] = []
    limits: list[str] = []
    if sizes:
        limits.append(
            f"Sample sizes per bar range {min(sizes)}-{max(sizes)} tasks; the smallest bars "
            "move a lot on one unusually chatty task"
        )
    if zero_call:
        detail = ", ".join(f"{m} {c}" for m, c in sorted(zero_call.items(), key=lambda kv: -kv[1]))
        notes.append(
            f"{sum(zero_call.values())} censored row(s) recorded ZERO calls — the run stopped "
            f"before its first round-trip — and are EXCLUDED rather than averaged in as real "
            f"zeros ({detail})"
        )
    if censored_kept:
        limits.append(
            f"{censored_kept} row(s) that DID make calls still stopped on a resource limit, so "
            "their call counts are truncated: those bars are lower bounds, not completed runs"
        )
    if dropped:
        notes.append(
            f"{len(dropped)} enabled model(s) have no recorded call counts and are absent "
            f"from the left panel: {', '.join(dropped)}"
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def plot_timing(results_csv: Path, matrix: dict, tasks: list[str], out_path: Path) -> Path:
    """Two panels: avg calls/task per model (left) and per routed strategy (right)."""
    model_calls, zero_call, censored_kept = _model_calls(results_csv)
    candidates = config.enabled_models() or sorted(model_calls)
    models = [m for m in candidates if m in model_calls]
    dropped = [m for m in candidates if m not in model_calls]
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
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return plot_frame.save(
        fig,
        out_path,
        SPEC,
        extra=_annotations(models, model_calls, strat_calls, dropped, zero_call, censored_kept),
    )


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
    # The right panel replays routed strategies over the matrix — analytical, so it
    # defaults to the VALID set (complete challenges, censored + incomplete excluded).
    # The left panel reads results.csv directly in _model_calls (every measured run, all
    # arms) — a per-model latency characterization that is deliberately left raw.
    matrix = summary.load_scored_matrix(matrix_path)
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
