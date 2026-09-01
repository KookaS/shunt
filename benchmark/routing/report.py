#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from benchmark import config
from benchmark.plot_frame import Annotations
from benchmark.routing import (
    censoring,
    figures,
    impute,
    metrics,
    plot_style,
    repricing,
    selection_guard,
    summary,
)
from benchmark.routing.figures import arm_manipulation as fig_arms
from benchmark.routing.figures import cache_economics as fig_cache
from benchmark.routing.figures import complementarity as fig_complementarity
from benchmark.routing.figures import cost_quality_frontier as fig_frontier
from benchmark.routing.figures import decision_audit as fig_audit
from benchmark.routing.figures import evidence_basis as fig_evidence
from benchmark.routing.figures import kill_gate as fig_kill_gate
from benchmark.routing.figures import ladder_rungs as fig_ladder
from benchmark.routing.figures import live_gap as fig_live_gap
from benchmark.routing.figures import model_grid as fig_model_grid
from benchmark.routing.figures import oracle_gap as fig_oracle
from benchmark.routing.figures import pareto_dimensions as fig_pareto_dims
from benchmark.routing.figures import task_difficulty as fig_difficulty
from benchmark.routing.impute import ImputedMatrix
from benchmark.routing.metrics import _reward, compute_cost_decomposition
from benchmark.routing.plot_style import RawResults, row_real_cost, usd
from benchmark.routing.strategies import Strategy
from benchmark.routing.strategies.oracle import OracleRewardAware

# A (model, arm) cell below this many tasks is provisional HERE — stricter than
# plot_style's shared floor of 10, because at 10 a single cell was silently
# defining the arm frontier and moving AIQ by 0.03.
MIN_N_RELIABLE: Final[int] = 30


# ---------------------------------------------------------------------------
# Progress + memory diagnostics. This report holds the kNN family's ONNX embedders
# and their per-task index in RSS at once and has peaked near 5 GB; an OOM kill is
# SIGKILL, so nothing it was about to print survives. Every step therefore reports
# as it completes, on a line-buffered stdout, with the peak RSS so far — the last
# line printed names the step that was running when the kernel stepped in.
# ---------------------------------------------------------------------------

_OOM_HEADROOM_MB: Final[float] = 6000.0


def _peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _available_mb() -> float | None:
    """MemAvailable from /proc, or None where the kernel does not publish it."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _memory_preflight() -> None:
    """Warn BEFORE the embedders load when this host cannot hold the report."""
    available = _available_mb()
    if available is None:
        return
    print(f"Memory: {available:,.0f} MB available; this report has peaked near 5,000 MB.")
    if available < _OOM_HEADROOM_MB:
        print(
            f"WARNING: under {_OOM_HEADROOM_MB:,.0f} MB free — the kernel may OOM-kill this "
            "run (exit 137) mid-figure. Free memory or run it alone.",
            file=sys.stderr,
        )


def _step(label: str, value: object) -> None:
    """One completed step, with the peak RSS reached by the time it finished."""
    print(f"  {label:13}: {value}   [peak RSS {_peak_rss_mb():,.0f} MB]")


def _const_factory(strategy: Strategy) -> Callable[[], Strategy]:
    """Wrap an already-built strategy instance as a 0-arg factory — re-use, not
    rebuild, so a kNN strategy's embeddings/index aren't recomputed per plot.
    """
    return lambda: strategy


def _build_strategy_factories(gamma: float) -> dict[str, Callable[[], Strategy]]:
    """One source of truth for the regret plot's strategy set."""
    # The SAME config-enabled list run_eval.get_strategies reads for every
    # other plot (a hardcoded set here previously added Oracle-reward+Random
    # but omitted an enabled headline strategy — silently absent from the
    # regret plot while still shown everywhere else).
    # Oracle-reward is added unconditionally: the regret plot's internal
    # reference baseline every other strategy's regret is measured against,
    # independent of whether it is itself config-enabled for display.
    from benchmark.routing import run_eval

    factories: dict[str, Callable[[], Strategy]] = {
        s.name: _const_factory(s) for s in run_eval.get_strategies()
    }
    factories.setdefault("Oracle-reward", lambda: OracleRewardAware(gamma=gamma))
    return factories


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_matrix(path: Path) -> dict | None:
    try:
        return config.load_matrix(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _validate_rows(results: list[dict]) -> str | None:
    """Reason the row set cannot be plotted, or ``None`` if it is usable.

    Plotting reads TotalCost/AvgPerf% unguarded; a thin or malformed matrix must
    fail here with a diagnosis rather than half-write artifacts and KeyError later.
    """
    for row in results:
        for field in ("TotalCost", "AvgPerf%"):
            try:
                float(row[field])
            except (KeyError, TypeError, ValueError):
                return f"strategy {row.get('strategy', '?')!r} has no usable {field}"
    if not any(int(float(r.get("n_tasks", 0) or 0)) > 0 for r in results):
        return (
            "no strategy has a single scorable task — every chosen cell is a "
            "coverage gap. Run the matrix before reporting."
        )
    return None


def print_summary(results: list[dict[str, str]]) -> None:
    """Print the strategy ranking to stdout. Rows are derived in-memory from the
    results.csv source of truth (no committed derived CSV).
    """
    rows = sorted(results, key=lambda r: float(r.get("Reward", 0)), reverse=True)
    # Both cost models are columns, never one standing in for the other, and `subset_selected`
    # rides beside them so a coverage-selected row cannot be read as a full-sample row.
    cols = [
        "strategy",
        "n_pass",
        "AvgPerf%",
        "TotalCost",
        "TotalCost_cacheaware",
        "AvgCost",
        "Reward",
        "Pareto",
        "subset_selected",
    ]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))
    for line in selection_guard.rows_footer(rows):
        print(line)


# ---------------------------------------------------------------------------
# Model palette + arm ranks — computed once, shared by every arm-aware plot.
# ---------------------------------------------------------------------------


def _model_colors() -> dict[str, str]:
    """Okabe-Ito hue per model, fixed by price order (cheapest -> priciest)."""
    try:
        order = config.enabled_models()
    except Exception:  # noqa: BLE001 (registry optional at plot time)
        order = []
    return plot_style.model_color_map(order)


def _arm_ranks() -> dict[tuple[str, str], int]:
    """(model, arm_id) -> within-model rank, from the registry's reasoning bracket."""
    ranks: dict[tuple[str, str], int] = {}
    try:
        resolved = config.resolved_models()
    except Exception:  # noqa: BLE001 (registry optional at plot time)
        return ranks
    for model, mc in resolved.items():
        if mc.reasoning is None:
            continue
        for arm in mc.reasoning.arms:
            ranks[(model, arm.id)] = arm.rank
    return ranks


# ---------------------------------------------------------------------------
# Footer content. Static construction facts live in the module-level FigureSpec
# constants; anything that depends on the DATA (counts, coverage, imputation,
# whether arm variation exists at all) is computed per run and merged in as
# Annotations, so a caveat can never go stale when the matrix grows.
# ---------------------------------------------------------------------------

_CI_NOTE = "Error bars and bands are 95% Wilson binomial CIs on the pass rate."


def _merge_annotations(*parts: Annotations) -> Annotations:
    """Concatenate several Annotations (plot_frame drops duplicates on merge)."""
    return Annotations(
        definitions=tuple(d for p in parts for d in p.definitions),
        notes=tuple(n for p in parts for n in p.notes),
        limitations=tuple(lim for p in parts for lim in p.limitations),
    )


def _row_coverage_annotations(ns: list[int]) -> Annotations:
    """Whether the plotted strategy rows really share one task denominator."""
    scored = [n for n in ns if n > 0]
    empty = sum(1 for n in ns if n <= 0)
    notes: list[str] = []
    limits: list[str] = []
    if scored and len(set(scored)) == 1:
        # Only the SCORED rows were checked — saying "every plotted strategy" while
        # zero-n rows sit on the canvas contradicts the no-evidence limitation below.
        subject = (
            f"Every scored strategy ({len(scored)} of {len(ns)} plotted)"
            if empty
            else "Every plotted strategy"
        )
        notes.append(f"{subject} is scored on the same {scored[0]} task(s).")
    elif scored:
        limits.append(
            f"Strategies are scored on uneven task counts (n={min(scored)}-{max(scored)}), "
            "so their totals are not strictly comparable."
        )
    if empty:
        limits.append(
            f"{empty} strategy row(s) have no scorable task and sit at \\$0 / 0% — "
            "that is absence of evidence, not efficiency."
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _single_arm_limits(raw: RawResults | None, synthesized: bool = False) -> tuple[str, ...]:
    """The 'only the default arm has data' caveat, when that is true of this cache.

    Silent on a SYNTHESIZED cache: collapsing every model to one "default" column is
    an artifact of the missing per-arm cache, not a fact about the sampled arms.
    """
    if synthesized or raw is None or not raw or not plot_style.is_single_arm(raw):
        return ()
    return (
        "Every model has exactly one sampled arm on this data (arm sweep pending), "
        "so nothing here shows reasoning-effort variation.",
    )


def _banner_annotations(banner: str | None) -> Annotations:
    """The equal-coverage/imputation disclosure, as a footer note instead of a caption."""
    return Annotations(notes=(banner,) if banner else ())


def _provisional_limits(points: list[dict], what: str) -> tuple[str, ...]:
    """How many plotted (model, arm) cells rest on fewer than MIN_N_PROVISIONAL tasks."""
    if not points:
        return ()
    thin = sum(1 for p in points if p["provisional"])
    if not thin:
        return ()
    return (
        f"{thin} of {len(points)} {what} rest on fewer than "
        f"{MIN_N_RELIABLE} tasks (drawn hollow) — their pass rate is provisional.",
    )


def _evaluate_strategies(
    factory_map: dict[str, Callable[[], Strategy]],
    matrix: dict,
    tasks: list[str],
) -> dict[str, tuple[list[tuple[str, str, bool, float]], set[str]]]:
    """Per strategy: ``(decisions, unscorable)`` from ``summary.evaluate``."""
    # Delegates rather than re-implementing. The private copy that used to live here
    # dropped a cascade's failed-probe cost and skipped the censoring check, so the
    # regret plot charged the within-task kNN cascade $3.09 less than strategy_summary.csv did and
    # printed a headline regret the CSV contradicted.
    return {
        name: summary.evaluate(factory(), matrix, tasks) for name, factory in factory_map.items()
    }


def _compute_per_task_regret(
    strategy_decisions: list[tuple[str, str, bool, float]],
    oracle_decisions: list[tuple[str, str, bool, float]],
    gamma: float = 0.1,
    excluded: set[str] | None = None,
) -> np.ndarray:
    """Cumulative regret over scorable tasks only; a task in ``excluded`` (a
    coverage gap) is DROPPED, never scored fail@$0. Filtering both series on the
    same task id keeps strategy and oracle position-aligned."""
    excluded = excluded or set()
    regrets: list[float] = []
    for sd, od in zip(strategy_decisions, oracle_decisions, strict=True):
        if sd[0] in excluded:
            continue
        sr = _reward(sd[2], sd[3], gamma)
        or_ = _reward(od[2], od[3], gamma)
        regrets.append(or_ - sr)
    return np.cumsum(regrets)


def _arm_oracle_decisions(
    raw: RawResults, tasks: list[str], gamma: float
) -> list[tuple[str, str, bool, float]]:
    """Best REALIZED (model, arm) per task among whichever arms were actually
    sampled for it — the ceiling given the sparse p(arm|model) coverage,
    not a full-bracket oracle.
    """
    decisions: list[tuple[str, str, bool, float]] = []
    for tid in tasks:
        per_model = raw.get(tid, {})
        best_key, best_reward, best_outcome = "", -math.inf, (False, 0.0)
        for model, per_arm in per_model.items():
            for arm, row in per_arm.items():
                passed, cost = bool(row.get("pass")), row_real_cost(row)
                r = _reward(passed, cost, gamma)
                if r > best_reward:
                    best_reward, best_key, best_outcome = r, f"{model}:{arm}", (passed, cost)
        decisions.append((tid, best_key, best_outcome[0], best_outcome[1]))
    return decisions


def _arm_bandit_score(
    opt: tuple[str, str, dict], history: dict[tuple[str, str], list[float]]
) -> float:
    hist = history.get((opt[0], opt[1]))
    return (sum(hist) / len(hist)) if hist else math.inf


def _arm_bandit_decisions(
    raw: RawResults, tasks: list[str], gamma: float
) -> list[tuple[str, str, bool, float]]:
    """Illustrative optimistic-greedy bandit over sampled (model, arm) options.
    Single consumer (this plot) — not a production Strategy (the arm sampler
    itself is likewise kept inline for the same one-consumer reason).
    """
    history: dict[tuple[str, str], list[float]] = {}
    decisions: list[tuple[str, str, bool, float]] = []
    for tid in tasks:
        per_model = raw.get(tid, {})
        options = [(m, a, row) for m, per_arm in per_model.items() for a, row in per_arm.items()]
        if not options:
            decisions.append((tid, "", False, 0.0))
            continue
        model, arm, row = max(options, key=lambda opt: _arm_bandit_score(opt, history))
        passed, cost = bool(row.get("pass")), row_real_cost(row)
        history.setdefault((model, arm), []).append(_reward(passed, cost, gamma))
        decisions.append((tid, f"{model}:{arm}", passed, cost))
    return decisions


# ---------------------------------------------------------------------------
# Measured vs imputed accounting. 409 of 1062 analytical cells are monotone-imputed
# and imputation is near-exclusively pass-filling, so "how much of this number is
# evidence" is a first-class question these figures must be able to answer.
# ---------------------------------------------------------------------------

# tid -> (passed, cost, imputed-anywhere-on-the-billed-path)
StrategyCells = dict[str, tuple[bool, float, bool]]


class _PathRecorder:
    """Wraps a strategy so ``summary.evaluate`` also yields each task's billed path."""

    # A cascade's reported cost sums every model it tried, but only the model it RETURNED
    # is visible in the decision tuple. Reading the imputed flag off that final cell alone
    # let projected dollars enter a panel labelled "no imputed cell on either side".
    # Recording the path during the one evaluate pass keeps that single-producer property
    # (no second, re-embedding evaluation) while making the flag cover what was billed.

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.paths: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return str(self._inner.name)  # type: ignore[attr-defined]

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        model: str = self._inner.select(task_id, task_meta, matrix)  # type: ignore[attr-defined]
        tried = getattr(self._inner, "cascade_tried_models", None)
        self.paths[task_id] = list(tried) if tried else [model]
        return model

    def __getattr__(self, item: str) -> object:
        # Forwards `cascade_total_cost` (set per select) to evaluate; absent on
        # single-shot strategies, where evaluate's getattr default takes over.
        return getattr(self._inner, item)


def _repriced_total(
    attempts: dict[str, list[summary.Attempt]], scored: Iterable[str]
) -> float | None:
    """The scored tasks' billed attempts at TODAY's cheapest listed price, or None."""
    # All-or-nothing over the SAME task set the recorded total covers, which is what makes the
    # two interchangeable on one axis. It goes None on two conditions and never fudges either:
    # an attempt on a model the price sheet does not price, and an attempt with no token counts
    # (an IMPUTED cell carries none -- `BilledAttempt.in_tok` is 0 there, and repricing that as
    # $0.00 would publish "this projected rung is free"). A partial total would be a smaller
    # number wearing the same axis label.
    rows = [
        {"model": a.model, "in_tok": a.in_tok, "out_tok": a.out_tok}
        for tid in scored
        for a in attempts.get(tid, [])
    ]
    if any(a["in_tok"] == 0 and a["out_tok"] == 0 for a in rows):
        return None
    return repricing.total_naive_cost(rows, "report.strategy_cells")


def strategy_cells(
    matrix: dict, tasks: list[str], strategies: Iterable[object]
) -> tuple[dict[str, tuple[StrategyCells, set[str]]], dict[str, float | None]]:
    """Per strategy: (pass, cost, imputed) per task, the unscorable set, and a repriced total."""
    # Cost is read from `real_cost` for a single-shot pick; a cascade keeps
    # summary.evaluate's cascade total so these numbers reconcile with
    # strategy_summary.csv rather than quietly forming a second accounting path.
    # `imputed` is PATH-AWARE: true when ANY cell the decision billed was projected.
    #
    # `evaluate_billed` rather than `evaluate` -- the SAME single evaluation pass, but it also
    # returns each task's billed attempts with their tokens, which is what a repriced total
    # needs and what the (pass, cost, imputed) triple drops. Nothing about the recorded cost
    # changes; the repriced total rides beside it (see `_repriced_total`).
    out: dict[str, tuple[StrategyCells, set[str]]] = {}
    repriced: dict[str, float | None] = {}
    for strategy in strategies:
        recorder = _PathRecorder(strategy)
        decisions, unscorable, attempts, _judge, _sessions = summary.evaluate_billed(
            recorder, matrix, tasks
        )
        is_cascade = getattr(strategy, "cascade_total_cost", None) is not None
        cells: StrategyCells = {}
        for tid, model, passed, cost in decisions:
            per_task = matrix.get("results", {}).get(tid, {})
            spend = float(cost) if is_cascade else row_real_cost(per_task.get(model, {}))
            path = recorder.paths.get(tid) or [model]
            imputed = any(bool(per_task.get(m, {}).get("imputed", False)) for m in path)
            cells[tid] = (bool(passed), spend, imputed)
        name = strategy.name  # type: ignore[attr-defined]
        out[name] = (cells, unscorable)
        repriced[name] = _repriced_total(attempts, (t for t in cells if t not in unscorable))
    return out, repriced


def _split_measured(cells: StrategyCells, unscorable: set[str]) -> dict[str, float]:
    """Measured vs imputed dollars and passes for one strategy's scored selections."""
    scored = [(p, c, i) for tid, (p, c, i) in cells.items() if tid not in unscorable]
    return {
        "measured_cost": sum(c for _p, c, i in scored if not i),
        "imputed_cost": sum(c for _p, c, i in scored if i),
        "measured_cells": float(sum(1 for _p, _c, i in scored if not i)),
        "imputed_cells": float(sum(1 for _p, _c, i in scored if i)),
        "measured_pass": float(sum(1 for p, _c, i in scored if p and not i)),
        "imputed_pass": float(sum(1 for p, _c, i in scored if p and i)),
    }


def paired_measured(
    router: str, baseline: str, by_strategy: dict[str, tuple[StrategyCells, set[str]]]
) -> dict[str, float] | None:
    """Router vs baseline on the tasks where NEITHER side billed a projected cell.

    This is the kill gate with the projection taken out. ``None`` when either side
    is absent or the overlap is empty.
    """
    # The imputed flag is path-aware (see strategy_cells), so a cascade that probed an
    # imputed cell on its way to a measured final pick is excluded here — the panel's
    # "no imputed cell on either side" title is then literally true of every dollar shown.
    if router not in by_strategy or baseline not in by_strategy:
        return None
    r_cells, r_un = by_strategy[router]
    b_cells, b_un = by_strategy[baseline]
    shared = [
        tid
        for tid in r_cells
        if tid in b_cells
        and tid not in r_un
        and tid not in b_un
        and not r_cells[tid][2]
        and not b_cells[tid][2]
    ]
    if not shared:
        return None
    r_pass = sum(1 for t in shared if r_cells[t][0])
    b_pass = sum(1 for t in shared if b_cells[t][0])
    only_b = sum(1 for t in shared if b_cells[t][0] and not r_cells[t][0])
    only_r = sum(1 for t in shared if r_cells[t][0] and not b_cells[t][0])
    return {
        "n": float(len(shared)),
        "router_cost": sum(r_cells[t][1] for t in shared),
        "baseline_cost": sum(b_cells[t][1] for t in shared),
        "router_pass": float(r_pass),
        "baseline_pass": float(b_pass),
        "baseline_only": float(only_b),
        "router_only": float(only_r),
        "mcnemar_p": metrics.mcnemar_exact_p(only_b, only_r),
    }


def _model_total_price(models: dict, name: str) -> float:
    info = models.get(name, {})
    return float(info.get("input_price", 0)) + float(info.get("output_price", 0))


def _synthesize_raw(results_2level: dict) -> RawResults:
    """Wrap a flattened challenge x model view as challenge x model x "default"
    for the arm-aware heatmap when no true per-arm cache is supplied — keeps the
    2-level test fixtures (and the old callers) working unchanged.
    """
    return {
        cid: {model: {"default": row} for model, row in per_model.items()}
        for cid, per_model in results_2level.items()
    }


def _sorted_arm_columns(
    columns: list[tuple[str, str]],
    models_meta: dict,
    arm_ranks: dict[tuple[str, str], int],
) -> list[tuple[str, str]]:
    """Column order: price ascending, then within-model arm rank ascending."""
    return sorted(
        columns,
        key=lambda c: (_model_total_price(models_meta, c[0]), c[0], arm_ranks.get(c, 0), c[1]),
    )


def arm_pair_contrasts(raw: RawResults, arm_ranks: dict[tuple[str, str], int]) -> list[dict]:
    """Every within-model (lower-rank, higher-rank) arm pair, on CO-MEASURED tasks only.

    The metamorphic check this replaces an unpaired marginal with: a within-model
    effect estimate must not change sign under restriction to the co-measured subset.
    """
    by_model: dict[str, list[str]] = {}
    for model, arm in plot_style.arm_columns(raw):
        by_model.setdefault(model, []).append(arm)
    out: list[dict] = []
    for model in sorted(by_model):
        arms = sorted(by_model[model], key=lambda a: arm_ranks.get((model, a), 0))
        for i, low in enumerate(arms):
            for high in arms[i + 1 :]:
                out.append(_arm_pair(raw, model, low, high))
    return [p for p in out if p["n"] > 0]


def _arm_pair(raw: RawResults, model: str, low: str, high: str) -> dict:
    """One paired arm contrast over the tasks where both arms ran."""
    # A CENSORED cell (step/wall/abandon stop) has an UNKNOWN true outcome, so scoring it
    # as a clean fail turns a resource limit into a capability claim — and on a paired
    # contrast it does so asymmetrically, because the higher arm runs longer and is
    # therefore censored more often. Dropping the pair is the only honest option here:
    # `censoring.is_censored` is the same predicate the kill gate and the strategy
    # evaluator use, so all three exclude the same cells.
    shared = [
        (per_model[model][low], per_model[model][high])
        for per_model in raw.values()
        if low in per_model.get(model, {})
        and high in per_model.get(model, {})
        and not censoring.is_censored(per_model[model][low])
        and not censoring.is_censored(per_model[model][high])
    ]
    n = len(shared)
    lost = sum(1 for lo, hi in shared if lo.get("pass") and not hi.get("pass"))
    won = sum(1 for lo, hi in shared if hi.get("pass") and not lo.get("pass"))
    low_pass = sum(1 for lo, _hi in shared if lo.get("pass"))
    high_pass = sum(1 for _lo, hi in shared if hi.get("pass"))
    delta = (high_pass - low_pass) / n * 100 if n else 0.0
    # Wald interval for the difference of CORRELATED proportions (the paired form —
    # an independent-samples interval would be far too narrow here).
    var = (lost + won - (won - lost) ** 2 / n) if n else 0.0
    half = 1.96 * math.sqrt(max(0.0, var)) / n * 100 if n else 0.0
    cost_delta = sum(row_real_cost(hi) - row_real_cost(lo) for lo, hi in shared) / n if n else 0.0
    return {
        "model": model,
        "low": low,
        "high": high,
        "n": n,
        "low_rate": low_pass / n * 100 if n else 0.0,
        "high_rate": high_pass / n * 100 if n else 0.0,
        "delta_pp": delta,
        "ci": (delta - half, delta + half),
        "violations": lost,
        "gains": won,
        "p": metrics.mcnemar_exact_p(lost, won),
        "cost_delta": cost_delta,
    }


def arm_pair_totals(pairs: list[dict]) -> dict[str, float]:
    """Pooled paired effect over every arm pair: net pp, violation rate, exact McNemar p."""
    n = sum(p["n"] for p in pairs)
    lost = sum(p["violations"] for p in pairs)
    won = sum(p["gains"] for p in pairs)
    return {
        "n": float(n),
        "violations": float(lost),
        "gains": float(won),
        "net_pp": (won - lost) / n * 100 if n else 0.0,
        "violation_rate": lost / n if n else 0.0,
        "p": metrics.mcnemar_exact_p(lost, won),
    }


def _monotonicity_annotations(pairs: list[dict]) -> Annotations:
    """The dataset-wide paired result — the honest headline this figure exists to state."""
    t = arm_pair_totals(pairs)
    n = int(t["n"])
    if not n:
        return Annotations(
            limitations=("No arm pair has a single co-measured task, so nothing here is paired.",)
        )
    verdict = (
        "indistinguishable from zero"
        if t["p"] > 0.05
        else ("positive" if t["net_pp"] > 0 else "negative")
    )
    notes = [
        f"DATASET-WIDE PAIRED RESULT: over {n} co-measured within-model arm pair-observations, "
        f"the net effect of more reasoning effort is {t['net_pp']:+.1f}pp "
        f"({int(t['gains'])} gains vs {int(t['violations'])} losses), exact McNemar two-sided "
        f"p={t['p']:.3f} — {verdict}.",
        f"Monotonicity is VIOLATED on {t['violation_rate'] * 100:.1f}% of co-measured pairs "
        f"({int(t['violations'])} of {n}): the higher-effort arm failed a task the lower-effort "
        "arm passed.",
        f"{len(pairs)} arm pair(s) plotted; co-measured n runs "
        f"{min(p['n'] for p in pairs)}-{max(p['n'] for p in pairs)}.",
    ]
    straddling = sum(1 for p in pairs if p["ci"][0] <= 0.0 <= p["ci"][1])
    limits: list[str] = []
    if straddling:
        limits.append(
            f"{straddling} of {len(pairs)} pair interval(s) straddle zero — on this data the "
            "arm dimension is not something to route on."
        )
    thin = [p for p in pairs if p["n"] < MIN_N_RELIABLE]
    if thin:
        limits.append(
            f"{len(thin)} of {len(pairs)} pair(s) rest on fewer than {MIN_N_RELIABLE} "
            "co-measured tasks (marked provisional) — their interval is wide enough to contain "
            "almost anything."
        )
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


# ---------------------------------------------------------------------------
# Equal-coverage imputation reporting — capability distribution,
# coverage audit, violation rate, cascade overhead-leak, per-stratum win-rates,
# and the disclosure banner that replaces the removed sparse-frontier (phantom) machinery.
# ---------------------------------------------------------------------------


# Report-time capability BANDS — an ordinal, metadata-described grouping of the measured
# capability rank (band 1 = weakest ... band N = strongest), NOT stored and NOT the routing
# unit. Bands carry NO semantic names: each is described only by its member models, price
# range, marginal-pass-rate range (+CI), n_models, and the share of tasks it is the weakest
# to solve. The band COUNT is data-driven — adjacent models whose capability CIs overlap
# merge into one band — never a fixed cap.


def capability_bands(rank: config.CapabilityRank) -> dict[str, int]:
    """Ordinal band index per model (1 = weakest ... N = strongest), merged on PAIRWISE
    CI overlap: every member of a band overlaps every other member."""
    # A set of intervals overlaps pairwise iff max(lo) <= min(hi), so the band carries a
    # running envelope and cuts when a candidate falls outside it. The previous rule
    # compared each model only with its predecessor, which chains: it put deepseek
    # [0.613, 0.745] and qwen [0.337, 0.547] — disjoint — in one band while the legend
    # told the reader a band means overlapping CIs.
    bands: dict[str, int] = {}
    band = 1
    envelope: tuple[float, float] | None = None
    for rm in rank.ordered:
        ev = rank.evidence.get(rm.model)
        if ev is not None:
            if envelope is None:
                envelope = (ev.ci_lo, ev.ci_hi)
            else:
                lo, hi = max(envelope[0], ev.ci_lo), min(envelope[1], ev.ci_hi)
                if lo > hi:
                    band += 1
                    lo, hi = ev.ci_lo, ev.ci_hi
                envelope = (lo, hi)
        bands[rm.model] = band
    return bands


def _band_order(bands: dict[str, int]) -> list[int]:
    """Unique band indices in weakest -> strongest order."""
    return sorted(set(bands.values()))


def band_metadata(
    rank: config.CapabilityRank,
    bands: dict[str, int],
    im: ImputedMatrix | None = None,
) -> dict[int, dict]:
    """Per-band metadata: member models (rank order), n_models, price_range [min,max],
    capability_range (marginal pass-rate min/max + CI envelope), and pct_tasks_min_solved
    (share of tasks whose crossover model τ falls in the band, when a matrix is given)."""
    members: dict[int, list[str]] = {}
    for rm in rank.ordered:
        b = bands.get(rm.model)
        if b is not None:
            members.setdefault(b, []).append(rm.model)
    tau_total = len(im.tau) if im is not None else 0
    tau_counts: dict[int, int] = {}
    if im is not None:
        for model in im.tau.values():
            b = bands.get(model) if model is not None else None
            if b is not None:
                tau_counts[b] = tau_counts.get(b, 0) + 1
    meta: dict[int, dict] = {}
    for b in _band_order(bands):
        models = members.get(b, [])
        evs = [rank.evidence[m] for m in models if m in rank.evidence]
        prices = [e.price for e in evs]
        rates = [e.pass_rate for e in evs]
        meta[b] = {
            "band": b,
            "models": models,
            "n_models": len(models),
            "price_range": [min(prices), max(prices)] if prices else None,
            "capability_range": {
                "pass_rate_min": round(min(rates), 4),
                "pass_rate_max": round(max(rates), 4),
                "ci_lo": round(min(e.ci_lo for e in evs), 4),
                "ci_hi": round(max(e.ci_hi for e in evs), 4),
            }
            if evs
            else None,
            "pct_tasks_min_solved": (
                round(tau_counts.get(b, 0) / tau_total, 4) if tau_total else None
            ),
        }
    return meta


def _band_label(b: int, meta: dict[int, dict]) -> str:
    """Compact axis label: the ordinal index plus the band's defining metadata (member
    models, price range, marginal pass-rate range) — never a semantic name."""
    m = meta.get(b, {})
    who = "\n".join(m.get("models") or ["?"])  # one per line: names collided at fontsize 7
    pr = m.get("price_range")
    # Escaped: an unescaped pair of `$` in one tick label is parsed as mathtext and
    # both currency markers are deleted, so the shipped axis read "2-4" not "$2-$4".
    price = f"{usd(pr[0], 2)}-{usd(pr[1], 2)}/Mtok" if pr else "price ?"
    cr = m.get("capability_range")
    rate = (
        f"marginal pass {cr['pass_rate_min'] * 100:.0f}-{cr['pass_rate_max'] * 100:.0f}%"
        if cr
        else "?"
    )
    return f"band {b}\n{who}\n{price}\n{rate}"


def _rank_bands() -> tuple[config.CapabilityRank, dict[str, int], list[int]]:
    """The live capability rank, its ordinal display bands, and their weak->strong order."""
    rank = config.capability_rank()
    bands = capability_bands(rank)
    return rank, bands, _band_order(bands)


def coverage_rows(im: ImputedMatrix, bands: dict[str, int], band_order: list[int]) -> list[dict]:
    """Per-band real / imputed / UNKNOWN cell counts — the audit that coverage is
    now equal and how much rests on imputation."""
    n_tasks = len(im.matrix)
    models_per_band: dict[int, int] = {}
    for b in bands.values():
        models_per_band[b] = models_per_band.get(b, 0) + 1
    real: dict[int, int] = {}
    imputed: dict[int, int] = {}
    for cells in im.matrix.values():
        for model, cell in cells.items():
            mb = bands.get(model)
            if mb is None:
                continue
            bucket = imputed if cell.get("imputed") else real
            bucket[mb] = bucket.get(mb, 0) + 1
    rows: list[dict] = []
    for b in band_order:
        if b not in models_per_band:
            continue
        r, i = real.get(b, 0), imputed.get(b, 0)
        expected = n_tasks * models_per_band[b]
        rows.append({"band": b, "real": r, "imputed": i, "unknown": max(0, expected - r - i)})
    return rows


def write_coverage_table(im: ImputedMatrix, out_dir: Path) -> Path:
    """Write the per-band coverage audit CSV (regenerable, gitignored)."""
    _rank, bands, order = _rank_bands()
    rows = coverage_rows(im, bands, order)
    path = out_dir / "coverage_table.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["band", "real", "imputed", "unknown"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def write_capability_evidence(out_dir: Path, im: ImputedMatrix | None = None) -> Path:
    """Write the per-model capability-rank evidence + ordinal-band grouping artifact (rank,
    price, pass-rate, CI, source, band metadata; regenerable, gitignored). Prints a loud
    finding if the derived strongest model is not the configured control."""
    rank, bands, _order = _rank_bands()
    knobs = config.capability_rank_config()
    control = config.frontier_model()
    if rank.ordered and rank.strongest() != control:
        print(
            f"  ⚠ FINDING: derived strongest model {rank.strongest()!r} != control_model "
            f"{control!r} — the kill-gate baseline may be mis-chosen (investigate, do not ignore)."
        )
    csv_path = config.results_csv_path()
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest() if csv_path.exists() else "absent"
    models = [
        {
            "model": rm.model,
            "n": rank.evidence[rm.model].n,
            "pass_rate": round(rank.evidence[rm.model].pass_rate, 4),
            "ci_lo": round(rank.evidence[rm.model].ci_lo, 4),
            "ci_hi": round(rank.evidence[rm.model].ci_hi, 4),
            "rank": rm.rank,
            "source": rm.source,
            "price": rank.evidence[rm.model].price,
            "band": bands.get(rm.model),
        }
        for rm in rank.ordered
    ]
    payload = {
        "generated_from": f"results.csv@{digest}",
        "knobs": {k: knobs[k] for k in ("K", "W", "min_pairs")},
        "control_model": control,
        "bands": list(band_metadata(rank, bands, im).values()),
        "models": models,
    }
    path = out_dir / "capability_evidence.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _violation_line(im: ImputedMatrix) -> str:
    """One-line monotonicity violation rate with a Wilson CI — 'measured' only when it was."""
    # violation_ci returns (0, 0, 0) at n=0; rendering that as v̂=0 [0,0] sells a
    # vacuous denominator as a perfect measurement.
    if im.n_multi_observed <= 0:
        return (
            "Monotonicity violation rate UNMEASURED — no task has two or more observed "
            "ranked models, so the axiom is assumed here, not verified"
        )
    v, lo, hi = impute.violation_ci(len(im.violations), im.n_multi_observed)
    return (
        f"Monotonicity violation rate v̂={v:.3f} 95% CI [{lo:.3f}, {hi:.3f}] "
        f"({len(im.violations)} of {im.n_multi_observed} multi-observed tasks) — "
        "measured, not assumed"
    )


def _disclosure_banner(im: ImputedMatrix | None, rows: list[dict] | None) -> str | None:
    """Coverage disclosure for the Pareto / cost=quality planes.

    Downgrades 'equal-coverage' to 'coverage-completed' with the residual UNKNOWN
    frontier count whenever per-strategy n_tasks are not actually equal (FIX B).
    """
    if im is None:
        return None
    _rank, bands, order = _rank_bands()
    strongest = order[-1] if order else None
    fr = next(
        (r for r in coverage_rows(im, bands, order) if r["band"] == strongest),
        None,
    )
    covered = (fr["real"] + fr["imputed"]) if fr else 0
    unknown = fr["unknown"] if fr else 0
    frac = (fr["imputed"] / covered * 100) if fr and covered else 0.0
    ns = [int(r.get("n_tasks", 0) or 0) for r in rows] if rows else []
    equal = bool(ns) and len(set(ns)) == 1
    if equal:
        head = (
            f"equal-coverage via monotone imputation — {frac:.0f}% of frontier cells "
            f"imputed (every strategy scored on n={ns[0]})."
        )
    else:
        rng = f"{min(ns)}–{max(ns)}" if ns else "?"
        head = (
            f"coverage-completed via monotone imputation — {frac:.0f}% of frontier cells "
            f"imputed, {unknown} still unmeasured (per-strategy n={rng}); true equal-coverage "
            "requires ladder collection."
        )
    if im.n_multi_observed > 0:
        v, _lo, _hi = impute.violation_ci(len(im.violations), im.n_multi_observed)
        axiom = (
            f" Monotonicity holds on {(1 - v) * 100:.0f}% of "
            f"{im.n_multi_observed} multi-observed task(s) (measured, not assumed)."
        )
    else:
        axiom = (
            " Monotonicity is UNVERIFIED here — no task has two or more observed ranked "
            "models, so the axiom is assumed."
        )
    # No directional claim here. The shipped banner asserted "imputation is
    # conservative, so a broken axiom only widens the router's lead"; that is not
    # demonstrated — the router takes free imputed passes too, and on the measured-only
    # overlap its lead is zero (kill_gate.png's measured-only row).
    # The fail-branch count is DERIVED from the matrix being disclosed. It was the frozen
    # literal "1 of 398", which the corpus had already moved past (5 of 410) while the
    # percentages either side of it were recomputed on every run.
    pass_filled, n_filled = fig_evidence.filled_outcomes(im.matrix)
    return (
        head + axiom + " NEARLY every imputed cell is filled pass=True at a median measured "
        f"price (the monotone ladder has a fail branch, and {n_filled - pass_filled} of "
        f"{n_filled} filled cells took it), for the router as well as for the baseline — "
        "see evidence_basis.png for how much of each strategy's number that is, and "
        "kill_gate.png's measured-only row for what "
        "survives when the projection is removed."
    )


def cascade_overhead(
    matrix: dict, im: ImputedMatrix, tasks: list[str], strategy: Strategy
) -> float:
    """Mean ``cascade_total_cost − single_shot_cost(τ)`` over solvable tasks — dollars a
    cascade burns on failed cheaper probes before the passing model."""
    results = matrix.get("results", {})
    overheads: list[float] = []
    for tid in tasks:
        tau = im.tau.get(tid)  # the crossover MODEL (weakest that solves the task), or None
        if tau is None:
            continue
        strategy.select(tid, matrix.get("tasks", {}).get(tid, {}), matrix)
        cascade_cost = getattr(strategy, "cascade_total_cost", None)
        if cascade_cost is None:
            continue
        single = row_real_cost(results.get(tid, {}).get(tau, {}))
        overheads.append(float(cascade_cost) - single)
    return sum(overheads) / len(overheads) if overheads else 0.0


def derive_tasks(matrix: dict, seed: int) -> list[str]:
    """Covered tasks (present in results.csv) sampled by ``sample_size`` — matches run_eval."""
    # Evaluate over the MEASURED denominator, not the full suite: an uncovered task is
    # unmeasured, NOT a failure — scoring the 151 unrun-of-200 as fail@$0 diluted every
    # plotted pass-rate ~4x and made report.py disagree with run_eval/plot_strategies.
    return config.sample_tasks(sorted(matrix.get("results", {}).keys()), seed=seed)


def derive_rows(
    matrix: dict, tasks: list[str], strategies: list[object] | None = None
) -> list[dict]:
    """Per-strategy summary rows derived in-memory from the results.csv cache."""
    # The single source of truth. Mirrors run_matrix.refresh_summary exactly (same
    # strategies, gamma, bootstrap, seed, and task set — see derive_tasks). Pass
    # `strategies` to reuse an already-built set: each kNN-family strategy embeds every
    # task description and builds its own HNSW index on first select, so building a
    # second set costs GBs of RSS for no new information.
    from benchmark.routing import run_eval

    bm = config.benchmark_params()
    strategies = strategies if strategies is not None else run_eval.get_strategies()
    return summary.compute_strategy_rows(
        matrix,
        tasks,
        strategies,
        gamma=config.gamma(),
        bootstrap=bm.get("bootstrap_iterations", 1000),
        seed=bm.get("seed", 42),
    )


def _load_raw_results() -> RawResults | None:
    """Best-effort challenge x model x arm cache (config.load_results()) — never
    fatal, since every arm plot degrades gracefully without it.
    """
    try:
        raw = config.load_results()
    except Exception:  # noqa: BLE001 (arm plots are all optional)
        return None
    return _only_enabled_models(raw) or None


def _only_enabled_models(raw: RawResults) -> RawResults:
    """Drop cells for models outside benchmark.yaml's enabled set."""
    # results.csv legitimately carries cells for models the benchmark does not evaluate — a
    # probe-only collection such as zai-glm-5.3-flash's free window. The arm plots and the
    # Arm-oracle / Arm-bandit strategies read this cache directly, so without this filter such
    # a model becomes an extra complementarity column and a rung those strategies can pick,
    # scoring them on a model the router cannot serve and the benchmark never enabled. Every
    # other routing analysis already scopes to `enabled_models()` (run_eval, report ordering,
    # compute_costs, sensitivity, plot_exploration, figures/context); this makes the arm path
    # agree with them. An empty/failed registry read leaves `raw` untouched rather than
    # silently emptying every arm plot.
    try:
        enabled = set(config.enabled_models())
    except Exception:  # noqa: BLE001 (registry optional at plot time)
        return raw
    if not enabled:
        return raw
    return {
        challenge: {model: arms for model, arms in by_model.items() if model in enabled}
        for challenge, by_model in raw.items()
    }


def _report_imputation_outputs(
    im: ImputedMatrix,
    matrix: dict,
    tasks: list[str],
    out_dir: Path,
    strategies: list[object] | None = None,
) -> None:
    """Emit the equal-coverage outputs: violation rate, coverage table, capability
    evidence, cascade overhead-leak. The per-band split is drawn by evidence_basis.png
    and the band histogram by task_difficulty.png, so neither is written twice."""
    from benchmark.routing import run_eval
    from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

    strategies = strategies if strategies is not None else run_eval.get_strategies()
    print(f"  {_violation_line(im)}")
    _rank, bands, order = _rank_bands()
    unknown_by_band = {r["band"]: r["unknown"] for r in coverage_rows(im, bands, order)}
    total_unknown = sum(unknown_by_band.values())
    print(
        f"  Residual UNKNOWN cells (excluded from scoring, NOT equal coverage yet): "
        f"{total_unknown} total; by band {unknown_by_band}"
    )
    print(f"  Coverage tbl : {write_coverage_table(im, out_dir)}")
    print(f"  Capab. evid. : {write_capability_evidence(out_dir, im)}")

    completed, _im = summary.complete_scored_matrix(matrix)
    overhead = cascade_overhead(completed, im, tasks, kNNCascadeStrategy(**config.knn_params()))
    print(f"  Cascade overhead-leak: USD {overhead:.4f}/task (failed cheaper probes before tau)")


def strategy_choices(
    matrix: dict, tasks: list[str], strategies: Iterable[object]
) -> dict[str, dict[str, str]]:
    """Per strategy: the model it chose per scorable task. The decision, not its outcome."""
    # `strategy_cells` records what a decision COST and whether it passed; the audit figure
    # needs what it PICKED, and re-running the strategies to get it would re-pay the
    # embedding peak. Both read the same single `summary.evaluate` pass per strategy.
    out: dict[str, dict[str, str]] = {}
    for strategy in strategies:
        decisions, unscorable = summary.evaluate(strategy, matrix, tasks)
        out[strategy.name] = {  # type: ignore[attr-defined]
            tid: model for tid, model, _passed, _cost in decisions if tid not in unscorable
        }
    return out


def _render_oracle_gap(
    ctx: figures.RoutingContext,
    matrix: dict | None,
    tasks: list[str],
    raw: RawResults | None,
    gamma: float,
) -> object:
    """Assemble the cost decomposition and the regret series, then draw oracle_gap.png."""
    from benchmark.runner import kill_gate as gate

    if matrix is None or not tasks:
        return "skipped (no matrix)"
    completed = ctx.completed or matrix
    control = gate.evaluate_control(completed, tasks, config.frontier_model() or "")
    router = gate.evaluate_router(completed, tasks)
    decomposition = compute_cost_decomposition(control, router)

    evaluated = _evaluate_strategies(_build_strategy_factories(gamma), completed, tasks)
    oracle_pair = evaluated.get("Oracle-reward")
    if oracle_pair is None:
        return "skipped (no oracle reference)"
    oracle, excluded = oracle_pair[0], set(oracle_pair[1])
    series: dict[str, list[tuple[str, str, bool, float]]] = {}
    for name, (decisions, unscorable) in evaluated.items():
        if name == "Oracle-reward":
            continue
        series[name] = decisions
        excluded |= set(unscorable)
    if raw:
        series["Arm-oracle"] = _arm_oracle_decisions(raw, tasks, gamma)
        series["Arm-bandit"] = _arm_bandit_decisions(raw, tasks, gamma)
    return (
        fig_oracle.render(ctx, decomposition, series, oracle, excluded, gamma)
        or "skipped (no equal-quality pair)"
    )


def paired_quality_contrast() -> str:
    """The Price-Cascade vs fixed-frontier paired quality headline (docs/benchmark.md)."""
    # Emits the exact sentence fragment for the "cheapest strategy that matches fixed-frontier
    # quality" claim from the committed corpus, regenerable byte-for-byte. Uses ONLY fixed
    # strategies (Price-Cascade, Always-Frontier) — no embeddings — so it runs offline and
    # deterministically: paired pass-rate delta + paired bootstrap CI (seed 42, 1000 draws,
    # the convention run_eval._paired_bootstrap_ci uses).
    import random

    from benchmark.routing.strategies.fixed import AlwaysFrontier
    from benchmark.routing.strategies.price_cascade import PriceCascade

    matrix = config.load_matrix()
    completed, _ = summary.complete_scored_matrix(matrix)
    tasks = sorted(completed["results"].keys())
    pc = PriceCascade(max_tries=3)
    af = AlwaysFrontier()
    pc_dec, pc_un = summary.evaluate(pc, completed, tasks)
    af_dec, af_un = summary.evaluate(af, completed, tasks)
    pc_pass = {d[0]: bool(d[2]) for d in pc_dec}
    af_pass = {d[0]: bool(d[2]) for d in af_dec}
    shared = [t for t in tasks if t not in pc_un and t not in af_un]
    diffs = [int(pc_pass[t]) - int(af_pass[t]) for t in shared]
    n = len(shared)
    delta = 100.0 * sum(diffs) / n
    rng = random.Random(42)
    n_boot = 1000
    means = sorted(
        100.0 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    a = int(0.025 * n_boot)
    lo, hi = means[a], means[n_boot - 1 - a]
    crosses_zero = lo <= 0 <= hi
    pc_cost = sum(r[3] for r in pc_dec if r[0] in shared)
    af_cost = sum(r[3] for r in af_dec if r[0] in shared)
    cheaper_pct = round((1 - pc_cost / af_cost) * 100) if af_cost else 0
    reading = "statistically equal" if crosses_zero else "not statistically equal"
    return (
        f"{delta:+.1f} pp, CI crosses zero → {reading}, at roughly "
        f"{cheaper_pct}% lower cost on the shared measurable set"
    )


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    # Line-buffered: a piped stdout otherwise buffers 8 KB, and an OOM SIGKILL
    # discards it — the first observed failure of this report printed NOTHING.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    config.load(config_path)

    ap = argparse.ArgumentParser(
        description="Generate comparison reports from routing benchmark results"
    )
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument(
        "--results",
        default=None,
        help="Optional: read a pre-derived strategy-summary CSV instead of deriving "
        "in-memory from results.csv (default: derive from the source of truth)",
    )
    ap.add_argument(
        "--out-dir",
        default="benchmark/routing/reports",
        help="Output directory for the derived CSV/JSON artifacts "
        "(default: benchmark/routing/reports/)",
    )
    ap.add_argument(
        "--figures-dir",
        default="docs/assets/figures/routing",
        help="Output directory for the PNGs. They live inside the published docs tree, one "
        "subdirectory per half, so the docs can link them relatively "
        "(default: docs/assets/figures/routing/)",
    )
    ap.add_argument(
        "--matrix",
        default=None,
        help="Path to task matrix JSON (enables per-task regret lines + heatmap)",
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _memory_preflight()

    # ONE strategy set for the whole run. Each kNN-family strategy embeds every task
    # description and builds its own HNSW index on first select, so re-instantiating
    # the set per consumer cost GBs of RSS and OOM-killed the report.
    from benchmark.routing import run_eval

    strategies = run_eval.get_strategies()

    # ONE load of the matrix for the whole run: it was loaded twice (once to derive the
    # rows, once for the plots), holding two full copies alongside the embedders.
    matrix = load_matrix(matrix_path) if matrix_path and matrix_path.exists() else None

    if args.results:
        results = load_results(Path(args.results))
        source = args.results
        tasks: list[str] = []
    else:
        if matrix is None or not matrix.get("results"):
            print(
                "No results yet — results.csv holds no rows. "
                "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
            )
            return
        tasks = derive_tasks(matrix, config.benchmark_params().get("seed", 42))
        results = derive_rows(matrix, tasks, strategies)
        # Validate BEFORE writing: a degenerate matrix must not leave a misleading
        # summary CSV behind on its way to a crash in the first plot.
        problem = _validate_rows(results) if results else None
        if problem:
            print(f"Refusing to report: {problem}", file=sys.stderr)
            return
        # Write a human-readable copy to reports/ — the one tracked file there, so a committed
        # figure can be checked against the numbers it was drawn from. Every row is
        # stamped with the SHIPPED selection path's instrument verdict; the figures' gate
        # certifies `select_from_rates`, which is a different rule.
        table = summary.certified_table(results)
        print(table.admissibility.reason)
        summary.write_summary_csv(table, out_dir / "strategy_summary.csv")
        source = "results.csv (derived in-memory)"

    if not results:
        print("No strategy rows to plot.", file=sys.stderr)
        return

    problem = _validate_rows(results)
    if problem:
        print(f"Refusing to report: {problem}", file=sys.stderr)
        return

    print(f"Loaded {len(results)} strategies from {source}")

    print()
    print("Strategy ranking (derived from the results.csv source of truth):")
    print_summary(results)
    print()

    raw_results = _load_raw_results()
    matrix_for_plots = matrix

    # Complete the matrix ONCE and keep the completed copy: the measured-vs-projected
    # cells below used to re-run the same completion, doubling the work and the peak.
    imputed: ImputedMatrix | None = None
    completed: dict | None = None
    if matrix_for_plots is not None:
        completed, imputed = summary.complete_scored_matrix(matrix_for_plots)
    banner = _disclosure_banner(imputed, results)

    # Per-task selections with the imputed flag, plus the model each strategy chose.
    # ONE evaluation pass feeds every figure below — a second one costs GBs of RSS.
    by_strategy: dict[str, tuple[StrategyCells, set[str]]] = {}
    repriced_totals: dict[str, float | None] = {}
    chosen: dict[str, dict[str, str]] = {}
    if completed is not None and tasks:
        by_strategy, repriced_totals = strategy_cells(completed, tasks, strategies)
        chosen = strategy_choices(completed, tasks, strategies)

    ctx = figures.build_context(
        out_dir=figures_dir,
        matrix=matrix_for_plots or {},
        completed=completed or {},
        imputed=imputed,
        tasks=tasks,
        rows=results,
        raw=raw_results,
        banner=banner,
        by_strategy=by_strategy,
        repriced_totals=repriced_totals,
    )

    _step("Kill gate", fig_kill_gate.render(ctx) or "skipped (no paired arm)")
    _step("Ladder rungs", fig_ladder.render(ctx) or "skipped (no priced target)")
    _step("Cost/quality", fig_frontier.render(ctx) or "skipped (no cost)")
    _step("Pareto dimensions", fig_pareto_dims.render(ctx) or "skipped (no live row)")
    _step("Model grid", fig_model_grid.render(ctx) or "skipped (no measured model)")
    _step("Live gap", fig_live_gap.render(ctx) or "skipped (no bound row)")
    _step("Cache econ", fig_cache.render(ctx) or "skipped (no priced row)")

    g = config.gamma()
    _step("Oracle gap", _render_oracle_gap(ctx, matrix_for_plots, tasks, raw_results, g))

    router_picks = chosen.get(figures.ROUTER_STRATEGY, {})
    _step("Decision audit", fig_audit.render(ctx, router_picks) or "skipped (no picks)")

    if imputed is not None and matrix_for_plots is not None:
        _report_imputation_outputs(imputed, matrix_for_plots, tasks, out_dir, strategies)
        _rank, bands, order = _rank_bands()
        _step(
            "Evidence basis",
            fig_evidence.render(ctx, coverage_rows(imputed, bands, order)) or "skipped",
        )
        _step("Task difficulty", fig_difficulty.render(ctx, bands, router_picks) or "skipped")

    if raw_results is not None:
        arm_ranks = _arm_ranks()
        columns = _sorted_arm_columns(
            plot_style.arm_columns(raw_results),
            (matrix_for_plots or {}).get("models", {}),
            arm_ranks,
        )
        _step("Complementarity", fig_complementarity.render(ctx, columns) or "skipped")
        contrasts = arm_pair_contrasts(raw_results, arm_ranks)
        _step(
            "Arm manipulation",
            fig_arms.render(ctx, contrasts, arm_pair_totals(contrasts)) or "skipped (no arm pairs)",
        )

    plt.close("all")
    print(f"Done. Peak RSS {_peak_rss_mb():,.0f} MB.")


if __name__ == "__main__":
    main()
