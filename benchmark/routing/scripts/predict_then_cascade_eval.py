#!/usr/bin/env python3
"""Offline eval of the predict-then-cascade gate: its cost/quality curve on the completed matrix."""

# Loads the completed matrix the way report.py does (config.load_matrix ->
# summary.complete_scored_matrix, the 200-task input of which 184 complete challenges are scored)
# and sweeps the fraction *f* of tasks a score-free placeholder gate sends cheap-direct. x = f,
# y = naive total cost and pass rate, for f in {0.0, 0.1, ..., 1.0}.
#
# MARKED POINTS
# - f=0 is Session-Cascade (today's shipped behaviour) and f=1 is Always-Cheap, so a chance-level
#   gate interpolates the mixture line between the two fixed policies. The eval asserts the two
#   endpoints reproduce those rows exactly (the no-regression property).
# - The oracle point is the PERFECT binary gate: it sends exactly the tasks the cheap model solves
#   cheap-direct and routes everything else through the same ladder. The oracle ties f=0 exactly
#   (same cost, same pass set) — the ladder already pays exactly one cheap attempt on every
#   cheap-solving task — so the gate idea's headroom is zero AT EQUAL QUALITY, not on cost alone:
#   the raw-cost axis's f>0 points below f=0 (e.g. f=0.3 at ~$1.17 vs f=0's $1.29) are cheaper
#   only at a lower pass rate, a quality tradeoff, not prediction headroom. The fixed-denominator
#   line (pass rate among Session-Cascade's own 142 scored tasks) removes the denominator drift,
#   because cheap-direct tasks are always scorable and the ladder's hard tail is not.
#
# Offline only: fixed strategies, no embeddings, no live API. Outputs to
# benchmark/routing/reports/ (gitignored; regenerable).

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Final

from matplotlib.axes import Axes

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec, Provenance
from benchmark.routing import summary
from benchmark.routing.figures.context import corpus_digest
from benchmark.routing.strategies.fixed import AlwaysCheap
from benchmark.routing.strategies.predict_then_cascade import (
    FractionalGate,
    PerfectCheapGate,
    PredictThenCascadeStrategy,
)
from benchmark.routing.strategies.session_cascade import SessionCascadeStrategy

_GENERATOR = "benchmark.routing.scripts.predict_then_cascade_eval"
_CHEAP_MODEL = "deepseek-v4-flash"
_DEFAULT_CSV = Path("benchmark/routing/reports/predict_then_cascade_curve.csv")
_DEFAULT_PLOT = Path("benchmark/routing/reports/predict_then_cascade_curve.png")

F_GRID: Final[tuple[float, ...]] = tuple(round(i * 0.1, 1) for i in range(11))

CSV_FIELDS: Final[tuple[str, ...]] = (
    "kind",
    "f",
    "strategy",
    "n_tasks",
    "n_unscorable",
    "n_pass",
    "AvgPerf%",
    "TotalCost",
    "TotalCost_cacheaware",
    "AvgCost",
    "fixed_denom_pass",
    "fixed_denom_n",
    "fixed_denom_rate",
)


def _curve_row(
    strat: object, kind: str, f: float | None, matrix: dict, tasks: list[str], fixed_set: set[str]
) -> dict:
    """One row: cost/quality metrics (raw + cache-aware) plus the fixed-denominator pass."""
    rows = summary.compute_strategy_rows(
        matrix, tasks, [strat], gamma=config.gamma(), bootstrap=1000, seed=42
    )
    r = rows[0]
    fixed_pass, fixed_n = _fixed_denominator(strat, fixed_set, matrix, tasks)
    return {
        "kind": kind,
        "f": "" if f is None else f"{f:.1f}",
        "strategy": r["strategy"],
        "n_tasks": r["n_tasks"],
        "n_unscorable": r["n_unscorable"],
        "n_pass": r["n_pass"],
        "AvgPerf%": r["AvgPerf%"],
        "TotalCost": r["TotalCost"],
        "TotalCost_cacheaware": r["TotalCost_cacheaware"],
        "AvgCost": r["AvgCost"],
        "fixed_denom_pass": fixed_pass,
        "fixed_denom_n": fixed_n,
        "fixed_denom_rate": round(fixed_pass / fixed_n * 100, 2) if fixed_n else 0.0,
    }


def _fixed_denominator(
    strat: object, fixed_set: set[str], matrix: dict, tasks: list[str]
) -> tuple[int, int]:
    """Passes/n among Session-Cascade's scored set — the like-for-like quality axis."""
    decisions, unscorable = summary.evaluate(strat, matrix, tasks)
    by_tid = {tid: passed for tid, _model, passed, _cost in decisions}
    keep = [tid for tid in fixed_set if tid not in unscorable]
    return sum(1 for tid in keep if by_tid.get(tid, False)), len(keep)


def _cheap_passing_tasks(completed: dict) -> set[str]:
    """Tasks the cheap model solves on the completed matrix — the oracle gate's membership."""
    return {
        tid
        for tid, cells in completed.get("results", {}).items()
        if cells.get(_CHEAP_MODEL, {}).get("pass", False)
    }


def _evaluate(matrix: dict, tasks: list[str]) -> list[dict]:
    """The curve rows: f-grid, oracle point, and the two fixed-policy endpoints."""
    # The COMPLETED matrix is the one compute_strategy_rows scores every row on, so the oracle
    # gate's cheap-set and the fixed-denominator helper must read it too — a raw-matrix helper
    # counted 9 imputed-only tasks as unscorable and shrank the fixed denominator to 133.
    completed, _im = summary.complete_scored_matrix(matrix)
    fixed_set = _session_cascade_scored(completed, tasks)
    rows: list[dict] = []
    for f in F_GRID:
        strat = PredictThenCascadeStrategy(gate=FractionalGate(f), label=f"PTC f={f:.1f}")
        rows.append(_curve_row(strat, "curve", f, completed, tasks, fixed_set))
    rows.append(
        _curve_row(
            PredictThenCascadeStrategy(
                gate=PerfectCheapGate(_cheap_passing_tasks(completed)), label="PTC oracle"
            ),
            "oracle",
            None,
            completed,
            tasks,
            fixed_set,
        )
    )
    rows.append(_curve_row(SessionCascadeStrategy(), "baseline", 0.0, completed, tasks, fixed_set))
    rows.append(_curve_row(AlwaysCheap(), "baseline", 1.0, completed, tasks, fixed_set))
    return rows


def _session_cascade_scored(completed: dict, tasks: list[str]) -> set[str]:
    """The tasks Session-Cascade can score — the fixed quality denominator for all rows."""
    _decisions, unscorable = summary.evaluate(SessionCascadeStrategy(), completed, tasks)
    return set(tasks) - unscorable


def _assert_degeneration(rows: list[dict]) -> None:
    """A chance-level gate at the endpoints MUST reproduce the fixed policies exactly."""
    by_kind_f = {(r["kind"], r["f"]): r for r in rows}
    f0 = by_kind_f[("curve", "0.0")]
    f1 = by_kind_f[("curve", "1.0")]
    sc = by_kind_f[("baseline", "0.0")]
    ac = by_kind_f[("baseline", "1.0")]
    for got, expected, label in (
        (
            (f0["AvgPerf%"], f0["TotalCost"], f0["n_tasks"]),
            (sc["AvgPerf%"], sc["TotalCost"], sc["n_tasks"]),
            "f=0 vs Session-Cascade",
        ),
        (
            (f1["AvgPerf%"], f1["TotalCost"], f1["n_tasks"]),
            (ac["AvgPerf%"], ac["TotalCost"], ac["n_tasks"]),
            "f=1 vs Always-Cheap",
        ),
    ):
        if got != expected:
            raise RuntimeError(f"gate does not degenerate: {label}: got {got}, expected {expected}")


def write_csv(rows: list[dict], path: Path) -> None:
    """The full curve, one row per f plus the oracle and endpoint baselines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _draw_curve(ax: Axes, rows: list[dict]) -> Annotations:
    """Cost-quality: the mixture line, the fixed-denominator line, and the marked points."""
    curve = [r for r in rows if r["kind"] == "curve"]
    oracle = next(r for r in rows if r["kind"] == "oracle")
    sc = next(r for r in rows if r["kind"] == "baseline" and r["f"] == "0.0")
    ac = next(r for r in rows if r["kind"] == "baseline" and r["f"] == "1.0")
    xs = [float(r["TotalCost"]) for r in curve]
    ys = [float(r["AvgPerf%"]) for r in curve]
    ax.plot(
        xs,
        ys,
        "-o",
        color="#1F4E79",
        markersize=4.5,
        linewidth=1.5,
        label="chance gate, f grid",
    )
    for r in curve:
        ax.annotate(
            f"f={r['f']}\nn={r['n_tasks']}",
            (float(r["TotalCost"]), float(r["AvgPerf%"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=6.5,
            color="#333333",
        )
    # Fixed-denominator quality: pass rate among Session-Cascade's own scored tasks.
    ax.plot(
        xs,
        [float(r["fixed_denom_rate"]) for r in curve],
        "--",
        color="#6A3D9A",
        linewidth=1.4,
        label="pass rate on Session-Cascade's scored set",
    )
    ax.scatter(
        [float(sc["TotalCost"])],
        [float(sc["AvgPerf%"])],
        marker="s",
        s=60,
        color="#B71C1C",
        zorder=6,
        label="f=0  Session-Cascade",
    )
    ax.scatter(
        [float(ac["TotalCost"])],
        [float(ac["AvgPerf%"])],
        marker="s",
        s=60,
        color="#2E7D32",
        zorder=6,
        label="f=1  Always-Cheap",
    )
    ax.scatter(
        [float(oracle["TotalCost"])],
        [float(oracle["AvgPerf%"])],
        marker="*",
        s=180,
        color="#FF8C00",
        zorder=7,
        label="oracle gate (perfect binary predictor)",
    )
    # Headroom: the region between the chance curve and the oracle's quality level.
    if xs:
        ax.fill_between(
            xs,
            ys,
            float(oracle["AvgPerf%"]),
            color="#B71C1C",
            alpha=0.12,
            label="headroom a real gate must earn",
        )
    ax.set_xlabel("total naive cost over the suite (USD)", fontsize=9)
    ax.set_ylabel("pass rate over the strategy's scored set (%)", fontsize=9)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.92)
    return Annotations(
        subtitle_facts=(
            f"n={sum(1 for r in curve) + 3} rows · {len(curve)} f-values + oracle + endpoints",
            f"oracle gate ties f=0 exactly (${float(oracle['TotalCost']):.4f}, "
            f"{float(oracle['AvgPerf%']):.1f}%) — zero prediction headroom at equal quality",
        ),
        notes=(
            "A perfect binary gate ties Session-Cascade exactly — same cost, same pass set: the "
            "ladder already pays one cheap attempt on every cheap-solving task, so routing those "
            "cheap-direct is cost-identical and the hard tail still climbs the same ladder. That "
            "zero headroom is AT EQUAL QUALITY: on the raw-cost axis a few f>0 points are cheaper "
            "than f=0 only at a lower pass rate (f=0.3 at ~$1.17 vs f=0's $1.29, ~92% vs 100%) — "
            "a quality tradeoff the gate would buy with misroutes, not prediction headroom. The "
            "headroom that is zero lives in the ladder ENTRY point, not in the binary decision.",
            "The pass-rate axis has a growing denominator: cheap-direct tasks are always scorable, "
            "while the ladder's hard tail (42 tasks ending on censored frontier cells) is "
            "unscorable, so n_scored rises from 142 at f=0 to 184 at f=1. The dashed line removes "
            "that drift by scoring every row on Session-Cascade's own 142-task set.",
        ),
        limitations=(
            "The completed matrix's hard tail is censored (42 of 45 cheap-failing tasks land on "
            "non-observation frontier cells), so the scored curve measures the gate's behaviour on "
            "the solvable majority, not its cost on the hard tail.",
            "The f-grid uses a deterministic per-task draw; a different draw moves the curve "
            "within the same mixture envelope.",
        ),
    )


def _figure(rows: list[dict], path: Path, matrix: dict, tasks: list[str]) -> None:
    spec = FigureSpec(
        title="The binary gate's curve: f=0 is Session-Cascade, f=1 is Always-Cheap, oracle = f=0",
        subtitle="One decision per session boundary (cache-safe); the ladder re-serves one model.",
        reading=(
            "The mixture line of a chance-level gate sweeping the fraction f of tasks sent "
            "cheap-direct. The perfect binary gate (oracle) lands exactly on the f=0 point."
        ),
        goal=(
            "A real difficulty score must move a point ABOVE the chance line toward the oracle; "
            "the gap between the curve and the oracle's quality level is the headroom it has to "
            "earn, and the oracle's position at f=0 says the gate cannot save cost AT EQUAL "
            "QUALITY — an f>0 point below f=0's cost always pays a lower pass rate."
        ),
        definitions=(
            ("f", "fraction of tasks the score-free placeholder gate sends cheap-direct"),
            ("cheap-direct", "one attempt on the cheapest model, no ladder, one session"),
            (
                "oracle gate",
                "a perfect binary predictor: cheap-direct exactly the tasks the cheap model solves",
            ),
        ),
        notes=("Offline eval over the completed matrix; no embeddings, no live API.",),
    )
    plot_frame.render(
        path,
        spec,
        lambda ax: _draw_curve(ax, rows),
        size=plot_frame.WIDE,
        provenance=Provenance(
            generator=_GENERATOR,
            data_digest=corpus_digest(matrix, tasks),
            manifest=path.parent / "figures.json",
        ),
    )


def _report(rows: list[dict]) -> None:
    curve = sorted((r for r in rows if r["kind"] == "curve"), key=lambda r: r["f"])
    sc = next(r for r in rows if r["kind"] == "baseline" and r["f"] == "0.0")
    oracle = next(r for r in rows if r["kind"] == "oracle")
    print("\n=== PREDICT-THEN-CASCADE CURVE (naive cost, completed matrix) ===")
    print(
        f"  {'f':>4} {'n_scored':>8} {'n_pass':>6} {'AvgPerf%':>8} {'TotalCost':>10} "
        f"{'fixed_denom%':>11}"
    )
    for r in curve:
        print(
            f"  {r['f']:>4} {r['n_tasks']:>8} {r['n_pass']:>6} {r['AvgPerf%']:>8.2f} "
            f"{float(r['TotalCost']):>10.4f} {r['fixed_denom_rate']:>11.2f}"
        )
    print(f"  f=0 = Session-Cascade : pass {sc['AvgPerf%']:.2f}% at ${float(sc['TotalCost']):.4f}")
    coincides = abs(float(oracle["TotalCost"]) - float(sc["TotalCost"])) < 1e-9
    print(
        f"  oracle gate (perfect) : pass {oracle['AvgPerf%']:.2f}% at "
        f"${float(oracle['TotalCost']):.4f} — "
        f"{'COINCIDES with f=0' if coincides else 'DIFFERS from f=0'}"
    )
    # Precise headline: zero headroom is AT EQUAL QUALITY. On the raw-cost axis some f>0 points
    # are cheaper than f=0 only at a lower pass rate — a quality tradeoff, not prediction headroom.
    cheapest = min((r for r in curve), key=lambda r: float(r["TotalCost"]))
    if float(cheapest["TotalCost"]) < float(sc["TotalCost"]) - 1e-9:
        print(
            f"  headline: the oracle ties f=0 exactly at {oracle['AvgPerf%']:.1f}% and "
            f"${float(oracle['TotalCost']):.4f} — zero headroom AT EQUAL QUALITY. On the raw-cost "
            f"axis f={cheapest['f']} is ${float(cheapest['TotalCost']):.4f}, cheaper than f=0 "
            f"only at {cheapest['AvgPerf%']:.1f}% pass — a quality tradeoff, not prediction "
            f"headroom."
        )
    else:
        print(
            f"  headline: the oracle ties f=0 exactly at {oracle['AvgPerf%']:.1f}% and "
            f"${float(oracle['TotalCost']):.4f} — zero headroom at equal quality, and no f>0 "
            f"point undercuts f=0's raw cost."
        )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="benchmark/benchmark.yaml")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output", default=str(_DEFAULT_CSV))
    ap.add_argument("--plot", default=str(_DEFAULT_PLOT))
    return ap.parse_args()


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    """Compute the gate's cost/quality curve and write the CSV + figure."""
    config.load(config_path)
    args = _parse_args()
    if args.config != config_path:
        config.load(args.config)

    matrix = config.load_matrix(Path(args.matrix) if args.matrix else None)
    tasks = config.sample_tasks(
        sorted(matrix["results"].keys()), seed=config.benchmark_params().get("seed", 42)
    )
    print(f"Loaded {len(tasks)} tasks from the matrix input.")
    rows = _evaluate(matrix, tasks)
    _assert_degeneration(rows)

    out = Path(args.output)
    write_csv(rows, out)
    print(f"CSV written to {out}")
    if args.plot:
        _figure(rows, Path(args.plot), matrix, tasks)
        print(f"Figure written to {args.plot}")
    _report(rows)


if __name__ == "__main__":
    main()
