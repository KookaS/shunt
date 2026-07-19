#!/usr/bin/env python3
"""Kill-gate automation: compare Shunt routing (blind cascade) against the
fixed-frontier baseline on N tasks (cost-at-equal-quality + pass-rate deltas,
bootstrap CIs). Exit: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (CI crosses zero).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

from benchmark import config
from benchmark.routing.metrics import compute_cost_decomposition as _compute_cost_decomposition
from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------


def _cost_per_1m_from_config(model: str) -> float:
    return config.cost_per_1m(model)


def _input_share_from_config(model: str) -> float:
    pricing = config.enabled_pricing()
    p = pricing.get(model, {})
    inp: float = p.get("input", 0)
    out: float = p.get("output", 0)
    total = inp + out
    return inp / total if total > 0 else 0.3


def _model_order(pricing: dict[str, dict]) -> list[str]:
    models = {k: v for k, v in pricing.items() if isinstance(v, dict)}
    return sorted(models, key=lambda m: config.cost_per_1m(m, models))


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


def select_tasks(matrix: dict[str, Any], n: int, seed: int = 42) -> list[str]:
    all_tasks = sorted(matrix.get("results", {}).keys())
    rng = random.Random(seed)
    rng.shuffle(all_tasks)
    return all_tasks[: min(n, len(all_tasks))]


# ---------------------------------------------------------------------------
# Arm evaluation
# ---------------------------------------------------------------------------


Decision = tuple[str, str, bool, float, int, int, int]
# Fields: task_id, model_name, passed, cost, in_tok, out_tok, calls


def _get_outcome(matrix: dict[str, Any], tid: str, model: str) -> dict[str, Any]:
    return matrix.get("results", {}).get(tid, {}).get(model, {})


def _make_decision(tid: str, model: str, outcome: dict[str, Any]) -> Decision:
    return (
        tid,
        model,
        outcome.get("pass", False),
        outcome.get("cost", 0.0),
        outcome.get("in_tok", 0),
        outcome.get("out_tok", 0),
        outcome.get("calls", 0),
    )


def evaluate_control(
    matrix: dict[str, Any], task_ids: list[str], frontier_model: str
) -> list[Decision]:
    """Control arm: fixed frontier model on all tasks."""
    decisions: list[Decision] = []
    for tid in task_ids:
        outcome = _get_outcome(matrix, tid, frontier_model)
        decisions.append(_make_decision(tid, frontier_model, outcome))
    return decisions


def evaluate_test(
    matrix: dict[str, Any], task_ids: list[str], pricing: dict[str, Any]
) -> list[Decision]:
    """Test arm: cheapest passing model per task (perfect-information baseline)."""
    model_order = _model_order(pricing)
    decisions: list[Decision] = []
    for tid in task_ids:
        task_results = matrix.get("results", {}).get(tid, {})
        chosen_model = model_order[-1] if model_order else ""
        chosen_pass = False

        for model in model_order:
            outcome = task_results.get(model, {})
            if outcome.get("pass", False):
                chosen_model = model
                chosen_pass = True
                chosen_cost = outcome.get("cost", 0.0)
                chosen_in = outcome.get("in_tok", 0)
                chosen_out = outcome.get("out_tok", 0)
                chosen_calls = outcome.get("calls", 0)
                break
        else:
            outcome = task_results.get(chosen_model, {})
            chosen_cost = outcome.get("cost", 0.0)
            chosen_in = outcome.get("in_tok", 0)
            chosen_out = outcome.get("out_tok", 0)
            chosen_calls = outcome.get("calls", 0)

        decisions.append(
            (tid, chosen_model, chosen_pass, chosen_cost, chosen_in, chosen_out, chosen_calls)
        )
    return decisions


def evaluate_knn_cascade(
    matrix: dict[str, Any],
    task_ids: list[str],
    strategy: kNNCascadeStrategy | None = None,
) -> list[Decision]:
    if strategy is None:
        knn = config.knn_params()
        strategy = kNNCascadeStrategy(
            k=knn.get("k", 20),
            max_tries=knn.get("max_tries", 3),
            success_rate_threshold=knn.get("success_rate_threshold", 0.6),
            min_samples=knn.get("min_samples", 3),
        )
    decisions: list[Decision] = []
    for tid in task_ids:
        task_meta = matrix.get("tasks", {}).get(tid, {})
        model = strategy.select(tid, task_meta, matrix)
        outcome = _get_outcome(matrix, tid, model)
        cost = strategy.cascade_total_cost
        decisions.append(
            (
                tid,
                model,
                outcome.get("pass", False),
                cost,
                outcome.get("in_tok", 0) if outcome else 0,
                outcome.get("out_tok", 0) if outcome else 0,
                outcome.get("calls", 0) if outcome else 0,
            )
        )
    return decisions


# ---------------------------------------------------------------------------
# Cache-aware costing
# ---------------------------------------------------------------------------


def compute_cache_costs(
    decisions: list[Decision],
    pricing: dict[str, Any],
    cache_hit_rate: float = 0.9,
    cache_discount: float = 0.5,
) -> tuple[float, float]:
    naive = sum(d[3] for d in decisions)
    if not decisions:
        return (0.0, 0.0)

    savings = 0.0
    prev_model: str | None = None
    for model, cost in ((d[1], d[3]) for d in decisions):
        if prev_model is not None and model == prev_model:
            share = _input_share_from_config(model)
            savings += cost * share * cache_hit_rate * cache_discount
        prev_model = model

    return (naive, naive - savings)


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def bootstrap_cost_delta(
    control: list[Decision],
    test: list[Decision],
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    pairs = [(td[3], cd[3]) for cd, td in zip(control, test, strict=True) if cd[2] == td[2]]
    if not pairs:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_eq": 0}

    n = len(pairs)
    deltas = [t - c for t, c in pairs]
    observed_mean = sum(deltas) / n

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_iterations):
        s = [rng.choice(deltas) for _ in range(n)]
        boot_means.append(sum(s) / n)

    boot_means.sort()
    std = math.sqrt(sum((d - observed_mean) ** 2 for d in deltas) / n) if n > 1 else 0.0

    return {
        "mean": observed_mean,
        "std": std,
        "ci_lower": boot_means[int(0.05 * n_iterations)],
        "ci_upper": boot_means[int(0.95 * n_iterations)],
        "n_eq": n,
    }


def bootstrap_pass_rate_delta(
    control: list[Decision],
    test: list[Decision],
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    n = len(control)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    control_pass = [1 if d[2] else 0 for d in control]
    test_pass = [1 if d[2] else 0 for d in test]
    observed_mean = (sum(test_pass) - sum(control_pass)) / n

    rng = random.Random(seed)
    boot_deltas: list[float] = []
    for _ in range(n_iterations):
        c_s = [rng.choice(control_pass) for _ in range(n)]
        t_s = [rng.choice(test_pass) for _ in range(n)]
        boot_deltas.append((sum(t_s) - sum(c_s)) / n)

    boot_deltas.sort()

    return {
        "mean": observed_mean,
        "std": math.sqrt(
            (sum(test_pass) / n * (1 - sum(test_pass) / n) / n)
            + (sum(control_pass) / n * (1 - sum(control_pass) / n) / n)
        )
        if n > 1
        else 0.0,
        "ci_lower": boot_deltas[int(0.05 * n_iterations)],
        "ci_upper": boot_deltas[int(0.95 * n_iterations)],
    }


def bootstrap_cost_ratio(
    control: list[Decision],
    test: list[Decision],
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    pairs = [
        (td[3], cd[3]) for cd, td in zip(control, test, strict=True) if cd[2] == td[2] and cd[3] > 0
    ]
    if not pairs:
        return {"mean": 1.0, "std": 0.0, "ci_lower": 1.0, "ci_upper": 1.0, "n_eq": 0}

    n = len(pairs)
    ratios = [t / c for t, c in pairs]
    observed_mean = sum(ratios) / n

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_iterations):
        s = [rng.choice(ratios) for _ in range(n)]
        boot_means.append(sum(s) / n)

    boot_means.sort()
    std = math.sqrt(sum((r - observed_mean) ** 2 for r in ratios) / n) if n > 1 else 0.0

    return {
        "mean": observed_mean,
        "std": std,
        "ci_lower": boot_means[int(0.05 * n_iterations)],
        "ci_upper": boot_means[int(0.95 * n_iterations)],
        "n_eq": n,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def decide_verdict(
    cost_delta: dict[str, float],
    pr_delta: dict[str, float],
    verifier_threshold: float,
    n: int,
) -> tuple[int, str]:
    if cost_delta["ci_lower"] < 0 < cost_delta["ci_upper"]:
        return (2, f"INCONCLUSIVE \u2014 extend N by 10 (current N={n})")

    if cost_delta["ci_upper"] < 0:
        quality_ok = pr_delta["ci_lower"] >= -verifier_threshold
        if quality_ok:
            return (0, "PASS \u2014 Shunt cheaper at equal or better quality")
        return (1, "FAIL \u2014 Shunt pass rate too low vs baseline")

    if cost_delta["ci_lower"] > 0:
        return (1, "FAIL \u2014 Shunt is more expensive at equal quality")

    return (
        2,
        f"INCONCLUSIVE \u2014 unexpected CI state "
        f"[{cost_delta['ci_lower']}, {cost_delta['ci_upper']}]",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_report(  # noqa: PLR0913
    control: list[Decision],
    test: list[Decision],
    pricing: dict[str, Any],
    n: int,
    verifier_threshold: float,
    cost_delta: dict[str, float],
    pr_delta: dict[str, float],
    cost_ratio: dict[str, float],
    cache_control: tuple[float, float],
    cache_test: tuple[float, float],
    frontier_model: str,
    decomposition: dict[str, float] | None = None,
    knn_cascade: list[Decision] | None = None,
    cache_knn: tuple[float, float] | None = None,
) -> str:
    n_actual = len(control)
    control_pass = sum(1 for d in control if d[2])
    test_pass = sum(1 for d in test if d[2])
    control_switches = sum(1 for i in range(1, len(control)) if control[i][1] != control[i - 1][1])
    test_switches = sum(1 for i in range(1, len(test)) if test[i][1] != test[i - 1][1])

    control_naive, control_cache = cache_control
    test_naive, test_cache = cache_test

    knn_naive, knn_cache = cache_knn if cache_knn else (0.0, 0.0)
    knn_pass = sum(1 for d in knn_cascade) if knn_cascade else 0
    knn_switches = (
        sum(1 for i in range(1, len(knn_cascade)) if knn_cascade[i][1] != knn_cascade[i - 1][1])
        if knn_cascade and len(knn_cascade) > 1
        else 0
    )

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"KILL GATE REPORT \u2014 Shunt vs Always-Frontier ({frontier_model})")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Tasks               : {n_actual} (requested: {n})")
    lines.append(f"Verifier threshold  : {verifier_threshold}")
    lines.append("")
    lines.append("\u2500" * 72)
    lines.append(
        "  "
        + f"{'Arm':<16} {'Pass Rate':<12} {'Naive Cost':<14}"
        + f" {'Cache-Aware Cost':<18} {'Switches':<10}"
    )
    lines.append("\u2500" * 72)
    lines.append(
        "  "
        + f"{'Control':<16} {control_pass:>2}/{n_actual:<3}"
        + f" ({control_pass / n_actual * 100:>5.1f}%)  "
        + f"${control_naive:<10.6f}  ${control_cache:<14.6f}"
        + f"  {control_switches:<10}"
    )
    lines.append(
        "  "
        + f"{'Test (cascade)':<16} {test_pass:>2}/{n_actual:<3}"
        + f" ({test_pass / n_actual * 100:>5.1f}%)  "
        + f"${test_naive:<10.6f}  ${test_cache:<14.6f}"
        + f"  {test_switches:<10}"
    )
    if knn_cascade is not None:
        lines.append(
            "  "
            + f"{'Test (kNN-casc)':<16} {knn_pass:>2}/{n_actual:<3}"
            + f" ({knn_pass / n_actual * 100:>5.1f}%)  "
            + f"${knn_naive:<10.6f}  ${knn_cache:<14.6f}"
            + f"  {knn_switches:<10}"
        )

    lines.append("")
    lines.append("Switch-tax detail:")
    items: list[tuple[str, float, float]] = [
        ("Control", control_naive, control_cache),
        ("Test", test_naive, test_cache),
    ]
    if knn_cascade is not None:
        items.append(("kNN-casc", knn_naive, knn_cache))
    for label, naive_amt, cache_amt in items:
        switch_tax = naive_amt - cache_amt
        pct = (switch_tax / naive_amt * 100) if naive_amt else 0.0
        lines.append(f"  {label:<20} switch-tax = ${switch_tax:<10.6f} ({pct:.2f}% of naive)")

    control_st = control_naive - control_cache
    test_st = test_naive - test_cache
    cmp = "more" if test_st > control_st else "less"
    delta_st = abs(test_st - control_st)
    lines.append(f"  {'Switch-tax delta':<20} ${delta_st:<10.6f} (test pays {cmp})")

    cache_ratio = test_cache / control_cache if control_cache else float("inf")
    naive_ratio = test_naive / control_naive if control_naive else float("inf")
    lines.append(f"  {'Cost ratio (cache-aware)':<20} {cache_ratio:<8.4f}")
    lines.append(f"  {'Cost ratio (naive)':<20} {naive_ratio:<8.4f}")

    if knn_cascade is not None and control_cache > 0:
        knn_cache_ratio = knn_cache / control_cache
        lines.append(f"  {'kNN-casc ratio (cache)':<20} {knn_cache_ratio:<8.4f}")

    lines.append("")
    lines.append("\u2500" * 72)
    lines.append("  Comparison Metrics (90% CI, bootstrap)")
    lines.append("\u2500" * 72)
    lines.append("")
    lines.append("  Cost delta per task (test \u2212 control, equal quality):")
    lines.append(f"    Mean    : ${cost_delta['mean']:.6f}")
    lines.append(f"    90% CI  : [${cost_delta['ci_lower']:.6f}, ${cost_delta['ci_upper']:.6f}]")
    lines.append(f"    N eq-q  : {cost_delta['n_eq']}")
    lines.append("")
    lines.append("  Cost ratio per task (test / control, equal quality):")
    lines.append(f"    Mean    : {cost_ratio['mean']:.4f}")
    lines.append(f"    90% CI  : [{cost_ratio['ci_lower']:.4f}, {cost_ratio['ci_upper']:.4f}]")
    lines.append(f"    N eq-q  : {cost_ratio['n_eq']}")
    lines.append("")
    pr_pct = pr_delta["mean"] * 100
    pr_lo = pr_delta["ci_lower"] * 100
    pr_hi = pr_delta["ci_upper"] * 100
    lines.append("  Pass rate delta (test \u2212 control):")
    lines.append(f"    Mean    : {pr_delta['mean']:.4f} ({pr_pct:.1f}%)")
    lines.append(
        f"    90% CI  : [{pr_delta['ci_lower']:.4f}, {pr_delta['ci_upper']:.4f}] "
        f"[{pr_lo:.1f}%, {pr_hi:.1f}%]"
    )

    if decomposition and decomposition.get("n_eq_pass", 0) > 0:
        lines.append("")
        lines.append("\u2500" * 72)
        lines.append("  Cost Decomposition (equal-quality tasks where both pass)")
        lines.append("\u2500" * 72)
        lines.append("")
        d = decomposition
        lines.append(f"    Tasks where both pass : {d['n_eq_pass']}")
        lines.append(f"    Direct savings        : ${d['total_direct_saving']:<10.6f}")
        dec = d["price_savings"] + d["volume_savings"] + d["interaction"]
        lines.append(f"    Decomposed savings    : ${dec:<10.6f}")
        lines.append("")
        lines.append(
            f"    Price effect   : ${d['price_savings']:<10.6f}  ({d['price_pct']:>6.2f}%)"
        )
        lines.append(
            f"    Volume effect  : ${d['volume_savings']:<10.6f}  ({d['volume_pct']:>6.2f}%)"
        )
        lines.append(
            f"    Interaction    : ${d['interaction']:<10.6f}  ({d['interaction_pct']:>6.2f}%)"
        )

    lines.append("")
    lines.append("\u2500" * 72)
    lines.append("  DATA CAVEAT: model availability determined by config.yaml.")
    lines.append("  Models absent from config.yaml's `models` list are excluded.")
    lines.append("\u2500" * 72)
    verdict_code, verdict_label = decide_verdict(cost_delta, pr_delta, verifier_threshold, n)
    lines.append(f"  Verdict : {verdict_label}")
    lines.append("\u2500" * 72)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_kill_gate(
    matrix: dict[str, Any],
    pricing: dict[str, Any],
    task_ids: list[str],
    verifier_threshold: float,
    frontier_model: str,
    n_iterations: int = 10000,
) -> tuple[int, str]:
    control = evaluate_control(matrix, task_ids, frontier_model)
    test = evaluate_test(matrix, task_ids, pricing)

    cost_delta = bootstrap_cost_delta(control, test, n_iterations=n_iterations)
    pr_delta = bootstrap_pass_rate_delta(control, test, n_iterations=n_iterations)
    cost_ratio = bootstrap_cost_ratio(control, test, n_iterations=n_iterations)

    cache_control = compute_cache_costs(control, pricing)
    cache_test = compute_cache_costs(test, pricing)

    decomposition = _compute_cost_decomposition(control, test)

    # kNN-cascade arm
    try:
        knn_decisions = evaluate_knn_cascade(matrix, task_ids)
        cache_knn = compute_cache_costs(knn_decisions, pricing)
    except Exception as exc:
        # Optional arm — must not abort the gate, but a dropped arm must be
        # visible, not silent (that silence hid the config.knn_params() bug).
        print(f"WARNING: kNN-cascade arm skipped: {exc!r}", file=sys.stderr)
        knn_decisions = None
        cache_knn = None

    report = _format_report(
        control=control,
        test=test,
        pricing=pricing,
        n=len(task_ids),
        verifier_threshold=verifier_threshold,
        cost_delta=cost_delta,
        pr_delta=pr_delta,
        cost_ratio=cost_ratio,
        cache_control=cache_control,
        cache_test=cache_test,
        frontier_model=frontier_model,
        decomposition=decomposition,
        knn_cascade=knn_decisions,
        cache_knn=cache_knn,
    )

    exit_code, _ = decide_verdict(cost_delta, pr_delta, verifier_threshold, len(task_ids))
    return exit_code, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(config_path: str = "benchmark/config.yaml") -> None:
    config.load(config_path)

    bm = config.benchmark_params()

    ap = argparse.ArgumentParser(
        description="Kill-gate automation: compare Shunt routing vs Always-Frontier baseline."
    )
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None, help="Matrix JSON path")
    ap.add_argument(
        "--pricing", default=None, help="Pricing JSON path (deprecated — uses config.yaml)"
    )
    ap.add_argument("--log", default="benchmark/runner/kill_gate.log", help="Output log path")
    ap.add_argument("--n", type=int, default=bm.get("n_default", 20), help="Number of tasks")
    ap.add_argument(
        "--verifier-confidence",
        type=float,
        default=0.6,
        help="Verifier confidence threshold (default: 0.6)",
    )
    ap.add_argument("--seed", type=int, default=bm.get("seed", 42), help="RNG seed")
    ap.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=bm.get("bootstrap_iterations", 10000),
        help="Bootstrap iterations",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)
        bm = config.benchmark_params()

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    log_path = Path(args.log)

    if not matrix_path.exists():
        print(f"Error: matrix not found: {matrix_path}", file=sys.stderr)
        sys.exit(2)

    matrix = config.load_matrix(matrix_path)

    # Pricing from config, not the model registry
    pricing = config.enabled_pricing()

    frontier = config.frontier_model()
    if frontier is None:
        print("Error: no frontier model available (no enabled models?)", file=sys.stderr)
        sys.exit(2)

    all_tasks = sorted(matrix.get("results", {}).keys())
    all_tasks = config.sample_tasks(all_tasks, seed=args.seed)
    task_ids = select_tasks({"results": {t: {} for t in all_tasks}}, args.n, seed=args.seed)
    if not task_ids:
        print(
            "No results yet — results.csv holds no rows, so there is nothing to gate. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        sys.exit(0)

    if args.verbose:
        print(
            f"Kill gate: {len(task_ids)} tasks, "
            f"frontier={frontier}, "
            f"threshold={args.verifier_confidence}, "
            f"bootstrap={args.bootstrap_iterations} iters"
        )

    exit_code, report = run_kill_gate(
        matrix=matrix,
        pricing=pricing,
        task_ids=task_ids,
        verifier_threshold=args.verifier_confidence,
        frontier_model=frontier,
        n_iterations=args.bootstrap_iterations,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report)
    print(report)
    print(f"Log written to {log_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
