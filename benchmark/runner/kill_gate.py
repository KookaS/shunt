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


Decision = tuple[str, str, bool, float, int, int, int, bool]
# Fields: task_id, model_name, passed, cost, in_tok, out_tok, calls, scorable
# ``scorable`` is False when the chosen (task, model) cell was never measured — a
# coverage gap. Such a decision is NOT a real fail@$0 and must be excluded from the
# equal-quality cost comparison, not silently paired as a fake-cheap datapoint.


def _get_outcome(matrix: dict[str, Any], tid: str, model: str) -> dict[str, Any]:
    return matrix.get("results", {}).get(tid, {}).get(model, {})


def _make_decision(tid: str, model: str, outcome: dict[str, Any]) -> Decision:
    # An empty outcome dict means the cell has no measured row (sparse coverage,
    # not a bug — the matrix is intentionally sparse), so the decision is unscorable.
    return (
        tid,
        model,
        outcome.get("pass", False),
        outcome.get("cost", 0.0),
        outcome.get("in_tok", 0),
        outcome.get("out_tok", 0),
        outcome.get("calls", 0),
        bool(outcome),
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
    """Oracle reference arm: cheapest passing model per task (perfect-information
    upper bound). NOT the verdict arm — see :func:`run_kill_gate`.
    """
    model_order = _model_order(pricing)
    decisions: list[Decision] = []
    for tid in task_ids:
        task_results = matrix.get("results", {}).get(tid, {})
        chosen_model = model_order[-1] if model_order else ""

        for model in model_order:
            outcome = task_results.get(model, {})
            if outcome.get("pass", False):
                chosen_model = model
                break
        else:
            outcome = task_results.get(chosen_model, {})

        decisions.append(_make_decision(tid, chosen_model, outcome))
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
                strategy.cascade_scorable,
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


def _scorable_pair(cd: Decision, td: Decision) -> bool:
    """A task pair enters the equal-quality comparison only when BOTH arms landed
    on a measured cell. A coverage gap in either arm makes the pair unscorable."""
    return bool(cd[7]) and bool(td[7])


def _scorable_subset(
    control: list[Decision], test: list[Decision]
) -> tuple[list[Decision], list[Decision]]:
    """Positionally-aligned pairs where BOTH arms landed on a measured cell."""
    # The cache-aware cost basis must drop unscorable ($0 coverage-gap) cells exactly
    # like the bootstrap pairing does: a task measured in one arm but not the other
    # biases the cache-aware ratio and can flip the verdict (see run_kill_gate).
    pairs = [(cd, td) for cd, td in zip(control, test, strict=True) if _scorable_pair(cd, td)]
    return [cd for cd, _ in pairs], [td for _, td in pairs]


def bootstrap_cost_delta(
    control: list[Decision],
    test: list[Decision],
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    pairs = [
        (td[3], cd[3])
        for cd, td in zip(control, test, strict=True)
        if _scorable_pair(cd, td) and cd[2] == td[2]
    ]
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
    scorable = [(cd, td) for cd, td in zip(control, test, strict=True) if _scorable_pair(cd, td)]
    n = len(scorable)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    control_pass = [1 if cd[2] else 0 for cd, _ in scorable]
    test_pass = [1 if td[2] else 0 for _, td in scorable]
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
        (td[3], cd[3])
        for cd, td in zip(control, test, strict=True)
        if _scorable_pair(cd, td) and cd[2] == td[2] and cd[3] > 0
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


def _cache_aware_ratio(
    cache_control: tuple[float, float], cache_router: tuple[float, float]
) -> float:
    """router cache-aware cost / control cache-aware cost. A zero-cost control
    yields +inf, which the guard treats as 'not cheaper' (conservative)."""
    control_cache = cache_control[1]
    router_cache = cache_router[1]
    return router_cache / control_cache if control_cache else float("inf")


def decide_verdict(
    cost_delta: dict[str, float],
    pr_delta: dict[str, float],
    verifier_threshold: float,
    n: int,
    cache_aware_ratio: float,
) -> tuple[int, str]:
    # The cost_delta / cost_ratio bootstrap CIs are on NAIVE per-task cost. The
    # kill-gate criterion is "beat fixed-frontier-WITH-caching", so a naive PASS is
    # gated below by a hard cache-aware guard. We deliberately do NOT bootstrap a CI
    # on cache-aware cost: cache savings are adjacency-dependent (they depend on
    # which model immediately preceded which), and paired bootstrap resampling
    # reshuffles the task sequence, destroying the very adjacency the savings are
    # computed from \u2014 a resampled "cache-aware cost" is not a well-defined
    # statistic. Hence the guard is a point-estimate comparison, applied only in the
    # PASS branch so it can only ever make the gate STRICTER, never spuriously fail
    # a genuinely-cheaper router.
    if cost_delta["ci_lower"] < 0 < cost_delta["ci_upper"]:
        return (2, f"INCONCLUSIVE \u2014 extend N by 10 (current N={n})")

    if cost_delta["ci_upper"] < 0:
        quality_ok = pr_delta["ci_lower"] >= -verifier_threshold
        if not quality_ok:
            return (1, "FAIL \u2014 Shunt pass rate too low vs baseline")
        if cache_aware_ratio >= 1.0:
            return (
                1,
                f"FAIL \u2014 cheaper on naive cost but not once caching is priced "
                f"(cache-aware ratio {cache_aware_ratio:.4f} \u2265 1.0)",
            )
        return (
            0,
            f"PASS \u2014 Shunt cheaper at equal or better quality "
            f"(cache-aware ratio {cache_aware_ratio:.4f})",
        )

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


def _arm_switches(decisions: list[Decision]) -> int:
    return sum(1 for i in range(1, len(decisions)) if decisions[i][1] != decisions[i - 1][1])


def _format_report(  # noqa: PLR0913
    control: list[Decision],
    router: list[Decision],
    oracle: list[Decision],
    n: int,
    verifier_threshold: float,
    cost_delta: dict[str, float],
    pr_delta: dict[str, float],
    cost_ratio: dict[str, float],
    cache_control: tuple[float, float],
    cache_router: tuple[float, float],
    cache_oracle: tuple[float, float],
    frontier_model: str,
    n_unscorable: int,
    cache_aware_ratio: float,
    decomposition: dict[str, float] | None = None,
) -> str:
    n_actual = len(control)
    control_pass = sum(1 for d in control if d[2])
    router_pass = sum(1 for d in router if d[2])
    oracle_pass = sum(1 for d in oracle if d[2])
    control_switches = _arm_switches(control)
    router_switches = _arm_switches(router)
    oracle_switches = _arm_switches(oracle)

    control_naive, control_cache = cache_control
    router_naive, router_cache = cache_router
    oracle_naive, oracle_cache = cache_oracle

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"KILL GATE REPORT \u2014 Shunt router vs Always-Frontier ({frontier_model})")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Tasks               : {n_actual} (requested: {n})")
    lines.append(f"Verifier threshold  : {verifier_threshold}")
    lines.append(f"Unscorable (cov gap): {n_unscorable}")
    lines.append("")
    lines.append("\u2500" * 72)
    lines.append(
        "  "
        + f"{'Arm':<18} {'Pass Rate':<12} {'Naive Cost':<14}"
        + f" {'Cache-Aware Cost':<18} {'Switches':<10}"
    )
    lines.append("\u2500" * 72)

    def _arm_line(label: str, passes: int, naive: float, cache: float, switches: int) -> str:
        return (
            "  "
            + f"{label:<18} {passes:>2}/{n_actual:<3}"
            + f" ({passes / n_actual * 100:>5.1f}%)  "
            + f"${naive:<10.6f}  ${cache:<14.6f}"
            + f"  {switches:<10}"
        )

    lines.append(_arm_line("Control", control_pass, control_naive, control_cache, control_switches))
    lines.append(
        _arm_line("Router (kNN-casc)", router_pass, router_naive, router_cache, router_switches)
    )
    lines.append(
        _arm_line("Oracle (ref)", oracle_pass, oracle_naive, oracle_cache, oracle_switches)
    )
    lines.append("  (Oracle = perfect-information upper bound; reference only, not the verdict.)")
    if n_unscorable:
        lines.append(
            f"  (Naive/Cache-Aware Cost exclude {n_unscorable} unscorable coverage-gap "
            f"task(s); pass counts are over all {n_actual}.)"
        )

    lines.append("")
    lines.append("Switch-tax detail:")
    items: list[tuple[str, float, float]] = [
        ("Control", control_naive, control_cache),
        ("Router", router_naive, router_cache),
        ("Oracle", oracle_naive, oracle_cache),
    ]
    for label, naive_amt, cache_amt in items:
        switch_tax = naive_amt - cache_amt
        pct = (switch_tax / naive_amt * 100) if naive_amt else 0.0
        lines.append(f"  {label:<20} switch-tax = ${switch_tax:<10.6f} ({pct:.2f}% of naive)")

    control_st = control_naive - control_cache
    router_st = router_naive - router_cache
    cmp = "more" if router_st > control_st else "less"
    delta_st = abs(router_st - control_st)
    lines.append(f"  {'Switch-tax delta':<20} ${delta_st:<10.6f} (router pays {cmp})")

    naive_ratio = router_naive / control_naive if control_naive else float("inf")
    lines.append(
        f"  {'Cost ratio (cache-aware)':<26} {cache_aware_ratio:<8.4f}"
        "  <- REAL kill-gate criterion (< 1.0 to pass)"
    )
    lines.append(f"  {'Cost ratio (naive)':<26} {naive_ratio:<8.4f}")

    lines.append("")
    lines.append("\u2500" * 72)
    lines.append("  Comparison Metrics (90% CI, bootstrap) \u2014 on NAIVE cost")
    lines.append("  (Bootstrap CIs are on naive cost only; cache-aware cost is not")
    lines.append("   bootstrappable \u2014 savings are adjacency-dependent. The cache-aware")
    lines.append("   ratio above is the real gate criterion; naive CIs are diagnostic.)")
    lines.append("\u2500" * 72)
    lines.append("")
    lines.append("  Cost delta per task (router \u2212 control, equal quality, NAIVE):")
    lines.append(f"    Mean    : ${cost_delta['mean']:.6f}")
    lines.append(f"    90% CI  : [${cost_delta['ci_lower']:.6f}, ${cost_delta['ci_upper']:.6f}]")
    lines.append(f"    N eq-q  : {cost_delta['n_eq']}")
    lines.append("")
    lines.append("  Cost ratio per task (router / control, equal quality):")
    lines.append(f"    Mean    : {cost_ratio['mean']:.4f}")
    lines.append(f"    90% CI  : [{cost_ratio['ci_lower']:.4f}, {cost_ratio['ci_upper']:.4f}]")
    lines.append(f"    N eq-q  : {cost_ratio['n_eq']}")
    lines.append("")
    pr_pct = pr_delta["mean"] * 100
    pr_lo = pr_delta["ci_lower"] * 100
    pr_hi = pr_delta["ci_upper"] * 100
    lines.append("  Pass rate delta (router \u2212 control):")
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
    _, verdict_label = decide_verdict(
        cost_delta, pr_delta, verifier_threshold, n, cache_aware_ratio
    )
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

    # The shipped kNN-cascade router is the VERDICT arm: the gate asks whether the
    # real router beats fixed-frontier-with-caching at equal quality. A router error
    # must surface as a real failure — no try/except swallowing it into a warning.
    router = evaluate_knn_cascade(matrix, task_ids)

    # Oracle (cheapest-passing) is a labelled reference upper bound only — it does
    # NOT drive the verdict.
    oracle = evaluate_test(matrix, task_ids, pricing)

    cost_delta = bootstrap_cost_delta(control, router, n_iterations=n_iterations)
    pr_delta = bootstrap_pass_rate_delta(control, router, n_iterations=n_iterations)
    cost_ratio = bootstrap_cost_ratio(control, router, n_iterations=n_iterations)

    # Cache-aware costs gate the PASS branch, so they MUST be measured over the same
    # scorable-pair set as the bootstraps — a $0 coverage-gap cell in either arm would
    # otherwise contaminate the ratio (see _scorable_subset).
    scorable_control, scorable_router = _scorable_subset(control, router)
    cache_control = compute_cache_costs(scorable_control, pricing)
    cache_router = compute_cache_costs(scorable_router, pricing)
    cache_oracle = compute_cache_costs(oracle, pricing)
    cache_aware_ratio = _cache_aware_ratio(cache_control, cache_router)

    decomposition = _compute_cost_decomposition(control, router)

    n_unscorable = sum(
        1 for cd, rd in zip(control, router, strict=True) if not _scorable_pair(cd, rd)
    )

    report = _format_report(
        control=control,
        router=router,
        oracle=oracle,
        n=len(task_ids),
        verifier_threshold=verifier_threshold,
        cost_delta=cost_delta,
        pr_delta=pr_delta,
        cost_ratio=cost_ratio,
        cache_control=cache_control,
        cache_router=cache_router,
        cache_oracle=cache_oracle,
        frontier_model=frontier_model,
        n_unscorable=n_unscorable,
        cache_aware_ratio=cache_aware_ratio,
        decomposition=decomposition,
    )

    exit_code, _ = decide_verdict(
        cost_delta, pr_delta, verifier_threshold, len(task_ids), cache_aware_ratio
    )
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
