#!/usr/bin/env python3
# The verdict is ALSO emitted as a tracked, deterministic JSON artifact
# (benchmark/runner/kill_gate_verdict.json) — the diffable record of the make-or-break
# verdict, unlike the free-form report log which is gitignored.
"""Kill-gate automation: SHIPPED kNN router vs fixed-frontier baseline on N tasks
(cost-at-equal-quality + pass-rate deltas, bootstrap CIs). Exit: 0 PASS,
1 FAIL or UNTESTED (empty corpus or coverage floor not met), 2 INCONCLUSIVE (CI crosses zero)."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from benchmark import config, validate_results
from benchmark.routing import censoring, summary
from benchmark.routing.cache_cost import (
    CachePrice,
    cache_aware_total,
    cache_cost_is_scoped_to_tasks,
    cache_prices,
)
from benchmark.routing.metrics import (
    _scorable_pair,
)
from benchmark.routing.metrics import (
    compute_cost_decomposition as _compute_cost_decomposition,
)
from benchmark.routing.strategies import BilledAttempt
from benchmark.routing.strategies.knn import kNNStrategy

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------


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

# Pre-registered kill-gate coverage floor (docs/benchmark-design.md §"Kill-gate coverage
# floor"). A verdict may be called only when >= this fraction of BOTH arms' cells are
# MEASURED (real provider calls, non-ERROR integrity). Below it the verdict is UNTESTED —
# not PASS, not FAIL. This is the wall that makes a headline whose decisive cells were all
# imputed fills impossible to publish again.
COVERAGE_FLOOR: float = 0.90


def _get_outcome(matrix: dict[str, Any], tid: str, model: str) -> dict[str, Any]:
    return matrix.get("results", {}).get(tid, {}).get(model, {})


def _make_decision(tid: str, model: str, outcome: dict[str, Any]) -> Decision:
    # An empty outcome dict means the cell has no measured row (sparse coverage,
    # not a bug — the matrix is intentionally sparse), so the decision is unscorable.
    # A CENSORED cell (resource-limit stop, unknown true outcome) is likewise unscorable:
    # scoring it as a clean pass=False would understate the frontier baseline's quality.
    # An IMPUTED cell (completed by the monotone-ladder imputer, `imputed: True`) is
    # also unscorable: its `pass`/`cost` are axiom-implied fills, not a real observation.
    # Treating an imputed cell as scorable is exactly the failure this gate exists to block —
    # publishing a verdict whose decisive cells were mostly imputed fills.
    measured = bool(outcome) and not censoring.is_censored(outcome)
    if measured and outcome.get("imputed") is True:
        measured = False
    return (
        tid,
        model,
        outcome.get("pass", False),
        outcome.get("cost", 0.0),
        outcome.get("in_tok", 0),
        outcome.get("out_tok", 0),
        outcome.get("calls", 0),
        measured,
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


def evaluate_router(
    matrix: dict[str, Any],
    task_ids: list[str],
    strategy: kNNStrategy | None = None,
) -> list[Decision]:
    """The PRE-REGISTERED verdict arm: the single-shot kNN selection rule, one decision/task."""
    # It used to be the within-task kNN cascade, which `LIVE_STRATEGIES` rejects at boot — the
    # gate was adjudicating "should we ship this" on a strategy the product will not run. The
    # within-task cascade's mid-session verify-and-escalate is not cache-safe; see
    # benchmark/routing/strategy_class.py for the blocker and its path to live.
    #
    # AND IT IS STILL NOT THE SHIPPED DEFAULT. `router.strategy` defaults to `session_cascade`
    # — the cheap-start ladder, which consults no neighbours — and the nearest selectable kNN
    # configuration, `knn_cascade`, is this pick PLUS the escalation ladder. Either way the
    # pre-registered arm adjudicates a configuration no operator can select. That is a
    # pre-existing defect the rename exposed, not one it created: the ladder has been on by
    # default since escalation shipped, so this arm has never been the shipped router. It is
    # deliberately NOT repointed, because moving a
    # pre-registered verdict arm after seeing the data rewrites the registered test. The default
    # is published beside it instead, on its own clearly-marked non-pre-registered row — see
    # benchmark/routing/figures/kill_gate.py `_default_basis`.
    if strategy is None:
        knn = config.knn_params()
        strategy = kNNStrategy(
            k=knn.get("k", 20),
            success_rate_threshold=knn.get("success_rate_threshold", 0.6),
            min_samples=knn.get("min_samples", 3),
        )
    decisions: list[Decision] = []
    for tid in task_ids:
        task_meta = matrix.get("tasks", {}).get(tid, {})
        model = strategy.select(tid, task_meta, matrix)
        decisions.append(_make_decision(tid, model, _get_outcome(matrix, tid, model)))
    return decisions


# ---------------------------------------------------------------------------
# Cache-aware costing
# ---------------------------------------------------------------------------


def _attempts_by_task(decisions: list[Decision]) -> dict[str, list[BilledAttempt]]:
    """Group decisions into per-task attempt sequences (a task is one session)."""
    # A `Decision` collapses a task to (id, model, passed, cost) and carries no token counts, so
    # these records declare zero tokens. `cache_aware_total` reads model and cost only, and a
    # zero-token record is inadmissible to the context cost model by construction — the gate is
    # never the path that publishes a context bracket.
    attempts: dict[str, list[BilledAttempt]] = {}
    for d in decisions:
        attempts.setdefault(d[0], []).append(BilledAttempt(model=d[1], cost=d[3]))
    return attempts


def _per_task_cache_costs(
    decisions: list[Decision], prices: dict[str, CachePrice]
) -> dict[str, float]:
    """Per-task cache-aware cost — one float per task, scoped to that task's own attempts."""
    return {
        tid: cache_aware_total(attempts, prices)
        for tid, attempts in _attempts_by_task(decisions).items()
    }


def compute_cache_costs(
    decisions: list[Decision],
    pricing: dict[str, Any],
    prices: dict[str, CachePrice] | None = None,
) -> tuple[float, float]:
    """(naive, cache-aware) total cost — scoped PER TASK, matching ``summary._cache_aware_cost``."""
    # The model itself lives in `benchmark.routing.cache_cost`, shared with the routing summary:
    # a cost that is cache-aware in this gate and cache-blind in the published strategy table is
    # two different verdicts wearing one name.
    #
    # SCOPING IS THE LOAD-BEARING CHOICE. A decision here is ONE attempt at ONE task, and a task
    # is one session — so the cache discount may only fire WITHIN a task (a cascade re-serving the
    # same model on consecutive attempts), never BETWEEN independent tasks that happen to route to
    # the same model. Flattening every decision into one attempt sequence (as this function once
    # did) handed a fixed-model baseline a discount on every task after the first: the control
    # arm's 20 consecutive kimi-k3 decisions banked 19 spurious discounts, deflating control cost
    # and inflating `cache_aware_ratio` — the gate's PASS criterion. Measured on this corpus the
    # flat version cut Always-Frontier from $88.61 to $23.51, exactly the treatment
    # `summary._cache_aware_cost` rejects in its own comment. Per-task scoping means each task's
    # single attempt carries no discount and the ratio collapses to the naive one.
    if not decisions:
        return (0.0, 0.0)
    if prices is None:
        prices = cache_prices(sorted({d[1] for d in decisions} | set(pricing)))
    return (
        sum(d[3] for d in decisions),
        sum(_per_task_cache_costs(decisions, prices).values()),
    )


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


# ``_scorable_pair`` is imported from ``benchmark.routing.metrics`` — the shared predicate
# the cost decomposition routes through, so the pairing and the decomposition can never
# drift apart.


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
    if n_iterations < 1:
        raise ValueError(f"a bootstrap needs at least 1 iteration; got {n_iterations}")
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
    """PAIRED bootstrap: one task resample per draw, both arms read at those tasks."""
    if n_iterations < 1:
        raise ValueError(f"a bootstrap needs at least 1 iteration; got {n_iterations}")
    scorable = [(cd, td) for cd, td in zip(control, test, strict=True) if _scorable_pair(cd, td)]
    n = len(scorable)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    # Both arms are scored on the SAME task set, so the comparison is paired and the
    # resampling unit is the TASK, not the arm. Resampling each arm independently drops
    # the strong positive across-arm correlation (same tasks, same difficulty) and
    # inflates the CI several-fold — on this gate that is a false-negative risk, not a
    # cosmetic one. Because the statistic is a difference of means,
    # mean(test[idx]) - mean(control[idx]) == mean(delta[idx]) exactly, so resampling the
    # per-task deltas IS the task-index resample — the same shape the sibling
    # `bootstrap_cost_delta` already uses.
    deltas = [(1 if td[2] else 0) - (1 if cd[2] else 0) for cd, td in scorable]
    observed_mean = sum(deltas) / n

    rng = random.Random(seed)
    boot_deltas: list[float] = []
    for _ in range(n_iterations):
        s = [rng.choice(deltas) for _ in range(n)]
        boot_deltas.append(sum(s) / n)

    boot_deltas.sort()

    # Standard ERROR of the paired mean delta (the previous formula summed the two arms'
    # independent binomial variances — the same unpaired assumption as the loop above).
    var = sum((d - observed_mean) ** 2 for d in deltas) / n if n > 1 else 0.0

    return {
        "mean": observed_mean,
        "std": math.sqrt(var / n),
        "ci_lower": boot_deltas[int(0.05 * n_iterations)],
        "ci_upper": boot_deltas[int(0.95 * n_iterations)],
    }


def bootstrap_cost_ratio(
    control: list[Decision],
    test: list[Decision],
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    if n_iterations < 1:
        raise ValueError(f"a bootstrap needs at least 1 iteration; got {n_iterations}")
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


def bootstrap_cache_aware_ratio(
    control: list[Decision],
    test: list[Decision],
    pricing: dict[str, Any],
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired-task bootstrap CI on the cache-aware ratio (router / control)."""
    # The ratio is the gate's criterion, and it carries a CI because cache cost is SCOPED PER
    # TASK: one task is one session, so a whole-task resample keeps each task's own attempt
    # sequence intact and the resampled ratio is the statistic it is named after. The shared
    # predicate ``cache_cost_is_scoped_to_tasks`` (the same check metrics._assert_cache_cost_scoping
    # runs) re-fires the old refusal as a raise if a future re-flattening ever makes the
    # statistic order-dependent again.
    scorable_control, scorable_test = _scorable_subset(control, test)
    prices = cache_prices(
        sorted({d[1] for d in scorable_control} | {d[1] for d in scorable_test} | set(pricing))
    )
    control_attempts = _attempts_by_task(scorable_control)
    test_attempts = _attempts_by_task(scorable_test)
    tids = sorted(control_attempts)
    if not tids:
        return {"mean": 1.0, "std": 0.0, "ci_lower": 1.0, "ci_upper": 1.0, "n_eq": 0}
    for label, attempts in (("control", control_attempts), ("router", test_attempts)):
        if not cache_cost_is_scoped_to_tasks(tids, attempts, prices):
            raise RuntimeError(
                f"refusing to bootstrap the cache-aware ratio: {label} cache cost is "
                "not scoped per task (a task's cost changed with the set/order of the "
                "other tasks, so a whole-task resample does not preserve the statistic)"
            )

    control_cost = {t: cache_aware_total(control_attempts[t], prices) for t in tids}
    test_cost = {t: cache_aware_total(test_attempts[t], prices) for t in tids}
    control_total = sum(control_cost.values())
    observed = sum(test_cost.values()) / control_total if control_total else float("inf")
    n = len(tids)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_iterations):
        sample = rng.choices(tids, k=n)
        denom = sum(control_cost[t] for t in sample)
        if denom <= 0:
            continue
        draws.append(sum(test_cost[t] for t in sample) / denom)
    draws.sort()
    if not draws:
        return {"mean": observed, "std": 0.0, "ci_lower": observed, "ci_upper": observed, "n_eq": n}
    draw_mean = sum(draws) / len(draws)
    return {
        "mean": observed,
        "std": math.sqrt(sum((d - draw_mean) ** 2 for d in draws) / len(draws)),
        "ci_lower": draws[int(0.05 * len(draws))],
        "ci_upper": draws[int(0.95 * len(draws))],
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


def _cache_ratio_text(cache_aware_ratio: float, cache_aware_ci: dict[str, float] | None) -> str:
    """The criterion line as printed in verdict labels: the ratio plus its CI when held."""
    text = f"cache-aware ratio {cache_aware_ratio:.4f}"
    if cache_aware_ci is not None:
        text += f" CI [{cache_aware_ci['ci_lower']:.4f}, {cache_aware_ci['ci_upper']:.4f}]"
    return text


def decide_verdict(
    cost_delta: dict[str, float],
    pr_delta: dict[str, float],
    verifier_threshold: float,
    n: int,
    cache_aware_ratio: float,
    cache_aware_ci: dict[str, float] | None = None,
) -> tuple[int, str]:
    # The cost_delta / cost_ratio bootstrap CIs are on NAIVE per-task cost. The
    # kill-gate criterion is "beat fixed-frontier-WITH-caching", so a naive PASS is
    # gated below by a hard cache-aware guard — and the cache-aware ratio now carries
    # its own CI. It is bootstrappable because cache cost is SCOPED PER TASK: one task
    # is one session, so a whole-task resample keeps each task's own attempt sequence
    # intact, within-task adjacency survives, and a resampled ratio is the statistic it
    # is named after. The old refusal was a consequence of the flattened cross-task
    # sequence, which the per-task scoping fix removed; the shared predicate
    # ``cache_cost_is_scoped_to_tasks`` (the check ``metrics._assert_cache_cost_scoping``
    # runs too) re-fires that refusal as a raise if scoping is ever broken again. The
    # guard remains a point-estimate comparison applied only in the PASS branch, so it
    # can only ever make the gate STRICTER — the CI is published alongside it, not used
    # to change the criterion.
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
                f"({_cache_ratio_text(cache_aware_ratio, cache_aware_ci)} \u2265 1.0)",
            )
        return (
            0,
            f"PASS \u2014 Shunt cheaper at equal or better quality "
            f"({_cache_ratio_text(cache_aware_ratio, cache_aware_ci)})",
        )

    if cost_delta["ci_lower"] > 0:
        return (1, "FAIL \u2014 Shunt is more expensive at equal quality")

    lo, hi = cost_delta["ci_lower"], cost_delta["ci_upper"]
    # A CI bound landing exactly on 0.0 is ROUTINE for a discrete paired bootstrap
    # over few tasks (a resample where no task differs), not an internal error. Name
    # each boundary state for what it is rather than reporting "unexpected".
    if lo == 0.0 and hi == 0.0:
        return (2, f"INCONCLUSIVE \u2014 zero measured cost difference (N={n})")
    if lo == 0.0:
        return (
            2,
            f"INCONCLUSIVE \u2014 no saving demonstrated: cost-delta CI [0.0, {hi}] lies "
            f"entirely \u2265 0, i.e. Shunt is weakly MORE expensive (the boundary of FAIL); "
            f"extend N by 10 (current N={n})",
        )
    if hi == 0.0:
        return (
            2,
            f"INCONCLUSIVE \u2014 weakly cheaper only: cost-delta CI [{lo}, 0.0] lies "
            f"entirely \u2264 0 but touches zero, so the saving is not significant; "
            f"extend N by 10 (current N={n})",
        )

    return (2, f"INCONCLUSIVE \u2014 non-finite CI state [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _arm_switches(decisions: list[Decision]) -> int:
    return sum(1 for i in range(1, len(decisions)) if decisions[i][1] != decisions[i - 1][1])


# The TRACKED verdict artifact (benchmark/runner/kill_gate_verdict.json). It is the
# diffable, deterministic record of the make-or-break verdict — deliberately NOT the
# free-form report (kill_gate.log, gitignored via `*.log`), because a verdict nobody can
# diff is a verdict nobody can audit. It must be a pure function of the committed inputs:
# no timestamps, no absolute paths, no wall-clock state — two regenerations over the same
# committed corpus must be byte-identical — that is the contract the tracked file holds.


def _ci_slice(ci: dict[str, float]) -> dict[str, float]:
    """The reviewer-facing subset of a bootstrap result (drops the derived std)."""
    return {"mean": ci["mean"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"]}


def _cache_leg(costs: tuple[float, float]) -> dict[str, float]:
    return {"naive": costs[0], "cache_aware": costs[1]}


def _build_verdict_payload(
    *,
    exit_code: int,
    verdict: str,
    frontier_model: str,
    n: int,
    n_scorable: int,
    n_unscorable: int,
    verifier_threshold: float,
    task_seed: int,
    bootstrap_iterations: int,
    cache_aware_ratio: float,
    cache_aware_ratio_ci: dict[str, float],
    naive_ratio: float,
    control_pass: int,
    router_pass: int,
    cost_delta: dict[str, float],
    cost_ratio: dict[str, float],
    pass_rate_delta: dict[str, float],
    cache_control: tuple[float, float],
    cache_router: tuple[float, float],
    cache_oracle: tuple[float, float],
    coverage_floor: float,
    control_coverage: float,
    router_coverage: float,
    n_control_measured: int,
    n_router_measured: int,
    coverage_tripped: bool,
    impute_enabled: bool,
) -> dict[str, Any]:
    """Build the deterministic, tracked verdict record. Pure function of the inputs — a
    reviewer diffs this file to see a verdict move, so it carries no run-local noise."""
    return {
        "verdict": verdict,
        "exit_code": exit_code,
        "frontier_model": frontier_model,
        "n": n,
        "n_scorable": n_scorable,
        "n_unscorable": n_unscorable,
        "verifier_threshold": verifier_threshold,
        "task_seed": task_seed,
        "bootstrap_iterations": bootstrap_iterations,
        "cache_aware_ratio": cache_aware_ratio,
        "cache_aware_ratio_ci": _ci_slice(cache_aware_ratio_ci),
        "naive_ratio": naive_ratio,
        "pass_rate": {"control": control_pass, "router": router_pass},
        "cost_delta": _ci_slice(cost_delta),
        "cost_ratio": _ci_slice(cost_ratio),
        "pass_rate_delta": _ci_slice(pass_rate_delta),
        "cache_control": _cache_leg(cache_control),
        "cache_router": _cache_leg(cache_router),
        "cache_oracle": _cache_leg(cache_oracle),
        "coverage": {
            "floor": coverage_floor,
            "control_measured": n_control_measured,
            "control_coverage": control_coverage,
            "router_measured": n_router_measured,
            "router_coverage": router_coverage,
            "tripped": coverage_tripped,
        },
        "impute_enabled": impute_enabled,
    }


def write_verdict_artifact(payload: dict[str, Any], path: Path) -> None:
    """Serialize the verdict record deterministically: sorted keys, fixed float
    formatting, trailing newline — two identical payloads produce identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    cache_aware_ci: dict[str, float] | None = None,
    decomposition: dict[str, float] | None = None,
    *,
    impute_enabled: bool = False,
    coverage_tripped: bool = False,
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
    if coverage_tripped:
        # The pre-registered floor (docs/benchmark-design.md 'Kill-gate coverage floor'): a
        # verdict is quotable only when >= 90% of BOTH arms' cells are MEASURED (real provider
        # calls, not imputed fills). Below it the gate has no verdict — it has a coverage gap.
        n_control_measured = sum(1 for d in control if d[7])
        n_router_measured = sum(1 for d in router if d[7])
        c_cov = n_control_measured / n_actual if n_actual else 0.0
        r_cov = n_router_measured / n_actual if n_actual else 0.0
        lines.append(
            "KILL GATE VERDICT: UNTESTED \u2014 coverage floor not met. "
            f"Pre-registered floor: {COVERAGE_FLOOR:.0%} measured on BOTH arms. "
            f"Control ({frontier_model}): {c_cov:.1%} measured ({n_control_measured}/{n_actual}). "
            f"Router: {r_cov:.1%} measured ({n_router_measured}/{n_actual}). Below the floor no "
            "PASS/FAIL may be quoted \u2014 this is not a verdict, it is a coverage gap. Close "
            "the measured-cell gap (see docs/benchmark-design.md 'Kill-gate coverage floor') "
            "before re-running."
        )
    # WHO handled the censored cells. With impute.enabled, main() completes the matrix FIRST, so
    # a censored cell is imputed and the censoring guard below is fallback-only; with it off the
    # guard is the whole mechanism. State which, so a reader cannot guess why unscorable was 0.
    impute_note = (
        "impute.enabled=true: the matrix was completed by the monotone-ladder imputer before "
        "scoring, so resource-limit cells are imputed; the censoring guard is the fallback"
        if impute_enabled
        else "impute.enabled=false: cells were scored raw, so the censoring guard excluded "
        "every resource-limit (CENSORED) cell"
    )
    lines.append(f"Censoring guard     : {impute_note}")
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
        _arm_line("Router (kNN, live)", router_pass, router_naive, router_cache, router_switches)
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
    if cache_aware_ci is not None:
        lines.append(
            f"  {'90% CI (bootstrap)':<26} "
            f"[{cache_aware_ci['ci_lower']:.4f}, {cache_aware_ci['ci_upper']:.4f}]"
        )
    lines.append(f"  {'Cost ratio (naive)':<26} {naive_ratio:<8.4f}")

    lines.append("")
    lines.append("\u2500" * 72)
    lines.append("  Comparison Metrics (90% CI, bootstrap) \u2014 naive and cache-aware")
    lines.append("  (Cache cost is scoped PER TASK \u2014 one task is one session \u2014 so a")
    lines.append("   whole-task resample keeps each task's own attempt sequence intact,")
    lines.append("   within-task adjacency survives, and the cache-aware ratio IS")
    lines.append("   bootstrappable. The shared per-task scoping guard re-fires the old")
    lines.append("   refusal as a raise if a future re-flattening ever breaks that.")
    lines.append("   The cache-aware ratio above is the real gate criterion; naive CIs")
    lines.append("   are diagnostic.)")
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
    lines.append("  DATA CAVEAT: model availability determined by benchmark.yaml.")
    lines.append("  Models absent from benchmark.yaml's `models` list are excluded.")
    lines.append("\u2500" * 72)
    if coverage_tripped:
        lines.append("  Verdict : UNTESTED (coverage floor not met — see banner above)")
        lines.append("\u2500" * 72)
        lines.append("")
        return "\n".join(lines)
    _, verdict_label = decide_verdict(
        cost_delta, pr_delta, verifier_threshold, n, cache_aware_ratio, cache_aware_ci
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
    *,
    verdict_path: Path | None = None,
    task_seed: int = 42,
) -> tuple[int, str]:
    control = evaluate_control(matrix, task_ids, frontier_model)

    # The shipped kNN router is the VERDICT arm: the gate asks whether the LIVE router
    # beats fixed-frontier-with-caching at equal quality. A router error must surface as
    # a real failure — no try/except swallowing it into a warning.
    router = evaluate_router(matrix, task_ids)

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
    # The cache-aware ratio carries its own paired-task bootstrap CI: cache cost is scoped
    # per task, so a whole-task resample is the statistic it is named after. The bootstrap
    # re-fires the refusal (raises) if a future re-flattening ever breaks that scoping.
    cache_aware_ci = bootstrap_cache_aware_ratio(
        control, router, pricing, n_iterations=n_iterations
    )

    decomposition = _compute_cost_decomposition(control, router)

    n_unscorable = sum(
        1 for cd, rd in zip(control, router, strict=True) if not _scorable_pair(cd, rd)
    )

    # Read live so the report cannot contradict the matrix that was actually scored: main()
    # completes the matrix when impute is on, so whether the censoring guard or the imputer
    # handled a censored cell is a property of the config at run time, not of this file.
    impute_enabled = bool(config.impute_config().get("enabled", False))

    # Pre-registered coverage floor: no verdict is quotable below 90% MEASURED on both arms.
    # `_scorable_subset` already dropped unscorable pairs from the cost basis; this is the
    # companion gate on the VERDICT ITSELF. A gate whose decisive cells are mostly imputed has
    # no verdict — it is UNTESTED, and the report must say so rather than let a
    # PASS/FAIL leak out of a thin matrix. Computed once here and passed into the report so the
    # banner and the return code can never disagree.
    n_scored = len(task_ids)
    n_control_measured = sum(1 for d in control if d[7])
    n_router_measured = sum(1 for d in router if d[7])
    control_coverage = n_control_measured / n_scored if n_scored else 0.0
    router_coverage = n_router_measured / n_scored if n_scored else 0.0
    coverage_tripped = min(control_coverage, router_coverage) < COVERAGE_FLOOR
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
        cache_aware_ci=cache_aware_ci,
        decomposition=decomposition,
        impute_enabled=impute_enabled,
        coverage_tripped=coverage_tripped,
    )

    if coverage_tripped:
        # A coverage-gap run is never a PASS/FAIL — exit code mirrors the UNTESTED verdict.
        exit_code = 1
        verdict_label = "UNTESTED"
    else:
        exit_code, verdict_label = decide_verdict(
            cost_delta,
            pr_delta,
            verifier_threshold,
            len(task_ids),
            cache_aware_ratio,
            cache_aware_ci,
        )

    # The naive ratio is a companion to the cache-aware one: when caching is not priced the
    # two collapse, and a reviewer needs both to see how much of the saving is cache-driven.
    naive_ratio = cache_router[0] / cache_control[0] if cache_control[0] else float("inf")
    n_scorable = len(scorable_control)

    if verdict_path is not None:
        payload = _build_verdict_payload(
            exit_code=exit_code,
            verdict=verdict_label,
            frontier_model=frontier_model,
            n=n_scored,
            n_scorable=n_scorable,
            n_unscorable=n_unscorable,
            verifier_threshold=verifier_threshold,
            task_seed=task_seed,
            bootstrap_iterations=n_iterations,
            cache_aware_ratio=cache_aware_ratio,
            cache_aware_ratio_ci=cache_aware_ci,
            naive_ratio=naive_ratio,
            control_pass=sum(1 for d in control if d[2]),
            router_pass=sum(1 for d in router if d[2]),
            cost_delta=cost_delta,
            cost_ratio=cost_ratio,
            pass_rate_delta=pr_delta,
            cache_control=cache_control,
            cache_router=cache_router,
            cache_oracle=cache_oracle,
            coverage_floor=COVERAGE_FLOOR,
            control_coverage=control_coverage,
            router_coverage=router_coverage,
            n_control_measured=n_control_measured,
            n_router_measured=n_router_measured,
            coverage_tripped=coverage_tripped,
            impute_enabled=impute_enabled,
        )
        write_verdict_artifact(payload, verdict_path)

    return exit_code, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    config.load(config_path)

    bm = config.benchmark_params()

    ap = argparse.ArgumentParser(
        description="Kill-gate automation: compare Shunt routing vs Always-Frontier baseline."
    )
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None, help="Matrix JSON path")
    ap.add_argument(
        "--pricing", default=None, help="Pricing JSON path (deprecated — uses benchmark.yaml)"
    )
    ap.add_argument("--log", default="benchmark/runner/kill_gate.log", help="Output log path")
    ap.add_argument(
        "--verdict",
        default="benchmark/runner/kill_gate_verdict.json",
        help="Deterministic verdict artifact path (tracked; diffable, unlike the log)",
    )
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
    verdict_path = Path(args.verdict)

    if not matrix_path.exists():
        print(f"Error: matrix not found: {matrix_path}", file=sys.stderr)
        sys.exit(2)

    # Pre-analysis data-integrity gate (fail closed): refuse to gate on poison data —
    # ERROR-severity rows (e.g. a paid model that ran but recorded real_cost==0) would
    # silently skew the cost verdict. Analysis does not run until the data is clean.
    _report, _blocking = validate_results.gate(
        config.results_csv_path(), dict(config.load_pricing())
    )
    if _blocking:
        print(validate_results.format_report(_report, config.results_csv_path()), file=sys.stderr)
        print(
            "Refusing to run the kill gate on data with ERROR-severity integrity violations.",
            file=sys.stderr,
        )
        sys.exit(2)

    matrix = config.load_matrix(matrix_path)
    # Complete-only + equal-coverage: with imputation enabled, drop every INCOMPLETE challenge
    # (unknown band still open) and fill every enabled tier on the survivors, so the gate is
    # decided over challenges whose crossover is actually established (no-op when impute is off).
    matrix, _im = summary.complete_scored_matrix(matrix)

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
        # An empty corpus is UNTESTED, not a PASS: exit 0 is the green code an automated
        # consumer reads as "the gate cleared", and a gate that measured nothing must not
        # read that way. 1 is the same code the coverage-floor path uses for UNTESTED.
        print(
            "No results yet — results.csv holds no rows, so there is nothing to gate. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        sys.exit(1)

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
        verdict_path=verdict_path,
        task_seed=args.seed,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report)
    print(report)
    print(f"Log written to {log_path}")
    print(f"Verdict artifact written to {verdict_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
