"""Canonical per-strategy benchmark summary derived from results.csv."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing import censoring
from benchmark.routing.impute import ImputedMatrix, complete_matrix
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
    "n_unscorable",
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


def complete_scored_matrix(matrix: dict) -> tuple[dict, ImputedMatrix | None]:
    """Complete ``matrix['results']`` to equal coverage under the monotone-ladder axiom.

    Returns ``(matrix, ImputedMatrix)``, or raw + ``None`` when imputation is off / no ladder.
    Only COMPLETE challenges survive; an open-UNKNOWN-band challenge is excluded entirely.
    """
    # Off, or a non-registry matrix (a synthetic test slice whose models aren't the
    # enabled ladder — nothing to complete from), reproduces raw-coverage scoring
    # exactly. The completed matrix drops disabled/foreign models and fills every
    # enabled model on every task, so all strategies score one equal-coverage set.
    if not config.impute_config().get("enabled", False):
        return matrix, None
    rank = config.capability_rank()
    ranked = {r.model for r in rank.ordered}
    results = matrix.get("results", {})
    if not any(m in ranked for cells in results.values() for m in cells):
        return matrix, None
    im = complete_matrix(results, rank)
    # Complete-only sampling: exclude every incomplete challenge (unknown band still open).
    completed = {tid: cells for tid, cells in im.matrix.items() if tid in im.complete}
    excluded = len(im.matrix) - len(completed)
    if excluded:
        print(
            f"  complete-only: {len(completed)} complete challenge(s) kept, "
            f"{excluded} incomplete excluded from analysis."
        )
    if config.impute_config().get("drop_unsolvable", False):
        completed = {tid: cells for tid, cells in completed.items() if im.tau.get(tid) is not None}
    return {**matrix, "results": completed}, im


def evaluate(strategy: object, matrix: dict, tasks: list[str]) -> tuple[list[Decision], set[str]]:
    """Run one strategy, returning ``(decisions, unscorable)``: ``(task, model,
    passed, cost)`` tuples plus the set of task ids whose chosen cell was never
    measured (a coverage gap, NOT a real fail@$0 — callers must exclude them)."""
    decisions: list[Decision] = []
    unscorable: set[str] = set()
    for tid in tasks:
        task_meta = matrix["tasks"].get(tid, {})
        model = strategy.select(tid, task_meta, matrix)  # type: ignore[attr-defined]
        outcome = matrix["results"].get(tid, {}).get(model, {})
        # An unmeasured cell (coverage gap) OR a CENSORED cell (resource-limit stop, unknown
        # true outcome) is unscorable: counting a censored cell as a clean pass=False would
        # understate the chosen model's quality. Callers exclude the unscorable set from metrics.
        if not outcome or censoring.is_censored(outcome):
            unscorable.add(tid)
        passed = outcome.get("pass", False)
        cascade_cost = getattr(strategy, "cascade_total_cost", None)
        cost = cascade_cost if cascade_cost is not None else outcome.get("cost", 0.0)
        decisions.append((tid, model, passed, cost))
    return decisions, unscorable


def compute_strategy_rows(
    matrix: dict,
    tasks: list[str],
    strategies: list[object],
    gamma: float = 0.1,
    bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Evaluate every strategy against the matrix and return summary rows (Reward-sorted).

    When ``impute.enabled``, the matrix is first completed to equal coverage so every
    strategy scores the SAME task set; with it off, behaviour is unchanged.
    """
    if not tasks:
        return []
    matrix, _imputed = complete_scored_matrix(matrix)
    oracle_decisions, oracle_unscorable = evaluate(OracleRewardAware(gamma=gamma), matrix, tasks)
    random.seed(seed)
    rows: list[dict] = []
    strat_metrics: dict[str, dict] = {}
    for strategy in strategies:
        rows.append(
            _strategy_row(
                strategy, matrix, tasks, oracle_decisions, oracle_unscorable, gamma, bootstrap
            )
        )
        # Zero-evidence rows (no scorable task) are kept in the output but NEVER
        # enter the Pareto comparison: (cost $0, pass 0%) is un-dominated by
        # construction, so including them certifies "measured nothing" as optimal.
        if int(rows[-1].get("n_tasks", 0) or 0) > 0:
            strat_metrics[rows[-1]["strategy"]] = {
                "AvgPerf%": rows[-1].get("AvgPerf%", 0.0),
                "TotalCost": rows[-1].get("TotalCost", 0.0),
            }
    pareto = compute_pareto(strat_metrics)
    for r in rows:
        r["Pareto"] = pareto.get(r["strategy"], False)
    rows.sort(key=lambda r: r.get("Reward", 0), reverse=True)
    return rows


def _strategy_row(  # noqa: PLR0913
    strategy: object,
    matrix: dict,
    tasks: list[str],
    oracle_decisions: list[Decision],
    oracle_unscorable: set[str],
    gamma: float,
    bootstrap: int,
) -> dict:
    decisions, strat_unscorable = evaluate(strategy, matrix, tasks)
    # A task is comparable only if BOTH the strategy and the oracle landed on a
    # measured cell; otherwise it is a coverage gap, not a real fail@$0, and must
    # not corrupt pass rate / cost. Filtering keeps strategy/oracle position-aligned.
    excluded = strat_unscorable | oracle_unscorable
    if excluded:
        kept = [i for i, d in enumerate(decisions) if d[0] not in excluded]
        decisions = [decisions[i] for i in kept]
        oracle_aligned = [oracle_decisions[i] for i in kept]
    else:
        oracle_aligned = oracle_decisions
    metrics = compute_metrics(decisions, gamma=gamma)
    comparison = compare_to_oracle(decisions, oracle_aligned, gamma=gamma)
    avgperf_ci, cumreg_ci = bootstrap_ci(decisions, oracle_aligned, bootstrap, gamma=gamma)
    return {
        "strategy": strategy.name,  # type: ignore[attr-defined]
        # Count every task dropped from THIS row's metrics — a coverage gap in the
        # strategy OR the oracle arm removes the pair, so report the union, not just
        # the strategy's own gaps (which undercounts when the oracle missed a cell).
        "n_unscorable": len(excluded),
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
