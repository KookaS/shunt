"""Canonical per-strategy benchmark summary derived from results.csv."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.admissibility import AdmissibilityResult
from benchmark.routing import selection_guard
from benchmark.routing.cache_cost import CachePrice, cache_aware_total, cache_prices
from benchmark.routing.impute import ImputedMatrix, complete_matrix, is_non_observation
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
    # BOTH cost models ship, never one silently standing in for the other. `TotalCost` is the raw
    # sum of billed attempts; `TotalCost_cacheaware` prices the repeat-model discount a provider
    # actually grants. A cascade re-serves one model on consecutive attempts by construction, so
    # the two differ most exactly where the ranking is decided — a reader must see which they hold.
    "TotalCost",
    "TotalCost_ci_lower",
    "TotalCost_ci_upper",
    "TotalCost_cacheaware",
    "TotalCost_cacheaware_ci_lower",
    "TotalCost_cacheaware_ci_upper",
    "AvgCost",
    "AvgCost_ci_lower",
    "AvgCost_ci_upper",
    "AvgCost_cacheaware",
    "AvgCost_cacheaware_ci_lower",
    "AvgCost_cacheaware_ci_upper",
    "Reward",
    "CumReg",
    "CumReg_ci_lower",
    "CumReg_ci_upper",
    "rAcc",
    # `Pareto` is decided on CACHE-AWARE cost; `Pareto_naive` repeats the frontier on raw sums so
    # a reader can see whether the cache assumption moved anyone.
    "Pareto",
    "Pareto_naive",
    # Whether this row was scored on the full sample or a coverage-selected slice of it, and the
    # measured difficulty gap when it was. See benchmark/routing/selection_guard.py.
    "subset_selected",
    "subset_note",
    # Every published row carries the verdict on the instrument that produced it. A reader who
    # greps ONE line still sees whether the routing pipeline behind it has been shown to recover
    # a signal it is known to contain — which is the difference between "kNN scored 78.53%" and
    # "kNN scored 78.53% on an instrument never shown to measure anything".
    "instrument_admissible",
    "instrument_verdict",
)
# Columns that are deterministic given the matrix (no bootstrap, no strategy-set
# dependency) — what the drift check compares.
DETERMINISTIC_FIELDS: Final[tuple[str, ...]] = (
    "n_pass",
    "AvgPerf%",
    "TotalCost",
    "TotalCost_cacheaware",
    "AvgCost",
    "AvgCost_cacheaware",
    "Reward",
    "CumReg",
    "rAcc",
)

Decision = tuple[str, str, bool, float]
# One billed attempt: (model, cost). A single-shot decision is one attempt; a cascade is as many
# as it made, in billing order.
Attempt = tuple[str, float]


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


def load_scored_matrix(path: str | Path | None = None) -> dict:
    """Load the matrix, reduce to the VALID set (complete, uncensored), and restrict to
    the deterministic sampled set."""
    # load_matrix -> complete_scored_matrix -> the SAME sample_tasks restriction run_eval
    # applies, so every analytical plot scores exactly the tasks run_eval/strategy_summary
    # do — never an off-sample row that leaked into results.csv.
    matrix = config.load_matrix(path)
    scored, _imputed = complete_scored_matrix(matrix)
    sampled = set(config.sample_tasks(sorted(matrix["results"].keys())))
    scored["results"] = {tid: cells for tid, cells in scored["results"].items() if tid in sampled}
    return scored


def evaluate(strategy: object, matrix: dict, tasks: list[str]) -> tuple[list[Decision], set[str]]:
    """Run one strategy, returning ``(decisions, unscorable)``: ``(task, model, passed,
    cost)`` tuples plus the task ids that cannot be scored — the chosen cell was never
    measured, or a cascade's PATH crossed an unmeasured cell (callers must exclude them)."""
    decisions, unscorable, _attempts = evaluate_billed(strategy, matrix, tasks)
    return decisions, unscorable


def evaluate_billed(
    strategy: object, matrix: dict, tasks: list[str]
) -> tuple[list[Decision], set[str], dict[str, list[Attempt]]]:
    """``evaluate`` plus each task's billed attempt sequence, from the SAME single pass.

    The cache-aware cost model needs the attempt ORDER, which the collapsed decision hides.
    """
    decisions: list[Decision] = []
    unscorable: set[str] = set()
    attempts: dict[str, list[Attempt]] = {}
    for tid in tasks:
        task_meta = matrix["tasks"].get(tid, {})
        model = strategy.select(tid, task_meta, matrix)  # type: ignore[attr-defined]
        outcome = matrix["results"].get(tid, {}).get(model, {})
        # A cascade bills an unmeasured intermediate cell at $0, so its aggregated cost
        # understates reality even when the FINAL cell is measured. `cascade_scorable` is the
        # strategy's own report of that; the kill gate already gates on it, and reading it here
        # makes it authoritative on the path that produces the published numbers too. Set during
        # `select` above, so it must be read after. Non-cascade strategies default to True.
        cascade_ok = bool(getattr(strategy, "cascade_scorable", True))
        # An unmeasured cell (coverage gap) OR any NON-OBSERVATION is unscorable: a censored
        # cell (resource-limit stop, unknown true outcome) and a never-executed cell (zero
        # priced calls AND $0 real spend) are both the absence of a measurement, so scoring
        # either as a clean pass=False understates the chosen model's quality and hands the
        # strategy a free fail@$0. This is the SAME predicate the cost model excludes on, applied
        # here so the supported `impute.enabled` toggle cannot change what counts as measured.
        if not outcome or is_non_observation(outcome) or not cascade_ok:
            unscorable.add(tid)
        passed = outcome.get("pass", False)
        cascade_cost = getattr(strategy, "cascade_total_cost", None)
        cost = cascade_cost if cascade_cost is not None else outcome.get("cost", 0.0)
        decisions.append((tid, model, passed, cost))
        # A cascade publishes its per-attempt billing; a single-shot decision IS one attempt.
        # Copied, because the strategy overwrites the attribute on the next `select`.
        billed = getattr(strategy, "cascade_attempts", None)
        attempts[tid] = list(billed) if billed else [(model, float(cost))]
    return decisions, unscorable, attempts


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
    # The PRE-completion matrix is kept as the selection guard's difficulty probe: completion
    # drops incomplete challenges, so asking the completed matrix why they were dropped finds no
    # cells to compare and the guard would report "nothing to say" about its own selection.
    probe = matrix
    matrix, _imputed = complete_scored_matrix(matrix)
    oracle_decisions, oracle_unscorable = evaluate(OracleRewardAware(gamma=gamma), matrix, tasks)
    random.seed(seed)
    prices = cache_prices(sorted(matrix.get("models", {})))
    rows = [
        _strategy_row(
            s,
            matrix,
            tasks,
            oracle_decisions,
            oracle_unscorable,
            gamma,
            bootstrap,
            prices,
            probe,
            seed=seed,
        )
        for s in strategies
    ]
    _apply_pareto(rows)
    rows.sort(key=lambda r: r.get("Reward", 0), reverse=True)
    return rows


def _apply_pareto(rows: list[dict]) -> None:
    """Stamp each row's Pareto membership under BOTH cost models."""
    # `Pareto` is decided on CACHE-AWARE cost, because that is the cost a deployment actually
    # pays and the question the frontier answers is which strategy to run. Naive cost overstates
    # exactly the strategies that re-serve one model — the cascades — so a naive frontier ranks
    # partly on cache blindness. `Pareto_naive` ships beside it so the assumption is auditable
    # rather than silent: a row where the two disagree is a row the cache model moved.
    axes = (("TotalCost_cacheaware", "Pareto"), ("TotalCost", "Pareto_naive"))
    for cost_field, out_field in axes:
        # Zero-evidence rows (no scorable task) are kept in the output but NEVER enter the
        # comparison: (cost $0, pass 0%) is un-dominated by construction, so including them
        # certifies "measured nothing" as optimal.
        metrics = {
            r["strategy"]: {"AvgPerf%": r.get("AvgPerf%", 0.0), "TotalCost": r.get(cost_field, 0.0)}
            for r in rows
            if int(r.get("n_tasks", 0) or 0) > 0
        }
        pareto = compute_pareto(metrics)
        for r in rows:
            r[out_field] = pareto.get(r["strategy"], False)


def _cache_aware_cost(
    decisions: list[Decision],
    attempts: dict[str, list[Attempt]],
    prices: dict[str, CachePrice],
) -> float:
    """Total cost of the scored tasks' billed attempts once repeat-model caching is priced."""
    # SCOPED PER TASK, and that is the load-bearing choice. A task here is one session, and a
    # cache discount exists because a PREFIX is still warm — so a cascade retrying inside a task
    # banks it, and two unrelated tasks that happen to route to the same model do not. Summing
    # the whole run as one sequence instead would hand a fixed-model baseline a discount on every
    # task after the first (measured: it cuts Always-Frontier from $88.61 to $23.51 on this
    # corpus), which is not caching — it is treating 175 independent sessions as one conversation.
    return sum(cache_aware_total(attempts.get(tid, []), prices) for tid, _m, _p, _c in decisions)


def _strategy_row(  # noqa: PLR0913
    strategy: object,
    matrix: dict,
    tasks: list[str],
    oracle_decisions: list[Decision],
    oracle_unscorable: set[str],
    gamma: float,
    bootstrap: int,
    prices: dict[str, CachePrice] | None = None,
    probe: dict | None = None,
    seed: int = 42,
) -> dict:
    decisions, strat_unscorable, attempts = evaluate_billed(strategy, matrix, tasks)
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
    cis = bootstrap_ci(
        decisions,
        oracle_aligned,
        bootstrap,
        gamma=gamma,
        seed=seed,
        attempts=attempts,
        prices=prices,
    )
    cache_total = _cache_aware_cost(decisions, attempts, prices or {})
    n = len(decisions)
    probe_matrix = matrix if probe is None else probe
    selection = selection_guard.assess(
        [d[0] for d in decisions],
        tasks,
        probe_matrix,
        selection_guard.reference_model(probe_matrix),
    )
    return {
        "strategy": strategy.name,  # type: ignore[attr-defined]
        # Count every task dropped from THIS row's metrics — a coverage gap in the
        # strategy OR the oracle arm removes the pair, so report the union, not just
        # the strategy's own gaps (which undercounts when the oracle missed a cell).
        "n_unscorable": len(excluded),
        **metrics,
        **comparison,
        "TotalCost_cacheaware": round(cache_total, 4),
        "TotalCost_cacheaware_ci_lower": cis.total_cost_cacheaware[0],
        "TotalCost_cacheaware_ci_upper": cis.total_cost_cacheaware[1],
        "AvgCost_cacheaware": round(cache_total / n, 6) if n else 0.0,
        "AvgCost_cacheaware_ci_lower": cis.avg_cost_cacheaware[0],
        "AvgCost_cacheaware_ci_upper": cis.avg_cost_cacheaware[1],
        "AvgPerf_ci_lower": cis.avgperf[0],
        "AvgPerf_ci_upper": cis.avgperf[1],
        "CumReg_ci_lower": cis.cumreg[0],
        "CumReg_ci_upper": cis.cumreg[1],
        "TotalCost_ci_lower": cis.total_cost[0],
        "TotalCost_ci_upper": cis.total_cost[1],
        "AvgCost_ci_lower": cis.avg_cost[0],
        "AvgCost_ci_upper": cis.avg_cost[1],
        "subset_selected": selection.is_subset,
        "subset_note": selection.note,
    }


@dataclass(frozen=True)
class StrategyTable:
    """Summary rows plus the verdict on the instrument that produced them."""

    # REQUIRED and FIRST, with no default, for the same reason `knn_nulls.TransferCurve` carries
    # one: the strategy table is the artifact the kill-gate comparison is read off, and it can now
    # only exist once the caller has stated whether the SHIPPED selection path
    # (`kNNStrategy` -> `RouterEngine.decide` -> `SelectionRule`) recovers a signal it is known to
    # contain and collapses to chance when that signal is destroyed. A defaulted field would have
    # let every existing call site keep compiling and keep publishing, which is precisely how the
    # gate went unwired the first time. The figures' gate does NOT cover this path: it certifies
    # `select_from_rates`, which documents three named divergences from the shipped rule. Build
    # the verdict with `instrument_control.strategy_instrument_admissibility`.
    admissibility: AdmissibilityResult
    rows: tuple[dict, ...]


def certified_table(
    rows: list[dict],
    *,
    k: int | None = None,
    threshold: float | None = None,
    min_samples: int | None = None,
) -> StrategyTable:
    """Pair ``rows`` with the shipped-path instrument verdict AT THE SAME kNN parameters.

    Defaults come from ``config.knn_params()``; pass overrides when the caller did.
    """
    # The parameters matter: a control run at other settings certifies an instrument nobody is
    # quoting, which is why they are threaded through rather than read from config unconditionally.
    from benchmark.routing.instrument_control import strategy_instrument_admissibility

    params = config.knn_params()
    return StrategyTable(
        admissibility=strategy_instrument_admissibility(
            k=int(params.get("k", 20)) if k is None else int(k),
            threshold=(
                float(params.get("success_rate_threshold", 0.6))
                if threshold is None
                else float(threshold)
            ),
            min_samples=(
                int(params.get("min_samples", 3)) if min_samples is None else int(min_samples)
            ),
        ),
        rows=tuple(rows),
    )


def write_summary_csv(table: StrategyTable, path: Path) -> None:
    """Write the strategy table to ``path``, stamping every row with the instrument verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = {
        "instrument_admissible": table.admissibility.admissible,
        "instrument_verdict": table.admissibility.headline,
    }
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS))
        w.writeheader()
        for row in table.rows:
            w.writerow({k: {**row, **stamp}.get(k, "") for k in SUMMARY_FIELDS})
