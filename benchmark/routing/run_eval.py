#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing import frontier_estimate, impute, selection_guard, summary
from benchmark.routing.metrics import discriminating_stats
from benchmark.routing.session_cascade_control import assert_ladder_quotable
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier, Random
from benchmark.routing.strategies.knn import kNNStrategy
from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy
from benchmark.routing.strategies.knn_difficulty import (
    DifficultyBandCascadeStrategy,
    knnDifficultyCascadeStrategy,
    knnDifficultyStrategy,
)
from benchmark.routing.strategies.knn_session_cascade import kNNSessionCascadeStrategy
from benchmark.routing.strategies.oracle import Oracle, OracleRewardAware
from benchmark.routing.strategies.price_cascade import PriceCascade
from benchmark.routing.strategies.session_cascade import (
    DEFAULT_LADDER,
    SessionCascadeStrategy,
)
from benchmark.routing.strategies.tier_classifier import TierClassifier
from benchmark.routing.strategy_class import is_live

# The selection knobs the kNN pick accepts. Named explicitly so an unrelated key added under
# `strategies.knn` cannot reach a constructor that would reject it at run time.
_KNN_KNOBS: Final[tuple[str, ...]] = ("k", "success_rate_threshold", "min_samples")


def _results_file(base_dir: Path, k: int, success_rate: float, min_samples: int) -> Path:
    sf = _fmt_flag(success_rate)
    return base_dir / f"results_k{k}_sr{sf}_ms{min_samples}.csv"


def load_matrix(path: Path) -> dict:
    return config.load_matrix(path)


def get_strategies(
    k: int | None = None,
    success_rate: float | None = None,
    min_samples: int | None = None,
    max_tries: int | None = None,
):
    """Read strategy list and params from config. CLI args override config values."""
    g = config.gamma()
    strat_cfg = config.strategies()
    enabled = strat_cfg.get("enabled", [])
    if not enabled:
        # The no-config fallback means "every registered strategy". It has to stay in step
        # with `registry` below — it silently omitted tier_classifier, so a run with no
        # config scored one fewer strategy than a reader of this list would expect.
        enabled = [
            "oracle",
            "oracle_reward",
            "always_cheap",
            "always_frontier",
            "random",
            "knn_semantic",
            "knn_semantic_cascade",
            "knn_semantic_cascade_withintask",
            "knn_difficulty",
            "knn_difficulty_cascade",
            "difficulty_band_cascade",
            "price_cascade",
            "session_cascade",
            "knn_semantic_tier",
        ]

    knn_p = dict(strat_cfg.get("knn_semantic", {}))
    cascade_p = dict(strat_cfg.get("knn_semantic", {}))
    cascade_p.update(strat_cfg.get("knn_semantic_cascade_withintask", {}))
    tier_p = dict(strat_cfg.get("knn_semantic", {}))
    tier_p.update(strat_cfg.get("knn_semantic_tier", {}))
    # The difficulty family reads its own knobs: the semantic kNN's thresholds describe an
    # embedding neighbourhood, and borrowing them would couple the two families' tuning.
    difficulty_p = dict(strat_cfg.get("knn_difficulty", {}))
    band_p = dict(strat_cfg.get("difficulty_band", {}))

    if k is not None:
        knn_p.setdefault("k", k)
        cascade_p.setdefault("k", k)
        tier_p.setdefault("k", k)
    if success_rate is not None:
        knn_p.setdefault("success_rate_threshold", success_rate)
        cascade_p.setdefault("success_rate_threshold", success_rate)
    if min_samples is not None:
        knn_p.setdefault("min_samples", min_samples)
        cascade_p.setdefault("min_samples", min_samples)
    if max_tries is not None:
        cascade_p.setdefault("max_tries", max_tries)
    # Price-Cascade takes only max_tries — it has no kNN knobs by construction, and
    # sharing the cascade's depth keeps the two cascades comparable at equal depth.
    price_p = {"max_tries": cascade_p.get("max_tries", 3)}
    # Session-Cascade takes the PRODUCT's escalation knobs, not the cascade's depth: its ladder
    # length is the model pool's, and what it is parameterised by is the live recurrence policy.
    session_p = dict(strat_cfg.get("session_cascade", {}))
    # The OPT-IN `knn_semantic_cascade` row is the kNN pick on top of that same ladder, so it
    # takes session knobs verbatim and adds only the selection knobs. No
    # `knn_semantic_cascade:` block of its own: a second copy of the ladder knobs is a second
    # ladder waiting to drift from the one the Session-Cascade row prices, and the two rows are
    # only comparable at one ladder.
    knn_session_p = {**{key: knn_p[key] for key in _KNN_KNOBS if key in knn_p}, **session_p}
    # The difficulty session-cadence rows mirror knn_semantic_cascade: the difficulty pick on the
    # SAME session ladder, so the three session-cadence rows are scored at one ladder.
    difficulty_session_p = {
        **{key: difficulty_p[key] for key in _KNN_KNOBS if key in difficulty_p},
        **session_p,
    }
    band_session_p = {**{key: band_p[key] for key in _KNN_KNOBS if key in band_p}, **session_p}
    # STRUCTURAL: refuse to build either session-cadence row at a ladder its positive control does
    # not cover, rather than produce a row nobody may quote. Raised here — before any evaluation —
    # so the failure costs nothing and cannot be mistaken for a result.
    # The fallback is the STRATEGY's own default, not a restated literal: gating on a ladder the
    # row would not have run is a green gate over an uncertified replay.
    ladder = str(session_p.get("ladder", DEFAULT_LADDER))
    for cadence_id in ("session_cascade", "knn_semantic_cascade"):
        if cadence_id in enabled:
            assert_ladder_quotable(ladder)

    registry: dict[str, Callable[[], object]] = {
        "oracle": lambda: Oracle(),
        "oracle_reward": lambda: OracleRewardAware(gamma=g),
        "always_cheap": lambda: AlwaysCheap(),
        "always_frontier": lambda: AlwaysFrontier(),
        "random": lambda: Random(seed=42),
        "knn_semantic": lambda: kNNStrategy(**knn_p),
        "knn_semantic_cascade": lambda: kNNSessionCascadeStrategy(**knn_session_p),
        "knn_semantic_cascade_withintask": lambda: kNNCascadeStrategy(**cascade_p),
        "knn_difficulty": lambda: knnDifficultyStrategy(**difficulty_p),
        "knn_difficulty_cascade": lambda: knnDifficultyCascadeStrategy(**difficulty_session_p),
        "difficulty_band_cascade": lambda: DifficultyBandCascadeStrategy(**band_session_p),
        "price_cascade": lambda: PriceCascade(**price_p),
        "session_cascade": lambda: SessionCascadeStrategy(**session_p),
        "knn_semantic_tier": lambda: TierClassifier(**tier_p),
    }

    return [registry[name]() for name in enabled if name in registry]


def _fmt_flag(v: float) -> str:
    s = f"{v:.1f}"
    return s.replace(".", "_")


def _build_arg_parser(
    config_path: str, bm: dict, knn_p: dict, cascade_p: dict
) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None, help="Matrix JSON path (default: challenges.json)")
    ap.add_argument("--output-dir", default=None, help="Output directory for results CSV")
    ap.add_argument("--gamma", type=float, default=config.gamma(), help="cost weight")
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=bm.get("bootstrap_iterations", 1000),
        help="bootstrap iterations for CI",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=bm.get("seed", 42),
        help="RNG seed for bootstrap reproducibility",
    )
    ap.add_argument("--knn-k", type=int, default=knn_p.get("k", 20), help="kNN neighbor count")
    ap.add_argument(
        "--knn-success-rate",
        type=float,
        default=knn_p.get("success_rate_threshold", 0.7),
        help=(
            "Min neighbor success rate for model eligibility. "
            "Lower=more cost savings, higher=more reliability."
        ),
    )
    ap.add_argument(
        "--knn-min-samples",
        type=int,
        default=knn_p.get("min_samples", 3),
        help="Min neighbor samples for model eligibility",
    )
    ap.add_argument(
        "--cascade-max-tries",
        type=int,
        default=cascade_p.get("max_tries", 3),
        help="Max models to try in the kNN-semantic-cascade (within-task) shortlist",
    )
    return ap


def _print_effective_sample(tasks: list[str]) -> None:
    """Report how much of the task set actually discriminates strategies."""
    stats = discriminating_stats(
        config.load_results(), tasks, config.enabled_models(), config.default_arm_ids()
    )
    print(
        f"Effective sample: {stats['n_discriminating']} discriminating / "
        f"{stats['n_fully_covered']} fully-covered "
        f"({stats['n_tasks']} with any coverage) — "
        f"{stats['n_all_pass']} all-pass, {stats['n_all_fail']} all-fail "
        f"carry no routing signal\n"
    )


def compute_frontier_gate(
    covariate: dict[str, float],
    frontier_pass: dict[str, int],
    discriminating: set[str],
    audit_ids: set[str],
    audit_fraction: float,
    router_pass: dict[str, int] | None = None,
    baseline_pass: dict[str, int] | None = None,
    margin: float = 0.05,
) -> dict:
    """Partial-coverage kill-gate summary from an adaptively-collected matrix.

    Assigns each labeled frontier cell its known sampling probability (discriminating
    -> 1.0, audit -> fraction), then runs the Q_F estimate, violation rate and paired tests.
    """
    labeled_outcome: dict[str, float] = {}
    labeled_prob: dict[str, float] = {}
    for tid, passed in frontier_pass.items():
        if tid in discriminating:
            labeled_prob[tid] = 1.0
        elif tid in audit_ids:
            labeled_prob[tid] = audit_fraction
        else:
            continue  # frontier cell of unknown provenance — excluded from the estimate
        labeled_outcome[tid] = float(passed)
    gate: dict = {"n_labeled": len(labeled_outcome)}
    if labeled_outcome:
        gate["q_f"] = frontier_estimate.ppi_frontier_quality(
            covariate, labeled_outcome, labeled_prob
        )
        gate["violation"] = frontier_estimate.frontier_violation_rate(
            labeled_outcome, covariate, sorted(audit_ids)
        )
    if router_pass and baseline_pass:
        gate["mcnemar"] = frontier_estimate.mcnemar_noninferiority(
            router_pass, baseline_pass, margin=margin
        )
        seq = None
        for tid in sorted(router_pass):
            if tid in baseline_pass:
                seq = frontier_estimate.update_confidence_sequence(
                    seq, router_pass[tid], baseline_pass[tid], margin
                )
        gate["sequence"] = seq
    return gate


def _frontier_gate_inputs(matrix: dict, tasks: list[str]) -> dict:
    """Build the covariate/strata/paired outcomes for the gate from the flattened matrix."""
    from benchmark.runner.collect import phase_a_models
    from benchmark.runner.sampling import in_frontier_audit

    knobs = config.collect_config()
    fraction = float(knobs.get("audit_fraction", 0.20))
    salt = str(knobs.get("audit_salt", "frontier-audit-v1"))
    cheap_mid = phase_a_models(str(knobs.get("phase_a_mode", "single")))
    control = config.frontier_model() or ""
    results = matrix.get("results", {})

    covariate: dict[str, float] = {}
    frontier_pass: dict[str, int] = {}
    discriminating: set[str] = set()
    audit_ids: set[str] = set()
    for tid in tasks:
        cells = results.get(tid, {})
        proxy = [cells[m]["pass"] for m in cheap_mid if m in cells]
        if len(proxy) < len(cheap_mid):
            continue  # not fully cheap+mid covered → outside the population
        covariate[tid] = sum(1.0 for p in proxy if p) / len(proxy)
        if len(set(proxy)) > 1:
            discriminating.add(tid)
        elif in_frontier_audit(tid, fraction, salt):
            audit_ids.add(tid)
        if control in cells:
            frontier_pass[tid] = int(bool(cells[control].get("pass")))

    router_pass, baseline_pass = _paired_outcomes(matrix, sorted(discriminating), control)
    return {
        "covariate": covariate,
        "frontier_pass": frontier_pass,
        "discriminating": discriminating,
        "audit_ids": audit_ids,
        "audit_fraction": fraction,
        "router_pass": router_pass,
        "baseline_pass": baseline_pass,
        "margin": float(knobs.get("noninferiority_margin", 0.05)),
    }


def _paired_outcomes(
    matrix: dict, disc: list[str], control: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Shipped kNN router vs fixed-frontier (control) realized pass on the disputed set."""
    # The arm has to be a strategy `LIVE_STRATEGIES` accepts, or the gate adjudicates
    # "should we ship this" on something the product refuses to run. It used to be
    # kNN-semantic-cascade, which is blocked (see benchmark/routing/strategy_class.py) — the same
    # substitution benchmark/runner/kill_gate.py was fixed away from. Single-shot: one
    # decision per task, so the contrast is single-shot-vs-single-shot rather than
    # best-of-N coverage against a single attempt.
    knn = config.knn_params()
    router = kNNStrategy(
        k=knn.get("k", 20),
        success_rate_threshold=knn.get("success_rate_threshold", 0.6),
        min_samples=knn.get("min_samples", 3),
    )
    results = matrix.get("results", {})
    router_pass: dict[str, int] = {}
    baseline_pass: dict[str, int] = {}
    for tid in disc:
        if control not in results.get(tid, {}):
            continue
        chosen = router.select(tid, matrix.get("tasks", {}).get(tid, {}), matrix)
        cell = results.get(tid, {}).get(chosen, {})
        router_pass[tid] = int(bool(cell.get("pass")))
        baseline_pass[tid] = int(bool(results[tid][control].get("pass")))
    return router_pass, baseline_pass


def _print_frontier_gate(gate: dict) -> None:
    """Print the frontier kill-gate summary (only when collect mode is enabled)."""
    print("\nFrontier kill-gate (partial-coverage estimators — collect mode):")
    print(f"  labeled frontier cells: {gate['n_labeled']}")
    if "q_f" in gate:
        q = gate["q_f"]
        ci = f"[{q.ci_lo:.3f}, {q.ci_hi:.3f}]"
        print(f"  Q_F (PPI++/AIPW) = {q.point:.3f}  CI {ci}  lam={q.lam:.2f}")
        v = gate["violation"]
        vci = f"[{v.ci_lo:.3f}, {v.ci_hi:.3f}]"
        print(f"  violation rate = {v.point:.3f}  CI {vci}  n={v.n_labeled}")
    if "mcnemar" in gate:
        m = gate["mcnemar"]
        print(f"  paired McNemar b={m.b} c={m.c}: {m.decision}  z={m.stat:.2f} p={m.p_value:.3f}")
    if gate.get("sequence") is not None:
        s = gate["sequence"]
        verdict = s.direction if s.decided else "undecided"
        sci = f"[{s.ci_lo:.2f}, {s.ci_hi:.2f}]"
        print(f"  confidence sequence: {verdict}  e={s.e_value:.2f} CI {sci}")


# Below this many shared scorable tasks a paired router-vs-frontier delta is too
# thin to trust — refuse the headline rather than print a bogus number (design R1).
_MIN_PAIRED: Final[int] = 10

# The fixed-frontier comparison arm, by display name.
_BASELINE_STRATEGY: Final[str] = "Always-Frontier"


def _pick_router(rows: list[dict]) -> str | None:
    """Best LIVE router by Reward — the headline must name something an operator can run."""
    # `is_live` is the product's own allowlist, so this can no longer drift from it. The
    # frontier baseline IS live; it is excluded because it is the thing being compared
    # against, not a candidate router.
    candidates = [
        r
        for r in rows
        if is_live(str(r["strategy"]))
        and str(r["strategy"]) != _BASELINE_STRATEGY
        and int(r.get("n_tasks", 0) or 0) > 0
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda r: float(r.get("Reward", 0)))["strategy"])


def _strategy_named(strategies: list, name: str) -> object | None:
    return next((s for s in strategies if getattr(s, "name", None) == name), None)


def _paired_bootstrap_ci(diffs: list[int], seed: int, n_boot: int = 1000) -> tuple[float, float]:
    """95% percentile CI (percentage points) for the mean paired pass-rate difference."""
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        100.0 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    a = int(0.025 * n_boot)
    return (means[a], means[n_boot - 1 - a])


def _paired_delta(
    completed: dict, tasks: list[str], router: object, frontier: object, seed: int
) -> dict | None:
    """Router-minus-frontier pass rate (pp) on the INTERSECTION of their scorable
    tasks, with a paired bootstrap CI. None when the two share no scorable task."""
    r_dec, r_uns = summary.evaluate(router, completed, tasks)
    f_dec, f_uns = summary.evaluate(frontier, completed, tasks)
    r_pass = {d[0]: bool(d[2]) for d in r_dec}
    f_pass = {d[0]: bool(d[2]) for d in f_dec}
    shared = [t for t in tasks if t not in r_uns and t not in f_uns]
    if not shared:
        return None
    diffs = [int(r_pass[t]) - int(f_pass[t]) for t in shared]
    delta = 100.0 * sum(diffs) / len(diffs)
    lo, hi = _paired_bootstrap_ci(diffs, seed)
    return {"n": len(shared), "delta": delta, "ci": (lo, hi)}


def _print_paired(router: str, label: str, res: dict, flip: bool = False) -> None:
    lo, hi = res["ci"]
    tag = "  (CI crosses zero — not significant)" if lo <= 0 <= hi else ""
    if flip:
        tag += "  ⚠ SIGN FLIPS vs all-tasks"
    print(
        f"  Paired contrast ({router} vs Always-Frontier, pass% on {res['n']} shared tasks) "
        f"[{label}]: {res['delta']:+.1f}pp 95% CI [{lo:+.1f}, {hi:+.1f}]{tag}"
    )


def _print_imputation_gate(
    matrix: dict, tasks: list[str], strategies: list, rows: list[dict], gamma: float, seed: int
) -> None:
    """Print the monotonicity violation rate (Wilson CI) and the PAIRED router-vs-frontier
    contrast (with a CI) on the shared scorable set, plus the violators-excluded sensitivity."""
    completed, im = summary.complete_scored_matrix(matrix)
    if im is None:
        return
    v, lo, hi = impute.violation_ci(len(im.violations), im.n_multi_observed)
    print(
        f"\nCoverage-completed imputation: monotonicity violation rate "
        f"v̂={v:.3f} 95% CI [{lo:.3f}, {hi:.3f}] "
        f"({len(im.violations)} of {im.n_multi_observed} multi-observed tasks)"
    )
    router_name = _pick_router(rows)
    router = _strategy_named(strategies, router_name) if router_name else None
    frontier = _strategy_named(strategies, "Always-Frontier")
    if router is None or frontier is None or router_name is None:
        return
    full = _paired_delta(completed, tasks, router, frontier, seed)
    if full is None or full["n"] < _MIN_PAIRED:
        got = full["n"] if full else 0
        print(
            f"  Paired contrast refused — {router_name} and Always-Frontier share only "
            f"{got} scorable task(s) (< {_MIN_PAIRED}); too few to pair honestly."
        )
        return
    _print_paired(router_name, "all tasks", full)
    violator_ids = {viol.task_id for viol in im.violations}
    kept = [t for t in tasks if t not in violator_ids]
    excl = _paired_delta(completed, kept, router, frontier, seed)
    if excl is not None and excl["n"] >= _MIN_PAIRED:
        _print_paired(
            router_name, "violators-excluded", excl, (full["delta"] >= 0) != (excl["delta"] >= 0)
        )


def _print_rows(rows: list[dict]) -> None:
    for r in rows:
        ci_ap = f"[{r['AvgPerf_ci_lower']:>5.2f},{r['AvgPerf_ci_upper']:>5.2f}]"
        ci_c = f"[{r['TotalCost_ci_lower']:>8.4f},{r['TotalCost_ci_upper']:>8.4f}]"
        ci_cr = f"[{r['CumReg_ci_lower']:>7.4f},{r['CumReg_ci_upper']:>7.4f}]"
        # The subset marker rides on the row itself, not only in a footer: a line copied out of
        # this table alone still carries the fact that it was not scored on the full sample.
        flag = " *SUBSET" if r.get("subset_selected") else ""
        print(
            f"  {r['strategy']:25}  AvgPerf={r['AvgPerf%']:>5.2f}%  {ci_ap}  "
            f"TotalCost=${r['TotalCost']:<8.4f} {ci_c}  "
            f"cache-aware=${r['TotalCost_cacheaware']:<8.4f}  "
            f"CumReg={r['CumReg']:<8.4f}  {ci_cr}  "
            f"rAcc={r['rAcc']:<6.4f}{flag}"
        )
    for line in selection_guard.rows_footer(rows):
        print(line)


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    config.load(config_path)
    bm = config.benchmark_params()
    strat_cfg = config.strategies()
    ap = _build_arg_parser(
        config_path,
        bm,
        strat_cfg.get("knn_semantic", {}),
        strat_cfg.get("knn_semantic_cascade_withintask", {}),
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)
        bm = config.benchmark_params()

    # Parameterized per-run CSVs are temporary artifacts — default to the
    # gitignored artifacts/ dir so they never pollute the tracked tree. The sole
    # committed source of truth is results.csv; the per-strategy summary is
    # regenerable (report.py / run_matrix write it to reports/, where it is the one
    # tracked file).
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "artifacts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    matrix = load_matrix(matrix_path)
    tasks = sorted(matrix["results"].keys())
    tasks = config.sample_tasks(tasks, seed=args.seed)
    strategies = get_strategies(
        k=args.knn_k,
        success_rate=args.knn_success_rate,
        min_samples=args.knn_min_samples,
        max_tries=args.cascade_max_tries,
    )

    if not tasks:
        print(
            "  No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    print(f"Evaluating {len(strategies)} strategies on {len(tasks)} tasks")
    _print_effective_sample(tasks)

    rows = summary.compute_strategy_rows(
        matrix, tasks, strategies, gamma=args.gamma, bootstrap=args.bootstrap, seed=args.seed
    )
    _print_rows(rows)
    _print_imputation_gate(matrix, tasks, strategies, rows, args.gamma, args.seed)

    if config.collect_enabled():
        gate = compute_frontier_gate(**_frontier_gate_inputs(matrix, tasks))
        _print_frontier_gate(gate)

    out_file = _results_file(output_dir, args.knn_k, args.knn_success_rate, args.knn_min_samples)
    table = summary.certified_table(
        rows,
        k=args.knn_k,
        threshold=args.knn_success_rate,
        min_samples=args.knn_min_samples,
    )
    print(f"\n{table.admissibility.reason}")
    summary.write_summary_csv(table, out_file)

    print(f"\nResults written to {out_file}")
    print(f"  k={args.knn_k}, sr={args.knn_success_rate}, ms={args.knn_min_samples}")


if __name__ == "__main__":
    main()
