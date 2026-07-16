#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from benchmark import config
from benchmark.routing import summary
from benchmark.routing.strategies.external_prior import ExternalPriorCascade
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier, Random
from benchmark.routing.strategies.knn import kNNStrategy
from benchmark.routing.strategies.knn_blended import kNNBlended
from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy
from benchmark.routing.strategies.oracle import Oracle, OracleRewardAware


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
        enabled = [
            "oracle",
            "oracle_reward",
            "always_cheap",
            "always_frontier",
            "random",
            "knn",
            "knn_cascade",
        ]

    knn_p = dict(strat_cfg.get("knn", {}))
    cascade_p = dict(strat_cfg.get("knn", {}))
    cascade_p.update(strat_cfg.get("knn_cascade", {}))
    ext_p = dict(strat_cfg.get("external_prior", {}))
    blend_p = dict(strat_cfg.get("knn_blended", {}))

    if k is not None:
        knn_p.setdefault("k", k)
        cascade_p.setdefault("k", k)
    if success_rate is not None:
        knn_p.setdefault("success_rate_threshold", success_rate)
        cascade_p.setdefault("success_rate_threshold", success_rate)
    if min_samples is not None:
        knn_p.setdefault("min_samples", min_samples)
        cascade_p.setdefault("min_samples", min_samples)
    if max_tries is not None:
        cascade_p.setdefault("max_tries", max_tries)

    registry: dict[str, Callable[[], object]] = {
        "oracle": lambda: Oracle(),
        "oracle_reward": lambda: OracleRewardAware(gamma=g),
        "always_cheap": lambda: AlwaysCheap(),
        "always_frontier": lambda: AlwaysFrontier(),
        "random": lambda: Random(seed=42),
        "knn": lambda: kNNStrategy(**knn_p),
        "knn_cascade": lambda: kNNCascadeStrategy(**cascade_p),
        "external_prior": lambda: ExternalPriorCascade(**ext_p),
        "knn_blended": lambda: kNNBlended(**blend_p),
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
        help="Max models to try in kNN-cascade shortlist",
    )
    return ap


def _print_rows(rows: list[dict]) -> None:
    for r in rows:
        ci_ap = f"[{r['AvgPerf_ci_lower']:>5.2f},{r['AvgPerf_ci_upper']:>5.2f}]"
        ci_cr = f"[{r['CumReg_ci_lower']:>7.4f},{r['CumReg_ci_upper']:>7.4f}]"
        print(
            f"  {r['strategy']:25}  AvgPerf={r['AvgPerf%']:>5.2f}%  {ci_ap}  "
            f"TotalCost=${r['TotalCost']:<8.4f}  "
            f"CumReg={r['CumReg']:<8.4f}  {ci_cr}  "
            f"rAcc={r['rAcc']:<6.4f}"
        )


def main(config_path: str = "benchmark/config.yaml") -> None:
    config.load(config_path)
    bm = config.benchmark_params()
    strat_cfg = config.strategies()
    ap = _build_arg_parser(
        config_path, bm, strat_cfg.get("knn", {}), strat_cfg.get("knn_cascade", {})
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)
        bm = config.benchmark_params()

    # Parameterized per-run CSVs are temporary artifacts — default to the
    # gitignored artifacts/ dir so they never pollute the tracked tree. The sole
    # committed source of truth is results.csv; the per-strategy summary is
    # regenerable (report.py / run_matrix write it to the gitignored reports/ dir).
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

    print(f"Evaluating {len(strategies)} strategies on {len(tasks)} tasks\n")

    rows = summary.compute_strategy_rows(
        matrix, tasks, strategies, gamma=args.gamma, bootstrap=args.bootstrap, seed=args.seed
    )
    _print_rows(rows)

    out_file = _results_file(output_dir, args.knn_k, args.knn_success_rate, args.knn_min_samples)
    summary.write_summary_csv(rows, out_file)

    print(f"\nResults written to {out_file}")
    print(f"  k={args.knn_k}, sr={args.knn_success_rate}, ms={args.knn_min_samples}")


if __name__ == "__main__":
    main()
