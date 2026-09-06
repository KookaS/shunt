#!/usr/bin/env python3
"""Re-score the ranker-family routing rows offline from the committed results matrix.

python -m benchmark.routing.scripts.score_ranker_strategy_rows --out <path.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing import summary
from benchmark.routing.session_cascade_control import assert_ladder_quotable
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier
from benchmark.routing.strategies.knn import kNNStrategy
from benchmark.routing.strategies.knn_difficulty import (
    DifficultyBandCascadeStrategy,
    knnDifficultyCascadeStrategy,
    knnDifficultyStrategy,
)
from benchmark.routing.strategies.knn_session_cascade import kNNSessionCascadeStrategy
from benchmark.routing.strategies.oracle import Oracle
from benchmark.routing.strategies.price_cascade import PriceCascade
from benchmark.routing.strategies.ranker_defer import RankerDeferCascadeStrategy
from benchmark.routing.strategies.ranker_difficulty import (
    RankerDifficultyCascadeStrategy,
    RankerDifficultyStrategy,
)
from benchmark.routing.strategies.session_cascade import (
    DEFAULT_LADDER,
    SessionCascadeStrategy,
)

# WHY THIS SCRIPT EXISTS. The ranker rows (`Ranker-Difficulty`, `Ranker-Difficulty-cascade`,
# `Ranker-Defer-cascade`) are not in `benchmark.yaml`'s `strategies.enabled` list, so `run_eval`
# never emits them and there is otherwise no reproducible path to the numbers they are quoted
# at. This script is that path: it builds the ranker rows alongside the reference rows they are
# read against and scores all of them in ONE process, through the SAME summary path the
# committed reports use -- `config.load_matrix()` + `config.sample_tasks(seed=...)` +
# `summary.compute_strategy_rows`, with imputation on and the configured bootstrap -- so a
# ranker row and a baseline row it is compared against are always scored on the same completed
# task set. Nothing here measures anything new: it is an offline replay over committed
# measurements, it makes no API calls, and it writes only to the path given on the command line.

# The selection knobs a kNN-shaped pick accepts (mirrors run_eval's list): named explicitly
# so an unrelated key added under a family's block cannot reach a constructor that rejects it.
_KNN_KNOBS: Final[tuple[str, ...]] = ("k", "success_rate_threshold", "min_samples")

# REPRODUCTION ASSERTION. Committed reference values for the row every ranker row is read
# against: `Session-Cascade` is re-scored in the same process and checked against them
# (cache-aware $23.4019 at 97.24% AvgPerf, tolerance 0.01). A run that cannot reproduce these is
# scoring a different matrix than the published rows were scored on -- a matrix, ladder or knob
# that has drifted since those were published fails the run loudly (non-zero exit) instead of
# silently re-publishing a number nobody can reproduce.
_REFERENCE_ROW: Final[str] = "Session-Cascade"
_REFERENCE_CACHEAWARE: Final[float] = 23.4019
_REFERENCE_AVGPERF: Final[float] = 97.24
_REFERENCE_TOL: Final[float] = 0.01


# KNOBS FROM CONFIG, NEVER LITERALS. Every strategy below is constructed from the same
# `strategies:` blocks `run_eval.get_strategies` reads, assembled the same way (selection knobs
# from the family's own block, ladder knobs from `session_cascade`), so the session-cadence rows
# are all scored at one ladder.
def build_strategies() -> list[object]:
    """The ranker rows plus the reference rows, at the knobs `run_eval` builds them with."""
    strat_cfg = config.strategies()
    knn_p = dict(strat_cfg.get("knn_semantic", {}))
    difficulty_p = dict(strat_cfg.get("knn_difficulty", {}))
    band_p = dict(strat_cfg.get("difficulty_band", {}))
    ranker_diff_p = dict(strat_cfg.get("ranker_difficulty", {}))
    ranker_defer_p = dict(strat_cfg.get("ranker_defer", {}))
    session_p = dict(strat_cfg.get("session_cascade", {}))

    def _session(selection: dict) -> dict:
        return {**{key: selection[key] for key in _KNN_KNOBS if key in selection}, **session_p}

    # Refuse to score any session-cadence row at a ladder its positive control does not
    # cover, before any evaluation runs -- the same structural guard run_eval applies.
    assert_ladder_quotable(str(session_p.get("ladder", DEFAULT_LADDER)))

    return [
        Oracle(),
        AlwaysFrontier(),
        AlwaysCheap(),
        PriceCascade(
            max_tries=strat_cfg.get("knn_semantic_cascade_withintask", {}).get("max_tries", 3)
        ),
        kNNStrategy(**knn_p),
        kNNSessionCascadeStrategy(**_session(knn_p)),
        knnDifficultyStrategy(**difficulty_p),
        knnDifficultyCascadeStrategy(**_session(difficulty_p)),
        DifficultyBandCascadeStrategy(**_session(band_p)),
        RankerDifficultyStrategy(**ranker_diff_p),
        RankerDifficultyCascadeStrategy(**_session(ranker_diff_p)),
        RankerDeferCascadeStrategy(
            **{
                **{k: ranker_defer_p[k] for k in ("defer_threshold",) if k in ranker_defer_p},
                **session_p,
            }
        ),
        SessionCascadeStrategy(**session_p),
    ]


def check_reproduction(rows: list[dict]) -> str | None:
    """None when the reference row reproduces its committed values, else the reason it did not."""
    row = next((r for r in rows if r["strategy"] == _REFERENCE_ROW), None)
    if row is None:
        return f"{_REFERENCE_ROW} was not scored — nothing anchors these rows to the committed run"
    cave = float(row["TotalCost_cacheaware"])
    perf = float(row["AvgPerf%"])
    if (
        abs(cave - _REFERENCE_CACHEAWARE) > _REFERENCE_TOL
        or abs(perf - _REFERENCE_AVGPERF) > _REFERENCE_TOL
    ):
        return (
            f"{_REFERENCE_ROW} re-scored at cache-aware ${cave:.4f} / {perf:.2f}% but the "
            f"committed reference is ${_REFERENCE_CACHEAWARE:.4f} / {_REFERENCE_AVGPERF:.2f}% "
            "— the matrix or the knobs have drifted; these rows are not comparable to the "
            "published ones"
        )
    return None


def _print_rows(rows: list[dict]) -> None:
    for r in rows:
        print(
            f"  {r['strategy']:26} n={r['n_tasks']:>4} perf={r['AvgPerf%']:>6.2f}% "
            f"naive=${r['TotalCost']:>8.4f} cave=${r['TotalCost_cacheaware']:>8.4f} "
            f"sess={r['sessions_mean']:>5.3f} p95={r['sessions_p95']:>3.0f} "
            f"cost_cv={r['cost_cv']:>7.4f} judge=${r['judge_label_cost']:>6.4f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score the ranker routing rows and their reference rows into one CSV"
    )
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to config YAML")
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV path (no default: these rows are not a tracked artifact)",
    )
    ap.add_argument("--bootstrap", type=int, default=None, help="bootstrap iterations")
    args = ap.parse_args()

    config.load(args.config)
    bm = config.benchmark_params()
    seed = int(bm.get("seed", 42))
    bootstrap = int(
        args.bootstrap if args.bootstrap is not None else bm.get("bootstrap_iterations", 1000)
    )

    matrix = config.load_matrix()
    tasks = config.sample_tasks(sorted(matrix.get("results", {}).keys()), seed=seed)
    if not tasks:
        print("No results — results.csv holds no rows; nothing to score.")
        return 1
    strategies = build_strategies()

    print(
        f"Scoring {len(strategies)} strategies on {len(tasks)} offered tasks "
        f"(bootstrap={bootstrap}, seed={seed})"
    )
    rows = summary.compute_strategy_rows(
        matrix, tasks, strategies, gamma=config.gamma(), bootstrap=bootstrap, seed=seed
    )
    _print_rows(rows)

    drift = check_reproduction(rows)
    if drift is not None:
        print(f"REPRODUCTION FAILED: {drift}")
        return 1
    print(
        f"Reproduction: {_REFERENCE_ROW} matches its committed reference "
        f"(${_REFERENCE_CACHEAWARE:.4f} / {_REFERENCE_AVGPERF:.2f}%)."
    )

    table = summary.certified_table(rows)
    print(table.admissibility.reason)
    out = Path(args.out)
    summary.write_summary_csv(table, out)
    print(f"Wrote {out} — {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
