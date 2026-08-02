"""Per-model corpus coverage: is every configured model backed by enough collected data?"""

# THE GAP THIS CLOSES. Adding a name to `benchmark.yaml` `models:` changes what the harness is
# asked to evaluate, but nothing reported whether the live collection ever reached it. Both
# existing per-model views enumerate the DATA, not the pool, so a model with no data is invisible
# rather than flagged: `escalation.features.model_coverage` buckets the trajectories that exist,
# and `config.derive_capability_rank` quietly seats a model with too few cells at its
# price-implied rank (`source="price-prior"`). Neither says "the collection is incomplete".
#
# NO NEW THRESHOLDS. Both floors are the ones already pinned elsewhere, so a model passing here
# is a model the downstream analysis can actually measure rather than impute:
#   * routing — `capability_rank.K` default-arm cells, the existing gate between a measured rank
#     and a price prior;
#   * escalation — `prefix_eval.MIN_ROWS` trajectories admissible at the shallowest evaluated
#     depth, the existing floor below which `evaluate_depth` returns nothing.
#
# Reads only committed data (results.csv + the trajectory plane); no containers, no spend.
# Run it (never bare python3): uv run --extra benchmark python -m benchmark.model_coverage

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.escalation import features, prefix_eval, schema

_ABSENT: Final = "ABSENT"
_THIN: Final = "THIN"
_OK: Final = "OK"

# model -> (trajectories, scorable steps, trajectories admissible at the evaluated depth)
Corpus = dict[str, tuple[int, int, int]]


@dataclass(frozen=True)
class ModelRow:
    """One configured model's measured footprint on both corpora."""

    model: str
    routing_cells: int
    trajectories: int
    steps: int
    admissible: int

    def routing_status(self, floor: int) -> str:
        """ABSENT / THIN / OK against the capability-rank cell floor."""
        return _grade(self.routing_cells, floor)

    def escalation_status(self, floor: int) -> str:
        """ABSENT / THIN / OK against the prefix-eval admissible-trajectory floor."""
        return _grade(self.admissible, floor)


def _grade(count: int, floor: int) -> str:
    if count == 0:
        return _ABSENT
    return _THIN if count < floor else _OK


def routing_cells() -> dict[str, int]:
    """Default-arm measured cells per model, from the committed results.csv."""
    counts: dict[str, int] = {}
    for cells in config.flatten_default_arm(config.load_results()).values():
        for model in cells:
            counts[model] = counts.get(model, 0) + 1
    return counts


def escalation_counts(live_dir: Path, depth: int) -> Corpus:
    """Per model: (trajectories, scorable steps, trajectories admissible at ``depth``)."""
    # Admission is `prefix_eval.corpus_census`'s own rule, not a local copy: the per-step stamping
    # stage must have run (`is_stamped`) AND the run must carry `depth + MIN_WITHHELD` scorable
    # steps. A change to the anti-leak margin therefore moves this report with it.
    needed = depth + features.MIN_WITHHELD
    counts: Corpus = {}
    for path in sorted(live_dir.glob("*.jsonl")):
        traj = schema.load_jsonl(path)
        model = features.model_of(traj)
        scorable = len(features.scorable_steps(traj))
        admits = features.is_stamped(traj) and scorable >= needed
        trajectories, steps, admissible = counts.get(model, (0, 0, 0))
        counts[model] = (trajectories + 1, steps + scorable, admissible + admits)
    return counts


def build_rows(pool: list[str], cells: dict[str, int], corpus: Corpus) -> list[ModelRow]:
    """One row per CONFIGURED model — a model with no data still gets a row, at zero."""
    return [ModelRow(m, cells.get(m, 0), *corpus.get(m, (0, 0, 0))) for m in pool]


def unconfigured_models(pool: list[str], cells: dict[str, int], corpus: Corpus) -> list[str]:
    """Models carrying data that the config no longer enables — stale, not evaluated."""
    return sorted((set(cells) | set(corpus)) - set(pool))


def format_report(rows: list[ModelRow], cell_floor: int, traj_floor: int, depth: int) -> str:
    """The per-model table, with the floor each column is graded against spelled out."""
    head = f"{'model':<20} {'cells':>6} {'routing':>9} {'traj':>6} {'steps':>7} "
    head += f"{'adm@' + str(depth):>7} {'escalation':>11}"
    lines = ["Per-model corpus coverage", "=" * 26, head, "-" * len(head)]
    for row in rows:
        lines.append(
            f"{row.model:<20} {row.routing_cells:>6} {row.routing_status(cell_floor):>9} "
            f"{row.trajectories:>6} {row.steps:>7} {row.admissible:>7} "
            f"{row.escalation_status(traj_floor):>11}"
        )
    lines.append("")
    lines.append(
        f"floors: routing >= {cell_floor} default-arm cells (capability_rank.K) · "
        f"escalation >= {traj_floor} trajectories with >= {depth + features.MIN_WITHHELD} "
        f"scorable steps (prefix_eval.MIN_ROWS)"
    )
    return "\n".join(lines)


def _verdicts(rows: list[ModelRow], cell_floor: int, traj_floor: int) -> list[str]:
    """One line per model whose data cannot carry the analysis it is enabled for."""
    out: list[str] = []
    for row in rows:
        routing, escalation = row.routing_status(cell_floor), row.escalation_status(traj_floor)
        if routing != _OK:
            out.append(
                f"{row.model}: routing {routing} ({row.routing_cells} cells) — its capability "
                "rank is a price prior, not a measurement. Collect more cells for it."
            )
        if escalation != _OK:
            out.append(
                f"{row.model}: escalation {escalation} ({row.admissible} admissible "
                "trajectories) — a model-restricted escalation eval is not estimable on it."
            )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: print the table; exit nonzero iff a configured model is ABSENT or THIN."""
    parser = argparse.ArgumentParser(
        prog="benchmark.model_coverage",
        description="Flag configured models the live collection has not covered.",
    )
    parser.add_argument("--config", default="benchmark/benchmark.yaml", help="config YAML")
    here = Path(__file__).resolve().parent
    parser.add_argument("--live-dir", type=Path, default=here / "escalation" / "data" / "live")
    parser.add_argument("--depth", type=int, default=min(features.DEFAULT_DEPTHS))
    args = parser.parse_args(argv)

    if not args.live_dir.is_dir():
        # Otherwise every model reads ABSENT on escalation and the report blames the collection
        # for a mistyped path.
        print(f"trajectory plane not found: {args.live_dir}")  # noqa: T201 - CLI output
        return 2
    config.load(args.config)
    pool = config.enabled_models()
    cell_floor = int(config.capability_rank_config()["K"])
    cells, corpus = routing_cells(), escalation_counts(args.live_dir, args.depth)
    rows = build_rows(pool, cells, corpus)
    print(format_report(rows, cell_floor, prefix_eval.MIN_ROWS, args.depth))  # noqa: T201 - CLI
    stale = unconfigured_models(pool, cells, corpus)
    if stale:
        print(f"\nnot in `models:` (data kept, never evaluated): {', '.join(stale)}")  # noqa: T201
    verdicts = _verdicts(rows, cell_floor, prefix_eval.MIN_ROWS)
    if not verdicts:
        return 0
    print("\nINCOMPLETE COLLECTION — the live run has not covered every enabled model:")  # noqa: T201
    for line in verdicts:
        print(f"  {line}")  # noqa: T201
    return 1


if __name__ == "__main__":
    sys.exit(main())
