"""Select the discriminating 100 SWE-bench Verified tasks for the routing run.

Ranks Verified instances by ``p_frontier - p_cheap`` resolve gap (from public
``SWE-bench/experiments`` data), stratified ~30/50/20 easy/medium/hard.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from benchmark.runner import swebench_specs

# Substring cohorts for classifying an experiments submission dir by model tier.
DEFAULT_CHEAP_MARKERS = ("deepseek", "qwen", "qwen3", "kimi", "glm", "codestral", "mistral")
DEFAULT_FRONTIER_MARKERS = ("opus", "gpt-5", "gpt5", "sonnet", "o1", "o3", "gemini-2.5-pro")
DEFAULT_STRATA: Final = {"easy": 30, "medium": 50, "hard": 20}

# instance_id -> {submission -> resolved}
ResolveTable = dict[str, dict[str, bool]]


def _cohort_rate(row: dict[str, bool], cohort: set[str]) -> float | None:
    """Resolve rate over the cohort submissions present for this instance."""
    vals = [row[s] for s in row if s in cohort]
    return sum(vals) / len(vals) if vals else None


def discrimination_gap(
    row: dict[str, bool],
    cheap: set[str],
    frontier: set[str],
) -> float:
    """``p_frontier - p_cheap`` for one instance; 0.0 when a cohort is absent."""
    p_cheap = _cohort_rate(row, cheap)
    p_frontier = _cohort_rate(row, frontier)
    if p_cheap is None or p_frontier is None:
        return 0.0
    return p_frontier - p_cheap


def classify_submissions(
    submissions: Iterable[str],
    cheap_markers: tuple[str, ...] = DEFAULT_CHEAP_MARKERS,
    frontier_markers: tuple[str, ...] = DEFAULT_FRONTIER_MARKERS,
) -> tuple[set[str], set[str]]:
    """Split submission names into (cheap, frontier) cohorts by name substring."""
    cheap: set[str] = set()
    frontier: set[str] = set()
    for name in submissions:
        low = name.lower()
        if any(m in low for m in frontier_markers):
            frontier.add(name)
        elif any(m in low for m in cheap_markers):
            cheap.add(name)
    return cheap, frontier


def select(
    table: ResolveTable,
    difficulty: dict[str, str],
    cheap: set[str],
    frontier: set[str],
    strata: dict[str, int] | None = None,
) -> list[str]:
    """Return the stratified, gap-maximising instance ids (deterministic order)."""
    targets = strata or DEFAULT_STRATA
    scored: dict[str, list[tuple[float, str]]] = {k: [] for k in targets}
    for iid, row in table.items():
        stratum = difficulty.get(iid, "medium")
        if stratum not in scored:
            continue
        scored[stratum].append((discrimination_gap(row, cheap, frontier), iid))
    selected: list[str] = []
    for stratum, want in targets.items():
        ranked = sorted(scored.get(stratum, []), key=lambda t: (-t[0], t[1]))
        selected.extend(iid for _, iid in ranked[:want])
    return selected


def load_experiments_resolves(root: Path, subset: str = "verified") -> ResolveTable:
    """Read ``<root>/evaluation/<subset>/*/results/results.json`` into a table.

    Each submission's ``results.json`` lists the ``resolved`` instance ids; we
    record resolved True/False per instance across the union of instances seen.
    """
    base = root / "evaluation" / subset
    table: ResolveTable = {}
    submission_instances: dict[str, tuple[set[str], set[str]]] = {}
    for results_file in sorted(base.glob("*/results/results.json")):
        submission = results_file.parent.parent.name
        data = json.loads(results_file.read_text())
        resolved = set(data.get("resolved", []))
        no_gen = set(data.get("no_generation", [])) | set(data.get("no_logs", []))
        submission_instances[submission] = (resolved, no_gen)
    all_instances: set[str] = set()
    for resolved, no_gen in submission_instances.values():
        all_instances |= resolved | no_gen
    for iid in all_instances:
        table[iid] = {}
        for submission, (resolved, no_gen) in submission_instances.items():
            if iid in resolved:
                table[iid][submission] = True
            elif iid in no_gen:
                table[iid][submission] = False
    return table


def _difficulty_map_from_hf() -> dict[str, str]:
    from datasets import load_dataset

    ds = load_dataset(swebench_specs.DATASET_NAME, split=swebench_specs.DATASET_SPLIT)
    return {
        str(r["instance_id"]): swebench_specs.difficulty_stratum(str(r.get("difficulty", "")))
        for r in ds
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Select the discriminating 100 Verified tasks.")
    ap.add_argument("experiments_root", type=Path, help="Local clone of SWE-bench/experiments")
    ap.add_argument("--subset", default="verified", help="experiments subset dir")
    ap.add_argument("--materialize", action="store_true", help="Write specs for the selection")
    ap.add_argument("--namespace", default=swebench_specs.DEFAULT_NAMESPACE)
    args = ap.parse_args()

    table = load_experiments_resolves(args.experiments_root, args.subset)
    cheap, frontier = classify_submissions({s for row in table.values() for s in row})
    difficulty = _difficulty_map_from_hf()
    selected = select(table, difficulty, cheap, frontier)
    print(
        f"Selected {len(selected)} instances "
        f"(cheap cohort={len(cheap)}, frontier cohort={len(frontier)})"
    )
    for iid in selected:
        print(f"  {iid}  [{difficulty.get(iid, '?')}]")
    if args.materialize:
        paths = swebench_specs.materialize(selected, namespace=args.namespace)
        print(f"Materialised {len(paths)} specs -> {swebench_specs.spec_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
