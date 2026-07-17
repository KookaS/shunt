"""Select the discriminating SWE-bench Verified tasks for the routing run.

Uses ``routing_headroom`` (``max(0, p_frontier - p_solve)``) as the
selection signal — never the degenerate ``p_cheap`` from the leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.runner import swebench_specs

_EXT_CSV: Final = (
    Path(__file__).resolve().parent.parent / "routing" / "data" / "external_swebench.csv"
)

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


def routing_headroom(p_solve: float, p_frontier: float) -> float:
    """Escalation value: ``max(0, p_frontier - p_solve)`` — peaks in the routable middle band."""
    return max(0.0, p_frontier - p_solve)


def load_external_rates(csv_path: Path = _EXT_CSV) -> dict[str, tuple[float, float]]:
    """{instance_id: (p_solve, p_frontier)} from the corrected external CSV."""
    out: dict[str, tuple[float, float]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            ps, pf = row.get("p_solve", ""), row.get("p_frontier", "")
            if not (ps and pf):
                continue
            try:  # skip a malformed row rather than aborting the whole load
                out[row["instance_id"]] = (float(ps), float(pf))
            except ValueError:
                continue
    return out


def rank_by_headroom(
    rates: dict[str, tuple[float, float]],
) -> list[tuple[str, float, float, float]]:
    """(instance_id, headroom, p_solve, p_frontier) sorted by descending headroom
    (ties broken by instance_id for determinism)."""
    scored = [(iid, routing_headroom(ps, pf), ps, pf) for iid, (ps, pf) in rates.items()]
    return sorted(scored, key=lambda t: (-t[1], t[0]))


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


def _current_task_ids() -> list[str]:
    """Instance ids in the committed challenges manifest (empty if unreadable)."""
    try:
        return sorted(json.loads(config.challenges_path().read_text()).get("tasks", {}))
    except (FileNotFoundError, ValueError, KeyError):
        return []


def headroom_dry_run(top_n: int = 10) -> int:
    """Print the top-N routing-headroom tasks and the current selection's headroom,
    so the owner can compare without re-running the (costly) live matrix."""
    rates = load_external_rates()
    if not rates:
        print(f"No external rates at {_EXT_CSV} — cannot rank by headroom.")
        return 1
    ranked = rank_by_headroom(rates)
    current = set(_current_task_ids())
    print(f"Top {top_n} by routing_headroom = max(0, p_frontier - p_solve):")
    for iid, hr, ps, pf in ranked[:top_n]:
        mark = "  <-- already selected" if iid in current else ""
        print(f"  {iid:34} headroom={hr:.2f}  p_solve={ps:.2f} p_frontier={pf:.2f}{mark}")
    if current:
        picked = [(i, *rates[i]) for i in sorted(current) if i in rates]
        cur_mean = sum(routing_headroom(ps, pf) for _, ps, pf in picked) / max(1, len(picked))
        top_mean = sum(hr for _, hr, _, _ in ranked[:top_n]) / max(1, top_n)
        print(f"\nCurrent selection mean headroom: {cur_mean:.3f}  (n={len(picked)})")
        print(
            f"Headroom-optimized top-{top_n} mean: {top_mean:.3f}  "
            f"→ {top_mean / cur_mean:.1f}x more routable"
            if cur_mean
            else ""
        )
    return 0


def enrich_challenges_with_rates(path: Path | None = None) -> int:
    """Add derived ``p_solve``/``p_frontier`` to each task in the challenges manifest
    (join on instance_id; re-runnable). These are DERIVED from external_swebench.csv —
    regenerate whenever that CSV changes; never hand-edit."""
    manifest_path = path or config.challenges_path()
    rates = load_external_rates()
    data = json.loads(manifest_path.read_text())
    n = 0
    for iid, task in data.get("tasks", {}).items():
        if iid in rates:
            ps, pf = rates[iid]
            task["p_solve"], task["p_frontier"] = round(ps, 4), round(pf, 4)
            task["routing_headroom"] = round(routing_headroom(ps, pf), 4)
            n += 1
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    print(f"Enriched {n} tasks with derived p_solve/p_frontier/routing_headroom → {manifest_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Select the discriminating Verified tasks.")
    ap.add_argument(
        "experiments_root", type=Path, nargs="?", help="Local SWE-bench/experiments clone"
    )
    ap.add_argument("--subset", default="verified", help="experiments subset dir")
    ap.add_argument("--materialize", action="store_true", help="Write specs for the selection")
    ap.add_argument("--namespace", default=swebench_specs.DEFAULT_NAMESPACE)
    ap.add_argument(
        "--headroom", type=int, metavar="N", help="Dry-run: rank top-N by routing headroom"
    )
    ap.add_argument(
        "--enrich-challenges", action="store_true", help="Add derived p_solve to challenges.json"
    )
    args = ap.parse_args()

    config.load()
    if args.headroom is not None:
        return headroom_dry_run(args.headroom)
    if args.enrich_challenges:
        return enrich_challenges_with_rates()
    if args.experiments_root is None:
        ap.error("experiments_root is required unless --headroom/--enrich-challenges is given")

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
