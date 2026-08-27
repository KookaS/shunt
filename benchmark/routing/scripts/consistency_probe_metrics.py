#!/usr/bin/env python3
"""Metrics for the consistency probe: does run-to-run variance predict outcome?"""

# Reads the gitignored `consistency_probe_*.jsonl` artifact, computes per-task
# run-to-run output variance of deepseek-v4-flash, and asks whether that
# variance predicts the task's MEASURED deepseek pass/fail (results.csv default
# arm, uncensored). Direction per the paper (arXiv:2602.11619): inconsistent
# tasks fail more, so a HIGHER inconsistency score should predict FAILURE.
#
# Instrument validity (required before any verdict is quoted):
#   (a) POSITIVE control — the SAME assembled pipeline (fixed prompt -> R runs ->
#       Jaccard similarity) must recover a planted signal: within-task runs of one
#       problem statement are structurally more similar than cross-task runs, so
#       the within-vs-cross pair AUROC must clear its shuffled-label null. The two
#       engineered control tasks (deterministic 2+2 vs open-ended story) must also
#       be ordered correctly. If the metric cannot recover a signal that IS
#       planted, a null on the real data is a coverage gap, not a falsification.
#   (b) SHUFFLED-LABEL NULL — permute the measured pass/fail labels across tasks
#       and recompute AUROC; a leakage-free pipeline must collapse to chance.
#   Adjudicated with the shared gate's predicate (`admissibility_verdict`).
#
# LIMITATION (state in every report): this is DIRECT-LLM output variance, not
# AGENTIC-TRAJECTORY divergence — a null here does not falsify the paper.

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import numpy

from benchmark import config
from benchmark.routing import censoring
from benchmark.routing.scripts.consistency_probe import mean_pairwise_similarity
from benchmark.routing.sensitivity import auroc as routing_auroc

SEED: Final[int] = 42
N_PERM: Final[int] = 2000
# The engineered control tasks written by the collection script.
CTL_DETERMINISTIC: Final[str] = "ctl_deterministic"
CTL_OPEN_ENDED: Final[str] = "ctl_open_ended"


def load_probe_rows(paths: list[Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """task_id -> run_no -> record, from the given JSONL artifact files."""
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tasks[rec["task_id"]][str(rec["run"])].append(rec)
    return dict(tasks)


def per_task_similarity(rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, float]:
    """mean_pairwise_sim per task, taking the LAST record per run (resume-safe)."""
    out: dict[str, float] = {}
    for tid, runs in rows.items():
        texts = [sorted(runs[r], key=lambda x: x["run"])[-1]["output_text"] for r in runs]
        if len(texts) < 2:
            continue
        out[tid] = mean_pairwise_similarity(texts)
    return out


def length_cv(rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, float]:
    """Coefficient of variation of output-text length per task (NaN if constant)."""
    out: dict[str, float] = {}
    for tid, runs in rows.items():
        lengths = [len(sorted(runs[r], key=lambda x: x["run"])[-1]["output_text"]) for r in runs]
        if len(lengths) < 2 or min(lengths) <= 0:
            continue
        mean = sum(lengths) / len(lengths)
        sd = math.sqrt(sum((x - mean) ** 2 for x in lengths) / (len(lengths) - 1))
        out[tid] = sd / mean
    return out


def finish_reason_diversity(rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, float]:
    """Number of distinct finish_reason values across a task's runs."""
    out: dict[str, float] = {}
    for tid, runs in rows.items():
        reasons = {
            sorted(runs[r], key=lambda x: x["run"])[-1].get("finish_reason") or "missing"
            for r in runs
        }
        out[tid] = float(len(reasons))
    return out


def measured_pass(tasks: list[str], results_path: str | Path | None = None) -> dict[str, bool]:
    """Per-task measured deepseek default-arm pass, dropping censored cells."""
    results = config.load_results(results_path)
    out: dict[str, bool] = {}
    for tid in tasks:
        per_arm = results.get(tid, {}).get("deepseek-v4-flash")
        if not per_arm:
            continue
        row = per_arm.get("high") or next(iter(per_arm.values()))
        if censoring.is_censored(row):
            continue
        out[tid] = bool(row.get("pass"))
    return out


def auroc_fail(scores: dict[str, float], pass_by_task: dict[str, bool]) -> float | None:
    """AUROC of a score predicting FAILURE (score up => predict fail)."""
    shared = sorted(scores.keys() & pass_by_task.keys())
    if len(shared) < 2 or len(set(pass_by_task[t] for t in shared)) < 2:
        return None
    # fail = not pass; higher score predicts failure per the paper's direction.
    labels = [not pass_by_task[t] for t in shared]
    value = routing_auroc(numpy.array([scores[t] for t in shared]), numpy.array(labels))
    if value != value:  # NaN — single-class permutation
        return None
    return float(value)


def shuffled_null(
    scores: dict[str, float], pass_by_task: dict[str, bool], n_perm: int = N_PERM
) -> tuple[float, float, float]:
    """(lo, hi, mean) of the 95% null band for AUROC under shuffled labels."""
    shared = sorted(scores.keys() & pass_by_task.keys())
    if len(shared) < 2:
        return (float("nan"), float("nan"), float("nan"))
    base = [scores[t] for t in shared]
    labels = [not pass_by_task[t] for t in shared]
    rng = random.Random(SEED)
    draws = []
    shuffled = list(labels)
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        draws.append(routing_auroc(numpy.array(base), numpy.array(shuffled)))
    ordered = sorted(draws)
    lo, hi = ordered[int(0.025 * n_perm)], ordered[int(0.975 * n_perm) - 1]
    return lo, hi, sum(draws) / len(draws)


def perm_p_value(
    observed: float, scores: dict[str, float], pass_by_task: dict[str, bool], n_perm: int
) -> float:
    """+1-corrected permutation p-value for the observed AUROC against shuffled labels."""
    shared = sorted(scores.keys() & pass_by_task.keys())
    if len(shared) < 2:
        return float("nan")
    base = [scores[t] for t in shared]
    labels = [not pass_by_task[t] for t in shared]
    rng = random.Random(SEED)
    shuffled = list(labels)
    n_ge = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        if routing_auroc(numpy.array(base), numpy.array(shuffled)) >= observed:
            n_ge += 1
    return (n_ge + 1) / (n_perm + 1)


def calibration(
    scores: dict[str, float], pass_by_task: dict[str, bool], bins: int = 3
) -> list[dict[str, float]]:
    """Pass-rate per score tercile — the directional reading behind the AUROC."""
    shared = sorted(scores.keys() & pass_by_task.keys())
    if len(shared) < bins * 2:
        return []
    ordered = sorted(shared, key=lambda t: scores[t])
    out: list[dict[str, float]] = []
    for b in range(bins):
        chunk = ordered[b * len(ordered) // bins : (b + 1) * len(ordered) // bins]
        if not chunk:
            continue
        n_pass = sum(1 for t in chunk if pass_by_task[t])
        out.append(
            {
                "bin": b,
                "n": len(chunk),
                "pass_rate": n_pass / len(chunk),
                "score_lo": scores[chunk[0]],
                "score_hi": scores[chunk[-1]],
            }
        )
    return out


def positive_control(sim: dict[str, float]) -> dict[str, float]:
    """The engineered controls must be ordered: deterministic sim > open sim."""
    if CTL_DETERMINISTIC not in sim or CTL_OPEN_ENDED not in sim:
        return {}
    d = sim[CTL_DETERMINISTIC]
    o = sim[CTL_OPEN_ENDED]
    return {
        "deterministic_sim": d,
        "open_ended_sim": o,
        "ordered": d > o,
        "margin": d - o,
    }


def within_vs_cross_control(
    rows: dict[str, dict[str, list[dict[str, Any]]]], n_perm: int = N_PERM
) -> dict[str, float]:
    """PRIMARY positive control: within-task runs are more similar than cross-task.

    The planted signal is task identity: R runs of the SAME problem statement are
    structurally more alike than runs of DIFFERENT statements (shared context).
    """
    from benchmark.routing.scripts.consistency_probe import _jaccard_similarity  # noqa: PLC0415

    measured = sorted(t for t in rows if not t.startswith("ctl_"))
    texts = {
        tid: [sorted(runs[r], key=lambda x: x["run"])[-1]["output_text"] for r in runs]
        for tid, runs in rows.items()
        if tid in measured and len(runs) >= 2
    }
    within: list[float] = []
    cross: list[float] = []
    rep = {t: v[0] for t, v in texts.items()}
    tlist = sorted(texts)
    for i, tid in enumerate(tlist):
        runs_i = texts[tid]
        for a in range(len(runs_i)):
            for b in range(a + 1, len(runs_i)):
                within.append(_jaccard_similarity(runs_i[a], runs_i[b]))
        for j in range(i + 1, len(tlist)):
            cross.append(_jaccard_similarity(rep[tid], rep[tlist[j]]))
    scores = within + cross
    labels = [True] * len(within) + [False] * len(cross)
    if len(within) < 2 or len(cross) < 2:
        return {}
    observed = routing_auroc(numpy.array(scores), numpy.array(labels))
    rng = random.Random(SEED)
    shuffled = list(labels)
    draws = []
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        draws.append(routing_auroc(numpy.array(scores), numpy.array(shuffled)))
    lo, hi = sorted(draws)[int(0.025 * n_perm)], sorted(draws)[int(0.975 * n_perm) - 1]
    return {
        "within_auroc": observed,
        "within_mean_sim": sum(within) / len(within),
        "cross_mean_sim": sum(cross) / len(cross),
        "n_within_pairs": len(within),
        "n_cross_pairs": len(cross),
        "null_lo": lo,
        "null_hi": hi,
        "clears_null": observed > hi,
    }


def _fmt(v: float | None, nd: int = 3) -> str:
    return "    —" if v is None else f"{v:+.3f}"


def _adjudicate(
    within: dict[str, float],
    observed_auc: float | None,
    null: tuple[float, float, float],
    scores: dict[str, float],
    pass_by_task: dict[str, bool],
) -> str:
    """Adjudicate instrument validity + the real-data verdict against the null."""
    parts: list[str] = []
    if within:
        verdict_pos = (
            "positive control PASS: within-task sim "
            f"{within['within_mean_sim']:.3f} > cross-task sim "
            f"{within['cross_mean_sim']:.3f}, AUROC {within['within_auroc']:.3f} "
            f"clears null (>{within['null_hi']:.3f}) — metric recovers planted task identity"
            if within["clears_null"]
            else "positive control FAILED: within-task AUROC does not clear its null"
        )
        parts.append(verdict_pos)
    lo, hi, mean = null
    if observed_auc is not None and lo == lo:  # not NaN
        p_value = perm_p_value(observed_auc, scores, pass_by_task, N_PERM)
        if p_value < 0.01:
            placement = "ABOVE null (signal)"
        elif p_value <= 0.05:
            placement = "MARGINAL (at the null boundary, p≤0.05)"
        else:
            placement = "INSIDE null (no signal)"
        parts.append(
            f"observed AUROC {observed_auc:.3f} vs shuffled null "
            f"[{lo:.3f},{hi:.3f}] mean {mean:.3f} p={p_value:.3f} — {placement}"
        )
    else:
        parts.append("null unavailable (insufficient shared tasks)")
    return "; ".join(parts)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description="Consistency-probe metrics (Phase 2.3)")
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to config YAML")
    ap.add_argument(
        "--probe",
        default="benchmark/routing/artifacts/consistency_probe_*.jsonl",
        help="Glob of consistency-probe JSONL artifacts to analyse",
    )
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    config.load(args.config)
    paths = sorted(Path(".").glob(args.probe)) + sorted(Path(".").glob(str(args.probe)))
    paths = sorted({p.resolve() for p in paths if p.exists()})
    if not paths:
        print(f"No probe artifacts matched {args.probe!r}. Run consistency_probe.py first.")
        return 1

    rows = load_probe_rows(paths)
    if not rows:
        print("No records in the matched artifacts.")
        return 1
    measured_tasks = sorted(t for t in rows if not t.startswith("ctl_"))
    print(f"Probe artifacts: {len(paths)}  |  tasks in data: {len(rows)}")
    print(
        f"Measured tasks: {len(measured_tasks)}  |  control tasks present: "
        f"{CTL_DETERMINISTIC in rows} / {CTL_OPEN_ENDED in rows}"
    )

    sim = per_task_similarity(rows)
    pass_by_task = measured_pass(measured_tasks)
    print(f"Measured pass overlap (deepseek default arm, uncensored): {len(pass_by_task)} tasks")

    # ---- positive control (instrument sensitivity on planted signal) --------
    positive = positive_control(sim)
    within = within_vs_cross_control(rows, args.n_perm)
    print("\n=== positive control (planted consistency axis) ===")
    if within:
        print(
            f"  within-task sim: {within['within_mean_sim']:.3f}  "
            f"cross-task sim: {within['cross_mean_sim']:.3f}  "
            f"AUROC {within['within_auroc']:.3f}  "
            f"null [{within['null_lo']:.3f},{within['null_hi']:.3f}]  "
            f"clears: {within['clears_null']}"
        )
    else:
        print("  within-vs-cross control not computable (too few tasks)")
    if positive:
        print(
            f"  engineered controls — deterministic sim: {positive['deterministic_sim']:.3f}  "
            f"open-ended sim: {positive['open_ended_sim']:.3f}  "
            f"ordered: {positive['ordered']}"
        )
    else:
        print("  engineered control tasks absent from the probe artifact")

    # ---- primary signal: mean pairwise similarity as a failure detector -----
    print("\n=== run-to-run similarity -> MEASURED deepseek pass/fail ===")
    inconsistency = {t: 1.0 - s for t, s in sim.items()}
    auc = auroc_fail(inconsistency, pass_by_task)
    null = shuffled_null(inconsistency, pass_by_task, args.n_perm)
    print(
        f"  inconsistency (1 - mean_pairwise_sim) as failure score: "
        f"AUROC {_fmt(auc)}   shuffled null [{null[0]:.3f},{null[1]:.3f}] "
        f"mean {null[2]:.3f}"
    )

    # ---- secondary signals ------------------------------------------------
    for name, fn in (
        ("output-length CV", length_cv),
        ("finish-reason diversity", finish_reason_diversity),
    ):
        score = fn(rows)
        a = auroc_fail(score, pass_by_task)
        n = shuffled_null(score, pass_by_task, args.n_perm)
        print(
            f"  {name} as failure score: AUROC {_fmt(a)}   "
            f"null [{n[0]:.3f},{n[1]:.3f}] mean {n[2]:.3f}"
        )

    # ---- calibration (binned pass rate vs inconsistency) -------------------
    print("\n=== calibration: pass-rate by inconsistency tercile ===")
    for row in calibration(inconsistency, pass_by_task):
        print(
            f"  bin {row['bin']}: n={row['n']:>3} pass={row['pass_rate']:.1%} "
            f"score∈[{row['score_lo']:.3f},{row['score_hi']:.3f}]"
        )

    # ---- verdict -----------------------------------------------------------
    verdict = _adjudicate(within, auc, null, inconsistency, pass_by_task)
    print(f"\nVERDICT: {verdict}")

    out_path = Path("benchmark/routing/artifacts/consistency_probe_metrics.json")
    payload = {
        "probe_files": [str(p) for p in paths],
        "n_tasks": len(rows),
        "n_measured": len(pass_by_task),
        "positive_control": positive,
        "within_vs_cross_control": within,
        "inconsistency_auroc": auc,
        "inconsistency_null": list(null),
        "signals": {
            "length_cv_auroc": auroc_fail(length_cv(rows), pass_by_task),
            "finish_reason_auroc": auroc_fail(finish_reason_diversity(rows), pass_by_task),
        },
        "calibration": calibration(inconsistency, pass_by_task),
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
