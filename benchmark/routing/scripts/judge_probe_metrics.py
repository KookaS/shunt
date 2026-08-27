#!/usr/bin/env python3
"""Metrics for the judge probe (rounds 1+2): do judge labels predict outcomes?"""

# Reads the gitignored `judge_probe_*.jsonl` artifacts and compares each judge's
# difficulty label against real measured pass/fail (deepseek-v4-flash, kimi-k3),
# against the other judges (Fleiss kappa / unanimous agreement), and against the
# known baselines:
#   - embedding LOO R^2  -0.040   null [-0.119, -0.006]
#   - human 3-level tag   R^2 +0.130
#
# The per-judge correlation is reported BOTH in-sample (r2) and leave-one-out
# (r2_loo); the verdict reads the anchor judge against the embedding baseline ON THE
# SAME BASIS — LOO R^2 vs a LOO shuffled-label null, alongside the same-pipeline LOO
# human-tag control.
#
# Round-2 protocol: per-judge runs (R=2 cheap, R=3 anchor, R=1 gpt-5.6-sol) under the
# metadata-augmented prompt (prompt_version 2). Labels are AGGREGATED per (task,
# judge) as the mean difficulty over that judge's round-2 runs; run-to-run stability
# is reported separately (unanimous binarized-label rate + cross-round agreement).
# Round-1 records (bare prompt, one run) are used only as a cross-prompt stability
# reference, never mixed into the round-2 mean.
#
# Instrument validity: the verdict is only admissible alongside (a) a shuffled-label
# null and (b) a positive control — the human difficulty tag through the SAME
# pipeline, which must clear its own null (reproduce a positive R^2). The
# router-analysis (LLM-as-router) step derives each judge's implicit "will a strong
# model solve this?" from the difficulty score, thresholds it, and scores the
# resulting router decision against the measured cheapest-sufficient model and
# against deepseek's measured pass/fail. Judge outputs are synthetic training-signal
# data; nothing here writes results.csv.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score

from benchmark import config
from benchmark.routing import censoring

HUMAN_TAG_ORDER: Final[dict[str, int]] = {"easy": 1, "medium": 2, "hard": 3}
N_PERM: Final[int] = 1000
SEED: Final[int] = 42
ROUND2_PROMPT_VERSION: Final[int] = 2
HARD_THRESHOLD: Final[float] = 4.0
# The measured frontier/control model the LLM-router escalates to, and the cheap
# model it defaults to — both are among the served/measured benchmark models.
FRONTIER_MODEL: Final[str] = "kimi-k3"
CHEAP_MODEL: Final[str] = "deepseek-v4-flash"

# Baseline LOO R^2 values the probe must be read against (see module docstring).
BASELINE_EMBEDDING_R2: Final[float] = -0.040
BASELINE_EMBEDDING_NULL: Final[tuple[float, float]] = (-0.119, -0.006)
BASELINE_HUMAN_TAG_R2: Final[float] = 0.130


def glob_probe_artifacts(pattern: str) -> list[Path]:
    """Resolve a probe-artifact glob, whether relative, absolute, wildcarded or exact.

    ``Path.glob`` rejects absolute patterns and a bare exact path, so an exact path is
    existence-checked and a wildcard globs from the directory of its first wildcard component.
    """
    if not any(ch in pattern for ch in "*?["):
        p = Path(pattern)
        return [p.resolve()] if p.exists() else []
    parts = Path(pattern).parts
    first_wild = next(i for i, part in enumerate(parts) if any(ch in part for ch in "*?["))
    base = Path(*parts[:first_wild]) if first_wild else Path(".")
    rest = "/".join(parts[first_wild:])
    return sorted({p.resolve() for p in base.glob(rest) if p.exists()})


def load_records(paths: list[Path]) -> list[dict]:
    """Every parsed (task, judge, run) record; missing prompt_version = round 1."""
    records: list[dict] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("parsed") and rec.get("difficulty") is not None:
                    rec.setdefault("prompt_version", 1)
                    records.append(rec)
    return records


def aggregate(records: list[dict]) -> dict[tuple[str, str], dict]:
    """(task, judge) -> {mean, n_runs, hardest} over a set of records."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in records:
        groups[(rec["task_id"], rec["judge"])].append(float(rec["difficulty"]))
    return {
        key: {"mean": sum(v) / len(v), "n_runs": len(v), "hardest": max(v)}
        for key, v in groups.items()
    }


def measured_pass(tasks: list[str], model: str) -> dict[str, bool]:
    """Per-task measured pass for *model*, dropping censored cells (unknown truth)."""
    results = config.load_results()
    out: dict[str, bool] = {}
    for tid in tasks:
        per_arm = results.get(tid, {}).get(model)
        if not per_arm:
            continue
        row = next(iter(per_arm.values()))
        if censoring.is_censored(row):
            continue
        out[tid] = bool(row.get("pass"))
    return out


def cheapest_sufficient(tasks: list[str]) -> dict[str, str | None]:
    """Per-task cheapest MEASURED model that passed, or None (censored cells dropped)."""
    results = config.load_results()
    prices = {m: config.cost_per_1m(m) for m in config.enabled_models()}
    out: dict[str, str | None] = {}
    for tid in tasks:
        best: str | None = None
        for model, per_arm in results.get(tid, {}).items():
            if model not in prices:
                continue
            row = next(iter(per_arm.values()))
            if censoring.is_censored(row) or not row.get("pass"):
                continue
            if best is None or prices[model] < prices[best]:
                best = model
        out[tid] = best
    return out


def human_tag(tasks: list[str]) -> dict[str, float]:
    """The task's human 3-level difficulty stratum as an ordinal (easy=1..hard=3)."""
    challenges = config.load_challenges()
    meta = challenges.get("tasks", {})
    out: dict[str, float] = {}
    for tid in tasks:
        stratum = meta.get(tid, {}).get("difficulty_stratum")
        if stratum in HUMAN_TAG_ORDER:
            out[tid] = float(HUMAN_TAG_ORDER[stratum])
    return out


def _leave_one_out_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Exact OLS leave-one-out R^2 on a single-feature fit (PRESS, hat-matrix)."""
    n = len(x)
    design = np.column_stack([np.ones(n), x.ravel()])
    hat = design @ np.linalg.pinv(design.T @ design) @ design.T
    leverage = np.diag(hat)
    resid = y - LinearRegression().fit(x, y).predict(x)
    pred = y - resid / (1.0 - leverage)
    for i in np.where(np.abs(1.0 - leverage) < 1e-9)[0]:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred[i] = LinearRegression().fit(x[mask], y[mask]).predict(x[i : i + 1])[0]
    return float(r2_score(y, pred))


def label_r2_auc(labels: dict[str, float], pass_by_task: dict[str, bool]) -> dict:
    """R^2 (linear fit of pass on label) and ROC AUC (label as score) on overlap.

    r2 is the in-sample fit; r2_loo is the leave-one-out value, the like-for-like
    comparison basis for the embedding LOO baseline.
    """
    shared = sorted(labels.keys() & pass_by_task.keys())
    if len(shared) < 2 or len(set(pass_by_task[t] for t in shared)) < 2:
        return {
            "n": len(shared),
            "r2": None,
            "r2_loo": None,
            "auc": None,
            "pass_rate": None,
        }
    x = np.array([labels[t] for t in shared], dtype=float).reshape(-1, 1)
    y = np.array([int(pass_by_task[t]) for t in shared], dtype=float)
    model = LinearRegression().fit(x, y)
    r2 = r2_score(y, model.predict(x))
    auc = float(roc_auc_score(y, x.ravel()))
    return {
        "n": len(shared),
        "r2": float(r2),
        "r2_loo": _leave_one_out_r2(x, y) if len(shared) >= 3 else None,
        "auc": auc,
        "pass_rate": float(y.mean()),
    }


def shuffled_null(
    labels: dict[str, float],
    pass_by_task: dict[str, bool],
    n_perm: int = N_PERM,
    loo: bool = False,
) -> tuple[float, float, float]:
    """95% null band for R^2 when the labels are shuffled across tasks."""
    shared = sorted(labels.keys() & pass_by_task.keys())
    if len(shared) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    x0 = np.array([labels[t] for t in shared], dtype=float)
    y = np.array([int(pass_by_task[t]) for t in shared], dtype=float)
    draws = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        xp = x0[rng.permutation(len(x0))].reshape(-1, 1)
        if loo:
            draws[i] = _leave_one_out_r2(xp, y)
        else:
            draws[i] = r2_score(y, LinearRegression().fit(xp, y).predict(xp))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return (float(lo), float(hi), float(draws.mean()))


def fleiss_kappa(matrix: np.ndarray) -> float:
    """Fleiss' kappa for *n* subjects x *k* raters on 2 categories."""
    n, k = matrix.shape[0], matrix.sum(axis=1).max()
    if n == 0 or k == 0:
        return float("nan")
    p_j = matrix.sum(axis=0) / (n * k)
    if np.any(p_j == 0):
        return float("nan")
    p_i = ((matrix**2).sum(axis=1) - k) / (k * (k - 1))
    p_bar = float(p_i.mean())
    p_e = float((p_j**2).sum())
    if p_e == 1.0:
        return 0.0
    return float((p_bar - p_e) / (1.0 - p_e))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two raters' binary labels."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    p_a = float(np.mean(a == 1))
    p_b = float(np.mean(b == 1))
    pe = p_a * p_b + (1 - p_a) * (1 - p_b)
    if pe == 1.0:
        return 0.0
    return float((po - pe) / (1 - pe))


def run_stability(records: list[dict], judges: list[str]) -> dict[str, dict]:
    """Per-judge run-to-run stability on the R runs of the round-2 protocol."""
    by_judge: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_judge[rec["judge"]][rec["task_id"]].append(float(rec["difficulty"]))
    out: dict[str, dict] = {}
    for judge in judges:
        tasks = by_judge.get(judge, {})
        multi = {t: v for t, v in tasks.items() if len(v) >= 2}
        if not multi:
            out[judge] = {"n_tasks": len(tasks), "n_multi": 0}
            continue
        hard = {t: [int(d >= HARD_THRESHOLD) for d in v] for t, v in multi.items()}
        unanimous = float(np.mean([len(set(h)) == 1 for h in hard.values()]))
        pair_agrees: list[float] = []
        sds: list[float] = []
        for t, v in multi.items():
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    pair_agrees.append(float(hard[t][i] == hard[t][j]))
            if len(v) > 1:
                sds.append(float(np.std(v)))
        out[judge] = {
            "n_tasks": len(tasks),
            "n_multi": len(multi),
            "unanimous_hard_rate": unanimous,
            "pairwise_agree_rate": float(np.mean(pair_agrees)),
            "mean_std_across_runs": float(np.mean(sds)),
        }
    return out


def router_analysis(
    labels: dict[str, float], deepseek: dict[str, bool], kimi: dict[str, bool]
) -> dict:
    """Router-analysis: difficulty >= threshold -> kimi, else deepseek; co-measured tasks."""
    co = sorted({t for t in labels if t in deepseek and t in kimi})
    if len(co) < 2:
        return {
            "n": len(co),
            "n_easy": 0,
            "n_hard": 0,
            "cheap_pick_pass_rate": None,
            "frontier_pick_pass_rate": None,
            "router_solve_rate": None,
            "auc_vs_deepseek": None,
        }
    easy = [t for t in co if labels[t] < HARD_THRESHOLD]
    hard = [t for t in co if labels[t] >= HARD_THRESHOLD]
    easy_ok = sum(1 for t in easy if deepseek[t]) / len(easy) if easy else float("nan")
    hard_ok = sum(1 for t in hard if kimi[t]) / len(hard) if hard else float("nan")
    router_solve = (sum(1 for t in easy if deepseek[t]) + sum(1 for t in hard if kimi[t])) / len(co)
    auc = None
    if len(set(deepseek[t] for t in co)) >= 2:
        auc = float(roc_auc_score([int(deepseek[t]) for t in co], [labels[t] for t in co]))
    return {
        "n": len(co),
        "n_easy": len(easy),
        "n_hard": len(hard),
        "cheap_pick_pass_rate": easy_ok,  # precision of the "send to deepseek" decision
        "frontier_pick_pass_rate": hard_ok,  # coverage of the "escalate to kimi" decision
        "router_solve_rate": router_solve,
        "auc_vs_deepseek": auc,
    }


def _fmt(v: float | None, nd: int = 3) -> str:
    return "    —" if v is None else f"{v:+.3f}"


def decide_verdict(
    r2_loo: float | None,
    n: int,
    null_loo_hi: float,
    human_r2_loo: float | None,
    positive_control_ok: bool,
) -> str:
    """Overall verdict for the anchor judge, compared on the LOO basis (see docstring)."""
    if (
        r2_loo is None or n < 30 or human_r2_loo is None or null_loo_hi != null_loo_hi  # NaN guard
    ):
        return "INCONCLUSIVE"
    if not positive_control_ok:
        return "INVALID (positive control failed — instrument cannot be read)"
    if r2_loo <= null_loo_hi:
        return "FAIL (anchor R2 inside the shuffled-label null)"
    if r2_loo >= human_r2_loo:
        return "PASS (meets or exceeds the same-pipeline human tag)"
    return "MARGINAL (above null, below the same-pipeline human tag)"


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(
        description="Judge-probe metrics (probe-agreement + router-analysis)"
    )
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to config YAML")
    ap.add_argument(
        "--probe",
        default="benchmark/routing/artifacts/judge_probe_*.jsonl",
        help="Glob of judge-probe JSONL artifacts to analyse",
    )
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    config.load(args.config)
    paths = glob_probe_artifacts(args.probe)
    if not paths:
        print(f"No probe artifacts matched {args.probe!r}. Run judge_probe.py first.")
        return 1

    records = load_records(paths)
    if not records:
        print("No parsed judge records in the matched artifacts.")
        return 1
    r2_records = [r for r in records if r["prompt_version"] == ROUND2_PROMPT_VERSION]
    r1_records = [r for r in records if r["prompt_version"] != ROUND2_PROMPT_VERSION]
    agg = aggregate(r2_records)
    judges = sorted({j for (_t, j) in agg})
    tasks = sorted({t for (t, _j) in agg})
    print(
        f"Probe artifacts: {len(paths)}  |  parsed records: {len(records)} "
        f"(round-2 {len(r2_records)}, round-1 {len(r1_records)})"
    )
    print(f"Round-2 tasks with a label: {len(tasks)}  |  judges: {', '.join(judges)}")

    deepseek = measured_pass(tasks, CHEAP_MODEL)
    kimi = measured_pass(tasks, FRONTIER_MODEL)
    human = human_tag(tasks)
    cheapest = cheapest_sufficient(tasks)
    print(
        f"Measured overlap — deepseek-v4-flash: {len(deepseek)} tasks, "
        f"kimi-k3: {len(kimi)} tasks; human tag on {len(human)}."
    )
    print()

    # ---- run-to-run stability (round-2 runs) ----------------------------------
    print("=== run-to-run stability (round-2 runs per judge) ===")
    stability = run_stability(r2_records, judges)
    for judge in judges:
        s = stability.get(judge, {})
        if s.get("n_multi", 0) == 0:
            print(f"  {judge:<16} n_tasks={s['n_tasks']}  (single run — no stability)")
            continue
        print(
            f"  {judge:<16} n={s['n_tasks']} multi={s['n_multi']}  "
            f"unanimous-hard {s['unanimous_hard_rate']:.1%}  "
            f"pairwise-{s['pairwise_agree_rate']:.1%}  "
            f"mean std {s['mean_std_across_runs']:.2f}"
        )
    for judge in judges:
        r1_judge = {
            rec["task_id"]: rec["difficulty"] for rec in r1_records if rec["judge"] == judge
        }
        r2_judge = {t: agg[(t, judge)]["mean"] for (t, j) in agg if j == judge}
        if r1_judge and r2_judge:
            shared = sorted(r1_judge.keys() & r2_judge.keys())
            a = np.array([int(r1_judge[t] >= HARD_THRESHOLD) for t in shared])
            b = np.array([int(r2_judge[t] >= HARD_THRESHOLD) for t in shared])
            print(
                f"  cross-prompt (round-1 vs round-2, {len(shared)} tasks): {judge:<16} "
                f"binarized Cohen κ {cohen_kappa(a, b):+.3f}"
            )
    print()

    # ---- judge-vs-judge agreement on the AGGREGATED round-2 labels -------------
    shared_all = sorted({t for t in tasks if all((t, j) in agg for j in judges)})
    if len(judges) >= 2 and shared_all:
        hard = np.array(
            [[int(agg[(t, j)]["mean"] >= HARD_THRESHOLD) for j in judges] for t in shared_all],
            dtype=float,
        )
        kappa = fleiss_kappa(np.column_stack([hard.sum(axis=1), len(judges) - hard.sum(axis=1)]))
        agree = np.maximum(hard.sum(axis=1), len(judges) - hard.sum(axis=1))
        unanimous = float(np.mean(agree == len(judges)))
        print(f"=== judge-vs-judge agreement on {len(shared_all)} tasks (hard = mean >= 4) ===")
        print(
            f"Fleiss' kappa (all {len(judges)} judges): {kappa:+.3f}   "
            f"unanimous agreement: {unanimous:.1%}"
        )
        for i, ja in enumerate(judges):
            for jb in judges[i + 1 :]:
                a = hard[:, i]
                b = hard[:, judges.index(jb)]
                print(f"  Cohen's kappa {ja:<14} vs {jb:<14}: {cohen_kappa(a, b):+.3f}")
        print()

    # ---- per-judge aggregated label vs MEASURED pass ---------------------------
    print("=== per-judge (mean difficulty, round-2) vs MEASURED pass ===")
    header = (
        f"{'judge':<16} {'target':<18} {'n':>4} {'R2':>8} {'R2_loo':>8} {'AUC':>7} {'pass%':>6}"
    )
    print(header)
    print("-" * len(header))
    summary: dict[str, dict] = {}
    for judge in judges:
        labels = {t: agg[(t, judge)]["mean"] for (t, j) in agg if j == judge}
        for target, pass_by_task in ((CHEAP_MODEL, deepseek), (FRONTIER_MODEL, kimi)):
            res = label_r2_auc(labels, pass_by_task)
            summary.setdefault(judge, {})[target] = res
            lo, hi, _mean = shuffled_null(labels, pass_by_task, args.n_perm)
            lo_loo, hi_loo, _mean_loo = shuffled_null(labels, pass_by_task, args.n_perm, loo=True)
            res["null"] = (lo, hi)
            res["null_loo"] = (lo_loo, hi_loo)
            null_s = f"[{lo:+.3f},{hi:+.3f}]" if lo == lo else "   n/a "
            null_loo_s = f"[{lo_loo:+.3f},{hi_loo:+.3f}]" if lo_loo == lo_loo else "   n/a "
            print(
                f"{judge:<16} {target:<18} {res['n']:>4} {_fmt(res['r2']):>8} "
                f"{_fmt(res['r2_loo']):>8} "
                f"{res['auc'] if res['auc'] is not None else float('nan'):>7.3f} "
                f"{res['pass_rate'] * 100 if res['pass_rate'] is not None else float('nan'):>6.1f} "
                f"  nullR2 {null_s}  nullR2_loo {null_loo_s}"
            )
        vals = [agg[(t, judge)]["mean"] for (t, j) in agg if j == judge]
        hard_rate = sum(1 for v in vals if v >= HARD_THRESHOLD) / len(vals)
        mean_d = sum(vals) / len(vals)
        print(f"  (difficulty distribution: mean {mean_d:.2f}, hard-rate {hard_rate:.1%})")
    print()

    # ---- human-tag positive control + baselines --------------------------------
    print("=== positive control (human 3-level tag, same pipeline) + baselines ===")
    for target, pass_by_task in ((CHEAP_MODEL, deepseek), (FRONTIER_MODEL, kimi)):
        res = label_r2_auc(human, pass_by_task)
        lo, hi, _m = shuffled_null(human, pass_by_task, args.n_perm)
        lo_loo, hi_loo, _ml = shuffled_null(human, pass_by_task, args.n_perm, loo=True)
        print(
            f"human tag -> {target:<18} n={res['n']:>3} R2={_fmt(res['r2'])} "
            f"R2_loo={_fmt(res['r2_loo'])} "
            f"AUC={res['auc'] if res['auc'] is not None else float('nan'):.3f} "
            f"nullR2 [{lo:+.3f},{hi:+.3f}] nullR2_loo [{lo_loo:+.3f},{hi_loo:+.3f}]"
        )
    print(
        f"Known baselines: embedding LOO R2 {BASELINE_EMBEDDING_R2:+.3f} "
        f"null {BASELINE_EMBEDDING_NULL}; human 3-level tag R2 {BASELINE_HUMAN_TAG_R2:+.3f}"
    )
    print()

    # ---- family-bias structure ---------------------------------------------------
    print("=== family-bias structure (systematic judge disagreement) ===")
    for judge in judges:
        vals = [agg[(t, judge)]["mean"] for (t, j) in agg if j == judge]
        dist = Counter(int(v) for v in vals)
        dist_s = " ".join(f"{k}:{dist.get(k, 0)}" for k in range(1, 6))
        print(
            f"  {judge:<16} n={len(vals):>3}  difficulty dist {{{dist_s}}}  "
            f"mean={sum(vals) / len(vals):.2f} "
            f"hard%={sum(1 for v in vals if v >= HARD_THRESHOLD) / len(vals):.1%}"
        )

    # Binned pass rate per difficulty level — the directional reading behind R^2.
    print("\n=== binned pass-rate by difficulty (deepseek-v4-flash) ===")
    for judge in judges:
        labels = {t: agg[(t, judge)]["mean"] for (t, j) in agg if j == judge}
        cells: list[tuple[str, int, int]] = []
        for level in (1.0, 2.0, 3.0, 4.0, 5.0):
            subset = [t for t in labels if labels[t] >= level - 0.5 and labels[t] < level + 0.5]
            subset = [t for t in subset if t in deepseek]
            if subset:
                n_pass = sum(1 for t in subset if deepseek[t])
                cells.append((f"{level:.0f}", n_pass, len(subset)))
        row = "  ".join(f"d={lv}:{np}/{n} ({np / n:.0%})" for lv, np, n in cells)
        print(f"  {judge:<16} {row}")
    print()

    # ---- router-analysis ----------------------------------------------------------
    print(
        "=== router-analysis: LLM-as-router (difficulty >= 4 -> escalate to kimi-k3, "
        "else deepseek) ==="
    )
    router_rows: dict[str, dict] = {}
    co_measured = sorted({t for t in tasks if t in deepseek and t in kimi})
    ds_only = (
        sum(1 for t in co_measured if deepseek[t]) / len(co_measured)
        if co_measured
        else float("nan")
    )
    ki_only = (
        sum(1 for t in co_measured if kimi[t]) / len(co_measured) if co_measured else float("nan")
    )
    print(
        f"  co-measured n={len(co_measured)} — deepseek-only solve rate {ds_only:.1%}; "
        f"kimi-only solve rate {ki_only:.1%}"
    )
    for judge in judges:
        labels = {t: agg[(t, judge)]["mean"] for (t, j) in agg if j == judge}
        ra = router_analysis(labels, deepseek, kimi)
        router_rows[judge] = ra
        # Under/over-provisioning vs the measured cheapest-sufficient model.
        co = {t: labels[t] for t in co_measured if t in labels}
        under = sum(1 for t in co if labels[t] < HARD_THRESHOLD and cheapest.get(t) != CHEAP_MODEL)
        over = sum(1 for t in co if labels[t] >= HARD_THRESHOLD and cheapest.get(t) == CHEAP_MODEL)
        under_n = sum(1 for t in co if labels[t] < HARD_THRESHOLD)
        over_n = sum(1 for t in co if labels[t] >= HARD_THRESHOLD)
        print(
            f"  {judge:<16} easy={ra['n_easy']} hard={ra['n_hard']}  "
            f"cheap-pick pass {ra['cheap_pick_pass_rate']:.1%}  "
            f"frontier-pick pass {ra['frontier_pick_pass_rate']:.1%}  "
            f"router-solve {ra['router_solve_rate']:.1%}  "
            f"AUC(->deepseek) {ra['auc_vs_deepseek']:.3f}  "
            f"under-provision {under}/{under_n}  over-provision {over}/{over_n}"
        )
    print()

    # ---- probe-gate verdict on the REVISED (aggregated, round-2) labels ----------
    anchor = summary.get("claude-sonnet-5", {}).get(CHEAP_MODEL, {})
    r2_loo = anchor.get("r2_loo")
    null_loo_hi = anchor.get("null_loo", (float("nan"), float("nan")))[1]
    human_res = label_r2_auc(human, deepseek)
    human_r2_loo = human_res.get("r2_loo")
    human_null_loo_hi = shuffled_null(human, deepseek, args.n_perm, loo=True)[1]

    # Positive control: the same-pipeline human tag must clear its own null, or the
    # instrument cannot be read.
    if human_r2_loo is not None and human_null_loo_hi == human_null_loo_hi:
        control_ok = human_r2_loo > human_null_loo_hi
        print(
            f"positive control (human tag, LOO) above its null: {'PASS' if control_ok else 'FAIL'}"
        )
    else:
        control_ok = False
        print("positive control (human tag, LOO) above its null: n/a (insufficient data)")

    verdict = decide_verdict(r2_loo, anchor.get("n", 0), null_loo_hi, human_r2_loo, control_ok)
    print(
        f"VERDICT (anchor claude-sonnet-5 {CHEAP_MODEL} LOO R2 vs LOO null + same-pipeline "
        f"human tag): {verdict}"
    )

    # Persist a gitignored summary for the owner.
    out_path = Path("benchmark/routing/artifacts/judge_probe_metrics.json")
    per_judge: dict[str, dict] = {}
    for judge, targets in summary.items():
        per_judge[judge] = {"correlation": {}, "router": router_rows.get(judge)}
        for target, m in targets.items():
            clean = {k: v for k, v in m.items() if k not in ("null", "null_loo")}
            clean["null"] = list(m["null"])
            clean["null_loo"] = list(m["null_loo"])
            per_judge[judge]["correlation"][target] = clean
    payload = {
        "probe_files": [str(p) for p in paths],
        "n_records": len(records),
        "n_round2_records": len(r2_records),
        "per_judge": per_judge,
        "stability": {
            j: {k: v for k, v in s.items() if k != "n_tasks"} for j, s in stability.items()
        },
        "agreement": {
            "fleiss_kappa": kappa if len(judges) >= 2 and shared_all else None,
            "unanimous_agreement": unanimous if len(judges) >= 2 and shared_all else None,
        },
        "positive_control": {
            "human_tag_r2": human_res.get("r2"),
            "human_tag_r2_loo": human_res.get("r2_loo"),
            "clears_loo_null": control_ok if human_r2_loo is not None else None,
        },
        "baselines": {
            "embedding_loo_r2": BASELINE_EMBEDDING_R2,
            "embedding_loo_null": list(BASELINE_EMBEDDING_NULL),
            "human_tag_r2": human_res.get("r2"),
            "human_tag_r2_loo": human_res.get("r2_loo"),
            "human_tag_r2_published": BASELINE_HUMAN_TAG_R2,
        },
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
