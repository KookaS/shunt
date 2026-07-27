#!/usr/bin/env python3
"""Plots for the external SWE-bench signal — separately and alongside our runs."""

from __future__ import annotations

# Three deterministic figures (Agg PNGs embed no timestamp → byte-stable on regen):
# * external_difficulty.png   — SWE-bench values ALONE: p_solve/p_cheap/p_frontier +
#                               routing-headroom (p_frontier − p_solve).
# * ours_vs_external.png      — TOGETHER on our tasks: external p_solve vs our cheap-model
#                               pass; per-task bars for a small set, a 2x2 agreement
#                               matrix at scale (exposes field-easy-yet-we-fail cases).
# * heldout_generalization.png — out-of-sample test over the held-out (unmeasured)
#                               external instances.
# Light plots need only the CSVs; the held-out plot needs the embedding cache.
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec

_EXT_CSV = Path(__file__).resolve().parents[1] / "data" / "external_swebench.csv"

_DIFFICULTY_SPEC = FigureSpec(
    reading=(
        "Four histograms over SWE-bench Verified leaderboard instances; each bar counts "
        "instances, the x axis is a resolve rate in 0-1. p_solve (blue) is the share of ALL "
        "leaderboard submissions that solved the instance and is the only usable difficulty "
        "signal. p_cheap (grey) and p_frontier (orange) are the same rate within the "
        "open-weight and the proprietary cohort. The fourth panel is routing headroom, "
        "p_frontier minus p_solve: above 0 means the frontier cohort out-resolves the field "
        "on that instance."
    ),
    goal=(
        "Look for a fat left tail in p_solve (genuinely hard instances) and mass above 0 in "
        "the headroom panel — headroom piled at 0 means there is nothing worth routing."
    ),
    definitions=(
        ("instance", "one SWE-bench Verified task"),
        ("resolve rate", "share of leaderboard submissions whose patch resolved the instance"),
        ("headroom", "p_frontier minus p_solve: how far the frontier cohort beats the field"),
    ),
    notes=(
        "This is a PRIOR from other people's leaderboard submissions, not our own measured runs.",
        "The older gap = p_frontier minus p_cheap was signal-free (it reduced to p_frontier "
        "minus 1) and was replaced by headroom.",
    ),
    limitations=(
        "Leaderboard submissions are a self-selected, biased population: nobody publishes "
        "their worst run.",
        "The cheap/frontier cohort split is a heuristic tiering applied upstream, not a "
        "property of the benchmark.",
        "Blank cells are dropped rather than imputed, so the four panels do not share an n.",
    ),
)

_OURS_BARS_SPEC = FigureSpec(
    reading=(
        "One pair of bars per task, one task per x tick. Blue is the SWE-bench leaderboard "
        "resolve rate p_solve for that instance (0-1); green is our own cheapest enabled "
        "model's outcome on it, drawn as 1.0 for a pass and 0 for a fail. The dotted grey "
        "line at 0.5 is the cut a router keying on the prior alone would use to decide "
        "whether to escalate. A tall blue bar beside a zero green bar is the dangerous case: "
        "the field finds it easy, we fail."
    ),
    goal=(
        "Look at the red callouts — each marks a task the leaderboard calls easy where our "
        "cheap model actually fails, which a prior-only router would silently ship."
    ),
    definitions=(
        ("p_solve", "share of SWE-bench leaderboard submissions that solved the instance"),
        ("escalate threshold", "the 0.5 cut on the prior above which a router would stay cheap"),
    ),
    limitations=(
        "PRIOR LEAKAGE: our tasks are drawn from the same SWE-bench Verified set the prior is "
        "computed over, so this is not independent evidence.",
        "A pass/fail bit is being compared against a continuous rate at an arbitrary 0.5 cut.",
        "Failure callouts are capped so the figure stays readable — not every failure is "
        "annotated.",
    ),
)

_OURS_AGREEMENT_SPEC = FigureSpec(
    reading=(
        "A 2x2 count of our own tasks. Rows split them by the leaderboard prior — EASY is "
        "p_solve >= 0.5, HARD is below it. Columns split them by whether our cheapest enabled "
        "model passed. The number in each cell is a task count; the fill shade and the "
        "colourbar encode the same count. The red-ringed (EASY, FAILS) cell is the point of "
        "the figure: tasks the leaderboard calls easy where our cheap model fails, so a router "
        "keying only on the prior would not escalate and would silently ship a failure."
    ),
    goal=(
        "Look at the red-ringed top-right cell and hope it is near empty; mass on the diagonal "
        "means the prior agrees with our own outcomes."
    ),
    definitions=(
        ("p_solve", "share of SWE-bench leaderboard submissions that solved the instance"),
        ("EASY / HARD", "the prior's own 0.5 cut on p_solve, not a label the benchmark ships"),
    ),
    limitations=(
        "PRIOR LEAKAGE: our tasks are drawn from the same SWE-bench Verified set the prior is "
        "computed over, so this is not independent evidence.",
        "A pass/fail bit is being compared against a continuous rate at an arbitrary 0.5 cut.",
        "Cell shading is auto-scaled to the largest cell, so shade compares cells within this "
        "figure only.",
    ),
)

_HELDOUT_SPEC = FigureSpec(
    reading=(
        "Three panels over the held-out SWE-bench Verified instances, leave-one-out. Panel 0 "
        "is the headline and is threshold-free: each dot is one instance at (mean p_solve of "
        "its 20 embedding neighbours, its own p_solve), with a dashed y = x. Dots along that "
        "line would mean neighbours predict difficulty; a flat cloud squeezed toward the mean "
        "means they do not. Panel 1 is each router's tier accuracy against the oracle. Panel 2 "
        "is each router's average reward, resolve rate minus 0.1 times cost."
    ),
    goal=(
        "Read panel 0 first for spread along y = x, then panel 2 for the gap between the "
        "Reward-Oracle bound and Always-Cheap — that gap is the most any router could win here."
    ),
    definitions=(
        ("leave-one-out", "an instance's neighbours never include the instance itself"),
        ("tier accuracy", "share of instances routed to the tier the oracle would have picked"),
        ("reward", "resolve rate minus 0.1 x cost, so higher is better"),
        ("Reward-Oracle", "hindsight-perfect router: an upper bound, not something achievable"),
    ),
    notes=(
        "corr(k=1) is the honest nearest-neighbour test; corr(k=20) is INFLATED by regression "
        "toward the global mean, because averaging 20 neighbours pulls every prediction to the "
        "middle. Read k=1 first.",
        "Panel 1 does not show Neighbour tying Always-Cheap on merit: a k=20 mean cannot cross "
        "the 0.5 threshold, so Neighbour escalates nothing and IS Always-Cheap — a non-test.",
    ),
    limitations=(
        "A correlation interval that excludes zero means the signal is DETECTABLE, not that it "
        "is USEFUL — the decision that matters is the reward gap in panel 2.",
        "Everything here is scored on external leaderboard resolve rates, not on our own "
        "verified pass/fail outcomes.",
        "Rewards depend on the two-tier cost model, so the panel-2 ranking moves when prices move.",
    ),
)


def _floats(rows: list[dict], col: str) -> np.ndarray:
    return np.array([float(r[col]) for r in rows if r.get(col) not in ("", None)], dtype=float)


def plot_external_difficulty(ext_csv: Path, out_dir: Path) -> Path:
    """Distributions of the external per-instance resolve rates (500 instances)."""
    rows = list(csv.DictReader(ext_csv.open()))
    pc = _floats(rows, "p_cheap")
    degenerate = bool(len(pc) and np.allclose(pc, 1.0))
    # Real routing headroom per instance: how far a frontier model out-resolves the
    # field. Replaces the signal-free `gap` (= p_frontier − p_cheap ≡ p_frontier − 1).
    headroom = np.array(
        [
            float(r["p_frontier"]) - float(r["p_solve"])
            for r in rows
            if r.get("p_frontier") not in ("", None) and r.get("p_solve") not in ("", None)
        ],
        dtype=float,
    )
    # Validated categorical palette (dataviz reference). p_cheap is drawn in MUTED
    # GRAY on purpose — it carries no signal, so it must not read as a live series.
    series = [
        (
            "p_solve — the usable difficulty signal",
            _floats(rows, "p_solve"),
            "#2a78d6",  # slot 1 blue — the signal downstream keys on
        ),
        (
            "p_cheap (open-weight) — DEGENERATE ≡ 1.0, carries NO signal"
            if degenerate
            else "p_cheap (open-weight)",
            pc,
            "#898781",  # muted gray — a dead column, not a live series
        ),
        ("p_frontier (proprietary cohort)", _floats(rows, "p_frontier"), "#eb6834"),  # slot 2
        ("routing headroom = p_frontier − p_solve  (the real gap)", headroom, "#1baf7a"),  # slot 3
    ]
    xlabels = (
        "resolve rate  (fraction of leaderboard submissions that solved the instance)",
        "resolve rate  (open-weight cohort — all 1.0, see note)",
        "resolve rate  (proprietary/frontier cohort)",
        "p_frontier − p_solve  ( >0 ⇒ frontier beats the field ⇒ routable )",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panel_ns = []
    for ax, (title, vals, color), xlabel in zip(axes.ravel(), series, xlabels, strict=True):
        panel_ns.append(len(vals))
        vmin = float(vals.min()) if vals.size else 0.0  # guard: empty/all-blank column
        median = float(np.median(vals)) if vals.size else float("nan")
        ax.hist(
            vals,
            bins=np.linspace(min(0.0, vmin), 1.0, 21),
            color=color,
            edgecolor="white",
        )
        ax.set_xlim(min(-0.05, vmin), 1.0)
        ax.set_title(f"{title}  (n={len(vals)}, median={median:.2f})", fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("# instances", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    pcheap_note = (
        "cheap cohort reports only\nRESOLVED instances →\n"
        "rate ≡ 1.0, carries no signal.\nDownstream keys on p_solve."
    )
    axes.ravel()[1].text(
        0.5,
        0.55,
        pcheap_note,
        transform=axes.ravel()[1].transAxes,
        ha="center",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#D55E00", alpha=0.9),
    )
    fig.suptitle(
        "SWE-bench Verified LEADERBOARD difficulty prior — per-instance resolve rates "
        f"(n={len(rows)}) · a prior from others' submissions, NOT our own runs"
    )
    fig.tight_layout()
    routable = int((headroom > 0).sum())
    notes = [
        f"{len(rows)} leaderboard instances read; {routable} of {len(headroom)} scored ones "
        f"have headroom above 0, so that is the routable share.",
    ]
    limits = [f"Panel n differs across the four columns: {', '.join(str(v) for v in panel_ns)}."]
    if degenerate:
        limits.append(
            "p_cheap is degenerate at exactly 1.0 — the open-weight cohort reports only "
            "RESOLVED instances — so it carries no signal and is drawn grey; everything "
            "downstream keys on p_solve."
        )
    return plot_frame.save(
        fig,
        out_dir / "external_difficulty.png",
        _DIFFICULTY_SPEC,
        extra=Annotations(notes=tuple(notes), limitations=tuple(limits)),
    )


def _cheapest_cheap_model() -> str:
    """Name of the cheapest model on the `cheap` tier — the one 'our cheap model' means."""
    pricing = config.load_pricing()
    cheap = [m for m, i in pricing.items() if isinstance(i, dict) and i.get("tier") == "cheap"]
    return min(
        cheap,
        key=lambda m: (
            pricing[m].get("input_cost_per_1m", 0) + pricing[m].get("output_cost_per_1m", 0)
        ),
        default="deepseek-v4-flash",
    )


def _our_cheap_pass(results_csv: Path, cheapest: str) -> dict[str, bool]:
    """{instance_id: did our cheapest model pass} from results.csv."""
    out: dict[str, bool] = {}
    for r in csv.DictReader(results_csv.open()):
        if r["model"] == cheapest:
            out[r["challenge_id"]] = str(r["pass"]).lower() in ("true", "1")
    return out


_BARS_MAX_TASKS = 40  # beyond this, per-task bars are unreadable → 2x2 agreement view
_FAIL_ANNOT_CAP = 8  # cap per-task failure callouts so the small-N figure stays clean


def plot_ours_vs_external(results_csv: Path, ext_csv: Path, out_dir: Path) -> Path:
    """External field-wide resolve rate vs our own cheapest-model pass, on our tasks.

    Per-task bars for a small set; a 2x2 agreement matrix once there are too many
    tasks to draw one bar each — so the figure stays readable at 500+ tasks.
    """
    ext = {r["instance_id"]: r for r in csv.DictReader(ext_csv.open())}
    cheapest = _cheapest_cheap_model()
    ours = _our_cheap_pass(results_csv, cheapest)
    ids = sorted(ours)
    # p_solve (field-wide difficulty) — p_cheap is degenerate (see load_external_priors).
    ext_rate = np.array(
        [float(ext[i]["p_solve"]) if i in ext and ext[i]["p_solve"] else np.nan for i in ids]
    )
    our_pass = np.array([1.0 if ours[i] else 0.0 for i in ids])

    path = out_dir / "ours_vs_external.png"
    if len(ids) <= _BARS_MAX_TASKS:
        return _draw_ours_bars(ids, ext_rate, our_pass, path, cheapest)
    return _draw_ours_agreement(ext_rate, our_pass, path, cheapest)


def _draw_ours_bars(
    ids: list[str], ext_rate: np.ndarray, our_pass: np.ndarray, path: Path, cheapest: str
) -> Path:
    """Per-task grouped bars (small N): external p_solve vs our cheap-model outcome."""
    n = len(ids)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(min(24.0, max(8.0, 0.55 * n)), 6))
    ax.bar(
        x - 0.2,
        ext_rate,
        0.4,
        label="SWE-bench leaderboard resolve rate p_solve (0–1, blue)",
        color="#2196F3",
    )
    ax.bar(
        x + 0.2,
        our_pass,
        0.4,
        label="our cheapest model: PASS→1.0, FAIL→0 (green)",
        color="#4CAF50",
    )
    # Annotate failures — misleading (field-easy ∧ we-fail) first, capped so it stays clean.
    fails = [
        (int(xi), float(pr)) for xi, p, pr in zip(x, our_pass, ext_rate, strict=True) if p == 0
    ]
    fails.sort(key=lambda t: np.isnan(t[1]) or t[1] < 0.5)  # misleading (easy∧fail) first
    for xi, pr in fails[:_FAIL_ANNOT_CAP]:
        misleading = not np.isnan(pr) and pr >= 0.5
        col = "#B71C1C" if misleading else "#616161"
        note = (
            "we FAIL but field\nsays EASY → prior\nMISLEADS the router"
            if misleading
            else "we FAIL; field\nalso finds it hard"
        )
        ax.annotate(
            note,
            xy=(xi + 0.2, 0.02),
            xytext=(xi + 0.2, 0.30 if misleading else 0.55),
            ha="center",
            fontsize=7,
            color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9),
        )
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7, label="escalate threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([i.split("__")[-1] for i in ids], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("resolve rate (blue)  /  our pass=1·fail=0 (green)")
    n_fail = int((our_pass == 0.0).sum())
    n_mis = int(((our_pass == 0.0) & (ext_rate >= 0.5)).sum())
    ax.set_title(
        f"SWE-bench Verified leaderboard prior (p_solve) vs our cheap model — {n} tasks\n"
        f"{n_fail}/{n} fail our cheap model (the routing headroom); "
        f"{n_mis} look EASY to the leaderboard → prior would mislead",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    n_missing = int(np.isnan(ext_rate).sum())
    notes = [
        f"'Our cheap model' is exactly one model, {cheapest} — the cheapest on the `cheap` "
        f"tier — not a cohort.",
        f"{n_fail} of {n} tasks fail it; {n_mis} of those look EASY to the leaderboard.",
    ]
    limits = [f"Drawn one bar per task because there are {n} tasks (at most {_BARS_MAX_TASKS})."]
    if len(fails) > _FAIL_ANNOT_CAP:
        limits.append(
            f"{len(fails)} tasks fail but only {_FAIL_ANNOT_CAP} carry a callout, worst-case first."
        )
    if n_missing:
        limits.append(
            f"{n_missing} of {n} tasks have no leaderboard row, so their blue bar is "
            f"absent rather than zero."
        )
    return plot_frame.save(
        fig, path, _OURS_BARS_SPEC, extra=Annotations(notes=tuple(notes), limitations=tuple(limits))
    )


def _draw_ours_agreement(
    ext_rate: np.ndarray, our_pass: np.ndarray, path: Path, cheapest: str
) -> Path:
    """2x2 agreement matrix (large N): field EASY/HARD × our cheap PASS/FAIL counts."""
    from matplotlib.patches import Rectangle

    valid = ~np.isnan(ext_rate)
    easy, passed = ext_rate >= 0.5, our_pass == 1.0
    counts = np.array(
        [
            [int((valid & easy & passed).sum()), int((valid & easy & ~passed).sum())],
            [int((valid & ~easy & passed).sum()), int((valid & ~easy & ~passed).sum())],
        ]
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = float(counts.max()) or 1.0
    im = ax.imshow(counts, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["our cheap model\nPASSES", "our cheap model\nFAILS"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        [
            "SWE-bench prior says\nEASY (p_solve ≥ 0.5)",
            "SWE-bench prior says\nHARD (p_solve < 0.5)",
        ]
    )
    ax.set_xlabel("our own outcome on our tasks", fontsize=9)
    ax.set_ylabel("leaderboard difficulty prior", fontsize=9)
    key_cell = (0, 1)  # EASY-prior yet we FAIL → the prior misleads
    for i in range(2):
        for j in range(2):
            # Contrast-aware ink: white on the darker (higher-count) fills, dark on light
            # ones — never dark-on-dark. The key "misleading" cell always reads red.
            shade = counts[i, j] / vmax
            color = "#B71C1C" if (i, j) == key_cell else "white" if shade > 0.55 else "#0b0b0b"
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=26, color=color)
    ax.add_patch(Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor="#B71C1C", lw=3))
    # Callout INSIDE the key cell, below its count — no collision with the title.
    ax.text(
        1,
        0.30,
        "prior says EASY yet we FAIL\n→ prior MISLEADS the router",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#B71C1C",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("# tasks in cell", fontsize=8)
    n, mis = int(valid.sum()), int(counts[0, 1])
    ax.set_title(
        f"Leaderboard difficulty prior vs our cheap model, on our own tasks — {n} tasks\n"
        f"prior = fraction of SWE-bench Verified leaderboard submissions that solved each "
        f"instance (p_solve)\n"
        f"good = a router keying only on the prior escalates correctly; the red cell "
        f"({mis}) is where it would wrongly NOT escalate",
        fontsize=8.5,
    )
    fig.tight_layout()
    dropped = int((~valid).sum())
    notes = [
        f"'Our cheap model' is exactly one model, {cheapest} — the cheapest on the `cheap` "
        f"tier — not a cohort.",
        f"Drawn as an agreement matrix rather than per-task bars because there are more than "
        f"{_BARS_MAX_TASKS} tasks; the four cells sum to {n}.",
    ]
    limits = []
    if n:
        notes.append(
            f"The red cell holds {mis} of {n} tasks ({mis / n * 100:.0f}%) — where a prior-only "
            f"router would wrongly stay cheap."
        )
    else:
        # Every cell is 0: no task carries a prior, so there is no agreement to read.
        # The figure still renders (empty) rather than crashing, and says why.
        limits.append(
            f"NO TASK MATCHED THE LEADERBOARD PRIOR: all {dropped} task(s) are missing a "
            "leaderboard row, so every cell is 0 and this figure shows no agreement — read "
            "nothing from it until the external CSV covers these instances."
        )
    if dropped and n:
        limits.append(
            f"{dropped} task(s) had no leaderboard row and are excluded from all four cells."
        )
    return plot_frame.save(
        fig,
        path,
        _OURS_AGREEMENT_SPEC,
        extra=Annotations(notes=tuple(notes), limitations=tuple(limits)),
    )


def plot_heldout(out_dir: Path, *, strict: bool = False) -> Path | None:
    """Held-out generalization figure. Deletes any stale PNG rather than silently
    keeping it when the embedding backend is unavailable (a stale image must never
    survive a regen). Any OTHER failure is a real bug and propagates.
    """
    from benchmark.routing import heldout_eval

    path = out_dir / "heldout_generalization.png"
    try:
        rep = heldout_eval.evaluate_heldout()
    except (ImportError, OSError) as exc:
        # ONLY a genuinely missing dependency degrades to a skip: no fastembed/hnswlib
        # (ImportError) or no reachable model cache (OSError). A blanket `except
        # Exception` here reported an empty held-out set (IndexError) as an
        # "embedding cache absent" skip and kept the exit code green for months.
        if strict:
            raise
        # Do NOT leave a stale artifact behind — a regen that can't recompute must
        # remove the old image so nobody publishes a figure that predates the code.
        path.unlink(missing_ok=True)
        print(f"  heldout   : skipped + removed stale PNG ({type(exc).__name__}: {exc})")
        return None
    return _draw_heldout(rep, path)


def _draw_heldout(rep, path: Path) -> Path:  # noqa: ANN001 (HeldoutReport, local import)
    """Three panels: the headline corr-scatter, plus the two-trap bar charts."""
    by = {r.strategy: r for r in rep.rows}
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 0 — THE headline, threshold-free result: own vs neighbour-mean p_solve.
    a0.scatter(rep.nbr_psolve, rep.own_psolve, s=14, alpha=0.5, color="#455A64", edgecolors="none")
    lo = float(min(rep.nbr_psolve.min(), rep.own_psolve.min()))
    a0.plot([lo, 1], [lo, 1], ls="--", color="#D55E00", lw=1, label="y = x (perfect clustering)")
    a0.set_xlabel("neighbour-mean p_solve (LOO, k=20) — compressed by 20-neighbour averaging")
    a0.set_ylabel("own p_solve")
    a0.set_title(
        f"Neighbour signal is weak, not zero — too small to route on\n"
        f"corr(k=1)={rep.corr_k1:.2f} (anti-predictive) · corr(k=20)={rep.corr:.2f} "
        f"95%CI[{rep.corr_ci[0]:.2f},{rep.corr_ci[1]:.2f}] (excludes 0)",
        fontsize=9,
    )
    a0.legend(fontsize=7, loc="upper left")
    a0.grid(True, alpha=0.3)

    names = [r.strategy for r in rep.rows]
    colors = ["#2196F3", "#4CAF50", "#455A64", "#9C27B0"]
    a1.bar(names, [r.accuracy for r in rep.rows], color=colors, edgecolor="white")
    a1.set_title(f"Tier-accuracy vs oracle (n={rep.n})", fontsize=9)
    a1.set_ylim(0, 1)
    # Trap 1: Neighbour ≡ Always-Cheap is a NON-test (it escalates 0/n by construction).
    trap1 = (
        f"Neighbour escalates {rep.neighbour_escalations}/{rep.n}\n"
        "(≡ Always-Cheap by construction —\nk=20 mean can't cross 0.5)"
    )
    a1.annotate(
        trap1,
        xy=(1, by["Neighbour"].accuracy),
        xytext=(0.5, 0.35),
        textcoords="axes fraction",
        fontsize=6.5,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.2", fc="#FFF3E0", ec="#D55E00", alpha=0.9),
    )
    a2.bar(names, [r.avg_reward for r in rep.rows], color=colors, edgecolor="white")
    a2.set_title("Avg reward (γ=0.1) — cost-model dependent", fontsize=9)
    # Trap 2: Reward-Oracle (true bound) edges Always-Cheap; Oracle-tier-acc loses.
    trap2 = (
        f"Reward-Oracle {by['Reward-Oracle'].avg_reward:.3f} vs\n"
        f"Always-Cheap {by['Always-Cheap'].avg_reward:.3f}\n→ real headroom is ~0.01"
    )
    a2.annotate(
        trap2,
        xy=(3, by["Reward-Oracle"].avg_reward),
        xytext=(0.5, 0.25),
        textcoords="axes fraction",
        fontsize=6.5,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#333", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.2", fc="#ECEFF1", ec="#455A64", alpha=0.9),
    )
    for ax in (a1, a2):
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
    headroom = by["Reward-Oracle"].avg_reward - by["Always-Cheap"].avg_reward
    fig.suptitle(
        f"Out-of-sample generalization ({rep.n} held-out SWE-bench Verified, leave-one-out): "
        f"embedding neighbours carry a detectable but routing-useless difficulty signal "
        f"(reward headroom {headroom:.3f})"
    )
    fig.tight_layout()
    return plot_frame.save(fig, path, _HELDOUT_SPEC, extra=_heldout_annotations(rep, headroom))


def _heldout_annotations(rep, headroom: float) -> Annotations:  # noqa: ANN001 (HeldoutReport)
    """Runtime notes/limits: the measured headroom, the k=1 corr, the degenerate row."""
    return Annotations(
        notes=(
            f"{rep.n} held-out instances, {rep.n_hard} of them hard (own p_solve below 0.5).",
            f"Rank-AUC of the neighbour signal flagging an own-hard instance is {rep.auc:.2f}, "
            f"where 0.50 is no signal at all.",
        ),
        limitations=(
            f"The measured reward headroom is {headroom:.3f} — the whole prize a perfect "
            f"router could win over Always-Cheap on this data, far below any threshold worth "
            f"acting on.",
            f"corr(k=1) is {rep.corr_k1:.2f}, so the honest nearest-neighbour test is near "
            f"zero even though corr(k=20)={rep.corr:.2f} with 95%CI "
            f"[{rep.corr_ci[0]:.2f},{rep.corr_ci[1]:.2f}] excludes 0.",
            f"The Neighbour router escalated {rep.neighbour_escalations} of {rep.n} instances, "
            f"so its bars are Always-Cheap's bars — a degenerate non-test, not a tie.",
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot the external SWE-bench signal (with our runs).")
    ap.add_argument("--out-dir", default="benchmark/routing/reports", help="output dir")
    ap.add_argument("--config", default="benchmark/benchmark.yaml")
    args = ap.parse_args()
    config.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = config.results_csv_path()

    print(f"  external  : {plot_external_difficulty(_EXT_CSV, out_dir)}")
    print(f"  ours-vs-ext: {plot_ours_vs_external(results_csv, _EXT_CSV, out_dir)}")
    held = plot_heldout(out_dir)
    if held:
        print(f"  heldout   : {held}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
