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

from benchmark import config

_EXT_CSV = Path(__file__).resolve().parents[1] / "data" / "external_swebench.csv"


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
    series = [
        (
            "p_solve (all 134 cohorts) — the usable difficulty signal",
            _floats(rows, "p_solve"),
            "#455A64",
        ),
        (
            "p_cheap (open-weight) — DEGENERATE ≡ 1.0, carries NO signal"
            if degenerate
            else "p_cheap (open-weight)",
            pc,
            "#2196F3",
        ),
        ("p_frontier (proprietary cohort)", _floats(rows, "p_frontier"), "#F44336"),
        ("routing headroom = p_frontier − p_solve  (the real gap)", headroom, "#4CAF50"),
    ]
    xlabels = (
        "resolve rate  (fraction of leaderboard submissions that solved the instance)",
        "resolve rate  (open-weight cohort — all 1.0, see note)",
        "resolve rate  (proprietary/frontier cohort)",
        "p_frontier − p_solve  ( >0 ⇒ frontier beats the field ⇒ routable )",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, vals, color), xlabel in zip(axes.ravel(), series, xlabels, strict=True):
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
    fig.suptitle("External SWE-bench difficulty prior — per-instance resolve rates (n=500)")
    fig.tight_layout()
    path = out_dir / "external_difficulty.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _our_cheap_pass(results_csv: Path) -> dict[str, bool]:
    """{instance_id: did our cheapest model pass} from results.csv."""
    pricing = config.load_pricing()
    cheap = [m for m, i in pricing.items() if isinstance(i, dict) and i.get("tier") == "cheap"]
    cheapest = min(
        cheap,
        key=lambda m: (
            pricing[m].get("input_cost_per_1m", 0) + pricing[m].get("output_cost_per_1m", 0)
        ),
        default="deepseek-v4-flash",
    )
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
    ours = _our_cheap_pass(results_csv)
    ids = sorted(ours)
    # p_solve (field-wide difficulty) — p_cheap is degenerate (see load_external_priors).
    ext_rate = np.array(
        [float(ext[i]["p_solve"]) if i in ext and ext[i]["p_solve"] else np.nan for i in ids]
    )
    our_pass = np.array([1.0 if ours[i] else 0.0 for i in ids])

    path = out_dir / "ours_vs_external.png"
    if len(ids) <= _BARS_MAX_TASKS:
        _draw_ours_bars(ids, ext_rate, our_pass, path)
    else:
        _draw_ours_agreement(ext_rate, our_pass, path)
    return path


def _draw_ours_bars(ids: list[str], ext_rate: np.ndarray, our_pass: np.ndarray, path: Path) -> None:
    """Per-task grouped bars (small N): external p_solve vs our cheap-model outcome."""
    n = len(ids)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(min(24.0, max(8.0, 0.55 * n)), 6))
    ax.bar(
        x - 0.2, ext_rate, 0.4, label="external resolve rate p_solve (0–1, blue)", color="#2196F3"
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
        f"External prior vs our actual — cheapest model on {n} tasks\n"
        f"{n_fail}/{n} fail our cheap model (the routing headroom); "
        f"{n_mis} look EASY to the field → prior would mislead",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_ours_agreement(ext_rate: np.ndarray, our_pass: np.ndarray, path: Path) -> None:
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
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(counts, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["our cheap model\nPASSES", "our cheap model\nFAILS"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["field says\nEASY (p_solve≥0.5)", "field says\nHARD (p_solve<0.5)"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(counts[i, j]),
                ha="center",
                va="center",
                fontsize=22,
                color="#B71C1C" if (i == 0 and j == 1) else "#333",
            )
    ax.add_patch(Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor="#B71C1C", lw=3))
    n, mis = int(valid.sum()), int(counts[0, 1])
    ax.set_title(
        f"External prior vs our actual — agreement on {n} tasks\n"
        f"red box: field says EASY yet our cheap model FAILS ({mis}) → "
        "a prior-only router would wrongly NOT escalate",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    _draw_heldout(rep, path)
    return path


def _draw_heldout(rep, path: Path) -> None:  # noqa: ANN001 (HeldoutReport, local import)
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
    fig.suptitle(
        "Out-of-sample generalization (~490 held-out SWE-bench Verified, leave-one-out): "
        "embedding neighbours carry a detectable but routing-useless difficulty signal "
        "(reward headroom ~0.01)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
