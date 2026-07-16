#!/usr/bin/env python3
"""Plots for the external SWE-bench signal — separately and alongside our runs."""

from __future__ import annotations

# Three deterministic figures (Agg PNGs embed no timestamp → byte-stable on regen):
# * external_difficulty.png   — SWE-bench values ALONE: p_solve/p_cheap/p_frontier/gap.
# * ours_vs_external.png      — TOGETHER on our 10: external p_solve vs our cheap-model
#                               pass (exposes sphinx: external says easy, our scaffold fails).
# * heldout_generalization.png — out-of-sample test over the ~490 held-out instances.
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
    series = [
        ("p_solve (all cohorts)", _floats(rows, "p_solve"), "#455A64"),
        ("p_cheap (open-weight)", _floats(rows, "p_cheap"), "#2196F3"),
        ("p_frontier (proprietary)", _floats(rows, "p_frontier"), "#F44336"),
        ("gap = p_frontier - p_cheap", _floats(rows, "gap"), "#4CAF50"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, vals, color) in zip(axes.ravel(), series, strict=True):
        ax.hist(vals, bins=20, color=color, edgecolor="white")
        ax.set_title(f"{title}  (n={len(vals)}, median={np.median(vals):.2f})", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("External SWE-bench difficulty prior — per-instance resolve rates")
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


def plot_ours_vs_external(results_csv: Path, ext_csv: Path, out_dir: Path) -> Path:
    """Our 10: external cheap-cohort rate vs our own cheapest-model pass."""
    ext = {r["instance_id"]: r for r in csv.DictReader(ext_csv.open())}
    ours = _our_cheap_pass(results_csv)
    ids = sorted(ours)
    labels = [i.split("__")[-1] for i in ids]
    # p_solve (field-wide difficulty) — p_cheap is degenerate (see load_external_priors).
    ext_rate = [float(ext[i]["p_solve"]) if i in ext and ext[i]["p_solve"] else np.nan for i in ids]
    our_pass = [1.0 if ours[i] else 0.0 for i in ids]

    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - 0.2, ext_rate, 0.4, label="external field-wide resolve rate (p_solve)", color="#2196F3"
    )
    ax.bar(x + 0.2, our_pass, 0.4, label="our cheapest model passed (1/0)", color="#4CAF50")
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7, label="escalate threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("resolve rate / pass")
    ax.set_title("External prior vs our actual — cheapest model on our 10 tasks")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "ours_vs_external.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heldout(out_dir: Path) -> Path | None:
    """Held-out generalization bars; None (with a note) if embeddings unavailable."""
    from benchmark.routing import heldout_eval

    try:
        rep = heldout_eval.evaluate_heldout()
    except Exception as exc:  # noqa: BLE001 (embedding cache / HF absent — skip the heavy plot)
        print(f"  heldout   : skipped ({type(exc).__name__}: {exc})")
        return None
    names = [r.strategy for r in rep.rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2196F3", "#4CAF50", "#455A64", "#9C27B0"]
    a1.bar(names, [r.accuracy for r in rep.rows], color=colors, edgecolor="white")
    a1.set_title(f"Tier-accuracy vs oracle (n={rep.n})")
    a1.set_ylim(0, 1)
    a2.bar(names, [r.avg_reward for r in rep.rows], color=colors, edgecolor="white")
    a2.set_title("Avg reward (γ=0.1) — cost-model dependent")
    for ax in (a1, a2):
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(
        "Out-of-sample: difficulty does NOT cluster — "
        f"corr(neighbour, own p_solve)={rep.corr:.2f}, AUC={rep.auc:.2f} (0.5=no signal)"
    )
    fig.tight_layout()
    path = out_dir / "heldout_generalization.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot the external SWE-bench signal (with our runs).")
    ap.add_argument("--out-dir", default="benchmark/routing/reports", help="output dir")
    ap.add_argument("--config", default="benchmark/config.yaml")
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
