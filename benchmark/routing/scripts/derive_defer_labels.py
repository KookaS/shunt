#!/usr/bin/env python3
"""Derive a per-task defer-label CSV from MEASURED rung-0 (cheapest enabled) cells.

A defer label answers "should this task have been escalated off the cheapest model?": it
is the rung-0 model's real measured outcome on the task — cheap_pass 1 if that model
solved it, 0 otherwise. It is the training/eval target for the tiny cross-encoder ranker
experiment that predicts deferral better than embedding kNN. The table is built ONLY from
genuine measurements:

  * results.csv holds real measurements by construction — monotone-ladder imputation is
    in-memory only (impute.py) and never persists — so a committed results.csv row for
    the rung-0 model's canonical default-arm cell is a real observation by definition.
    No imputed cell can enter the table, because there is none to read.
  * Non-observations are excluded, never labelled: a CENSORED stop (step/wall/abandon,
    censoring.is_censored) or a zero-work row (impute.is_zero_work) has an unknown true
    outcome, so pass=False there would fabricate a defer label.
  * The rung-0 model is DERIVED, never hardcoded, as config.enabled_models()[0] — the
    benchmark's own price rank (ascending list price, name tie-break), the ordering the
    eval's ladder and difficulty-family pick consume.
  * cost_usd is the provider's measured bill (real_cost), read via
    plot_style.row_real_cost — never the estimated_cost price sheet.
  * The cell mirrors the scoring view (config.load_results + flatten_default_arm): the
    DECLARED default reasoning arm when measured, else the sole cached arm.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing import impute, plot_style

COLUMNS: Final[tuple[str, ...]] = ("challenge_id", "cheap_model", "cheap_pass", "cost_usd")


def _defer_rows(
    cheap: str, raw: dict, flat: dict
) -> tuple[list[dict], list[tuple[str, str]], dict[int, int]]:
    """(rows, excluded, arms_per_task) over the MEASURED task set (``raw``'s keys)."""
    rows: list[dict] = []
    excluded: list[tuple[str, str]] = []
    arms_per_task: Counter = Counter()
    for cid in sorted(raw):
        per_arm = raw[cid].get(cheap)
        cell = (flat.get(cid) or {}).get(cheap)
        if cell is None:
            reason = (
                "no rung-0 row in results.csv"
                if per_arm is None
                else "no canonical default-arm rung-0 cell (default arm unmeasured, >1 cached arm)"
            )
            excluded.append((cid, reason))
            continue
        if impute.is_non_observation(cell):
            excluded.append((cid, "rung-0 cell is a censored or zero-work non-observation"))
            continue
        arms_per_task[len(per_arm or {})] += 1
        rows.append(
            {
                "challenge_id": cid,
                "cheap_model": cheap,
                "cheap_pass": int(bool(cell.get("pass", False))),
                "cost_usd": plot_style.row_real_cost(cell),
            }
        )
    return rows, excluded, dict(arms_per_task)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Derive benchmark/routing/data/defer_labels.csv from MEASURED rung-0 cells"
    )
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to config YAML")
    ap.add_argument(
        "--out",
        default="benchmark/routing/data/defer_labels.csv",
        help="Output path for the committed defer-label CSV",
    )
    args = ap.parse_args()
    config.load(args.config)

    raw = config.load_results()
    if not raw:
        print("No results — results.csv holds no rows; nothing to derive.")
        return 1
    cheap = config.enabled_models()[0]
    flat = config.flatten_default_arm(raw)
    rows, excluded, arms_per_task = _defer_rows(cheap, raw, flat)
    if not rows:
        print(f"No task has a MEASURED default-arm cell for rung-0 {cheap}; wrote nothing.")
        return 1

    out = Path(args.out)
    _write_csv(out, rows)
    n = len(rows)
    fails = sum(1 for r in rows if r["cheap_pass"] == 0)
    default_arm = config.default_arm_ids([cheap])[cheap]
    picked_default = sum(
        1
        for r in rows
        if (flat.get(r["challenge_id"]) or {}).get(cheap, {}).get("reasoning") == default_arm
    )
    print(f"rung-0 (cheapest enabled) model: {cheap}  (default arm {default_arm!r})")
    print(
        f"Wrote {out} — {n} defer-labelled tasks, {fails} cheap failures "
        f"(defer share {fails / n:.1%}), {n - fails} cheap passes"
    )
    print(
        "rung-0 arms observed per task: "
        + ", ".join(f"{k}-arm:{v}" for k, v in sorted(arms_per_task.items()))
        + f"; {picked_default}/{n} rows at the declared default arm"
    )
    for reason in sorted({r for _c, r in excluded}):
        how_many = sum(1 for _c, r in excluded if r == reason)
        print(f"excluded {how_many} task(s): {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
