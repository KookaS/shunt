#!/usr/bin/env python3
"""Backfill ``cached_in_tok`` on measured results.csv rows from archived message lists.

Every live cell's full conversation is persisted to
``benchmark/runner/artifacts/message_lists/<trajectory_id>.json``, and the runner now
sums ``usage.prompt_tokens_details.cached_tokens`` into the row's ``cached_in_tok``
column (`infer._sum_cached_in_tokens`). Rows collected BEFORE that extractor existed
carry a blank ``cached_in_tok`` even though the archived lists hold the numbers —
a measured per-model cache-hit rate (not yet consumed: the cost axis still uses an
assumed constant) is therefore missing for exactly the model populations that were
measured first.

This script repairs that gap WITHOUT re-running any cell: it re-derives
``cached_in_tok`` from the already-paid-for message lists and writes it back into
results.csv. It is REAL-ONLY — the value written is the same sum the live runner
would have recorded under the presence-strict rule (every usage-carrying message in
the cell must report the field, else the cell stays MISSING, never measured-zero).

Default is a DRY RUN that reports what would change and the pooled rate that results;
pass ``--write`` to mutate results.csv.

Usage:
    python -m benchmark.routing.scripts.backfill_cached_tokens --message-lists \
        benchmark/runner/artifacts/message_lists [--write]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Final

from benchmark import config

ARM_BY_FILE_SUFFIX: Final[tuple[str, str]] = ("__", ".json")


def cached_from_messages(messages: Any) -> int | None:
    """The presence-strict cached-token sum for one archived message list."""
    if not isinstance(messages, list):
        return None
    total = 0
    seen = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        response = (msg.get("extra") or {}).get("response")
        if not response:
            continue
        usage = response.get("usage") or {}
        if not usage:
            continue
        details = usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict) or "cached_tokens" not in details:
            return None
        seen += 1
        total += int(details["cached_tokens"] or 0)
    return total if seen else None


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row["challenge_id"],
        str(row.get("lane") or row["model"]),
        str(row.get("reasoning") or ""),
    )


def scan_message_lists(root: Path) -> dict[tuple[str, str, str], int | None]:
    """Map every archive under ``root`` to its cell key and cached-token sum (None = no usage)."""
    out: dict[tuple[str, str, str], int | None] = {}
    for path in sorted(root.glob("*.json")):
        stem = path.stem
        cid, model, arm = stem.rsplit("__", 2)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        messages = data.get("messages") if isinstance(data, dict) else None
        out[(cid, model, arm)] = cached_from_messages(messages)
    return out


def backfill_summary(results_path: Path, message_root: Path, *, write: bool) -> dict[str, Any]:
    """Re-derive ``cached_in_tok`` from the archives; dry-run or write to results.csv."""
    rows = list(integrity_read(results_path))
    index = {_row_key(r): r for r in rows}
    archive = scan_message_lists(message_root)
    matched = unmatched = blanked = 0
    pooled_cached = 0
    pooled_in = 0
    per_model: dict[str, list[int]] = {}
    updates = 0
    for key, cached in sorted(archive.items()):
        row = index.get(key)
        if row is None:
            unmatched += 1
            continue
        matched += 1
        if cached is None:
            # Archive with no usage at all: nothing measured, column stays blank.
            blanked += 1
            continue
        in_tok = int(row.get("in_tok") or 0)
        pooled_cached += cached
        pooled_in += in_tok
        bucket = per_model.setdefault(key[1], [0, 0])
        bucket[0] += cached
        bucket[1] += in_tok
        if str(row.get("cached_in_tok") or "") != str(cached):
            updates += 1
            if write:
                row["cached_in_tok"] = str(cached)
    if write:
        write_results(results_path, rows)
    return {
        "results_csv": str(results_path),
        "archived_cells": len(archive),
        "matched_rows": matched,
        "row_misses": unmatched,
        "no_usage_archives": blanked,
        "cells_to_update": updates if not write else 0,
        "cells_written": updates if write else 0,
        "pooled_rate": (pooled_cached / pooled_in) if pooled_in else None,
        "per_model": {
            model: {"cached": b[0], "in_tok": b[1], "rate": b[0] / b[1] if b[1] else None}
            for model, b in sorted(per_model.items())
        },
    }


def integrity_read(results_path: Path) -> list[dict[str, str]]:
    """Every row of results.csv, header order preserved."""
    if not results_path.exists():
        return []
    with results_path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_results(results_path: Path, rows: list[dict[str, str]]) -> None:
    """Rewrite results.csv with the same header order and row order."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with results_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--message-lists",
        default="benchmark/runner/artifacts/message_lists",
        help="Root of the archived message-list json files",
    )
    ap.add_argument("--results", default=None, help="results.csv path (default: config's)")
    ap.add_argument(
        "--write",
        action="store_true",
        help="Mutate results.csv (default: dry run reporting what would change)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    config.load("benchmark/benchmark.yaml")
    results_path = Path(args.results) if args.results else config.results_csv_path()
    root = Path(args.message_lists)
    if not root.is_dir():
        print(f"message-list root not found: {root}")
        return 2
    summary = backfill_summary(results_path, root, write=args.write)
    print(f"mode: {'WRITE' if args.write else 'DRY-RUN'} on {summary['results_csv']}")
    for k in ("archived_cells", "matched_rows", "row_misses", "no_usage_archives"):
        print(f"  {k}: {summary[k]}")
    if args.write:
        print(f"  cells_written: {summary['cells_written']}")
    else:
        print(f"  cells_to_update: {summary['cells_to_update']}")
    if summary["pooled_rate"] is not None:
        print(f"  pooled_rate after backfill: {summary['pooled_rate']:.4f}")
    for model, stats in (summary["per_model"] or {}).items():
        r = stats["rate"]
        print(f"  {model}: rate={r:.4f}" if r is not None else f"  {model}: no measurable in_tok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
