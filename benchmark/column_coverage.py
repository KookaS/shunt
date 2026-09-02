"""Which optional results.csv columns are actually populated, and on which cells."""

# THE GAP THIS CLOSES. The optional-column classes (see `benchmark.routing.integrity`) are
# blank by construction on every legacy row, and blank means MISSING FOREVER — never zero. So
# "can I ask this question of the corpus?" has no answer anywhere: a consumer either finds a
# value or raises through `validate.require_measured`, and nothing tells a human WHICH cells
# would have to be re-run to make the question answerable. This report is that answer, and it
# is deliberately a report rather than a gate: a blank optional column is a coverage fact, not
# a defect, so nothing here fails.
#
# Sibling of `benchmark.model_coverage`, and the same shape: reads only committed data
# (results.csv), no containers, no spend, exits 0 unless it cannot read the corpus.
# Run it (never bare python3): uv run --extra benchmark python -m benchmark.column_coverage

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing import integrity

# Column classes, spelled the way a reader of the report needs them. The column NAMES are
# never restated here — they are imported from `integrity`, so a column added to the schema
# appears in this report automatically and cannot be silently omitted from it.
_KEY: Final = "key"
_MEASUREMENT: Final = "measurement-optional"
_PROVENANCE: Final = "provenance-optional"

REPORTS_DIR: Final[Path] = Path(__file__).resolve().parent / "routing" / "reports"
CSV_NAME: Final[str] = "column_coverage.csv"
JSON_NAME: Final[str] = "column_coverage.json"


def column_classes() -> dict[str, str]:
    """Every optional column mapped to its class, derived from the schema declaration."""
    classes = {integrity.REPLICATE_COLUMN: _KEY}
    classes.update(dict.fromkeys(integrity.MEASUREMENT_OPTIONAL_COLUMNS, _MEASUREMENT))
    classes.update(dict.fromkeys(integrity.PROVENANCE_OPTIONAL_COLUMNS, _PROVENANCE))
    return classes


def _populated(row: dict[str, str], column: str) -> bool:
    """A column is populated on a row iff it carries a non-blank value."""
    # `rep` is the one exception: a blank normalises to 0, which IS a value (the first
    # observation), so the replicate key is populated on every row by definition.
    if column == integrity.REPLICATE_COLUMN:
        return True
    return bool(str(row.get(column, "") or "").strip())


@dataclass(frozen=True)
class ColumnRow:
    """One optional column's corpus-wide fill, and which models carry it at all."""

    column: str
    column_class: str
    n_rows: int
    n_populated: int
    populated_models: tuple[str, ...]

    @property
    def n_blank(self) -> int:
        """Rows on which the column is MISSING (never "zero")."""
        return self.n_rows - self.n_populated

    @property
    def pct(self) -> float:
        """Share of rows carrying a value, 0.0 on an empty corpus."""
        return round(100.0 * self.n_populated / self.n_rows, 2) if self.n_rows else 0.0


def column_rows(rows: list[dict[str, str]]) -> list[ColumnRow]:
    """Table 1: per optional column, how much of the corpus carries it."""
    classes = column_classes()
    out: list[ColumnRow] = []
    for column, column_class in classes.items():
        hits = [r for r in rows if _populated(r, column)]
        models = tuple(sorted({str(r.get("model", "")) for r in hits}))
        out.append(ColumnRow(column, column_class, len(rows), len(hits), models))
    return out


def cell_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Table 2: per (model, arm, rep), which optional columns are populated and which are not."""
    # This is the table that answers "which cells would I have to re-run to get ttft_s on
    # deepseek" without anyone opening a 1265-row CSV: find the (model, arm, rep) groups whose
    # `missing` list names the column, and re-run their `n_rows` cells.
    classes = column_classes()
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            str(row.get("model", "")),
            str(row.get("reasoning") or integrity.DEFAULT_REASONING),
            integrity.rep_index(row),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (model, arm, rep), group in sorted(groups.items()):
        populated = [c for c in classes if all(_populated(r, c) for r in group)]
        partial = [
            c for c in classes if c not in populated and any(_populated(r, c) for r in group)
        ]
        missing = [c for c in classes if c not in populated and c not in partial]
        out.append(
            {
                "model": model,
                "reasoning": arm,
                "rep": rep,
                "n_rows": len(group),
                "populated": populated,
                "partial": partial,
                "missing": missing,
            }
        )
    return out


def write_reports(
    columns: list[ColumnRow], cells: list[dict[str, object]], out_dir: Path
) -> tuple[Path, Path]:
    """Write both tables to the gitignored regenerable reports dir; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / CSV_NAME, out_dir / JSON_NAME
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["column", "class", "n_rows", "n_populated", "n_blank", "pct", "populated_models"]
        )
        for row in columns:
            writer.writerow(
                [
                    row.column,
                    row.column_class,
                    row.n_rows,
                    row.n_populated,
                    row.n_blank,
                    row.pct,
                    " ".join(row.populated_models),
                ]
            )
    payload = {
        "columns": [
            {
                "column": r.column,
                "class": r.column_class,
                "n_rows": r.n_rows,
                "n_populated": r.n_populated,
                "n_blank": r.n_blank,
                "pct": r.pct,
                "populated_models": list(r.populated_models),
            }
            for r in columns
        ],
        "cells": cells,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path


def format_report(columns: list[ColumnRow]) -> str:
    """The per-column table (the per-cell table is JSON/CSV only — it is too wide for stdout)."""
    head = f"{'column':<24} {'class':<21} {'populated':>10} {'blank':>7} {'pct':>7}  models"
    lines = ["Optional-column coverage", "=" * 24, head, "-" * len(head)]
    for row in columns:
        models = ", ".join(row.populated_models) if row.populated_models else "-"
        lines.append(
            f"{row.column:<24} {row.column_class:<21} {row.n_populated:>10} "
            f"{row.n_blank:>7} {row.pct:>6.1f}%  {models}"
        )
    lines.append("")
    lines.append(
        "blank means MISSING, never zero: aggregate an optional column only through "
        "benchmark.routing.validate.require_measured"
    )
    return "\n".join(lines)


def run(results_csv: Path, out_dir: Path = REPORTS_DIR) -> list[ColumnRow]:
    """Build both tables from the raw corpus and write them; returns the per-column table."""
    # ALL reps, deliberately: this report is ABOUT the rows, so it cannot use either raw-reader
    # policy — a rep-0-only view would report the replicates' coverage as if it did not exist.
    rows = integrity.all_rows(results_csv)
    columns = column_rows(rows)
    write_reports(columns, cell_rows(rows), out_dir)
    return columns


def main(argv: list[str] | None = None) -> int:
    """CLI: write both tables and print the per-column one. Exits 1 only if the corpus is absent."""
    parser = argparse.ArgumentParser(
        prog="benchmark.column_coverage",
        description="Report which optional results.csv columns are populated, and where.",
    )
    parser.add_argument("--config", default="benchmark/benchmark.yaml", help="config YAML")
    parser.add_argument("--results", type=Path, default=None, help="results.csv (default: config)")
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    config.load(args.config)
    results_csv = args.results or config.results_csv_path()
    if not results_csv.exists():
        print(f"no results corpus at {results_csv}")  # noqa: T201 - CLI output
        return 1
    print(format_report(run(results_csv, args.out_dir)))  # noqa: T201 - CLI output
    print(f"\nwrote {args.out_dir / CSV_NAME} and {args.out_dir / JSON_NAME}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
