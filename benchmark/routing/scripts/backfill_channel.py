#!/usr/bin/env python3
"""Backfill the OBSERVED ``channel`` / ``channel_source`` columns on the results CSVs.

The listing ``billing`` field (registry/overlay/synthesized row) is the ENTITLEMENT; the
observed ``channel`` is what each committed row's own evidence says happened. This script
applies the one pre-registered rule in ``benchmark.routing.channel`` to every historical row
and writes the two columns back, so a corpus reader never has to re-derive them.

Rule precedence (first match wins):

    real_cost > 0                          -> channel=paid,   source=real_cost
    calls == 0                             -> channel=blank,  source=unobserved
    inside a declared free window          -> channel=free,   source=declared_free_window
    row lives in results_free.csv          -> channel=free,   source=results_free_file
    listing billing == free                -> channel=free,   source=overlay_billing_free
    otherwise                              -> channel=blank,  source=no_evidence

There is no `-explabs` suffix fallback: the `explabs` provider ships, so `declared_billing`
resolves every `-explabs` lane as a free collection listing, and a lane whose listing does not
resolve is reported blank rather than asserted free.

Usage:
    python -m benchmark.routing.scripts.backfill_channel            # both committed CSVs
    python -m benchmark.routing.scripts.backfill_channel --check    # report only
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing import channel, integrity

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS: Final[tuple[Path, ...]] = (
    REPO_ROOT / "benchmark" / "routing" / "results.csv",
    REPO_ROOT / "benchmark" / "routing" / "results_free.csv",
)
DEFAULT_OVERLAY: Final[Path] = REPO_ROOT / "configs" / "free-tier" / "overlay.yaml"
FREE_FILE_NAME: Final[str] = "results_free.csv"


def _to_float(raw: object) -> float:
    """Parse a CSV cell as a float; blank/non-numeric is 0 (the column's write-time default)."""
    try:
        return float(str(raw).strip()) if str(raw).strip() else 0.0
    except ValueError:
        return 0.0


def _to_int(raw: object) -> int:
    """Parse a CSV cell as an int; blank/non-numeric is 0."""
    return int(_to_float(raw))


def backfill_rows(rows: list[dict], *, in_free_file: bool) -> list[dict]:
    """Return *rows* with `channel` / `channel_source` recomputed (other columns untouched)."""
    for row in rows:
        lane = str(row.get("lane") or row.get("model") or "")
        observed, source = channel.observed_channel(
            lane,
            _to_float(row.get("real_cost")),
            _to_int(row.get("calls")),
            str(row.get("computed_at") or ""),
            in_free_file=in_free_file,
        )
        row[integrity.CHANNEL_COLUMN] = observed
        row[integrity.CHANNEL_SOURCE_COLUMN] = source
    return rows


def backfill_file(path: Path, *, write: bool) -> tuple[Counter, Counter, int]:
    """Backfill one CSV; return (channel counts, source counts, row count)."""
    if not path.exists():
        return Counter(), Counter(), 0
    in_free_file = path.name == FREE_FILE_NAME
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    for column in integrity.CHANNEL_COLUMNS:
        if column not in header:
            header.append(column)
    rows = backfill_rows(rows, in_free_file=in_free_file)
    if write:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    channels = Counter(str(row.get(integrity.CHANNEL_COLUMN) or "(blank)") for row in rows)
    sources = Counter(str(row.get(integrity.CHANNEL_SOURCE_COLUMN) or "(blank)") for row in rows)
    return channels, sources, len(rows)


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        default=list(DEFAULT_RESULTS),
        help="results CSV(s) to backfill (default: both committed CSVs)",
    )
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    parser.add_argument(
        "--overlay",
        type=Path,
        default=DEFAULT_OVERLAY,
        help="non-shipped free overlay used to resolve listing billing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Backfill every target and report the per-file counts."""
    args = _arg_parser().parse_args(argv)
    if args.overlay.exists():
        config.set_free_registry(args.overlay)
    for path in args.results:
        channels, sources, total = backfill_file(path, write=not args.check)
        if total == 0:
            print(f"{path}: absent, skipped", file=sys.stderr)
            continue
        channel_text = ", ".join(f"{k}={v}" for k, v in sorted(channels.items()))
        source_text = ", ".join(f"{k}={v}" for k, v in sorted(sources.items()))
        verb = "would backfill" if args.check else "backfilled"
        print(f"{path}: {verb} {total} row(s)")
        print(f"  channel: {channel_text}")
        print(f"  channel_source: {source_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
