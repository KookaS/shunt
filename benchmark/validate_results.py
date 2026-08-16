"""Pre-analysis data-integrity gate for benchmark/routing/results.csv."""

# Scans the committed results.csv against the ERROR/WARN invariants in
# ``benchmark.routing.validate`` and exits NONZERO on any ERROR, so analysis
# (kill-gate, summary, reports) refuses to run on poison data — fail closed.
# Run it (never bare python3): uv run --extra benchmark python -m benchmark.validate_results

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from benchmark import config
from benchmark.routing import validate
from benchmark.routing.validate import Severity, ValidationReport


def load_raw_rows(path: str | Path) -> list[dict]:
    """Read results.csv as a list of raw string-valued row dicts (no coercion)."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def gate(path: str | Path, pricing: dict) -> tuple[ValidationReport, bool]:
    """Validate results.csv at ``path``; return (report, blocking?)."""
    report = validate.validate_results(load_raw_rows(path), pricing)
    return report, validate.has_blocking_violations(report)


def format_report(report: ValidationReport, path: str | Path) -> str:
    """Human-readable render of a ValidationReport for the CLI."""
    lines = [
        "Benchmark data-integrity report",
        "=" * 34,
        f"results: {path}",
        f"rows: {report.total_rows}  offending rows: {len(report.offending)}",
        f"violations by severity: {report.count_by_severity() or '{}'}",
        f"violations by code: {report.count_by_code() or '{}'}",
    ]
    if not report.offending:
        lines.append("CLEAN: no data-integrity violations.")
        return "\n".join(lines)
    lines.append("")
    lines.append("offending rows:")
    for row in report.offending:
        for v in row.violations:
            lines.append(f"  row {row.index} [{v.severity.value}/{v.code}] {v.message}")
    return "\n".join(lines)


def _severity_word(report: ValidationReport) -> str:
    return (
        "ERROR"
        if any(v.severity is Severity.ERROR for v in report.all_violations)
        else "clean/WARN-only"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: print the report; exit nonzero iff there is a blocking ERROR."""
    parser = argparse.ArgumentParser(
        prog="benchmark.validate_results",
        description="Fail-closed data-integrity gate over benchmark results.csv.",
    )
    parser.add_argument("--config", default="benchmark/benchmark.yaml", help="config YAML")
    parser.add_argument("--results", default=None, help="results.csv path (default: harness path)")
    args = parser.parse_args(argv)

    config.load(args.config)
    path = args.results or config.results_csv_path()
    report, blocking = gate(path, dict(config.load_pricing()))
    print(format_report(report, path))  # noqa: T201 - CLI output
    if blocking:
        print(  # noqa: T201
            "\nDATA-INTEGRITY ALARM: ERROR-severity violations found — analysis refuses to run "
            f"on poison data ({_severity_word(report)}). Fix the rows before scaling."
        )
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
