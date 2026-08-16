"""Cost-reconciliation loop: close the OPEN accounting loop on live-benchmark spend."""

# Requesty's cache-aware ``usage.cost`` (stored as ``real_cost`` in
# ``benchmark/routing/results.csv``) is the only cost signal, and there is NO
# balance API — so tracked cost can silently diverge from what was actually billed.
# A real run once spent ~$35 while the harness tracked $1.25 (a 28x undercount that
# nothing noticed). This module closes the loop two ways:
#
# 1. Tracked-vs-billed reconciliation. Sum ``real_cost`` over a time window and
#    compare to a ground-truth billed amount the owner reads off the Requesty
#    dashboard. ALARM (nonzero exit) when ``abs(ratio - 1) > tolerance``.
# 2. Independent cross-check. Re-estimate cost from token counts x posted registry
#    list price and compare to summed ``usage.cost`` per model. Cache discounts mean
#    these never match exactly, so only a LARGE gap (default >3x) or zero usage on
#    nonzero tokens is flagged — catching mixed cost basis and providers that
#    report zero usage.
#
# It also surfaces the exact bug class from the 28x miss: rows with ``real_cost == 0`` but
# ``calls > 0`` on a PAID, non-censored model (an accounting hole) vs ``real_cost == 0`` with
# ``calls == 0`` (a genuine non-run). A reaped CENSORED cell and a free/unpriced model are
# EXEMPT — the same exemptions ``routing.validate`` applies — so a legit $0 row never false-alarms.
#
# Run it via the benchmark extra (do NOT use bare python3): invoke
# ``python -m benchmark.cost_reconcile`` under ``uv run --extra benchmark`` with
# ``--billed``, ``--timestamp``, ``--start`` and ``--end`` (see argparse below).
# ``--billed`` is the owner-supplied dashboard figure; ``--timestamp`` is required
# (the module never calls ``datetime.now`` — the harness forbids it in some contexts
# and it hurts testability). Each run appends one row to the append-only ledger under
# ``benchmark/routing/artifacts/`` (gitignored). Exits nonzero on any alarm.

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import config
from benchmark.routing import censoring, validate

_EPS = 1e-12
DEFAULT_TOLERANCE = 0.15
DEFAULT_GAP_THRESHOLD = 3.0
_LEDGER_NAME = "cost_reconcile_ledger.jsonl"


@dataclass(frozen=True)
class ResultRow:
    """One results.csv row reduced to the cost-relevant fields."""

    model: str
    in_tok: int
    out_tok: int
    calls: int
    real_cost: float
    estimated_cost: float
    computed_at: str
    passed: bool = False
    timeout_flag: bool = False
    stop_reason: str = ""


@dataclass(frozen=True)
class TrackedSummary:
    """Summed tracked cost plus the two zero-cost row populations."""

    tracked_sum: float
    row_count: int
    zero_cost_with_calls: int
    zero_cost_no_calls: int


@dataclass(frozen=True)
class Reconciliation:
    """Tracked-vs-billed comparison and its alarm verdict."""

    tracked_sum: float
    billed_amount: float
    ratio: float
    abs_divergence: float
    tolerance: float
    alarm: bool


@dataclass(frozen=True)
class ModelCrossCheck:
    """Per-model token-price estimate vs summed usage.cost and its anomaly flag."""

    model: str
    priced: bool
    in_tok: int
    out_tok: int
    token_estimate: float
    usage_cost: float
    gap_factor: float
    zero_usage_with_tokens: bool
    anomaly: bool


# ── loading ─────────────────────────────────────────────────────────────────


def _to_int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except (TypeError, ValueError):
        return 0


def _to_float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except (TypeError, ValueError):
        return 0.0


def _to_bool(value: object) -> bool:
    return str(value or "").strip().lower() in ("true", "1", "yes")


def _parse_row(raw: dict[str, str]) -> ResultRow:
    return ResultRow(
        model=str(raw.get("model", "")),
        in_tok=_to_int(raw.get("in_tok")),
        out_tok=_to_int(raw.get("out_tok")),
        calls=_to_int(raw.get("calls")),
        # real_cost is the ground-truth cache-aware usage.cost; fall back to cost.
        real_cost=_to_float(raw.get("real_cost") or raw.get("cost")),
        estimated_cost=_to_float(raw.get("estimated_cost")),
        computed_at=str(raw.get("computed_at", "")),
        passed=_to_bool(raw.get("pass")),
        timeout_flag=_to_bool(raw.get("timeout_flag")),
        stop_reason=str(raw.get("stop_reason") or ""),
    )


def _within(stamp: str, start: datetime | None, end: datetime | None) -> bool:
    if start is None and end is None:
        return True
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return False  # a windowed query cannot place an unparseable row
    if start is not None and moment < start:
        return False
    return not (end is not None and moment > end)


def load_rows(
    path: str | Path, start: datetime | None = None, end: datetime | None = None
) -> list[ResultRow]:
    """Read results.csv rows, optionally filtered to a computed_at [start, end] window."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[ResultRow] = []
    with p.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            row = _parse_row(raw)
            if _within(row.computed_at, start, end):
                rows.append(row)
    return rows


def _is_accounting_hole(row: ResultRow, pricing: dict[str, dict[str, Any]]) -> bool:
    """A real accounting hole: paid model ran (calls>0) at $0 and is NOT a censored stop.

    Applies the SAME exemptions as ``validate._check_accounting`` — a reaped CENSORED cell
    (step/wall/abandoned) and a free/unpriced model legitimately record $0, so neither counts.
    """
    if row.calls <= 0 or row.real_cost > _EPS:
        return False
    censored = censoring.is_censored(
        {"pass": row.passed, "timeout_flag": row.timeout_flag, "stop_reason": row.stop_reason}
    )
    if censored:
        return False
    return validate.is_paid_model(row.model, pricing)


def summarize_tracked(
    rows: list[ResultRow], pricing: dict[str, dict[str, Any]] | None = None
) -> TrackedSummary:
    """Sum real_cost and split zero-cost rows into holes vs non-runs (validate's exemptions)."""
    priced = pricing or {}
    total = 0.0
    holes = 0
    non_runs = 0
    for row in rows:
        total += row.real_cost
        if row.real_cost > _EPS:
            continue
        if row.calls <= 0:
            non_runs += 1
        elif _is_accounting_hole(row, priced):
            holes += 1
    return TrackedSummary(
        tracked_sum=total,
        row_count=len(rows),
        zero_cost_with_calls=holes,
        zero_cost_no_calls=non_runs,
    )


# ── reconciliation ───────────────────────────────────────────────────────────


def reconcile(
    tracked_sum: float, billed_amount: float, tolerance: float = DEFAULT_TOLERANCE
) -> Reconciliation:
    """Compare tracked_sum to billed_amount; alarm when abs(ratio-1) > tolerance."""
    if billed_amount > _EPS:
        ratio = tracked_sum / billed_amount
    else:
        # no billed amount: any tracked spend is an unexplained divergence
        ratio = 1.0 if tracked_sum <= _EPS else float("inf")
    return Reconciliation(
        tracked_sum=tracked_sum,
        billed_amount=billed_amount,
        ratio=ratio,
        abs_divergence=abs(tracked_sum - billed_amount),
        tolerance=tolerance,
        alarm=abs(ratio - 1.0) > tolerance,
    )


# ── independent cross-check ──────────────────────────────────────────────────


@dataclass
class _Agg:
    in_tok: int = 0
    out_tok: int = 0
    usage_cost: float = 0.0


def _aggregate(rows: list[ResultRow]) -> dict[str, _Agg]:
    by_model: dict[str, _Agg] = {}
    for row in rows:
        agg = by_model.setdefault(row.model, _Agg())
        agg.in_tok += row.in_tok
        agg.out_tok += row.out_tok
        agg.usage_cost += row.real_cost
    return by_model


def _token_estimate(agg: _Agg, price: dict[str, Any] | None) -> float:
    if price is None:
        return 0.0
    in_rate = float(price.get("input_cost_per_1m", 0.0))
    out_rate = float(price.get("output_cost_per_1m", 0.0))
    return agg.in_tok / 1e6 * in_rate + agg.out_tok / 1e6 * out_rate


def _gap_factor(token_estimate: float, usage_cost: float) -> float:
    hi = max(token_estimate, usage_cost)
    lo = min(token_estimate, usage_cost)
    if lo > _EPS:
        return hi / lo
    return float("inf") if hi > _EPS else 1.0


def _check_one(
    model: str, agg: _Agg, price: dict[str, Any] | None, gap_threshold: float
) -> ModelCrossCheck:
    priced = price is not None
    token_estimate = _token_estimate(agg, price)
    gap = _gap_factor(token_estimate, agg.usage_cost)
    zero_usage = agg.usage_cost <= _EPS and (agg.in_tok + agg.out_tok) > 0
    # anomaly: unexplained spend (unpriced), a large basis gap, or zero usage on work
    anomaly = zero_usage or (not priced and agg.usage_cost > _EPS)
    if priced and gap > gap_threshold:
        anomaly = True
    return ModelCrossCheck(
        model=model,
        priced=priced,
        in_tok=agg.in_tok,
        out_tok=agg.out_tok,
        token_estimate=token_estimate,
        usage_cost=agg.usage_cost,
        gap_factor=gap,
        zero_usage_with_tokens=zero_usage,
        anomaly=anomaly,
    )


def cross_check(
    rows: list[ResultRow],
    pricing: dict[str, dict[str, Any]],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> list[ModelCrossCheck]:
    """Per-model token-price estimate vs summed usage.cost; flag large gaps as anomalies."""
    aggregated = _aggregate(rows)
    checks = [
        _check_one(model, agg, pricing.get(model), gap_threshold)
        for model, agg in aggregated.items()
    ]
    return sorted(checks, key=lambda c: c.model)


# ── ledger ───────────────────────────────────────────────────────────────────


def build_ledger_entry(
    rec: Reconciliation,
    window_start: datetime | None,
    window_end: datetime | None,
    timestamp: str,
) -> dict[str, Any]:
    """One append-only ledger record; ratio/inf serialized as a JSON-safe string."""
    ratio: Any = rec.ratio if rec.ratio not in (float("inf"), float("-inf")) else "inf"
    return {
        "timestamp": timestamp,
        "window_start": window_start.isoformat() if window_start else None,
        "window_end": window_end.isoformat() if window_end else None,
        "tracked_sum": rec.tracked_sum,
        "billed_amount": rec.billed_amount,
        "ratio": ratio,
        "abs_divergence": rec.abs_divergence,
        "tolerance": rec.tolerance,
        "alarm": rec.alarm,
    }


def append_ledger(path: str | Path, entry: dict[str, Any]) -> bool:
    """Append entry as one JSONL line unless an identical record already exists.

    Returns True if written, False if it was a no-op duplicate (idempotent-safe).
    """
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip() and json.loads(line) == entry:
                return False
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return True


def _default_ledger_path() -> Path:
    return config.results_csv_path().parent / "artifacts" / _LEDGER_NAME


# ── registry pricing (indirection point for tests) ───────────────────────────


def _registry_pricing() -> dict[str, dict[str, Any]]:
    """Priced registry models keyed by name — real registry, patched in tests."""
    return dict(config.load_pricing())


# ── reporting + CLI ──────────────────────────────────────────────────────────


def _format_report(
    summ: TrackedSummary,
    rec: Reconciliation | None,
    checks: list[ModelCrossCheck],
) -> str:
    lines = ["Cost reconciliation report", "=" * 34]
    lines.append(f"rows: {summ.row_count}  tracked_sum: ${summ.tracked_sum:.4f}")
    lines.append(
        f"zero-cost rows: {summ.zero_cost_with_calls} SUSPICIOUS (calls>0, accounting "
        f"hole), {summ.zero_cost_no_calls} benign (calls==0, non-run)"
    )
    if rec is not None:
        verdict = "ALARM" if rec.alarm else "ok"
        lines.append(
            f"tracked ${rec.tracked_sum:.4f} vs billed ${rec.billed_amount:.4f} "
            f"→ ratio {rec.ratio:.4f} (|abs| ${rec.abs_divergence:.4f}, "
            f"tol {rec.tolerance:g}) → {verdict}"
        )
    else:
        lines.append("tracked-vs-billed: skipped (no --billed given)")
    lines.append("per-model cross-check (token list-price estimate vs usage.cost):")
    for c in checks:
        gap = "inf" if c.gap_factor == float("inf") else f"{c.gap_factor:.2f}x"
        flag = " ANOMALY" if c.anomaly else ""
        priced = "" if c.priced else " [UNPRICED]"
        lines.append(
            f"  {c.model}{priced}: est ${c.token_estimate:.4f} vs usage "
            f"${c.usage_cost:.4f} (gap {gap}){flag}"
        )
    return "\n".join(lines)


def _parse_stamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.cost_reconcile",
        description="Reconcile tracked benchmark cost against a billed ground truth.",
    )
    parser.add_argument("--results", default=None, help="results.csv path (default: harness path)")
    parser.add_argument("--billed", type=float, default=None, help="owner-read billed USD amount")
    parser.add_argument("--start", default=None, help="window start (ISO 8601), inclusive")
    parser.add_argument("--end", default=None, help="window end (ISO 8601), inclusive")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD)
    parser.add_argument("--ledger", default=None, help="ledger JSONL path")
    parser.add_argument(
        "--timestamp", required=True, help="ISO 8601 reconciliation time (no datetime.now)"
    )
    return parser


def _run(args: argparse.Namespace) -> tuple[str, bool]:
    results_path = args.results or config.results_csv_path()
    start = _parse_stamp(args.start)
    end = _parse_stamp(args.end)
    rows = load_rows(results_path, start=start, end=end)
    pricing = _registry_pricing()
    summ = summarize_tracked(rows, pricing)
    checks = cross_check(rows, pricing, gap_threshold=args.gap_threshold)
    rec = None
    if args.billed is not None:
        rec = reconcile(summ.tracked_sum, args.billed, tolerance=args.tolerance)
        entry = build_ledger_entry(rec, start, end, args.timestamp)
        append_ledger(args.ledger or _default_ledger_path(), entry)
    report = _format_report(summ, rec, checks)
    alarm = (
        (rec is not None and rec.alarm)
        or any(c.anomaly for c in checks)
        or summ.zero_cost_with_calls > 0
    )
    return report, alarm


def main(argv: list[str] | None = None) -> int:
    """CLI: print the reconciliation report; exit nonzero on any alarm/anomaly/hole."""
    args = _build_parser().parse_args(argv)
    report, alarm = _run(args)
    print(report)  # noqa: T201 - CLI output
    if alarm:
        print("\nRECONCILE ALARM: cost accounting diverged — investigate before scaling.")  # noqa: T201
    return 1 if alarm else 0


if __name__ == "__main__":
    sys.exit(main())
