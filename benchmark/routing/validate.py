"""Data-integrity validator for benchmark results rows."""

# Encodes what a VALID results.csv row means as machine-checkable invariants that
# FAIL LOUD — enforced both at write-time (``DataIntegrityError``) and before any
# analysis (``benchmark.validate_results``). Born from the $35 miss: a live run
# wrote PAID frontier cells with ``real_cost == 0`` and nothing noticed because
# every consumer trusted the CSV.

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

from benchmark.routing import censoring

# Violation codes — one per invariant (stable identifiers for reporting/tests).
ACCOUNTING_HOLE: Final[str] = "ACCOUNTING_HOLE"
UNSOLVED_NOT_RUN: Final[str] = "UNSOLVED_NOT_RUN"
BAD_STOP_REASON: Final[str] = "BAD_STOP_REASON"
PASS_STOP_MISMATCH: Final[str] = "PASS_STOP_MISMATCH"
TIMEOUT_FLAG_MISMATCH: Final[str] = "TIMEOUT_FLAG_MISMATCH"
MALFORMED_NUMERIC: Final[str] = "MALFORMED_NUMERIC"
MALFORMED_TIMESTAMP: Final[str] = "MALFORMED_TIMESTAMP"
SUSPICIOUS_ZERO: Final[str] = "SUSPICIOUS_ZERO"

# Registry price fields (both the load_pricing and the _pricing_dict spellings).
_PRICE_KEYS: Final[tuple[str, ...]] = (
    "input_cost_per_1m",
    "output_cost_per_1m",
    "input",
    "output",
)
# Numeric columns that must parse as a finite value >= 0.
_NUMERIC_FIELDS: Final[tuple[str, ...]] = ("real_cost", "in_tok", "out_tok", "calls")
# A stop_reason marking a row whose API was unusable (never a real run); exempt from
# the SUSPICIOUS_ZERO warn like a censored row is (it neither ran nor is a clean fail).
_API_UNUSABLE_REASON: Final[str] = "api_unusable"


class Severity(StrEnum):
    """A violation is either a hard ERROR (blocks) or an advisory WARN."""

    ERROR = "ERROR"
    WARN = "WARN"


@dataclass(frozen=True)
class Violation:
    """One failed invariant on one row: its severity, stable code, and message."""

    severity: Severity
    code: str
    message: str


@dataclass(frozen=True)
class RowViolations:
    """The violations found on a single row, tagged with its 0-based index."""

    index: int
    violations: list[Violation]


@dataclass(frozen=True)
class ValidationReport:
    """The verdict over a set of rows: every offending row and its violations."""

    total_rows: int
    offending: list[RowViolations]

    @property
    def all_violations(self) -> list[Violation]:
        """Flat list of every violation across every offending row."""
        return [v for row in self.offending for v in row.violations]

    def count_by_code(self) -> dict[str, int]:
        """Violation counts keyed by code."""
        return dict(Counter(v.code for v in self.all_violations))

    def count_by_severity(self) -> dict[str, int]:
        """Violation counts keyed by severity name (ERROR/WARN)."""
        return dict(Counter(v.severity.value for v in self.all_violations))

    @property
    def error_count(self) -> int:
        """Number of ERROR-severity violations."""
        return sum(1 for v in self.all_violations if v.severity is Severity.ERROR)


def _num(value: object) -> float | None:
    """Parse a cell as a finite number >= 0; None if malformed/negative/NaN/inf."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _bool(value: object) -> bool:
    """CSV truthiness matching the rest of the harness (``_bool_field``)."""
    return str(value or "").strip().lower() in ("true", "1", "yes")


def is_paid_model(model: str, pricing: dict) -> bool:
    """True iff the registry lists a positive input OR output price for the model.

    A genuinely FREE model (all prices 0) or an unpriced/absent model is NOT paid,
    so a zero-cost row for it never trips the accounting-hole ERROR (conservative).
    """
    info = pricing.get(model)
    if not isinstance(info, dict):
        return False
    return any(_positive(info.get(key)) for key in _PRICE_KEYS)


def _positive(value: object) -> bool:
    """True iff value parses as a number > 0."""
    try:
        return float(value) > 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _valid_utc_timestamp(value: object) -> bool:
    """True iff value is an ISO-8601 timestamp carrying a zero (UTC) offset."""
    text = str(value or "").strip()
    if not text:
        return False
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return False
    return moment.tzinfo is not None and moment.utcoffset() == timedelta(0)


def _check_wellformed(row: dict) -> list[Violation]:
    """Numeric columns finite >= 0 and computed_at a valid ISO-8601 UTC timestamp."""
    out: list[Violation] = []
    for field_name in _NUMERIC_FIELDS:
        if _num(row.get(field_name)) is None:
            out.append(
                Violation(
                    Severity.ERROR,
                    MALFORMED_NUMERIC,
                    f"{field_name}={row.get(field_name)!r} is not a finite number >= 0",
                )
            )
    if not _valid_utc_timestamp(row.get("computed_at")):
        out.append(
            Violation(
                Severity.ERROR,
                MALFORMED_TIMESTAMP,
                f"computed_at={row.get('computed_at')!r} is not an ISO-8601 UTC timestamp",
            )
        )
    return out


def _check_schema(row: dict, derived: str) -> list[Violation]:
    """stop_reason in-vocabulary, pass<=>solved, timeout_flag<=>wall/abandon stop."""
    out: list[Violation] = []
    raw_stop = str(row.get("stop_reason") or "")
    if raw_stop and raw_stop not in censoring.STOP_REASONS:
        msg = f"stop_reason={raw_stop!r} not in vocabulary"
        out.append(Violation(Severity.ERROR, BAD_STOP_REASON, msg))
    passed = _bool(row.get("pass"))
    if passed != (derived == censoring.SOLVED):
        out.append(
            Violation(
                Severity.ERROR,
                PASS_STOP_MISMATCH,
                f"pass={passed} but stop_reason={derived!r} (pass<=>solved violated)",
            )
        )
    if _bool(row.get("timeout_flag")) != censoring.timeout_flag_for(derived):
        out.append(
            Violation(
                Severity.ERROR,
                TIMEOUT_FLAG_MISMATCH,
                f"timeout_flag={_bool(row.get('timeout_flag'))} inconsistent with "
                f"stop_reason={derived!r}",
            )
        )
    return out


def _check_accounting(row: dict, derived: str, pricing: dict) -> Violation | None:
    """The $35 fingerprint: PAID model, calls>0, real_cost==0, and NOT censored."""
    # Censored rows (resource-limit stops) are EXEMPT: a reaped cell may have run
    # (calls>0) yet legitimately never harvested its cost, so a $0 censored row is
    # not an accounting hole. Unpriced/free models are exempt via is_paid_model.
    calls = _num(row.get("calls"))
    real_cost = _num(row.get("real_cost"))
    if calls is None or real_cost is None or calls <= 0 or real_cost != 0:
        return None
    if censoring.is_censored_reason(derived):
        return None
    if not is_paid_model(str(row.get("model", "")), pricing):
        return None
    return Violation(
        Severity.ERROR,
        ACCOUNTING_HOLE,
        f"paid model {row.get('model')!r} ran ({int(calls)} calls) but real_cost==0",
    )


def _check_ranness(row: dict, derived: str) -> Violation | None:
    """A genuine capability fail (unsolved) must have actually run (calls>0)."""
    calls = _num(row.get("calls"))
    if derived == censoring.UNSOLVED and (calls is None or calls <= 0):
        return Violation(
            Severity.ERROR,
            UNSOLVED_NOT_RUN,
            f"stop_reason=unsolved but calls={row.get('calls')!r} (never ran to genuinely fail)",
        )
    return None


def _check_suspicious(row: dict, derived: str) -> Violation | None:
    """WARN: a $0/0-call row that neither ran nor was censored (legacy/odd row)."""
    calls = _num(row.get("calls"))
    real_cost = _num(row.get("real_cost"))
    if calls is None or real_cost is None or calls != 0 or real_cost != 0:
        return None
    raw_stop = str(row.get("stop_reason") or "")
    if censoring.is_censored_reason(derived) or raw_stop == _API_UNUSABLE_REASON:
        return None
    return Violation(
        Severity.WARN,
        SUSPICIOUS_ZERO,
        f"row for {row.get('model')!r} neither ran (calls=0) nor was censored, yet real_cost=0",
    )


def validate_row(row: dict, pricing: dict) -> list[Violation]:
    """Every invariant violation on one results row (ERROR and WARN)."""
    derived = censoring.derive_stop_reason(
        passed=_bool(row.get("pass")),
        timeout_flag=_bool(row.get("timeout_flag")),
        stop_reason=str(row.get("stop_reason") or ""),
    )
    out: list[Violation] = []
    out.extend(_check_wellformed(row))
    out.extend(_check_schema(row, derived))
    for maybe in (
        _check_accounting(row, derived, pricing),
        _check_ranness(row, derived),
        _check_suspicious(row, derived),
    ):
        if maybe is not None:
            out.append(maybe)
    return out


def validate_results(rows: list[dict], pricing: dict) -> ValidationReport:
    """Validate every row; report each offending row with its violations."""
    offending: list[RowViolations] = []
    for index, row in enumerate(rows):
        violations = validate_row(row, pricing)
        if violations:
            offending.append(RowViolations(index, violations))
    return ValidationReport(total_rows=len(rows), offending=offending)


def has_blocking_violations(report: ValidationReport) -> bool:
    """True iff any ERROR-severity violation exists (analysis must fail closed)."""
    return report.error_count > 0


class DataIntegrityError(RuntimeError):
    """A row failed an ERROR invariant — the run aborts rather than persist poison.

    Loud by design: a run producing corrupt data SHOULD stop (this would have aborted
    the $35 run on row 1). Carries the offending violations and the row.
    """

    def __init__(self, violations: list[Violation], row: dict | None = None) -> None:
        self.violations = violations
        self.row = row
        detail = "; ".join(f"[{v.code}] {v.message}" for v in violations)
        super().__init__(f"row failed data-integrity invariants: {detail}")


def enforce_row(row: dict, pricing: dict) -> None:
    """Raise DataIntegrityError on any ERROR violation (WARN does not abort)."""
    errors = [v for v in validate_row(row, pricing) if v.severity is Severity.ERROR]
    if errors:
        raise DataIntegrityError(errors, row)
