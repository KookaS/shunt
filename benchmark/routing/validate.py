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
from typing import TYPE_CHECKING, Final

from benchmark.routing import censoring, integrity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Violation codes — one per invariant (stable identifiers for reporting/tests).
ACCOUNTING_HOLE: Final[str] = "ACCOUNTING_HOLE"
UNSOLVED_NOT_RUN: Final[str] = "UNSOLVED_NOT_RUN"
BAD_STOP_REASON: Final[str] = "BAD_STOP_REASON"
PASS_STOP_MISMATCH: Final[str] = "PASS_STOP_MISMATCH"
TIMEOUT_FLAG_MISMATCH: Final[str] = "TIMEOUT_FLAG_MISMATCH"
MALFORMED_NUMERIC: Final[str] = "MALFORMED_NUMERIC"
MALFORMED_TIMESTAMP: Final[str] = "MALFORMED_TIMESTAMP"
SUSPICIOUS_ZERO: Final[str] = "SUSPICIOUS_ZERO"
MISSING_COLLECTION_FIELD: Final[str] = "MISSING_COLLECTION_FIELD"
MALFORMED_OPTIONAL: Final[str] = "MALFORMED_OPTIONAL"
MISSING_MEASUREMENT: Final[str] = "MISSING_MEASUREMENT"
REPLICATE_MISKEYED: Final[str] = "REPLICATE_MISKEYED"
BRACKET_OVER_COVERAGE: Final[str] = "BRACKET_OVER_COVERAGE"

# Promotional $0 windows: (model, first UTC date, last UTC date), both inclusive.
# A model is priced at its REAL rate in the registry — so a future run is billed, gated by
# the uncached-budget wall and caught by the accounting check — while cells genuinely
# collected inside the window keep a true real_cost of 0. Scoped by model AND by date, so
# a paid run on the same id outside the window is still an ACCOUNTING_HOLE. Add a row only
# with the provider's published promo dates; never to silence a harvesting failure.
_FREE_WINDOWS: Final[tuple[tuple[str, str, str], ...]] = (
    # OpenRouter listed z-ai/glm-5.3-flash (then stealth/ox-alpha) at $0 for this window;
    # the 41 committed cells all fall inside it. Price source in the registry entry.
    ("zai-glm-5.3-flash", "2026-08-20", "2026-08-26"),
)

# Registry price fields (both the load_pricing and the _pricing_dict spellings).
_PRICE_KEYS: Final[tuple[str, ...]] = (
    "input_cost_per_1m",
    "output_cost_per_1m",
    "input",
    "output",
)
# Numeric columns that must parse as a finite value >= 0.
# DO NOT ADD AN OPTIONAL COLUMN HERE. `enforce_row` is fail-closed and runs on every write
# (run_matrix._build_row), so requiring a column that 1265 committed rows leave blank would
# ERROR on all of them and abort every future write. Optional columns are checked
# presence-tolerantly by `_check_optional_measurements` instead.
_NUMERIC_FIELDS: Final[tuple[str, ...]] = ("real_cost", "in_tok", "out_tok", "calls")
# Collection-param provenance columns: every row MUST carry these keys so a
# reader can tell the regime a cell was collected under (step_limit/cost_limit/scaffold
# version / sampling kwargs / prompt). Absent key = schema error; EMPTY value is legal and
# is exactly the grandfather rule for legacy rows (an empty anchor degrades to a staleness
# no-op in run_matrix, mirroring _arm_stale/_image_stale).
_REQUIRED_COLLECTION_FIELDS: Final[tuple[str, ...]] = (
    "step_limit",
    "cost_limit",
    "scaffold_version",
    "sampling_hash",
    "prompt_hash",
)
# The MEASUREMENT-OPTIONAL columns that are a TIMING, and so must carry the two provenance
# labels that make them poolable. Deliberately a subset of MEASUREMENT_OPTIONAL_COLUMNS:
# `cached_in_tok` and `retry_count` are counts, not seconds, and neither serving mode nor
# latency source says anything about them.
_LATENCY_COLUMNS: Final[tuple[str, ...]] = ("wall_clock_s", "ttft_s", "latency_per_call_s")

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


def _check_collection_fields(row: dict) -> list[Violation]:
    """Every row carries the six collection-param keys (absent key is an ERROR; empty is legal)."""
    out: list[Violation] = []
    for field_name in _REQUIRED_COLLECTION_FIELDS:
        if field_name not in row:
            out.append(
                Violation(
                    Severity.ERROR,
                    MISSING_COLLECTION_FIELD,
                    f"row lacks required collection-param column {field_name!r}",
                )
            )
    return out


def _check_optional_measurements(row: dict) -> list[Violation]:
    """Optional columns are PRESENCE-TOLERANT: blank is legal, a present value must be sane."""
    # Blank means MISSING, forever — never 0. So this can use neither
    # `_REQUIRED_COLLECTION_FIELDS`' "the key must exist" rule nor `_NUMERIC_FIELDS`' "it must
    # parse" rule; it checks only the rows that actually carry a value. `rep` rides the same
    # code because it is well-formedness of a new column, not a measurement claim: a
    # non-negative integer, or blank meaning 0.
    out: list[Violation] = []
    for field_name in integrity.MEASUREMENT_OPTIONAL_COLUMNS:
        raw = str(row.get(field_name, "") or "").strip()
        if not raw:
            continue
        if _num(raw) is None:
            out.append(
                Violation(
                    Severity.ERROR,
                    MALFORMED_OPTIONAL,
                    f"{field_name}={row.get(field_name)!r} is present but not a finite "
                    "number >= 0 (blank is legal and means MISSING)",
                )
            )
    raw_rep = str(row.get(integrity.REPLICATE_COLUMN, "") or "").strip()
    if raw_rep and not raw_rep.isdigit():
        out.append(
            Violation(
                Severity.ERROR,
                MALFORMED_OPTIONAL,
                f"rep={row.get(integrity.REPLICATE_COLUMN)!r} is not a non-negative integer "
                "(blank is legal and normalises to 0)",
            )
        )
    return out


def _check_latency_labels(row: dict) -> list[Violation]:
    """A timing must be LABELLED, and its labels must be in vocabulary. Presence-tolerant."""
    # Two distinct failures, both silent without this:
    #
    #   * an out-of-vocabulary label. `serving_mode` and `provider_latency_source` exist to
    #     keep incomparable latency populations apart (local batch-1 vs a batched hosted API;
    #     client wall-clock vs a provider-reported field). A free-text value defeats that, so
    #     anything outside `integrity.SERVING_MODES` / `integrity.LATENCY_SOURCES` is refused.
    #   * an UNLABELLED timing. A row carrying a latency but no `serving_mode` or no
    #     `provider_latency_source` is a number that cannot be pooled safely by anyone,
    #     because nothing says which population or which measurement it belongs to.
    #
    # Blank stays legal throughout: the 1265 committed rows carry no timing and no label, and
    # that is the correct permanent state for them.
    out: list[Violation] = []
    vocabularies = (
        ("serving_mode", integrity.SERVING_MODES),
        ("provider_latency_source", integrity.LATENCY_SOURCES),
    )
    for field_name, allowed in vocabularies:
        raw = str(row.get(field_name, "") or "").strip()
        if raw and raw not in allowed:
            out.append(
                Violation(
                    Severity.ERROR,
                    MALFORMED_OPTIONAL,
                    f"{field_name}={raw!r} is not in the declared vocabulary {list(allowed)} "
                    "(blank is legal and means MISSING)",
                )
            )
    has_timing = any(str(row.get(field_name, "") or "").strip() for field_name in _LATENCY_COLUMNS)
    if has_timing:
        for field_name, _ in vocabularies:
            if not str(row.get(field_name, "") or "").strip():
                out.append(
                    Violation(
                        Severity.ERROR,
                        MALFORMED_OPTIONAL,
                        f"row carries a latency measurement but no {field_name!r}. An "
                        "unlabelled timing cannot be pooled with any other timing — a local "
                        "batch-1 second and a hosted batched second are different quantities, "
                        "as are a client wall-clock second and a provider-reported one.",
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


def _in_free_window(model: str, computed_at: str) -> bool:
    """True iff the row was collected inside a provider's genuinely-$0 promotional window."""
    for name, start, end in _FREE_WINDOWS:
        if name == model and start <= computed_at[: len(start)] <= end:
            return True
    return False


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
    if _in_free_window(str(row.get("model", "")), str(row.get("computed_at", ""))):
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
    out.extend(_check_collection_fields(row))
    out.extend(_check_optional_measurements(row))
    out.extend(_check_latency_labels(row))
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


def require_measured(
    values: Sequence[float | int | None], field: str, consumer: str
) -> list[float]:
    """Every value is a real measurement, or raise — never aggregate over a missing one."""
    # THE READ-SIDE HALF of the optional-column contract, and the one the codebase already
    # gets wrong for the columns that predate it: `config.load_results` reads
    # `float(row.get("estimated_cost") or 0.0)`, which turns "I have no record" into "it cost
    # nothing". Copying that idiom for an optional column is exactly the failure this exists
    # to stop, because an optional column is blank on ~1265 committed rows by construction.
    #
    # So a consumer aggregating an optional column calls this FIRST. When it raises, the
    # consumer's correct response is to OMIT the column and publish its `n` — the same shape
    # `summary._context_columns` uses for the context-cost bracket, and for the same reason: a
    # default is an affirmative claim, and publishing one with n=0 behind it is a fabrication,
    # not a conservative choice.
    n_missing = sum(1 for v in values if v is None)
    if n_missing:
        raise DataIntegrityError(
            [
                Violation(
                    Severity.ERROR,
                    MISSING_MEASUREMENT,
                    f"{consumer} tried to aggregate {field!r} over {len(values)} value(s) of "
                    f"which {n_missing} are MISSING (blank). A blank optional column is not "
                    "zero — omit the column and publish its n instead.",
                )
            ]
        )
    return [float(v) for v in values if v is not None]


def enforce_row(row: dict, pricing: dict) -> None:
    """Raise DataIntegrityError on any ERROR violation (WARN does not abort)."""
    errors = [v for v in validate_row(row, pricing) if v.severity is Severity.ERROR]
    if errors:
        raise DataIntegrityError(errors, row)


def enforce_bracket_coverage(
    bracket_tasks: Sequence[str], billed_models: Mapping[str, Sequence[str]], matrix: dict
) -> None:
    """Refuse a context-cost bracket that rests on a path through an imputed cell."""
    # WHY THIS IS NOT A COUNT CEILING. The obvious ceiling — "no more than the tasks measured on
    # every model" — is wrong by a factor of two here, because a bracket needs tokens only on the
    # models a task's realized path actually BILLED, which is one to three of six. The real
    # invariant is per-path, so this is what is enforced: no task counted in a bracket may have
    # billed an IMPUTED cell, because an imputed cell carries no token columns at all and would
    # enter the cost model as a free, context-less attempt.
    #
    # It is also a genuine cross-check rather than a restatement. `context_cost` selects its
    # subset on the TOKEN columns; this re-derives the same exclusion from the independent
    # `imputed` flag the completion stamps. The two agreeing is evidence; the two disagreeing is
    # the bug (a real cell whose tokens were never recorded, or an imputed cell that acquired
    # some) and it fails closed.
    #
    # AND WHY A TASK WITH NO BILLED MODELS IS AN OFFENDER, NOT A PASS. `billed_models.get(tid, ())`
    # makes `any()` False, so a task absent from the mapping used to satisfy this gate vacuously —
    # an EMPTY mapping passed for every task in the bracket. A bracket task always billed at least
    # one attempt by construction (it is token-complete), so an empty path is a broken caller, and
    # a gate that reads "nothing to check" as "checked" is the failure this one exists to stop.
    results = matrix.get("results", {})
    pathless = sorted(tid for tid in bracket_tasks if not billed_models.get(tid))
    offenders = sorted(
        tid
        for tid in bracket_tasks
        if any(
            bool((results.get(tid, {}).get(model) or {}).get("imputed"))
            for model in billed_models.get(tid, ())
        )
    )
    n_scored = len(results)
    if len(set(bracket_tasks)) > n_scored:
        raise DataIntegrityError(
            [
                Violation(
                    Severity.ERROR,
                    BRACKET_OVER_COVERAGE,
                    f"context-cost bracket claims {len(set(bracket_tasks))} task(s) but the "
                    f"scored matrix holds {n_scored}",
                )
            ]
        )
    if offenders:
        head = ", ".join(offenders[:5])
        raise DataIntegrityError(
            [
                Violation(
                    Severity.ERROR,
                    BRACKET_OVER_COVERAGE,
                    f"{len(offenders)} task(s) in a context-cost bracket billed an IMPUTED "
                    f"cell, which carries no measured tokens: {head}. The token-complete "
                    "filter has been bypassed.",
                )
            ]
        )
    # LAST, so the more specific over-coverage and IMPUTED diagnoses keep naming the defect.
    if pathless:
        raise DataIntegrityError(
            [
                Violation(
                    Severity.ERROR,
                    BRACKET_OVER_COVERAGE,
                    f"{len(pathless)} task(s) in a context-cost bracket billed NO model: "
                    f"{', '.join(pathless[:5])}. The bracket cannot be checked against the "
                    "imputed flag, so it fails closed rather than passing vacuously.",
                )
            ]
        )
