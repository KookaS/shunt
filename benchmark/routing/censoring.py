"""Stop-reason vocabulary and CENSORED-data helpers for the benchmark.

A cell that stopped on a resource limit (step/wall/hard-abandon) is CENSORED: its
true pass/fail is unknown and it must not be scored as a clean capability fail.
"""

from __future__ import annotations

from typing import Final

# The fixed stop_reason vocabulary. Every produced row carries exactly one.
SOLVED: Final[str] = "solved"  # harness resolved=True
UNSOLVED: Final[str] = "unsolved"  # agent finished (submitted/natural), harness did not resolve
STEP_LIMIT: Final[str] = "step_limit"  # agent hit its step (or cost) limit — CENSORED
WALL_LIMIT: Final[str] = "wall_limit"  # agent hit its graceful wall_time_limit_seconds — CENSORED
ABANDONED: Final[str] = "abandoned"  # external hard watchdog reaped it mid-run — CENSORED

STOP_REASONS: Final[frozenset[str]] = frozenset(
    {SOLVED, UNSOLVED, STEP_LIMIT, WALL_LIMIT, ABANDONED}
)
# CENSORED reasons: the true outcome is unknown, so the cell is neither an observed
# pass nor an observed fail (excluded from completeness crossovers and quality denominators).
CENSORED_REASONS: Final[frozenset[str]] = frozenset({STEP_LIMIT, WALL_LIMIT, ABANDONED})
# Back-compat: timeout_flag is True iff the stop was a wall/abandon timeout (NOT step_limit,
# which is a graceful non-timeout censored stop).
_TIMEOUT_FLAG_REASONS: Final[frozenset[str]] = frozenset({WALL_LIMIT, ABANDONED})

# mini-swe-agent's own exit_status (agent.run() return) → censored stop_reason, on the SUCCESS
# path where the scaffold caught its InterruptAgentFlow and returned normally. ``LimitsExceeded``
# covers BOTH the step and cost limits (query() raises it for either); both are resource-limit
# censored stops, mapped to ``step_limit``. ``TimeExceeded`` (a LimitsExceeded subclass) sets a
# DISTINCT exit_status string, so the graceful wall is separable from the step/cost limit.
_EXIT_STATUS_CENSORED: Final[dict[str, str]] = {
    "LimitsExceeded": STEP_LIMIT,
    "TimeExceeded": WALL_LIMIT,
}


def stop_reason_from_run(*, resolved: bool, exit_status: str) -> str:
    """Map a completed agent run (harness resolved + scaffold exit_status) to a stop_reason."""
    if resolved:
        return SOLVED
    return _EXIT_STATUS_CENSORED.get(exit_status, UNSOLVED)


def timeout_flag_for(stop_reason: str) -> bool:
    """Back-compat timeout_flag: True iff the stop was a wall/abandon timeout."""
    return stop_reason in _TIMEOUT_FLAG_REASONS


def is_censored_reason(stop_reason: str) -> bool:
    """True iff a stop_reason is CENSORED (resource-limit stop, unknown true outcome)."""
    return stop_reason in CENSORED_REASONS


def derive_stop_reason(*, passed: bool, timeout_flag: bool, stop_reason: str = "") -> str:
    """A row's stop_reason, deriving one for legacy rows that lack the column.

    An explicit stored value always wins; otherwise: solved if passed, else wall_limit
    (censored) if timeout_flag, else unsolved.
    """
    if stop_reason:
        return stop_reason
    if passed:
        return SOLVED
    if timeout_flag:
        return WALL_LIMIT
    return UNSOLVED


def is_censored(row: dict) -> bool:
    """True iff a results row/cell is a CENSORED stop (handles legacy rows via derivation)."""
    reason = derive_stop_reason(
        passed=bool(row.get("pass", False)),
        timeout_flag=bool(row.get("timeout_flag", False)),
        stop_reason=str(row.get("stop_reason") or ""),
    )
    return is_censored_reason(reason)
