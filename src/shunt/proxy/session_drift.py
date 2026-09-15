"""Cross-session drift: session-close counter summaries and their K-window aggregate.

The proxy may consult the last few closed sessions on the same repo at the NEXT session
boundary — never mid-turn and never on the wire — to ask whether carry-cost counters are
drifting. Only four whitelisted behavioural counters are read: ``is_revert``,
``retry_count``, ``loop_signal`` and the live structured ``WIRE_TOOL_ERROR_COUNT``. No
model-authored text enters a feature. Everything here is pure: the caller owns storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from shunt.proxy.wire_signals import WIRE_TOOL_ERROR_COUNT

# The task's pre-registered window grid: the last 3-10 same-repo sessions.
MIN_WINDOW: Final[int] = 3
MAX_WINDOW: Final[int] = 10
WINDOW_GRID: Final[tuple[int, ...]] = tuple(range(MIN_WINDOW, MAX_WINDOW + 1))

# The four whitelisted counters, in a fixed order so feature vectors never reorder.
COUNTER_NAMES: Final[tuple[str, ...]] = (
    "is_revert",
    "retry_count",
    "loop_signal",
    "wire_tool_error_count",
)
_COUNTER_COUNT: Final[int] = len(COUNTER_NAMES)


class CounterStep(Protocol):
    """The per-step behavioural fields a session summary reads — nothing free-text.

    Declared as read-only properties so a plain dataclass field satisfies it structurally.
    """

    @property
    def is_revert(self) -> bool: ...

    @property
    def retry_count(self) -> int: ...

    @property
    def loop_signal(self) -> bool: ...


@dataclass(frozen=True)
class SessionCounters:
    """One closed session's four whitelisted counter aggregates, plus its step count."""

    is_reverts: int
    retry_total: int
    loop_signals: int
    wire_tool_errors: int
    n_steps: int

    def counts(self) -> tuple[int, ...]:
        """The four counter aggregates in ``COUNTER_NAMES`` order."""
        return (self.is_reverts, self.retry_total, self.loop_signals, self.wire_tool_errors)

    def rates(self) -> tuple[float, ...]:
        """The four counters as per-step rates (0.0 when the session had no steps)."""
        denominator = self.n_steps if self.n_steps > 0 else 1
        return tuple(count / denominator for count in self.counts())


def summarize(records: Sequence[CounterStep], *, wire_tool_errors: int = 0) -> SessionCounters:
    """Aggregate the three per-step counters plus the session-level wire error peak."""
    return SessionCounters(
        is_reverts=sum(1 for record in records if record.is_revert),
        retry_total=sum(int(record.retry_count) for record in records),
        loop_signals=sum(1 for record in records if record.loop_signal),
        wire_tool_errors=max(int(wire_tool_errors), 0),
        n_steps=len(records),
    )


def wire_errors_from_metadata(metadata: dict[str, object]) -> int:
    """The session's peak ``WIRE_TOOL_ERROR_COUNT``, or 0 when the signal was never seen."""
    value = metadata.get(WIRE_TOOL_ERROR_COUNT, 0)
    return int(value) if isinstance(value, (int, float)) else 0


@dataclass(frozen=True)
class DriftFeatures:
    """The windowed drift features over the K sessions preceding a session boundary."""

    window: int
    mean_rates: tuple[float, ...]
    drift_level: float
    drift_slope: float

    def to_dict(self) -> dict[str, object]:
        """A JSON-safe projection naming each counter's mean rate."""
        out: dict[str, object] = {"window": self.window, "drift_level": round(self.drift_level, 6)}
        out["drift_slope"] = round(self.drift_slope, 6)
        for name, rate in zip(COUNTER_NAMES, self.mean_rates, strict=True):
            out[f"{name}_mean_rate"] = round(rate, 6)
        return out


def window_features(window: Sequence[SessionCounters]) -> DriftFeatures:
    """The drift level and slope of a non-empty window of same-repo session summaries.

    ``drift_level`` is the sum of the four per-counter window means; ``drift_slope`` is the
    sum of each counter's last-minus-first rate, so a stable window scores slope 0.
    """
    if not window:
        raise ValueError("drift window must not be empty")
    count = len(window)
    rates = [summary.rates() for summary in window]
    means = tuple(sum(rate[index] for rate in rates) / count for index in range(_COUNTER_COUNT))
    last, first = rates[-1], rates[0]
    slope = sum(last[index] - first[index] for index in range(_COUNTER_COUNT))
    return DriftFeatures(
        window=count,
        mean_rates=means,
        drift_level=sum(means),
        drift_slope=slope,
    )


def drift_level(window: Sequence[SessionCounters]) -> float:
    """The primary scalar drift score of a session window."""
    return window_features(window).drift_level


__all__ = [
    "COUNTER_NAMES",
    "MAX_WINDOW",
    "MIN_WINDOW",
    "WINDOW_GRID",
    "CounterStep",
    "DriftFeatures",
    "SessionCounters",
    "drift_level",
    "summarize",
    "window_features",
    "wire_errors_from_metadata",
]
