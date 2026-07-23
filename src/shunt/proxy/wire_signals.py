"""Structured wire-signal collection — the weak, quarantined Tier-1 prior.

Reads ONLY non-model-authored fields (``tool_result.is_error``, terminal ``stop_reason``),
never prose; the weak prior stays quarantined until an off-wire Tier-2 corroborates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from shunt.session import Session

# session.metadata keys carrying the accumulated structured signals (the collector→
# coordinator contract). Kept here so both the writer (router) and reader (coordinator)
# agree without importing each other.
WIRE_TOOL_ERROR_COUNT: Final = "wire_tool_error_count"
WIRE_TERMINAL_STOP: Final = "wire_terminal_stop"

# Weak by design: the Tier-1 prior is quarantined and never drives routing.
_WIRE_TIER1_CONFIDENCE: Final = 0.3

# Anthropic stop_reasons that reopen the agent loop — a prior is never finalized on them.
_OPEN_STOP_REASONS: Final[frozenset[str]] = frozenset({"tool_use", "pause_turn"})


class WireSignalCollector:
    """Accumulate structured, non-model-authored wire signals onto ``session.metadata``."""

    def observe_tool_errors(self, request_body: dict[str, Any], session: Session) -> None:
        """Count Anthropic ``tool_result`` blocks flagged ``is_error`` in the request history.

        The full turn history is resent each turn, so the count is monotone; ``max`` keeps
        the running peak without double-counting a block seen on an earlier turn.
        """
        count = _count_is_error_tool_results(request_body.get("messages", []))
        if count:
            prev = int(session.metadata.get(WIRE_TOOL_ERROR_COUNT, 0))
            session.metadata[WIRE_TOOL_ERROR_COUNT] = max(prev, count)

    def observe_terminal_stop(self, stop_reason: str | None, session: Session) -> None:
        """Record a terminal (Anthropic-normalized) stop_reason.

        A loop-reopening stop (``tool_use``/``pause_turn``) or a missing reason contributes
        no prior — the agent turn is not finished.
        """
        if not stop_reason or stop_reason in _OPEN_STOP_REASONS:
            return
        session.metadata[WIRE_TERMINAL_STOP] = stop_reason


def _count_is_error_tool_results(messages: object) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("is_error") is True
            ):
                count += 1
    return count


def derive_wire_tier1_outcome(metadata: dict[str, Any]) -> tuple[str, float] | None:
    """Weak Tier-1 ``(outcome, confidence)`` from accumulated wire signals, or None.

    Any structured tool error ⇒ weak ``failure``; else a clean terminal close ⇒ weak
    ``weak_success``; no structured signal ⇒ None (never guess from self-narration).
    """
    if int(metadata.get(WIRE_TOOL_ERROR_COUNT, 0)) > 0:
        return "failure", _WIRE_TIER1_CONFIDENCE
    if metadata.get(WIRE_TERMINAL_STOP):
        return "weak_success", _WIRE_TIER1_CONFIDENCE
    return None
