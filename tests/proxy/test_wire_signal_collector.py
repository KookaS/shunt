from __future__ import annotations

from datetime import UTC, datetime

from shunt.proxy.wire_signals import (
    WIRE_TERMINAL_STOP,
    WIRE_TOOL_ERROR_COUNT,
    WireSignalCollector,
    derive_wire_tier1_outcome,
)
from shunt.session import Session


def _session() -> Session:
    return Session(session_id="s1", tool_identity="t", start_time=datetime.now(UTC))


def _tool_result_body(*, is_error: bool) -> dict:
    """An Anthropic request whose user turn carries a tool_result block (harness-authored)."""
    return {
        "messages": [
            {"role": "user", "content": "run the tests"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": is_error,
                        "content": "pytest: 3 failed",
                    }
                ],
            },
        ]
    }


def test_is_error_tool_result_is_counted() -> None:
    collector = WireSignalCollector()
    session = _session()
    collector.observe_tool_errors(_tool_result_body(is_error=True), session)
    assert session.metadata[WIRE_TOOL_ERROR_COUNT] == 1


def test_clean_tool_result_records_no_error() -> None:
    collector = WireSignalCollector()
    session = _session()
    collector.observe_tool_errors(_tool_result_body(is_error=False), session)
    assert WIRE_TOOL_ERROR_COUNT not in session.metadata


def test_terminal_stop_recorded() -> None:
    collector = WireSignalCollector()
    session = _session()
    collector.observe_terminal_stop("end_turn", session)
    assert session.metadata[WIRE_TERMINAL_STOP] == "end_turn"


def test_open_loop_stop_reasons_are_ignored() -> None:
    # tool_use / pause_turn reopen the agent loop — no prior is finalized on them.
    collector = WireSignalCollector()
    for reason in ("tool_use", "pause_turn"):
        session = _session()
        collector.observe_terminal_stop(reason, session)
        assert WIRE_TERMINAL_STOP not in session.metadata


def test_none_stop_reason_is_ignored() -> None:
    collector = WireSignalCollector()
    session = _session()
    collector.observe_terminal_stop(None, session)
    assert WIRE_TERMINAL_STOP not in session.metadata


def test_error_count_is_monotone_across_resent_history() -> None:
    # Coding agents resend the full history each turn; the running peak must not double-count.
    collector = WireSignalCollector()
    session = _session()
    collector.observe_tool_errors(_tool_result_body(is_error=True), session)
    collector.observe_tool_errors(_tool_result_body(is_error=True), session)
    assert session.metadata[WIRE_TOOL_ERROR_COUNT] == 1


def test_assistant_prose_never_produces_a_label() -> None:
    # Self-narration is NOT a label. A response whose assistant text says "all tests
    # passed" (and a tool_result whose TEXT says "Error:") must not create a failure/success
    # signal — only the structured is_error flag and terminal stop_reason are read.
    collector = WireSignalCollector()
    session = _session()
    body = {
        "messages": [
            {"role": "assistant", "content": "All tests passed, everything works correctly."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        # No is_error flag; the prose says "Error:" but that is not structured.
                        "content": "Error: Traceback (most recent call last)",
                    }
                ],
            },
        ]
    }
    collector.observe_tool_errors(body, session)
    collector.observe_terminal_stop("end_turn", session)
    # The prose "Error:" produced NO error signal; only the clean terminal stop remains.
    assert WIRE_TOOL_ERROR_COUNT not in session.metadata
    derived = derive_wire_tier1_outcome(session.metadata)
    assert derived is not None
    assert derived[0] == "weak_success"  # from the terminal stop, never from the prose


def test_derive_failure_from_tool_error() -> None:
    metadata = {WIRE_TOOL_ERROR_COUNT: 2, WIRE_TERMINAL_STOP: "end_turn"}
    derived = derive_wire_tier1_outcome(metadata)
    assert derived is not None
    assert derived[0] == "failure"  # a structured error outranks the clean-close prior


def test_derive_none_without_signal() -> None:
    assert derive_wire_tier1_outcome({}) is None
