"""Tools must reach the upstream model — Shunt fronts tool-calling coding agents."""

from __future__ import annotations

from typing import Any, Final

from shunt.proxy.router import (
    _anthropic_request_to_openai,
    _anthropic_tool_choice_to_openai,
    _anthropic_tools_to_openai,
)

OPENAI_TOOL: Final[dict[str, Any]] = {
    "type": "function",
    "function": {"name": "Read", "description": "Read a file", "parameters": {"type": "object"}},
}
ANTHROPIC_TOOL: Final[dict[str, Any]] = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}


class TestAnthropicToolConversion:
    def test_tools_are_converted_to_the_openai_function_shape(self) -> None:
        # Without this a Claude Code request reached the model with NO tools, and the
        # model answered with an invented "<tool_calls>" string instead of calling one.
        kwargs = _anthropic_request_to_openai(
            {"messages": [{"role": "user", "content": "hi"}], "tools": [ANTHROPIC_TOOL]}
        )

        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": ANTHROPIC_TOOL["input_schema"],
                },
            }
        ]

    def test_a_tool_without_a_schema_still_gets_a_parameters_object(self) -> None:
        assert _anthropic_tools_to_openai([{"name": "Ping"}])[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_malformed_tool_entries_are_skipped_not_crashed(self) -> None:
        assert _anthropic_tools_to_openai(["nope", {"no_name": 1}, ANTHROPIC_TOOL]) == [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": ANTHROPIC_TOOL["input_schema"],
                },
            }
        ]

    def test_absent_tools_add_no_key(self) -> None:
        kwargs = _anthropic_request_to_openai({"messages": []})

        assert "tools" not in kwargs and "tool_choice" not in kwargs

    def test_tool_choice_mapping(self) -> None:
        assert _anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"
        assert _anthropic_tool_choice_to_openai({"type": "any"}) == "required"
        assert _anthropic_tool_choice_to_openai({"type": "tool", "name": "Read"}) == {
            "type": "function",
            "function": {"name": "Read"},
        }
        assert _anthropic_tool_choice_to_openai({"type": "bogus"}) is None
        assert _anthropic_tool_choice_to_openai(None) is None
