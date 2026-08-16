"""The fake upstream must never be able to hold an agentic client in a loop."""

# The stub is stateless, so an unconditional tool-call reply answers every turn with the
# same call and never a terminal text turn. Three handshake legs (opencode, claude-code,
# continue) spun on that for 2h16m of CI before a human cancelled the run. Two independent
# walls stand behind these tests: tool calls are OFF for the containerised stub, and a
# per-process budget bounds them even when they are ON.

from __future__ import annotations

import json
from typing import Any

import httpx

from tests.integrations.fake_upstream import FakeUpstream, _chat_completion_response


def _tools(name: str, required: list[str] | None = None) -> list[dict[str, Any]]:
    """One OpenAI-wire function-tool declaration, optionally with required keys."""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    if required is not None:
        parameters["required"] = required
        parameters["properties"] = {key: {"type": "string"} for key in required}
    return [{"type": "function", "function": {"name": name, "parameters": parameters}}]


def _post(upstream: FakeUpstream, body: dict[str, Any]) -> dict[str, Any]:
    """POST a non-streaming chat completion and return the parsed body."""
    resp = httpx.post(f"{upstream.base_url}/v1/chat/completions", json=body, timeout=10)
    resp.raise_for_status()
    return dict(resp.json())


def test_tool_calls_off_answers_text_even_when_the_request_declares_tools() -> None:
    """The regression: the Docker stub must answer text, so one round trip ends the leg."""
    payload = {"model": "m", "tools": _tools("bash", ["command"])}

    is_sse, frames = _chat_completion_response(payload, tool_calls=False)

    assert is_sse is False
    choice = json.loads(frames[0])["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "ok"


def test_tool_calls_off_answers_a_text_stream_when_streaming() -> None:
    """Same for the SSE path — a streamed leg must reach `stop`, not another tool call."""
    payload = {"model": "m", "stream": True, "tools": _tools("bash", ["command"])}

    is_sse, frames = _chat_completion_response(payload, tool_calls=True)

    assert is_sse is True
    assert any('"finish_reason": "tool_calls"' in frame for frame in frames)

    is_sse, frames = _chat_completion_response(payload, tool_calls=False)

    assert is_sse is True
    assert any('"finish_reason": "stop"' in frame for frame in frames)


def test_arguments_satisfy_a_required_key_the_canned_ones_do_not_cover() -> None:
    """The exact shape that broke opencode: bash declares `command`, not `path`."""
    payload = {"model": "m", "tools": _tools("bash", ["command"])}

    _, frames = _chat_completion_response(payload, tool_calls=True)

    call = json.loads(frames[0])["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    assert "command" in json.loads(call["function"]["arguments"])


def test_canned_arguments_survive_a_schema_that_does_not_contradict_them() -> None:
    """A tool declaring no `required` keeps `{"path": "main.py"}` — the suites assert it."""
    payload = {"model": "m", "tools": _tools("read_file")}

    _, frames = _chat_completion_response(payload, tool_calls=True)

    call = json.loads(frames[0])["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"path": "main.py"}


def test_the_budget_bounds_a_loop_even_with_tool_calls_enabled() -> None:
    """The wall that does not depend on configuration: past the cap, always text."""
    body = {"model": "m", "tools": _tools("read_file")}
    with FakeUpstream(max_tool_calls=1) as upstream:
        first = _post(upstream, body)
        second = _post(upstream, body)

    assert first["choices"][0]["finish_reason"] == "tool_calls"
    assert second["choices"][0]["finish_reason"] == "stop"


def test_a_request_without_tools_does_not_spend_the_budget() -> None:
    """Otherwise a chatty non-tool client would silently disarm the tool-call path."""
    with FakeUpstream(max_tool_calls=1) as upstream:
        plain = _post(upstream, {"model": "m"})
        called = _post(upstream, {"model": "m", "tools": _tools("read_file")})

    assert plain["choices"][0]["finish_reason"] == "stop"
    assert called["choices"][0]["finish_reason"] == "tool_calls"
