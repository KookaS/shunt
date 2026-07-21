"""The Anthropic surface must carry tool calls in both directions, not just text."""

# `/v1/messages` is the Claude Code path. It used to build a content list of exactly one
# `text` block and never read `message.tool_calls`, so every tool call was silently
# deleted and the agent could not act. Inbound `tool_use`/`tool_result` history was
# forwarded to OpenAI unconverted, so turn 2 of any tool conversation was malformed.

from __future__ import annotations

import json
from typing import Any

from shunt.proxy.router import (
    _anthropic_request_to_openai,
    _openai_chunk_to_anthropic_sse,
    _openai_response_to_anthropic,
    final_sse_events,
)


class _Obj:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def _response_with_tool_call() -> Any:
    call = _Obj(
        id="call_abc123",
        type="function",
        function=_Obj(name="read_file", arguments='{"path": "main.py"}'),
    )
    choice = _Obj(message=_Obj(content=None, tool_calls=[call]), finish_reason="tool_calls")
    return _Obj(id="resp_1", choices=[choice], model="qwen3.7-plus", usage=None)


# ── response direction ──────────────────────────────────────────────────────


def test_a_tool_call_survives_translation_to_anthropic() -> None:
    out = _openai_response_to_anthropic(_response_with_tool_call())
    blocks = [b for b in out["content"] if b.get("type") == "tool_use"]
    assert blocks, f"tool call dropped; content was {out['content']!r}"
    assert blocks[0]["name"] == "read_file"
    assert blocks[0]["id"] == "call_abc123"
    # Decoded, not the raw JSON string OpenAI sends.
    assert blocks[0]["input"] == {"path": "main.py"}


def test_tool_use_stop_reason_uses_the_anthropic_vocabulary() -> None:
    assert _openai_response_to_anthropic(_response_with_tool_call())["stop_reason"] == "tool_use"


def test_text_and_tool_calls_coexist_in_order() -> None:
    call = _Obj(id="c1", function=_Obj(name="ls", arguments="{}"))
    choice = _Obj(
        message=_Obj(content="let me look", tool_calls=[call]), finish_reason="tool_calls"
    )
    out = _openai_response_to_anthropic(_Obj(id="r", choices=[choice], model="m", usage=None))
    assert [b["type"] for b in out["content"]] == ["text", "tool_use"]


def test_malformed_tool_arguments_do_not_lose_the_turn() -> None:
    # A truncated argument string must degrade to an empty input, not raise.
    call = _Obj(id="c1", function=_Obj(name="ls", arguments='{"path": '))
    choice = _Obj(message=_Obj(content=None, tool_calls=[call]), finish_reason="tool_calls")
    out = _openai_response_to_anthropic(_Obj(id="r", choices=[choice], model="m", usage=None))
    assert out["content"][0]["input"] == {}


def test_a_plain_text_response_still_translates() -> None:
    choice = _Obj(message=_Obj(content="hello", tool_calls=None), finish_reason="stop")
    out = _openai_response_to_anthropic(_Obj(id="r", choices=[choice], model="m", usage=None))
    assert out["content"] == [{"type": "text", "text": "hello"}]
    assert out["stop_reason"] == "end_turn"


# ── request direction ───────────────────────────────────────────────────────


def test_assistant_tool_use_becomes_openai_tool_calls() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"p": "a.py"}}
                ],
            }
        ]
    }
    message = _anthropic_request_to_openai(body)["messages"][0]
    assert message["tool_calls"][0]["id"] == "tu_1"
    assert message["tool_calls"][0]["function"]["name"] == "read_file"
    # OpenAI wants arguments as a JSON string.
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"p": "a.py"}


def test_tool_result_becomes_a_separate_tool_role_message() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "file contents"},
                    {"type": "text", "text": "now fix it"},
                ],
            }
        ]
    }
    messages = _anthropic_request_to_openai(body)["messages"]
    # The result must precede the new user text, as OpenAI requires.
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "tu_1"
    assert messages[0]["content"] == "file contents"
    assert messages[1]["role"] == "user"


def test_block_shaped_tool_result_is_flattened_to_text() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": "line1"}],
                    }
                ],
            }
        ]
    }
    assert _anthropic_request_to_openai(body)["messages"][0]["content"] == "line1"


def test_plain_text_messages_are_unchanged() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert _anthropic_request_to_openai(body)["messages"] == [{"role": "user", "content": "hi"}]


def test_cache_control_markers_survive_on_text_blocks() -> None:
    # Cache-safety: a passthrough block must reach the upstream untouched.
    block = {"type": "text", "text": "big prefix", "cache_control": {"type": "ephemeral"}}
    body = {"messages": [{"role": "user", "content": [block]}]}
    assert _anthropic_request_to_openai(body)["messages"][0]["content"] == [block]


# ── streaming ───────────────────────────────────────────────────────────────


def _chunk(delta: dict[str, Any], finish: str | None = None) -> Any:
    return _Obj(
        choices=[_Obj(delta=_Obj(model_dump=lambda **_: delta), finish_reason=finish)],
        model="m",
        usage=None,
    )


def test_streamed_tool_call_emits_tool_use_and_input_json_deltas() -> None:
    state: dict[str, Any] = {}
    events: list[str] = []
    events += _openai_chunk_to_anthropic_sse(_chunk({"role": "assistant"}), state=state)
    events += _openai_chunk_to_anthropic_sse(
        _chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file"}}]}),
        state=state,
    )
    events += _openai_chunk_to_anthropic_sse(
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"p":'}}]}), state=state
    )
    events += _openai_chunk_to_anthropic_sse(
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}]}), state=state
    )
    events += _openai_chunk_to_anthropic_sse(_chunk({}, finish="tool_calls"), state=state)
    # message_delta/message_stop are deferred so the trailing usage-only chunk
    # can be folded in; the driver calls this once the stream is exhausted.
    events += final_sse_events(state)
    blob = "".join(events)

    assert '"type": "tool_use"' in blob
    assert '"name": "read_file"' in blob
    assert '"type": "input_json_delta"' in blob
    # The two argument fragments must both survive, so the client can reassemble.
    assert '{\\"p\\":' in blob or '{"p":' in blob
    assert '"stop_reason": "tool_use"' in blob
    # The tool block gets its own index, and both blocks are closed (text + tool).
    assert '"index": 1' in blob
    assert blob.count("event: content_block_stop") == 2


def test_streamed_text_is_unaffected() -> None:
    state: dict[str, Any] = {}
    events = _openai_chunk_to_anthropic_sse(_chunk({"role": "assistant"}), state=state)
    events += _openai_chunk_to_anthropic_sse(_chunk({"content": "hello"}), state=state)
    blob = "".join(events)
    assert '"type": "text_delta"' in blob
    assert "tool_use" not in blob


def _usage_chunk(prompt: int, completion: int) -> Any:
    """The trailing chunk stream_options.include_usage sends: usage, no choices."""
    return _Obj(
        choices=[],
        model="m",
        usage=_Obj(prompt_tokens=prompt, completion_tokens=completion),
    )


def test_streamed_usage_arrives_from_the_trailing_choices_less_chunk() -> None:
    # OpenAI sends token counts in a final chunk with `choices: []`, AFTER the
    # finish_reason chunk (whose own usage is null). That chunk used to hit an
    # early return, so every streamed Anthropic response reported `usage: {}`
    # to the client — Claude Code's token accounting silently read zero.
    state: dict[str, Any] = {}
    events = _openai_chunk_to_anthropic_sse(_chunk({"role": "assistant"}), state=state)
    events += _openai_chunk_to_anthropic_sse(_chunk({"content": "hi"}), state=state)
    events += _openai_chunk_to_anthropic_sse(_chunk({}, finish="stop"), state=state)
    events += _openai_chunk_to_anthropic_sse(_usage_chunk(1234, 56), state=state)
    events += final_sse_events(state)

    deltas = [e for e in events if e.startswith("event: message_delta")]
    assert len(deltas) == 1
    payload = json.loads(deltas[0].split("data: ", 1)[1].strip())
    assert payload["usage"] == {"input_tokens": 1234, "output_tokens": 56}
    assert payload["delta"]["stop_reason"] == "end_turn"
    assert sum(e.startswith("event: message_stop") for e in events) == 1


def test_stream_without_a_finish_reason_emits_no_terminal_events() -> None:
    # A truncated/aborted stream must not fabricate a message_stop.
    state: dict[str, Any] = {}
    _openai_chunk_to_anthropic_sse(_chunk({"content": "partial"}), state=state)
    assert final_sse_events(state) == []
