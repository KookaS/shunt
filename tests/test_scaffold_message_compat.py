"""The scaffold must not replay provider-specific message metadata to strict providers.

litellm attaches ``provider_specific_fields`` (and may nest it) to the assistant messages it
returns; mini-swe-agent appends those messages to its conversation and replays them on the
next turn, where an OpenAI-compatible provider such as Groq rejects the unknown property with
a 400. ``EnvKeyLitellmModel._query`` is the benchmark's only seam around a provider call, so
it sanitizes the OUTGOING copy there. No live call is made: the parent ``_query`` is stubbed.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmark.runner import scaffold_model


def _model() -> scaffold_model.EnvKeyLitellmModel:
    return scaffold_model.EnvKeyLitellmModel(model_name="acme/local-test")


def _capture_outgoing(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the parent provider call and capture the messages the seam actually sends."""
    seen: dict[str, Any] = {}

    def fake_query(self: Any, messages: Any, **kwargs: Any) -> str:
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return "response"

    monkeypatch.setattr(scaffold_model.LitellmModel, "_query", fake_query)
    return seen


def test_top_level_provider_specific_fields_is_stripped_before_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture_outgoing(monkeypatch)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
            "provider_specific_fields": {"cached_content": "xyz"},
        },
    ]
    assert _model()._query(messages) == "response"  # type: ignore[arg-type]
    sent = seen["messages"][1]
    assert "provider_specific_fields" not in sent
    # Ordinary fields and the tool call survive the cleaning untouched.
    assert sent["role"] == "assistant"
    assert sent["tool_calls"][0]["function"]["name"] == "bash"
    assert sent["tool_calls"][0]["function"]["arguments"] == "{}"


def test_nested_provider_specific_fields_is_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture_outgoing(monkeypatch)
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok", "provider_specific_fields": {"a": 1}},
            ],
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": "{}",
                        "provider_specific_fields": {"b": 2},
                    },
                }
            ],
        }
    ]
    _model()._query(messages)  # type: ignore[arg-type]
    sent = seen["messages"][0]
    assert "provider_specific_fields" not in sent
    assert "provider_specific_fields" not in sent["content"][0]
    assert "provider_specific_fields" not in sent["tool_calls"][0]
    assert "provider_specific_fields" not in sent["tool_calls"][0]["function"]


def test_ordinary_messages_pass_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_outgoing(monkeypatch)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "on it",
            "tool_calls": [{"id": "x", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "x", "content": "done"},
    ]
    _model()._query(messages)  # type: ignore[arg-type]
    assert seen["messages"] == messages


def test_the_scaffolds_own_message_list_is_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only the request copy is cleaned; the trajectory (and its persisted dump) keeps the
    # metadata litellm attached, so capture remains a faithful record of what was returned.
    _capture_outgoing(monkeypatch)
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "", "provider_specific_fields": {"k": 1}}
    ]
    _model()._query(messages)  # type: ignore[arg-type]
    assert messages[0]["provider_specific_fields"] == {"k": 1}


def test_reasoning_trace_keys_are_stripped_but_content_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Groq rejects `reasoning_content`/`reasoning` on a replayed assistant message after
    # `provider_specific_fields` was stripped (the second 400 observed live). The reasoning
    # trace is never re-submitted as input, so it must be dropped from the outgoing copy too,
    # while the assistant's actual content and tool call survive.
    seen = _capture_outgoing(monkeypatch)
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "on it",
            "reasoning_content": "let me think",
            "reasoning": "let me think",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash"}}],
        }
    ]
    _model()._query(messages)  # type: ignore[arg-type]
    sent = seen["messages"][0]
    assert "reasoning_content" not in sent
    assert "reasoning" not in sent
    assert sent["content"] == "on it"
    assert sent["tool_calls"][0]["function"]["name"] == "bash"
