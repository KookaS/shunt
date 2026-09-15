"""Degenerate-repetition-loop signal: N consecutive identical model actions.

The trigger reads only what the proxy already receives on every turn — the assistant
message content and the tool-call arguments — so it needs no verified outcome and can
fire before a run has failed anything. It is a *consecutive* rule: a different action
resets the run. The functions here are pure (no module state), so the wire path can
recompute the signal from the resent history without persisting a counter.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

# session.metadata key under which the proxy records the longest run it saw on a request.
REPETITION_RUN_KEY: Final[str] = "repetition_run"
# The smallest N the trigger can fire on: one action is not a repetition.
MIN_REPEAT: Final[int] = 2
_SEP: Final[str] = "\x1f"


def action_key(content: str, arguments: str | None) -> str:
    """Stable key for one ``(content, tool_call arguments)`` action pair."""
    # A hash keeps the key fixed-width and free of free text, so nothing authored by the
    # model is retained here and two distinct pairs cannot collide by concatenation.
    payload = f"{content}{_SEP}{arguments or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def longest_run(keys: list[str]) -> int:
    """The longest unbroken run of equal keys (1 for a unique list, 0 when empty)."""
    best = 0
    current = 0
    previous: str | None = None
    for key in keys:
        current = current + 1 if key == previous else 1
        best = max(best, current)
        previous = key
    return best


def first_fire_index(keys: list[str], n: int) -> int | None:
    """The 0-based index at which the trailing equal-key run first reaches *n*, or None."""
    if n < MIN_REPEAT:
        raise ValueError(f"n must be >= {MIN_REPEAT}, got {n}")
    current = 0
    previous: str | None = None
    for index, key in enumerate(keys):
        current = current + 1 if key == previous else 1
        previous = key
        if current >= n:
            return index
    return None


def action_keys(messages: object) -> list[str]:
    """One action key per assistant message in *messages*, in wire order."""
    if not isinstance(messages, list):
        return []
    keys: list[str] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            keys.append(action_key(_content_text(message.get("content")), _arguments_text(message)))
    return keys


def longest_run_from_messages(messages: object) -> int:
    """The longest consecutive identical-action run across a request's assistant turns."""
    return longest_run(action_keys(messages))


def _tool_arguments(message: dict[str, object]) -> list[str]:
    """Canonical argument strings for every tool call on one assistant message, in order."""
    out: list[str] = []
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                function = call.get("function")
                if isinstance(function, dict):
                    out.append(_canonical(function.get("arguments")))
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append(_canonical(block.get("input")))
    return out


def _arguments_text(message: dict[str, object]) -> str:
    """A single stable string for a message's tool-call arguments (empty when it has none)."""
    return _SEP.join(_tool_arguments(message))


def _canonical(value: object) -> str:
    """A stable string for one tool-call argument payload."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_text(content: object) -> str:
    """The text the assistant authored: a string, or the text blocks of a content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") in (None, "text")
        ]
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


__all__ = [
    "MIN_REPEAT",
    "REPETITION_RUN_KEY",
    "action_key",
    "action_keys",
    "first_fire_index",
    "longest_run",
    "longest_run_from_messages",
]
