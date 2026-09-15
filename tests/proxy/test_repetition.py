from __future__ import annotations

import pytest

from shunt.proxy.repetition import (
    MIN_REPEAT,
    action_key,
    action_keys,
    first_fire_index,
    longest_run,
    longest_run_from_messages,
)


def _assistant(content: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": arguments}}
        ],
    }


def test_action_key_is_stable_and_pair_sensitive() -> None:
    assert action_key("run tests", "ls") == action_key("run tests", "ls")
    # A different tool_call argument or a different content is a different action.
    assert action_key("run tests", "ls") != action_key("run tests", "ls -la")
    assert action_key("run tests", "ls") != action_key("", "ls")


def test_action_key_treats_null_and_empty_arguments_alike() -> None:
    assert action_key("x", None) == action_key("x", "")


def test_longest_run_counts_an_unbroken_run() -> None:
    assert longest_run(["a", "a", "a", "b", "a"]) == 3
    assert longest_run(["a", "b", "a", "b"]) == 1
    assert longest_run([]) == 0


def test_first_fire_index_requires_n_consecutive() -> None:
    keys = ["a", "a", "a"]
    assert first_fire_index(keys, 2) == 1  # the second identical action completes N=2
    assert first_fire_index(keys, 3) == 2
    assert first_fire_index(keys, 4) is None


def test_first_fire_index_only_accepts_the_first_fire() -> None:
    keys = ["x", "a", "a", "a"]
    assert first_fire_index(keys, 2) == 2
    assert first_fire_index(keys, 2) != 3


def test_a_break_resets_the_run() -> None:
    # N=2 with the two identical actions split by a different one must NOT fire.
    assert first_fire_index(["a", "b", "a"], 2) is None
    assert first_fire_index(["a", "a", "b", "a", "a"], 2) == 1


def test_first_fire_index_rejects_a_degenerate_threshold() -> None:
    with pytest.raises(ValueError):
        first_fire_index(["a", "a"], MIN_REPEAT - 1)


def test_action_keys_reads_only_assistant_turns() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        _assistant("", "ls"),
        {"role": "tool", "content": "output"},
        _assistant("", "ls"),
    ]
    keys = action_keys(messages)
    assert len(keys) == 2
    assert keys[0] == keys[1]
    assert longest_run(keys) == 2


def test_longest_run_from_messages_fires_on_n_not_n_minus_one() -> None:
    two = [_assistant("", "pytest"), _assistant("", "pytest")]
    assert longest_run_from_messages(two) == 2
    assert longest_run_from_messages([*two, _assistant("", "pytest")]) == 3
    mixed = [_assistant("", "pytest"), _assistant("", "ls"), _assistant("", "pytest")]
    assert longest_run_from_messages(mixed) == 1


def test_anthropic_tool_use_blocks_are_keyed() -> None:
    # The /v1/messages wire carries tool calls as tool_use content blocks.
    def anthropic(inputs: str) -> dict:
        return {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": inputs},
            ],
        }

    messages = [anthropic("cmd"), anthropic("cmd")]
    assert longest_run_from_messages(messages) == 2
    messages[-1]["content"][1]["input"] = "other"
    assert longest_run_from_messages(messages) == 1


def test_argument_dict_is_canonicalised() -> None:
    # Two equivalent dicts written in a different key order must map to one key.
    a = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"arguments": {"a": 1, "b": 2}}}],
    }
    b = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"arguments": {"b": 2, "a": 1}}}],
    }
    assert longest_run_from_messages([a, b]) == 2
