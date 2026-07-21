"""The routing embedding must carry the user's task, not the agent's system prompt."""

from __future__ import annotations

from shunt.proxy.router import _anthropic_request_to_openai, _routing_text_from_messages
from shunt.router.embedder import DEFAULT_MAX_EMBED_CHARS

# A coding agent's system prompt runs many times the embedder's clip window; this is
# the condition under which wire-order flattening silently produced a constant vector.
_HUGE_SYSTEM = "You are a coding agent. " + ("Follow the tool protocol. " * 1200)


def _embed_input(task: str) -> str:
    body = {
        "model": "m",
        "system": _HUGE_SYSTEM,
        "messages": [{"role": "user", "content": task}],
    }
    openai_kwargs = _anthropic_request_to_openai(body)
    return _routing_text_from_messages(openai_kwargs["messages"])[:DEFAULT_MAX_EMBED_CHARS]


def test_task_survives_a_system_prompt_larger_than_the_clip_window() -> None:
    assert len(_HUGE_SYSTEM) > DEFAULT_MAX_EMBED_CHARS
    assert "pagination" in _embed_input("Fix the flaky pagination test")


def test_different_tasks_do_not_collapse_to_the_same_embedding_input() -> None:
    # The regression this guards: every session embedded identically, so the kNN
    # neighbourhood was degenerate and routing carried no task signal at all.
    assert _embed_input("Fix the flaky pagination test") != _embed_input("Add a Redis cache")


def test_most_recent_task_leads() -> None:
    body = {
        "model": "m",
        "system": _HUGE_SYSTEM,
        "messages": [
            {"role": "user", "content": "first task: rename the module"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "now add retry logic"},
        ],
    }
    text = _routing_text_from_messages(_anthropic_request_to_openai(body)["messages"])
    assert text.startswith("now add retry logic")


def test_system_only_body_falls_back_rather_than_embedding_nothing() -> None:
    assert _routing_text_from_messages([{"role": "system", "content": "only sys"}]) == "only sys"


def test_empty_messages_are_safe() -> None:
    assert _routing_text_from_messages([]) == ""


def test_content_blocks_are_flattened() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "block task"}]}]
    assert "block task" in _routing_text_from_messages(messages)


def test_the_router_actually_routes_on_the_task_not_wire_order() -> None:
    """Guards the CALL SITE: testing the helper alone passed against the broken code."""
    from unittest.mock import patch

    from shunt.models.config import ModelPool
    from shunt.proxy.router import ProxyRouter
    from shunt.session import SessionManager

    router = ProxyRouter(
        model_pool=ModelPool(),
        session_manager=SessionManager(inactivity_timeout=900, grace_period=120),
        retry_count=1,
    )
    seen: list[str] = []

    def capture(session: object, prompt_text: str = "") -> str:
        seen.append(prompt_text)
        return "model-a"

    body = {
        "model": "auto",
        "system": _HUGE_SYSTEM,
        "messages": [{"role": "user", "content": "Fix the flaky pagination test"}],
    }
    with patch.object(router, "_get_or_lock_model", side_effect=capture):
        import asyncio
        import contextlib

        with contextlib.suppress(Exception):  # empty pool exhausts after the decision
            asyncio.run(router._route_with_fallback(_anthropic_request_to_openai(body), None))

    assert seen, "the decision was never made"
    assert "pagination" in seen[0][:DEFAULT_MAX_EMBED_CHARS]
