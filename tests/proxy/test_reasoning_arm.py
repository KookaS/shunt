"""B5 outbound: the proxy injects an escalated reasoning arm into the upstream request."""

# The escalated arm's raw API params reach the single upstream seam, override any client
# reasoning, and leave the served model (hence the cache namespace) unchanged. Upstream mocked.

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shunt.models.config import ModelConfig, ModelPool, ReasoningArm, ReasoningConfig
from shunt.proxy.router import ProxyRouter
from shunt.session import Session, SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"


def _response() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="ok", tool_calls=[]))]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, cost=0.0)
    resp.usage.prompt_tokens_details = MagicMock(cached_tokens=0)
    resp.model = "qwen3.7-plus"
    resp.id = "resp-1"
    return resp


@pytest.fixture
def router() -> ProxyRouter:
    # No engine → the default cold-start model is qwen3.7-plus, whose registry ladder is
    # nothink(rank0) / think(rank1). The router routes to it; we inject the escalated arm.
    return ProxyRouter(model_pool=ModelPool(), session_manager=SessionManager())


def _session(sm: SessionManager) -> Session:
    return sm.create_session("test-tool")


@pytest.mark.asyncio
async def test_escalated_arm_reaches_upstream_and_overrides_client(router: ProxyRouter) -> None:
    session = router._sessions.create_session("test-tool")
    session.metadata["reasoning_arm"] = "think"  # as set by an effort escalation on the decision
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "do it"}],
        "reasoning_effort": "low",  # client-supplied reasoning the arm must override
    }
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        _payload, model_name, _reason = await router.route_chat_completion(body, session)

    kwargs = mock_ac.call_args.kwargs
    assert kwargs["enable_thinking"] is True  # the "think" arm's api param reached upstream
    assert "reasoning_effort" not in kwargs  # client-supplied reasoning was overridden/removed
    assert model_name == "qwen3.7-plus"  # cache-safe: served model unchanged by the effort step


@pytest.mark.asyncio
async def test_no_arm_leaves_request_untouched(router: ProxyRouter) -> None:
    session = router._sessions.create_session("test-tool")  # no reasoning_arm on metadata
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "do it"}]}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        await router.route_chat_completion(body, session)
    assert "enable_thinking" not in mock_ac.call_args.kwargs


@pytest.mark.asyncio
async def test_escalated_arm_not_leaked_to_fallback_sibling() -> None:
    # B5 leak: the arm resolved for the HEAD model must not reach a fallback sibling that
    # coincidentally shares the arm id — the sibling gets its own defaults, no injection.
    head = ModelConfig(
        name="head",
        tier="cheap",
        provider="p",
        base_url="http://x",
        api_key_env_var="K",
        reasoning=ReasoningConfig(
            default_arm="low",
            arms=[
                ReasoningArm(id="low", rank=0, api={"reasoning_effort": "low"}),
                ReasoningArm(id="think", rank=1, api={"enable_thinking": True}),
            ],
        ),
    )
    sib = ModelConfig(
        name="sib",
        tier="cheap",
        provider="p",
        base_url="http://x",
        api_key_env_var="K",
        reasoning=ReasoningConfig(
            default_arm="think",  # SAME arm id as head, but the sibling must still not get it
            arms=[ReasoningArm(id="think", rank=0, api={"enable_thinking": True})],
        ),
    )
    pool = MagicMock(spec=ModelPool)
    pool.get_model.side_effect = lambda name: {"head": head, "sib": sib}.get(name)
    pool.fallback_chain.return_value = ["head", "sib"]
    pool.is_healthy.return_value = True

    router = ProxyRouter(model_pool=pool, session_manager=SessionManager())
    session = router._sessions.create_session("t")
    session.model_chosen = "head"  # locked; _get_or_lock_model returns it directly
    session.metadata["reasoning_arm"] = "think"  # escalated on head

    seen: list[tuple[str, dict[str, Any]]] = []

    async def fake_ac(config: ModelConfig, **kwargs: Any) -> MagicMock:
        seen.append((config.name, dict(kwargs)))
        if config.name == "head":
            raise RuntimeError("head is down")  # forces fallback to sib
        return _response()

    body: dict[str, Any] = {"messages": [{"role": "user", "content": "do it"}]}
    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=fake_ac)):
        _payload, model_name, _reason = await router.route_chat_completion(body, session)

    assert model_name == "sib"  # fell back
    head_kwargs = next(kw for name, kw in seen if name == "head")
    sib_kwargs = next(kw for name, kw in seen if name == "sib")
    assert head_kwargs.get("enable_thinking") is True  # head DID get its escalated arm
    assert "enable_thinking" not in sib_kwargs  # the sibling did NOT — no leak across fallback


def test_apply_reasoning_arm_skips_arm_foreign_to_the_served_model() -> None:
    # A fallback to a different model whose ladder lacks the arm id must not inject foreign params.
    pool = MagicMock(spec=ModelPool)
    pool.get_model.return_value = ModelConfig(
        name="other",
        tier="mid",
        provider="p",
        base_url="http://x",
        api_key_env_var="K",
        reasoning=ReasoningConfig(
            default_arm="a", arms=[ReasoningArm(id="a", rank=0, api={"reasoning_effort": "low"})]
        ),
    )
    router = ProxyRouter(model_pool=pool, session_manager=SessionManager())
    session = router._sessions.create_session("t")
    session.metadata["reasoning_arm"] = "think"  # not an arm of "other"
    kwargs: dict[str, Any] = {"messages": []}
    router._apply_reasoning_arm(kwargs, session, "other")
    assert kwargs == {"messages": []}  # untouched — foreign arm id skipped
