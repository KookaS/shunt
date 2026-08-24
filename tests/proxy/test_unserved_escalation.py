"""A rung that never reached the wire must fail loudly and must not be logged as escalated."""

# Two failure modes the fallback chain used to hide, both of which made the store record a
# successful escalation for a request no escalated params were ever sent on:
#   * a TypeError from the SDK (OUR kwargs, rejected before a byte left the process) was caught
#     as "model failed", walked the chain, and served a DIFFERENT model as if nothing happened;
#   * any fallback to a sibling left `auto_escalated: true` + `escalated_reasoning_arm` standing.

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shunt.models.config import ModelConfig, ModelPool, ReasoningArm, ReasoningConfig
from shunt.proxy.router import ProxyRouter, UnsupportedRequestParamError
from shunt.session import SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"


def _response() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="ok", tool_calls=[]))]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, cost=0.0)
    resp.usage.prompt_tokens_details = MagicMock(cached_tokens=0)
    resp.model = "head"
    resp.id = "resp-1"
    return resp


def _model(name: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="p",
        base_url="http://x",
        api_key_env_var="K",
        reasoning=ReasoningConfig(
            default_arm="low",
            arms=[
                ReasoningArm(id="low", rank=0, api={"reasoning_effort": "low"}),
                ReasoningArm(id="think", rank=1, api={"thinking": {"type": "enabled"}}),
            ],
        ),
    )


def _router() -> ProxyRouter:
    pool = MagicMock(spec=ModelPool)
    pool.get_model.side_effect = lambda name: _model(name) if name in {"head", "sib"} else None
    pool.fallback_chain.return_value = ["head", "sib"]
    pool.is_healthy.return_value = True
    return ProxyRouter(model_pool=pool, session_manager=SessionManager())


def _escalated_session(router: ProxyRouter) -> Any:
    session = router._sessions.create_session("t")
    session.model_chosen = "head"
    session.metadata["reasoning_arm"] = "think"
    session.decision_provenance = {
        "model_chosen": "head",
        "auto_escalated": True,
        "escalated_reasoning_arm": "think",
        "escalation_exploration": {"action": "raise_effort", "propensity": 0.2, "randomized": True},
    }
    return session


@pytest.mark.asyncio
async def test_a_rejected_kwarg_surfaces_and_never_falls_back() -> None:
    router = _router()
    session = _escalated_session(router)
    served: list[str] = []

    async def fake_ac(config: ModelConfig, **kwargs: Any) -> MagicMock:
        served.append(config.name)
        raise TypeError("create() got an unexpected keyword argument 'thinking'")

    body: dict[str, Any] = {"messages": [{"role": "user", "content": "do it"}]}
    with (
        patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=fake_ac)),
        pytest.raises(UnsupportedRequestParamError) as excinfo,
    ):
        await router.route_chat_completion(body, session)

    assert excinfo.value.status_code == 500  # ours, not a provider outage
    assert served == ["head"]  # the chain was NOT walked — no retry, no sibling
    prov = session.decision_provenance
    assert prov["auto_escalated"] is False
    assert "escalated_reasoning_arm" not in prov
    assert "escalation_not_served" in prov
    assert prov["escalation_exploration"]["randomized"] is False


@pytest.mark.asyncio
async def test_a_fallback_declaims_the_escalation_it_did_not_serve() -> None:
    router = _router()
    session = _escalated_session(router)

    async def fake_ac(config: ModelConfig, **kwargs: Any) -> MagicMock:
        if config.name == "head":
            raise RuntimeError("head is down")
        return _response()

    body: dict[str, Any] = {"messages": [{"role": "user", "content": "do it"}]}
    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=fake_ac)):
        _payload, model_name, _reason = await router.route_chat_completion(body, session)

    assert model_name == "sib"
    prov = session.decision_provenance
    assert prov["auto_escalated"] is False
    assert prov["fallback_chain_triggered"] is True
    assert "escalated_reasoning_arm" not in prov
    assert "sib" in prov["escalation_not_served"]


@pytest.mark.asyncio
async def test_a_served_rung_keeps_its_claim() -> None:
    router = _router()
    session = _escalated_session(router)
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "do it"}]}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        _payload, model_name, _reason = await router.route_chat_completion(body, session)

    assert model_name == "head"
    assert mock_ac.call_args.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert session.decision_provenance["auto_escalated"] is True
    assert session.decision_provenance["escalated_reasoning_arm"] == "think"


def _resumed_session(router: ProxyRouter) -> Any:
    """A session that INHERITED its arm from a stored row — what `_resume_locked_model` writes."""
    session = router._sessions.create_session("t")
    session.model_chosen = "head"
    session.metadata["reasoning_arm"] = "think"
    session.decision_provenance = {
        "model_chosen": "head",
        "selection_rule_used": "prefix_resume",
        "fallback_chain_triggered": False,
        "router_propensity": None,
        "auto_escalated": False,  # it inherited a rung; it did not climb one this turn
        "new_label_window": False,
        "escalated_reasoning_arm": "think",
    }
    return session


@pytest.mark.asyncio
async def test_a_fallback_on_a_resumed_session_keeps_the_inherited_arm() -> None:
    """Persisted provenance is the only restore source — popping the arm is not recoverable."""
    router = _router()
    session = _resumed_session(router)

    async def fake_ac(config: ModelConfig, **kwargs: Any) -> MagicMock:
        if config.name == "head":
            raise RuntimeError("head is down")
        return _response()

    body: dict[str, Any] = {"messages": [{"role": "user", "content": "do it"}]}
    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=fake_ac)):
        _payload, model_name, _reason = await router.route_chat_completion(body, session)

    assert model_name == "sib"
    prov = session.decision_provenance
    # The arm survives, so the next resume of this conversation still serves the escalated
    # effort instead of silently dropping to base — permanently, since nothing re-asserts it.
    assert prov["escalated_reasoning_arm"] == "think"
    # And nothing was de-claimed, because this turn claimed no escalation of its own.
    assert "escalation_not_served" not in prov
    assert prov["auto_escalated"] is False


@pytest.mark.asyncio
async def test_a_rank_rung_served_by_a_fallback_is_voided_like_an_effort_rung() -> None:
    """A rank escalation carries no arm — the rung claim must still be withdrawn."""
    router = _router()
    session = router._sessions.create_session("t")
    session.model_chosen = "head"
    session.decision_provenance = {
        "model_chosen": "head",
        "selection_rule_used": "auto_escalation",
        "pre_escalation_model": "cheap",
        "rank_escalation_reason": "same_verified_failure_x2",
        "auto_escalated": True,
        "new_label_window": True,
        "router_propensity": None,
    }

    async def fake_ac(config: ModelConfig, **kwargs: Any) -> MagicMock:
        if config.name == "head":
            raise RuntimeError("head is down")
        return _response()

    body: dict[str, Any] = {"messages": [{"role": "user", "content": "do it"}]}
    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=fake_ac)):
        _payload, model_name, _reason = await router.route_chat_completion(body, session)

    assert model_name == "sib"
    prov = session.decision_provenance
    assert prov["auto_escalated"] is False
    assert prov["fallback_chain_triggered"] is True
    assert "sib" in prov["escalation_not_served"]
