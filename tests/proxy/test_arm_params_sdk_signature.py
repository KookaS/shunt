"""Structural: every reasoning arm the SHIPPED registry declares must assemble into kwargs
the REAL installed OpenAI SDK accepts."""

# Why this test exists: the rest of the proxy suite mocks `_acompletion`, so a mock accepts any
# kwarg and 3900 green tests coexisted with a router that raised
# `AsyncCompletions.create() got an unexpected keyword argument 'thinking'` on the FIRST rung of
# the escalation ladder, for every thinking-capable model on the wire. The mock is the blind
# spot; `inspect.signature` on the installed SDK method is the thing that cannot be mocked.

from __future__ import annotations

import inspect

import pytest
from openai.resources.chat.completions import AsyncCompletions

from shunt.models.config import ModelPool, arm_api_params, default_registry_path
from shunt.proxy.router import (
    _FORWARDED_OPENAI_KEYS,
    ProxyRouter,
    sdk_completion_params,
    split_api_params,
)
from shunt.session import Session, SessionManager


def _shipped_pool() -> ModelPool:
    return ModelPool(str(default_registry_path()))


def _arm_cases() -> list[tuple[str, str]]:
    pool = _shipped_pool()
    cases: list[tuple[str, str]] = []
    for name in pool.model_names():
        config = pool.get_model(name)
        if config is None or config.reasoning is None:
            continue
        cases.extend((name, arm.id) for arm in config.reasoning.arms)
    return cases


def _bind(outbound: dict[str, object]) -> None:
    """Bind *outbound* to the SDK's real signature — TypeError if a kwarg is not accepted."""
    # `bind_partial` on the unbound method reproduces exactly the failure the wire produced:
    # the SDK's parameter list is explicit (no **kwargs), so an unknown name cannot bind.
    inspect.signature(AsyncCompletions.create).bind_partial(
        None, model="m", messages=[], **outbound
    )


def test_the_registry_has_arms_to_check() -> None:
    # Guards the vacuous pass: a registry that lost its reasoning blocks would make every
    # parametrised case below disappear and the file would still report green.
    cases = _arm_cases()
    assert len(cases) >= 10, f"expected the shipped registry to declare arms, got {cases}"


@pytest.mark.parametrize(("model", "arm"), _arm_cases())
def test_every_shipped_arm_assembles_into_acceptable_sdk_kwargs(model: str, arm: str) -> None:
    pool = _shipped_pool()
    router = ProxyRouter(model_pool=pool, session_manager=SessionManager())
    session: Session = router._sessions.create_session("signature-test")
    session.metadata["reasoning_arm"] = arm

    outbound: dict[str, object] = {"stream": False}
    router._apply_reasoning_arm(outbound, session, model)

    _bind(outbound)
    # Nothing provider-specific may travel as a kwarg: the split is an allowlist, so a param the
    # SDK does not name belongs in extra_body, which the SDK forwards into the JSON body.
    declared = set(arm_api_params(pool.get_model(model), arm))  # type: ignore[arg-type]
    native, extra = split_api_params(arm_api_params(pool.get_model(model), arm))  # type: ignore[arg-type]
    assert declared == set(native) | set(extra)
    assert set(native) <= sdk_completion_params()
    for key in extra:
        assert key in (outbound.get("extra_body") or {}), f"{key} was dropped, not routed"


def test_a_provider_specific_param_is_routed_to_extra_body() -> None:
    native, extra = split_api_params({"reasoning_effort": "high", "thinking": {"type": "enabled"}})
    assert native == {"reasoning_effort": "high"}
    assert extra == {"thinking": {"type": "enabled"}}


def test_the_router_is_the_sole_author_of_extra_body() -> None:
    """The arm's provider params land whole, because nothing else can put an extra_body there."""
    # This replaces a test that seeded `{"extra_body": {...}}` by hand and asserted the merge
    # branch: the router cannot build that state. `_route` starts the outbound dict from
    # `messages`/`stream` and copies only `_FORWARDED_OPENAI_KEYS` off the client body, and
    # `extra_body` is not one of them — so `_apply_reasoning_arm` always writes onto a clean
    # dict. Pinning the allowlist is what makes that a checked fact rather than a reading, and
    # is what would fail if the merge branch ever stopped being defensive.
    assert "extra_body" not in _FORWARDED_OPENAI_KEYS

    pool = _shipped_pool()
    router = ProxyRouter(model_pool=pool, session_manager=SessionManager())
    session = router._sessions.create_session("signature-test")
    session.metadata["reasoning_arm"] = "think"
    outbound: dict[str, object] = {"messages": [], "stream": False}  # the shape `_route` builds
    router._apply_reasoning_arm(outbound, session, "qwen3.7-plus")
    body = outbound["extra_body"]
    assert isinstance(body, dict)
    assert body["enable_thinking"] is True


def test_an_arm_that_sets_a_reserved_request_key_is_refused() -> None:
    """A registry `api:` blob may not overwrite the request itself — it is free-form text."""
    with pytest.raises(ValueError, match="reserved request key"):
        split_api_params({"messages": [{"role": "user", "content": "hijacked"}], "stream": True})
    # A provider param that merely SHARES the shape stays fine — the refusal is key-exact.
    native, extra = split_api_params({"reasoning_effort": "high"})
    assert native == {"reasoning_effort": "high"}
    assert extra == {}


def test_no_shipped_arm_sets_a_reserved_request_key() -> None:
    pool = _shipped_pool()
    for model in pool.model_names():
        config = pool.get_model(model)
        if config is None or config.reasoning is None:
            continue
        for arm in config.reasoning.arms:
            split_api_params(arm_api_params(config, arm.id))  # raises if one ever does


def test_the_allowlist_is_read_from_the_installed_sdk() -> None:
    # Pins the derivation, not a hardcoded list: `reasoning_effort` is a genuine SDK kwarg and
    # `thinking` / `enable_thinking` are not, and that must stay a fact about the SDK.
    allowed = sdk_completion_params()
    assert "reasoning_effort" in allowed
    assert "extra_body" in allowed
    assert "thinking" not in allowed
    assert "enable_thinking" not in allowed
