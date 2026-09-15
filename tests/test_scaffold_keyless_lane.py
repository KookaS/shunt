"""A keyless `key_optional` provider is admitted on the live path without a credential.

The scaffold reads its credential from the environment per request. For a provider that answers
a keyless request with 200 (Kilo Gateway's `:free` ids) a missing env var must NOT abort the run:
a harmless non-secret placeholder is injected instead. A provider that genuinely requires a key
keeps the fail-closed refusal, and nothing credential-shaped ever reaches the serialised config.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from benchmark import config
from benchmark.runner import infer, scaffold_model


class _Spec:
    instance_id = "psf__requests-1142"


def _keyless_row() -> dict[str, Any]:
    return {
        "provider": "kilo_gateway",
        "route": "openai/poolside/laguna-xs-2.1:free",
        "base_url": "https://api.kilo.ai/api/gateway",
        "api_key_env_var": "KILO_API_KEY",
        "key_optional": True,
    }


def _key_required_row() -> dict[str, Any]:
    return {
        "provider": "groq",
        "route": "openai/openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env_var": "GROQ_API_KEY",
        "key_optional": False,
    }


def _keyless_model() -> scaffold_model.EnvKeyLitellmModel:
    return scaffold_model.EnvKeyLitellmModel(
        model_name="openai/poolside/laguna-xs-2.1:free",
        api_key_env_var="KILO_API_KEY",
        key_optional=True,
    )


def _credential_free_block(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key_env_var": "KILO_API_KEY",
        "key_optional": True,
    }
    kwargs.update(overrides)
    return scaffold_model.credential_free_model_block(
        "openai/poolside/laguna-xs-2.1:free",
        {"api_base": "https://api.kilo.ai/api/gateway"},
        **kwargs,
    )


# --- the scaffold credential seam --------------------------------------------


def test_key_optional_missing_key_returns_the_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    assert _keyless_model()._credential_kwargs() == {
        scaffold_model.CREDENTIAL_KWARG: scaffold_model.ANONYMOUS_API_KEY
    }


def test_key_optional_missing_key_sends_the_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The placeholder must reach the OUTGOING request, not merely be computed."""
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    seen: dict[str, Any] = {}

    def fake_query(self: Any, messages: Any, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "response"

    monkeypatch.setattr(scaffold_model.LitellmModel, "_query", fake_query)
    _keyless_model()._query([{"role": "user", "content": "hi"}])  # type: ignore[arg-type]
    assert seen[scaffold_model.CREDENTIAL_KWARG] == scaffold_model.ANONYMOUS_API_KEY


def test_key_optional_prefers_a_real_key_when_one_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KILO_API_KEY", "real-key")
    assert _keyless_model()._credential_kwargs() == {scaffold_model.CREDENTIAL_KWARG: "real-key"}


def test_non_key_optional_missing_key_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    model = scaffold_model.EnvKeyLitellmModel(
        model_name="openai/openai/gpt-oss-120b",
        api_key_env_var="GROQ_API_KEY",
        key_optional=False,
    )
    with pytest.raises(scaffold_model.MissingScaffoldCredentialError):
        model._credential_kwargs()


# --- the serialised config stays credential-free -----------------------------


def test_the_key_optional_block_carries_the_flag_but_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    block = _credential_free_block()
    assert block["key_optional"] is True
    assert block["api_key_env_var"] == "KILO_API_KEY"
    assert "api_key" not in block["model_kwargs"]
    blob = json.dumps(block)
    assert scaffold_model.ANONYMOUS_API_KEY not in blob
    # The wall runs inside `credential_free_model_block`; pin it explicitly too.
    scaffold_model.assert_credential_free(block)


def test_the_serialised_model_config_carries_neither_key_nor_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through mini-swe-agent's OWN serialiser — the exact writer that leaked."""
    from minisweagent.models import get_model

    monkeypatch.delenv("KILO_API_KEY", raising=False)
    model_obj = get_model(config=dict(_credential_free_block()))
    blob = json.dumps(model_obj.serialize())
    assert scaffold_model.ANONYMOUS_API_KEY not in blob
    assert "KILO_API_KEY" in blob  # ...only the NAME is persisted
    assert model_obj._credential_kwargs() == {
        scaffold_model.CREDENTIAL_KWARG: scaffold_model.ANONYMOUS_API_KEY
    }


# --- the live path admits a keyless lane -------------------------------------


def test_litellm_model_target_keyless_lane_omits_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load_pricing", lambda *a, **k: {"m": _keyless_row()})
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    route, kwargs = infer.litellm_model_target("m")
    assert route == "openai/poolside/laguna-xs-2.1:free"
    assert kwargs == {"api_base": "https://api.kilo.ai/api/gateway"}


def test_litellm_model_target_key_required_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load_pricing", lambda *a, **k: {"m": _key_required_row()})
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(infer.MissingApiKeysError):
        infer.litellm_model_target("m")


def test_the_overlay_threads_key_optional_into_the_model_block() -> None:
    overlay = infer._scaffold_config_overlay(
        "openai/poolside/laguna-xs-2.1:free",
        {"api_base": "https://api.kilo.ai/api/gateway"},
        timeout=60,
        step_limit=3,
        cost_limit=0.01,
        trajectory_id="inst__m__default",
        api_key_env_var="KILO_API_KEY",
        key_optional=True,
    )
    assert overlay["model"]["key_optional"] is True
    assert overlay["model"]["api_key_env_var"] == "KILO_API_KEY"


def test_generate_patch_live_admits_a_keyless_key_optional_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load_pricing", lambda *a, **k: {"m": _keyless_row()})
    captured: dict[str, str] = {}

    def fake_invoke(spec: Any, model: str, scaffold: str, arm: str, **kwargs: Any) -> Any:
        captured["model"] = model
        return infer.AgentPatch(patch="", in_tok=0, out_tok=0, calls=0, cost=0.0)

    monkeypatch.setattr(infer, "_invoke_scaffold", fake_invoke)
    infer.generate_patch_live(_Spec(), "m", env={})  # type: ignore[arg-type]
    assert captured["model"] == "m"


def test_generate_patch_live_still_refuses_a_key_required_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load_pricing", lambda *a, **k: {"m": _key_required_row()})
    with pytest.raises(infer.MissingApiKeysError):
        infer.generate_patch_live(_Spec(), "m", env={})  # type: ignore[arg-type]
