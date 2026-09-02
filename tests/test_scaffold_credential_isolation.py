"""The live scaffold's model config must never carry a provider credential."""

# mini-swe-agent persists the model config verbatim into the trajectory dump
# (``LitellmModel.serialize()`` → ``self.config.model_dump(mode="json")`` → the JSON
# ``DefaultAgent.save`` writes to ``agent.output_path``). Anything in ``model_kwargs`` is therefore
# written to disk UNREDACTED, so an ``api_key`` passed there lands in plaintext in every
# message-list dump a live run produces. These tests assert the shipped path keeps the credential
# out of everything that gets serialised, and prove the assertion discriminates by showing the
# naive shape leaking through the same serialiser.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from benchmark import config
from benchmark.runner import infer, scaffold_model

# Assembled from halves so no literal key-shaped string lives in this file (gitleaks scans it).
_PLANTED_KEY = "rq-" + "K" * 40

# Same self-identifying shapes tests/test_step_snapshots.py scans dumps for.
_SECRET_SHAPE = re.compile(r"\brq-[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9]{20,}")


class _FakeEnv:
    """The minimum an agent needs to be constructed and serialised (no container)."""

    def serialize(self) -> dict[str, Any]:
        return {"info": {"config": {"environment": {"cwd": "/testbed"}}}}

    def execute(self, command: str) -> dict[str, Any]:
        return {"output": "", "returncode": 0}


def _hosted_model() -> str:
    """A registry model routed through a generic ``openai/`` surface (key passed explicitly)."""
    for name, info in config.load_pricing().items():
        if isinstance(info, dict) and str(info.get("route", "")).startswith("openai/"):
            return str(name)
    pytest.skip("no openai/-routed model in the registry")


def _overlay(model: str) -> dict[str, Any]:
    model_string, model_kwargs = infer.litellm_model_target(model)
    return infer._scaffold_config_overlay(
        model_string,
        model_kwargs,
        timeout=60,
        step_limit=3,
        cost_limit=0.01,
        trajectory_id="inst__model__default",
        api_key_env_var=infer._model_key_env_var(model),
    )


@pytest.fixture
def planted_env(monkeypatch: pytest.MonkeyPatch) -> str:
    model = _hosted_model()
    key_env = str(config.load_pricing()[model]["api_key_env_var"])
    monkeypatch.setenv(key_env, _PLANTED_KEY)
    return model


def test_the_live_call_kwargs_still_carry_the_key(planted_env: str) -> None:
    """Control: ``litellm_model_target`` is the LIVE-call path and must still authenticate."""
    _, model_kwargs = infer.litellm_model_target(planted_env)
    assert model_kwargs["api_key"] == _PLANTED_KEY


def test_scaffold_overlay_carries_no_credential(planted_env: str) -> None:
    blob = json.dumps(_overlay(planted_env))
    assert _PLANTED_KEY not in blob
    assert not _SECRET_SHAPE.search(blob)


def test_scaffold_overlay_names_the_env_var_instead(planted_env: str) -> None:
    """The credential is replaced by its env-var NAME, resolved at request time."""
    model_block = _overlay(planted_env)["model"]
    assert model_block["api_key_env_var"] == str(
        config.load_pricing()[planted_env]["api_key_env_var"]
    )
    assert "api_key" not in model_block["model_kwargs"]


def test_the_serialised_trajectory_config_carries_no_credential(planted_env: str) -> None:
    """End-to-end through mini-swe-agent's OWN serialiser — the exact writer that leaked."""
    from minisweagent.models import get_model

    model_obj = get_model(config=dict(_overlay(planted_env)["model"]))
    blob = json.dumps(model_obj.serialize())
    assert _PLANTED_KEY not in blob
    assert not _SECRET_SHAPE.search(blob)


def test_positive_control_the_naive_shape_does_leak(planted_env: str) -> None:
    """Negative control for the assertions above: without the fix the same serialiser leaks.

    This is the defect as it shipped — ``api_key`` inside ``model_kwargs`` on the stock
    ``LitellmModel``. If this ever stops leaking, the tests above have gone vacuous.
    """
    from minisweagent.models import get_model

    model_string, model_kwargs = infer.litellm_model_target(planted_env)
    leaky = get_model(
        config={
            "model_name": model_string,
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
            "model_class": "litellm",
        }
    )
    blob = json.dumps(leaky.serialize())
    assert _PLANTED_KEY in blob
    assert _SECRET_SHAPE.search(blob)


def test_the_key_is_injected_at_request_time(planted_env: str, monkeypatch) -> None:
    """The credential must still reach the provider — it moves, it is not dropped."""
    import litellm
    from minisweagent.models import get_model

    seen: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError("stop after capturing the request")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    model_obj = get_model(config=dict(_overlay(planted_env)["model"]))
    with pytest.raises(RuntimeError):
        model_obj._query([{"role": "user", "content": "hi"}])
    assert seen["api_key"] == _PLANTED_KEY


def test_a_credential_in_any_other_kwarg_is_refused(planted_env: str) -> None:
    """The wall: a NEW auth kwarg added later cannot silently reach the dump either."""
    with pytest.raises(scaffold_model.CredentialInScaffoldConfigError):
        scaffold_model.credential_free_model_block(
            "openai/x",
            {"api_base": "https://x", "extra_headers": {"Authorization": f"Bearer {_PLANTED_KEY}"}},
            api_key_env_var="REQUESTY_API_KEY",
        )


def test_the_wall_catches_a_credential_this_process_holds_but_cannot_shape_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider key with no self-identifying prefix is still caught, by value."""
    opaque = "Zq7" + "n" * 30
    monkeypatch.setenv("SOMEPROVIDER_API_KEY", opaque)
    with pytest.raises(scaffold_model.CredentialInScaffoldConfigError):
        scaffold_model.credential_free_model_block(
            "openai/x", {"api_base": "https://x", "auth": opaque}, api_key_env_var=None
        )


def test_the_wall_passes_a_clean_config() -> None:
    block = scaffold_model.credential_free_model_block(
        "openai/x", {"api_base": "https://x", "max_tokens": 128}, api_key_env_var="REQUESTY_API_KEY"
    )
    assert block["model_kwargs"] == {"api_base": "https://x", "max_tokens": 128}
    assert block["model_class"] == scaffold_model.MODEL_CLASS_PATH


def test_stripping_a_key_without_naming_its_env_var_is_refused() -> None:
    """Stripping the credential must never silently produce an unauthenticated run."""
    with pytest.raises(scaffold_model.CredentialInScaffoldConfigError):
        scaffold_model.credential_free_model_block(
            "openai/x", {"api_base": "https://x", "api_key": "whatever"}, api_key_env_var=None
        )


def test_the_written_dump_file_carries_no_credential(planted_env: str, tmp_path: Path) -> None:
    """The regression test proper: drive mini-swe-agent's OWN writer and read the bytes back."""

    # ``DefaultAgent.save`` is the exact call that produced the 42 plaintext occurrences across
    # 20 transcripts. Asserting on the file it writes, rather than on an intermediate dict, is
    # what makes this test insensitive to where in the scaffold the config is assembled.
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.utils.serialize import recursive_merge

    default_config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
    merged = recursive_merge(default_config, _overlay(planted_env))
    agent_config = {k: v for k, v in merged["agent"].items() if k != "output_path"}
    agent = DefaultAgent(get_model(config=dict(merged["model"])), _FakeEnv(), **agent_config)

    dump = tmp_path / "trajectory.json"
    agent.save(dump)
    written = dump.read_text(encoding="utf-8")
    assert _PLANTED_KEY not in written
    assert not _SECRET_SHAPE.search(written)
    # ...and it really is our credential-free model class doing the serialising.
    assert json.loads(written)["info"]["config"]["model_type"] == scaffold_model.MODEL_CLASS_PATH


def test_the_merged_config_is_rechecked_after_recursive_merge(planted_env: str) -> None:
    """`recursive_merge` merges the scaffold's own defaults UNDER the overlay, key by key."""

    # A credential in mini-swe-agent's shipped `swebench.yaml` `model_kwargs` would therefore
    # survive into the config the scaffold serialises even though the overlay was clean. The
    # post-merge check is what closes that; this proves it fires.
    from minisweagent.utils.serialize import recursive_merge

    poisoned = recursive_merge(
        {"model": {"model_kwargs": {"api_key": _PLANTED_KEY}}}, _overlay(planted_env)
    )
    with pytest.raises(scaffold_model.CredentialInScaffoldConfigError):
        scaffold_model.assert_credential_free(poisoned["model"], what="merged scaffold config")
