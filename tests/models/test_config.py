from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

import pytest
import yaml

from shunt.models.config import TIER_ORDER, ModelConfig, ModelPool, strict_yaml_load


def _write_yaml(path: str, data: dict) -> str:
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


DEFAULT_MODEL_NAMES: Final = [
    "qwen3.7-plus",
    "deepseek-v4-flash",
    "gpt-5-mini",
    "zai-glm-5.2",
    "kimi-k2.5",
    "kimi-k3",
    "claude-opus-4-6",
]


class TestStrictYamlLoad:
    def test_duplicate_top_level_key_is_rejected(self) -> None:
        # A copy-pasted duplicate provider/model row must fail loudly, not silently
        # shadow the earlier one (yaml.safe_load keeps last-wins).
        with pytest.raises(ValueError, match="duplicate key 'requesty'"):
            strict_yaml_load("providers:\n  requesty: {base_url: a}\n  requesty: {base_url: b}\n")

    def test_duplicate_nested_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate key 'tier'"):
            strict_yaml_load("models:\n  m:\n    tier: cheap\n    tier: mid\n")

    def test_valid_yaml_still_loads(self) -> None:
        assert strict_yaml_load("models:\n  m: {tier: cheap}\n") == {
            "models": {"m": {"tier": "cheap"}}
        }


class TestModelConfig:
    def test_minimal_config(self) -> None:
        cfg = ModelConfig(
            name="test-model",
            tier="cheap",
            provider="test",
            base_url="https://test.ai/v1",
            api_key_env_var="TEST_KEY",
        )
        assert cfg.name == "test-model"
        assert cfg.tier == "cheap"
        assert cfg.supports_streaming is True
        assert cfg.supports_cache_control is False

    def test_full_config(self) -> None:
        cfg = ModelConfig(
            name="test-model",
            tier="frontier",
            provider="test",
            base_url="https://test.ai/v1",
            api_key_env_var="TEST_KEY",
            supports_streaming=False,
            supports_cache_control=True,
        )
        assert cfg.supports_streaming is False
        assert cfg.supports_cache_control is True
        assert cfg.tier == "frontier"


class TestModelPoolLoad:
    def test_default_bundled_config(self) -> None:
        pool = ModelPool()
        for name in DEFAULT_MODEL_NAMES:
            model = pool.get_model(name)
            assert model is not None, f"Missing model {name}"
            assert model.name == name

    def test_custom_config_path(self) -> None:
        data = {
            "providers": {
                "test": {
                    "base_url": "https://test.ai/v1",
                    "api_key_env_var": "TEST_KEY",
                    "litellm_prefix": "openai",
                }
            },
            "models": {
                "test-model": {
                    "model_id": "test-model",
                    "tier": "cheap",
                    "provider": "test",
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = _write_yaml(f.name, data)

        try:
            pool = ModelPool(path)
            model = pool.get_model("test-model")
            assert model is not None
            assert model.provider == "test"
        finally:
            os.unlink(path)

    def test_env_config_dir(self) -> None:
        data = {
            "providers": {
                "env": {
                    "base_url": "https://env.ai/v1",
                    "api_key_env_var": "ENV_KEY",
                    "litellm_prefix": "openai",
                }
            },
            "models": {
                "env-model": {
                    "model_id": "env-model",
                    "tier": "mid",
                    "provider": "env",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            _write_yaml(str(config_path), data)

            old_env = os.environ.get("SHUNT_CONFIG_DIR")
            try:
                os.environ["SHUNT_CONFIG_DIR"] = tmpdir
                pool = ModelPool()
                model = pool.get_model("env-model")
                assert model is not None
                assert model.tier == "mid"
            finally:
                if old_env is not None:
                    os.environ["SHUNT_CONFIG_DIR"] = old_env
                else:
                    del os.environ["SHUNT_CONFIG_DIR"]

    def test_missing_config_file_falls_back_to_bundled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent = str(Path(tmpdir) / "nonexistent.yaml")
            pool = ModelPool(non_existent)
            assert pool.get_model("qwen3.7-plus") is not None

    def test_load_classmethod(self) -> None:
        pool = ModelPool.load()
        assert isinstance(pool, ModelPool)
        assert pool.get_model("deepseek-v4-flash") is not None


class TestTierLookup:
    def test_get_tier(self) -> None:
        pool = ModelPool()
        assert pool.get_tier("qwen3.7-plus") == "cheap"
        assert pool.get_tier("gpt-5-mini") == "mid"
        assert pool.get_tier("claude-opus-4-6") == "frontier"

    def test_get_tier_unknown_model(self) -> None:
        pool = ModelPool()
        assert pool.get_tier("nonexistent") is None

    def test_get_tier_models(self) -> None:
        pool = ModelPool()
        for tier in TIER_ORDER:
            models = pool.get_tier_models(tier)
            assert models, f"tier {tier} has no models"
            assert all(m.tier == tier for m in models)

    def test_pool_roster_is_exactly_the_declared_default(self) -> None:
        # DEFAULT_MODEL_NAMES pins the roster in BOTH directions: the per-name
        # checks catch a model that vanished, this catches one that appeared.
        # Deriving both sides from the pool would be self-referential.
        pool = ModelPool()
        assert _all_model_names(pool) == set(DEFAULT_MODEL_NAMES)

    def test_tiers_partition_the_pool(self) -> None:
        # Every model belongs to exactly one tier — no model is stranded in an
        # unknown tier, none is double-counted.
        pool = ModelPool()
        by_tier = [{m.name for m in pool.get_tier_models(t)} for t in TIER_ORDER]
        for i, left in enumerate(by_tier):
            for right in by_tier[i + 1 :]:
                assert not (left & right)


def _all_model_names(pool: ModelPool) -> set[str]:
    return {m.name for tier in TIER_ORDER for m in pool.get_tier_models(tier)}


class TestFallbackChain:
    def test_same_tier_fallback(self) -> None:
        pool = ModelPool()
        # cheap models: qwen3.7-plus, deepseek-v4-flash
        chain = pool.fallback_chain("qwen3.7-plus")
        assert chain[0] == "qwen3.7-plus"
        # deepseek-v4-flash is the other cheap model
        assert "deepseek-v4-flash" in chain[:3]
        # Exhaustive and duplicate-free, whatever the pool holds
        assert chain == list(dict.fromkeys(chain))
        assert set(chain) == _all_model_names(pool)

    def test_fallback_chain_same_tier_first(self) -> None:
        pool = ModelPool()
        chain = pool.fallback_chain("qwen3.7-plus")
        # Same tier models: qwen3.7-plus (first), deepseek-v4-flash
        cheap_models_in_chain = chain[:2]
        assert set(cheap_models_in_chain) == {"qwen3.7-plus", "deepseek-v4-flash"}

    def test_frontier_fallback(self) -> None:
        pool = ModelPool()
        chain = pool.fallback_chain("claude-opus-4-6")
        assert chain[0] == "claude-opus-4-6"
        assert chain == list(dict.fromkeys(chain))
        assert set(chain) == _all_model_names(pool)

    def test_unknown_model_returns_empty(self) -> None:
        pool = ModelPool()
        chain = pool.fallback_chain("nonexistent")
        assert chain == []


class TestHealthTracking:
    def test_all_healthy_by_default(self) -> None:
        pool = ModelPool()
        for name in DEFAULT_MODEL_NAMES:
            assert pool.is_healthy(name) is True

    def test_mark_unhealthy(self) -> None:
        pool = ModelPool()
        pool.mark_unhealthy("qwen3.7-plus")
        assert pool.is_healthy("qwen3.7-plus") is False

    def test_unknown_model_not_healthy(self) -> None:
        pool = ModelPool()
        assert pool.is_healthy("nonexistent") is False

    def test_health_check_interval_default(self) -> None:
        pool = ModelPool()
        assert pool.health_check_interval == 60

    def test_auto_recovery(self) -> None:
        pool = ModelPool()
        pool._health_check_interval = 0  # Immediate recovery
        pool.mark_unhealthy("qwen3.7-plus")
        assert pool.is_healthy("qwen3.7-plus") is True

    def test_mark_unhealthy_unknown_model_no_error(self) -> None:
        pool = ModelPool()
        pool.mark_unhealthy("nonexistent")
        assert pool.is_healthy("nonexistent") is False
