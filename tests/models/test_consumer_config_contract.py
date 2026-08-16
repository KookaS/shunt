"""Per-consumer config contract: isolation + production-completeness of used models.

Each consumer (router vs benchmark) reads only its own files, on the shared registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmark import config as benchmark_config
from shunt.models.config import ModelPool, load_registry
from shunt.router.policy import load_router_policy, packaged_policy_path


def _write_yaml(path: str, data: dict) -> str:
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


class TestConsumerConfigIsolation:
    """Each consumer reads only its own config files — SHUNT_CONFIG_DIR is router-only."""

    def test_router_consumer_ignores_a_planted_benchmark_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_yaml(
            str(tmp_path / "models.yaml"),
            {
                "providers": {
                    "planted": {
                        "base_url": "https://planted.test/v1",
                        "api_key_env_var": "PLANTED_KEY",
                        "litellm_prefix": "openai",
                    }
                },
                "models": {
                    "planted-model": {"model_id": "planted/planted-model", "provider": "planted"}
                },
            },
        )
        _write_yaml(
            str(tmp_path / "router.yaml"),
            {"router": {"strategy": "always_cheap", "models": ["planted-model"]}},
        )
        # benchmark-only-model exists ONLY here — a router that read this file would see it.
        _write_yaml(
            str(tmp_path / "benchmark.yaml"),
            {"models": ["benchmark-only-model", "planted-model"]},
        )
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
        pool = ModelPool()
        assert pool.get_model("planted-model") is not None
        assert pool.get_model("benchmark-only-model") is None
        policy = load_router_policy()
        assert policy.strategy == "always_cheap"
        assert policy.models == ["planted-model"]

    def test_benchmark_consumer_ignores_shunt_config_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = Path(benchmark_config.__file__).resolve().parent / "benchmark.yaml"
        _write_yaml(
            str(tmp_path / "router.yaml"),
            {"router": {"strategy": "always_cheap"}},
        )
        monkeypatch.setattr(benchmark_config, "_config", None)
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
        assert benchmark_config.load()["models"] == benchmark_config.load(path=repo)["models"]

    def test_consumer_config_files_are_distinct(self) -> None:
        router = packaged_policy_path()
        benchmark = Path(benchmark_config.__file__).resolve().parent / "benchmark.yaml"
        assert router.is_file()
        assert benchmark.is_file()
        assert router.name == "router.yaml"
        assert benchmark.name == "benchmark.yaml"
        assert router != benchmark


class TestProductionCompleteness:
    """The full production hyperparameter set every actually-used model must carry."""

    @staticmethod
    def _consumer_models() -> list[str]:
        router = load_router_policy(packaged_policy_path()).models
        benchmark = benchmark_config.enabled_models()
        return router + benchmark

    def test_both_consumers_name_nonempty_model_lists(self) -> None:
        assert load_router_policy(packaged_policy_path()).models
        assert benchmark_config.enabled_models()

    def test_every_used_model_is_registered_and_identified(self) -> None:
        registry = load_registry()
        for name in self._consumer_models():
            entry = registry.models.get(name)
            assert entry is not None, f"{name} absent from the registry"
            assert entry.model_id, f"{name} has an empty model_id"
            assert entry.provider, f"{name} has an empty provider"
            assert entry.version is not None, f"{name} is priced but declares no version"
            assert isinstance(entry.supports_streaming, bool), name
            assert isinstance(entry.supports_cache_control, bool), name

    def test_every_used_model_carries_full_pricing(self) -> None:
        registry = load_registry()
        for name in self._consumer_models():
            entry = registry.models.get(name)
            assert entry is not None, f"{name} absent from the registry"
            pricing = entry.pricing
            assert pricing is not None, f"{name} is routable but unpriced"
            assert pricing.input_cost_per_1m > 0, name
            assert pricing.output_cost_per_1m > 0, name
            assert pricing.cache_read_cost_per_1m is not None, f"{name} missing cache-read price"
            assert pricing.cache_read_cost_per_1m >= 0, name
            assert pricing.price_provider, f"{name} missing price_provider"
            assert pricing.price_source, f"{name} missing price_source"
            assert pricing.price_as_of, f"{name} missing price_as_of"

    def test_every_used_model_carries_full_reasoning_bracket(self) -> None:
        registry = load_registry()
        for name in self._consumer_models():
            entry = registry.models.get(name)
            assert entry is not None, f"{name} absent from the registry"
            reasoning = entry.reasoning
            assert reasoning is not None, f"{name} missing reasoning block"
            arm_ids = [arm.id for arm in reasoning.arms]
            assert reasoning.default_arm in arm_ids, f"{name} default_arm not among its arms"
            assert len(set(arm_ids)) == len(arm_ids), f"{name} has duplicate arm ids"
            assert len({arm.rank for arm in reasoning.arms}) == len(reasoning.arms), (
                f"{name} has duplicate arm ranks"
            )
            assert [arm.rank for arm in reasoning.arms] == list(range(len(reasoning.arms))), (
                f"{name} arms must be sorted by rank starting at 0"
            )
            for arm in reasoning.arms:
                assert arm.api, f"{name} arm {arm.id!r} has an empty api mapping"

    def test_both_consumer_sets_resolve(self) -> None:
        pool = ModelPool()
        pool.restrict_to_live(load_router_policy(packaged_policy_path()).models)
        benchmark_config.enabled_models()
