from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

import pydantic
import pytest
import yaml

from shunt.models.config import (
    UNDISCLOSED,
    ModelConfig,
    ModelEntry,
    ModelPool,
    ReasoningArm,
    ReasoningConfig,
    Size,
    arm_api_params,
    load_registry,
    resolve_models,
    strict_yaml_load,
)


def _write_yaml(path: str, data: dict) -> str:
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


DEFAULT_MODEL_NAMES: Final = [
    "qwen3.7-plus",
    "deepseek-v4-flash",
    # Measured 2026-09-03/04 and promoted to the live pool (SH015 triage KEEP on both strata).
    "deepseek-v4-pro",
    "gpt-5-mini",
    "zai-glm-5.2",
    "kimi-k2.5",
    "kimi-k3",
    # Frontier escalation tail added 2026-07-19 (router-only; benchmark-disabled).
    "gemini-3.1-pro",
    "gpt-5.6-sol",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-6",
    # JUDGE-ONLY (2026-08-25): never served by the router, never in the benchmark's
    # models list — registered so the judge-probe harness reads its pricing from the
    # same registry as the served models.
    "claude-sonnet-5",
    # JUDGE-ONLY (2026-08-26): the cheap OpenAI-5.6 difficulty judge — adopted as the
    # knn_difficulty label source over the claude-sonnet-5 anchor (see
    # benchmark/routing/data/judge_difficulty.json).
    "gpt-5.6-terra",
    # PROBE-ONLY (2026-08-26): the revealed identity of the retired stealth/ox-alpha.
    # Never served by the router, never in the benchmark's models list — registered so the
    # 41 cells collected during OpenRouter's $0 free window keep a priced registry row.
    "zai-glm-5.3-flash",
]


class TestStrictYamlLoad:
    def test_duplicate_top_level_key_is_rejected(self) -> None:
        # A copy-pasted duplicate provider/model row must fail loudly, not silently
        # shadow the earlier one (yaml.safe_load keeps last-wins).
        with pytest.raises(ValueError, match="duplicate key 'requesty'"):
            strict_yaml_load("providers:\n  requesty: {base_url: a}\n  requesty: {base_url: b}\n")

    def test_duplicate_nested_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate key 'provider'"):
            strict_yaml_load("models:\n  m:\n    provider: a\n    provider: b\n")

    def test_valid_yaml_still_loads(self) -> None:
        assert strict_yaml_load("models:\n  m: {provider: p}\n") == {
            "models": {"m": {"provider": "p"}}
        }


class TestModelConfig:
    def test_minimal_config(self) -> None:
        cfg = ModelConfig(
            name="test-model",
            provider="test",
            base_url="https://test.ai/v1",
            api_key_env_var="TEST_KEY",
        )
        assert cfg.name == "test-model"
        assert cfg.supports_streaming is True
        assert cfg.supports_cache_control is False

    def test_full_config(self) -> None:
        cfg = ModelConfig(
            name="test-model",
            provider="test",
            base_url="https://test.ai/v1",
            api_key_env_var="TEST_KEY",
            supports_streaming=False,
            supports_cache_control=True,
        )
        assert cfg.supports_streaming is False
        assert cfg.supports_cache_control is True


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
                assert model.provider == "env"
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


class TestCapabilityRank:
    """Price-implied capability rank (replaces the hand-assigned tiers)."""

    def test_ranked_models_ascending_by_total_price(self) -> None:
        # live models weakest -> strongest by input+output price, name tie-break.
        pool = ModelPool()
        pool.restrict_to_live(["kimi-k3", "qwen3.7-plus", "gpt-5-mini", "deepseek-v4-flash"])
        assert [m.name for m in pool.ranked_models()] == [
            "deepseek-v4-flash",  # 0.42
            "qwen3.7-plus",  # 1.60
            "gpt-5-mini",  # 2.25
            "kimi-k3",  # 18.0
        ]

    def test_rank_of_is_the_index(self) -> None:
        pool = ModelPool()
        pool.restrict_to_live(["kimi-k3", "deepseek-v4-flash"])
        assert pool.rank_of("deepseek-v4-flash") == 0
        assert pool.rank_of("kimi-k3") == 1
        assert pool.rank_of("nonexistent") is None

    def test_models_from_rank_is_the_tail(self) -> None:
        pool = ModelPool()
        pool.restrict_to_live(["kimi-k3", "deepseek-v4-flash", "qwen3.7-plus"])
        assert [m.name for m in pool.models_from_rank(1)] == ["qwen3.7-plus", "kimi-k3"]
        assert pool.models_from_rank(99) == []

    def test_name_tiebreak_on_equal_price(self) -> None:
        # claude-opus-4-6 and claude-opus-4-8 both total 30.0 → name tie-break.
        pool = ModelPool()
        pool.restrict_to_live(["claude-opus-4-8", "claude-opus-4-6"])
        assert [m.name for m in pool.ranked_models()] == [
            "claude-opus-4-6",
            "claude-opus-4-8",
        ]

    def test_unpriced_live_model_fails_loud(self) -> None:
        # a routable model without pricing is a config error, named (fail loud).
        data = {
            "providers": {
                "p": {
                    "base_url": "https://x/v1",
                    "api_key_env_var": "X",
                    "litellm_prefix": "openai",
                }
            },
            "models": {"unpriced": {"model_id": "u", "provider": "p"}},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = _write_yaml(f.name, data)
        try:
            pool = ModelPool(path)
            with pytest.raises(ValueError, match="unpriced"):
                pool.ranked_models()
        finally:
            os.unlink(path)

    def test_pool_roster_is_exactly_the_declared_default(self) -> None:
        pool = ModelPool()
        assert set(pool.model_names()) == set(DEFAULT_MODEL_NAMES)


class TestRestrictToLive:
    """`models:` in router.yaml — live-routing subset, separate from the benchmark's."""

    def test_empty_list_is_a_noop(self) -> None:
        pool = ModelPool()
        before = pool.model_names()
        pool.restrict_to_live([])
        assert pool.model_names() == before

    def test_filters_to_named_models_preserving_registry_order(self) -> None:
        pool = ModelPool()
        # Pass names out of registry order; the pool must not reorder by it.
        pool.restrict_to_live(["kimi-k3", "qwen3.7-plus", "gpt-5-mini"])
        assert pool.model_names() == ["qwen3.7-plus", "gpt-5-mini", "kimi-k3"]

    def test_ranked_models_reflects_the_subset(self) -> None:
        pool = ModelPool()
        pool.restrict_to_live(["kimi-k3", "qwen3.7-plus"])
        assert [m.name for m in pool.ranked_models()] == ["qwen3.7-plus", "kimi-k3"]

    def test_fallback_chain_only_contains_live_models(self) -> None:
        pool = ModelPool()
        pool.restrict_to_live(["qwen3.7-plus", "kimi-k3"])
        chain = pool.fallback_chain("qwen3.7-plus")
        assert set(chain) == {"qwen3.7-plus", "kimi-k3"}

    def test_health_entries_for_removed_models_are_dropped(self) -> None:
        pool = ModelPool()
        pool.restrict_to_live(["qwen3.7-plus"])
        assert pool.is_healthy("qwen3.7-plus") is True
        assert pool.is_healthy("kimi-k3") is False

    def test_unknown_model_name_raises_naming_the_offender(self) -> None:
        pool = ModelPool()
        with pytest.raises(ValueError, match="nonexistent-model"):
            pool.restrict_to_live(["qwen3.7-plus", "nonexistent-model"])

    def test_unknown_model_error_names_all_offenders(self) -> None:
        pool = ModelPool()
        with pytest.raises(ValueError) as exc_info:
            pool.restrict_to_live(["bogus-a", "bogus-b"])
        assert "bogus-a" in str(exc_info.value)
        assert "bogus-b" in str(exc_info.value)


class TestFallbackChain:
    def test_self_first_then_rank_neighbours(self) -> None:
        pool = ModelPool()
        # qwen3.7-plus (1.60) sits mid-registry; its nearest neighbours by total price are
        # deepseek-v4-pro (1.305) then gpt-5-mini (2.25), with zai-glm-5.3-flash (0.65) and
        # deepseek-v4-flash (0.42) further out. This ranks the whole registry, probe- and
        # judge-only rows included — the server restricts to the policy's live list before
        # serving (`proxy/server.py`), which is what keeps a probe-only row out of real routing.
        chain = pool.fallback_chain("qwen3.7-plus")
        assert chain[0] == "qwen3.7-plus"
        assert set(chain[:3]) == {"qwen3.7-plus", "deepseek-v4-pro", "gpt-5-mini"}
        # Exhaustive and duplicate-free, whatever the pool holds.
        assert chain == list(dict.fromkeys(chain))
        assert set(chain) == set(pool.model_names())

    def test_frontier_fallback(self) -> None:
        pool = ModelPool()
        chain = pool.fallback_chain("claude-opus-4-6")
        assert chain[0] == "claude-opus-4-6"
        assert chain == list(dict.fromkeys(chain))
        assert set(chain) == set(pool.model_names())

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


# ---------------------------------------------------------------------------
# ReasoningArm / ReasoningConfig schema
# ---------------------------------------------------------------------------


def _arm(id_: str, rank: int, **api: object) -> ReasoningArm:
    return ReasoningArm(id=id_, rank=rank, api=dict(api))


class TestReasoningConfigSchema:
    def test_valid_reasoning_block_parses(self) -> None:
        cfg = ReasoningConfig(
            default_arm="high",
            arms=[_arm("none", 0, enable_thinking=False), _arm("high", 1, enable_thinking=True)],
        )
        assert cfg.default_arm == "high"
        assert [a.id for a in cfg.arms] == ["none", "high"]

    def test_unknown_default_arm_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="default_arm"):
            ReasoningConfig(default_arm="nope", arms=[_arm("none", 0), _arm("high", 1)])

    def test_duplicate_arm_id_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="duplicate"):
            ReasoningConfig(default_arm="high", arms=[_arm("high", 0), _arm("high", 1)])

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ReasoningConfig.model_validate(
                {"default_arm": "high", "arms": [], "bogus": 1},
            )

    def test_absent_reasoning_is_none_on_model_entry(self) -> None:
        entry = ModelEntry(model_id="m", provider="p")
        assert entry.reasoning is None

    def test_model_entry_accepts_reasoning_block(self) -> None:
        cfg = ReasoningConfig(default_arm="high", arms=[_arm("none", 0), _arm("high", 1)])
        entry = ModelEntry(model_id="m", provider="p", reasoning=cfg)
        assert entry.reasoning is not None
        assert entry.reasoning.default_arm == "high"

    def test_model_config_mirrors_reasoning(self) -> None:
        cfg = ReasoningConfig(default_arm="high", arms=[_arm("none", 0), _arm("high", 1)])
        mc = ModelConfig(
            name="m",
            provider="p",
            base_url="https://x/v1",
            api_key_env_var="X",
            reasoning=cfg,
        )
        assert mc.reasoning is not None and mc.reasoning.default_arm == "high"

    def test_absent_reasoning_is_none_on_model_config(self) -> None:
        mc = ModelConfig(name="m", provider="p", base_url="https://x/v1", api_key_env_var="X")
        assert mc.reasoning is None


class TestDefaultRegistryHasReasoning:
    # JUDGE_ONLY_MODELS are deliberately exempt: they are registered so the judge-probe
    # harness reads their pricing from the registry, but they are never in router.yaml's
    # or benchmark.yaml's models list, so no reasoning bracket (a routing/benchmark-arm
    # concern) is required — see the JUDGE-ONLY comment in src/shunt/config/models.yaml.
    # NOTE: gpt-5.6-sol is deliberately NOT here — it is a served router model
    # (router.yaml `models:` list), so it keeps its reasoning bracket.
    JUDGE_ONLY_MODELS: Final = frozenset({"claude-sonnet-5", "gpt-5.6-terra"})
    # PROBE-ONLY rows are exempt for a different reason than the judge-only ones: their
    # committed result rows carry the legacy `reasoning="default"` placeholder, and declaring
    # a bracket now would re-alias those measured rows to an arm they never ran (see
    # benchmark/config.py `_alias_legacy_reasoning`). Also absent from router.yaml/benchmark.yaml.
    PROBE_ONLY_MODELS: Final = frozenset({"zai-glm-5.3-flash"})

    def test_every_default_model_declares_a_reasoning_block(self) -> None:
        pool = ModelPool()
        for name in DEFAULT_MODEL_NAMES:
            model = pool.get_model(name)
            assert model is not None
            if name in self.JUDGE_ONLY_MODELS or name in self.PROBE_ONLY_MODELS:
                assert model.reasoning is None, f"{name} must stay reasoning-free"
                continue
            assert model.reasoning is not None, f"{name} missing reasoning block"
            assert model.reasoning.default_arm in {a.id for a in model.reasoning.arms}

    def test_shipped_enabled_models_keep_effort_headroom_for_the_cache_safe_rung(self) -> None:
        # The effort-escalation ladder (`effort_then_rank`) steps the SAME model's reasoning arm
        # first precisely because that rung is cache-safe — but the rung only exists when the
        # shipped default is NOT already the top arm. A config edit that moves a multi-arm model's
        # default to its ceiling silently demotes `effort_then_rank` to `rank_only` on that model,
        # trading the cache-safe step for a cache-breaking one. This pins the shipped registry
        # against that: every ENABLED model with >=2 arms must default below its top arm.
        # Single-arm models (kimi-k3, claude-fable-5) have no effort ladder at all and are
        # exempt — they step rank directly, which the escalation docs state.
        from shunt.router.policy import load_router_policy, packaged_policy_path

        enabled = load_router_policy(packaged_policy_path()).models
        assert enabled, "the shipped router.yaml must name its live models"
        pool = ModelPool()
        headroomless: list[str] = []
        for name in enabled:
            model = pool.get_model(name)
            if model is None or model.reasoning is None or len(model.reasoning.arms) < 2:
                continue
            top = max(model.reasoning.arms, key=lambda a: a.rank).id
            if model.reasoning.default_arm == top:
                headroomless.append(name)
        assert not headroomless, (
            f"multi-arm enabled model(s) default at their top arm, so the cache-safe effort "
            f"rung never fires on them and effort_then_rank degrades to rank_only: "
            f"{sorted(headroomless)}"
        )


# ---------------------------------------------------------------------------
# D2 — arm_api_params resolver (the EXTRACT seam: benchmark + prod router)
# ---------------------------------------------------------------------------


class TestArmApiParams:
    def _model(self) -> ModelConfig:
        cfg = ReasoningConfig(
            default_arm="high",
            arms=[
                _arm("none", 0, enable_thinking=False),
                _arm("high", 1, enable_thinking=True),
            ],
        )
        return ModelConfig(
            name="m",
            provider="p",
            base_url="https://x/v1",
            api_key_env_var="X",
            reasoning=cfg,
        )

    def test_resolves_each_declared_arm(self) -> None:
        model = self._model()
        assert arm_api_params(model, "none") == {"enable_thinking": False}
        assert arm_api_params(model, "high") == {"enable_thinking": True}

    def test_unknown_arm_id_raises(self) -> None:
        model = self._model()
        with pytest.raises(ValueError, match="unknown reasoning arm"):
            arm_api_params(model, "max")

    def test_none_reasoning_returns_empty_for_default(self) -> None:
        mc = ModelConfig(name="m", provider="p", base_url="https://x/v1", api_key_env_var="X")
        assert arm_api_params(mc, "default") == {}

    def test_none_reasoning_raises_for_non_default_arm(self) -> None:
        mc = ModelConfig(name="m", provider="p", base_url="https://x/v1", api_key_env_var="X")
        with pytest.raises(ValueError, match="unknown reasoning arm"):
            arm_api_params(mc, "high")


class TestSizeMetadata:
    """The size axis is data with provenance, or it is not committed."""

    def test_moe_model_declares_active_below_total(self) -> None:
        size = Size(
            total_params=284_000_000_000,
            active_params=13_000_000_000,
            size_source="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash",
            size_as_of="2026-08-29",
        )
        assert size.active_params < size.total_params

    def test_active_above_total_is_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="exceeds"):
            Size(
                total_params=8_000_000_000,
                active_params=9_000_000_000,
                size_source="https://example.invalid",
                size_as_of="2026-08-29",
            )

    def test_zero_is_not_a_parameter_count(self) -> None:
        # A blank column is MISSING, never zero — the same rule the benchmark applies to
        # measured columns. `UNDISCLOSED` is the way to say "no vendor figure".
        with pytest.raises(pydantic.ValidationError, match="positive"):
            Size(
                total_params=0,
                active_params=0,
                size_source="https://example.invalid",
                size_as_of="2026-08-29",
            )

    def test_undisclosed_is_a_first_class_value(self) -> None:
        size = Size(
            total_params=UNDISCLOSED,
            active_params=UNDISCLOSED,
            size_source="https://platform.openai.com/docs/models",
            size_as_of="2026-08-29",
            size_note="OpenAI publishes no parameter count for any GPT-5 tier.",
        )
        assert size.total_params == "UNDISCLOSED"

    def test_a_bare_estimate_string_is_not_accepted(self) -> None:
        # Only the exact literal passes, so "~30B" or "probably 8B" cannot reach the figures.
        with pytest.raises(pydantic.ValidationError):
            Size(
                total_params="~30B",  # type: ignore[arg-type]
                active_params=UNDISCLOSED,
                size_source="https://example.invalid",
                size_as_of="2026-08-29",
            )

    def test_unsourced_size_is_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            Size(total_params=8_000_000_000, active_params=8_000_000_000)  # type: ignore[call-arg]

    def test_serving_mode_defaults_to_hosted(self) -> None:
        entry = ModelEntry(model_id="x", provider="p")
        assert entry.serving_mode == "hosted"

    def test_serving_mode_rejects_an_unknown_value(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ModelEntry(model_id="x", provider="p", serving_mode="on-prem")  # type: ignore[arg-type]

    def test_every_shipped_model_carries_sourced_size_and_serving_mode(self) -> None:
        # The registry is the figure's only size source; a row without provenance would put an
        # unsourced number on a published axis.
        registry = load_registry()
        for name, model in resolve_models(registry).items():
            assert model.size is not None, f"{name} declares no size"
            assert model.size.size_source.startswith("http"), name
            assert model.size.size_as_of, name
            assert model.serving_mode in ("hosted", "local"), name
