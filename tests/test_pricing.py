"""Tests for the pricing slice of the unified model registry (models.yaml)."""

from typing import Final

import pytest
import yaml
from pydantic import ValidationError

from benchmark import config
from benchmark.routing import cache_cost, integrity
from shunt.models.config import ModelPool, default_registry_path, parse_registry

REQUIRED_PRICING_FIELDS = (
    "input_cost_per_1m",
    "output_cost_per_1m",
    "version",
    "price_provider",
    "price_source",
    "price_as_of",
)

_UNPRICED_REGISTRY: Final = {
    "providers": {
        "p": {
            "base_url": "https://p.ai/v1",
            "api_key_env_var": "P_KEY",
            "litellm_prefix": "openai",
        }
    },
    "models": {"cheapo": {"model_id": "p/cheapo", "provider": "p"}},
}

# COLLECTION-ONLY (2026-09-06, free-tier probe): the nine `*-explabs` ids registered for the
# Experiential Labs $0 promotional channel. They must be priced (so `--extra-models` can
# collect through them) but are deliberately NOT enabled in benchmark.yaml, NOT in router.yaml's
# live pool, and never scored — the collection/analysis machinery excludes them by construction
# (`run_matrix._extra_models` unions them at the run site only). Their list price is an
# owner-directed recovered-list-price row that DISARMS the corpus accounting checks only while
# a dated free window holds (see benchmark/routing/validate.py), and the channel reports no
# cache-read rate and no reasoning bracket. The pricing censuses in
# this file therefore exempt this documented set (same mechanism as the JUDGE/PROBE-only rows
# below): a $0 promo row must never fail a "real quote" census that exists to keep genuinely
# priced rows honest.
COLLECTION_ONLY_MODELS: Final = frozenset(
    {
        "gpt-6-astra-explabs",
        "claude-fable-5.1-explabs",
        "gpt-5.6-luna-explabs",
        "qwen3.8-27b-explabs",
        "deepseek-v4-flash-explabs",
        "kimi-k3-explabs",
        "deepseek-v4-pro-explabs",
        "glm-5.3-explabs",
        "glm-5.3-flash-explabs",
    }
)

# AVAILABLE-ONLY (2026-09-13, W1): the six bare-id models promoted into the shipped registry
# under provider `explabs`. They are registered and priced so a user MAY select one by copying
# it into router.yaml, but they are deliberately NOT in router.yaml's live pool and NOT in
# benchmark.yaml's enabled set, so they never route by default and never score. They carry no
# `cache_read_cost_per_1m` (the gateway shares a cache namespace across tenants — the free-overlay
# HARD RULE 3), no `size` and no `reasoning` bracket: those are earned by a benchmark measurement,
# not asserted. The pricing censuses below exempt this documented set exactly as they exempt the
# JUDGE/PROBE-only rows.
AVAILABLE_ONLY_MODELS: Final = frozenset(
    {
        "gpt-6-astra",
        "claude-fable-5.1",
        "gpt-5.6-luna",
        "qwen3.8-27b",
        "glm-5.3",
    }
)


def _models() -> dict:
    """The priced models, as the benchmark sees them."""
    return config.load_pricing()


def _provider_row() -> dict:
    return {"base_url": "u", "api_key_env_var": "K", "litellm_prefix": "openai"}


class TestRegistryParses:
    def test_registry_is_valid_yaml_with_models(self):
        assert len(_models()) >= 6

    def test_no_orphaned_ood176_version(self):
        for name, info in _models().items():
            assert "ood176" not in str(info.get("version", "")), name


class TestRequiredFields:
    def test_every_priced_model_has_required_fields(self):
        for name, info in _models().items():
            for field in REQUIRED_PRICING_FIELDS:
                assert field in info, f"{name} missing {field}"

    def test_canonical_prices_are_positive_numbers(self):
        for name, info in _models().items():
            if name in COLLECTION_ONLY_MODELS:
                continue
            for key in ("input_cost_per_1m", "output_cost_per_1m"):
                val = info[key]
                assert isinstance(val, (int, float)) and val > 0, f"{name}.{key}={val!r}"

    def test_price_source_is_a_url(self):
        for name, info in _models().items():
            assert str(info["price_source"]).startswith("http"), name

    def test_every_model_carries_a_dated_price_note(self):
        # Prices are real Requesty router quotes — each documents its provenance.
        for name, info in _models().items():
            assert info.get("price_note"), f"{name} missing price_note"
            assert info.get("price_as_of"), f"{name} missing price_as_of"

    def test_cache_read_prices_are_positive_when_present(self):
        for name, info in _models().items():
            for key in ("cache_read_cost_per_1m", "cache_write_cost_per_1m"):
                if key in info:
                    val = info[key]
                    assert isinstance(val, (int, float)) and val > 0, f"{name}.{key}={val!r}"


class TestOptionalPricingIsTheNoBenchmarkPath:
    """`pricing` absent ⇒ routable but unscoreable. Both halves of the feature."""

    def test_unpriced_model_is_routable(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(_UNPRICED_REGISTRY, sort_keys=False))
        pool = ModelPool(str(path))
        # Routable = present in the pool; an unpriced model is unranked (ranking would
        # fail loud) but still loads and can be reached directly / by cold-start.
        assert pool.model_names() == ["cheapo"]
        assert pool.get_model("cheapo") is not None

    def test_unpriced_model_is_invisible_to_the_benchmark(self, tmp_path, monkeypatch):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(_UNPRICED_REGISTRY, sort_keys=False))
        monkeypatch.setattr(config, "_pricing", None)
        monkeypatch.setattr(config, "_pricing_path", lambda: path)
        monkeypatch.setattr(config, "_config", {"models": []})
        assert config.load_pricing() == {}
        assert config.enabled_models() == []


class TestVersionIsModelLevel:
    """`version` is a MODEL-IDENTITY attribute, not a pricing field — it lives on
    the model row, and a genuine model change means a new registry id, not a bump."""

    def _priced_model(self, **over) -> dict:
        row = {
            "model_id": "p/m",
            "provider": "p",
            "version": "m",
            "pricing": {
                "input_cost_per_1m": 0.1,
                "output_cost_per_1m": 0.2,
                "cache_read_cost_per_1m": 0.01,
                "price_provider": "p",
                "price_source": "https://p.ai",
                "price_as_of": "2026-07-18",
            },
        }
        row.update(over)
        return {"providers": {"p": _provider_row()}, "models": {"m": row}}

    def test_model_level_version_parses_and_surfaces(self, tmp_path, monkeypatch):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(self._priced_model(), sort_keys=False))
        registry = parse_registry(yaml.safe_load(path.read_text()))
        assert registry.models["m"].version == "m"
        monkeypatch.setattr(config, "_pricing", None)
        monkeypatch.setattr(config, "_pricing_path", lambda: path)
        assert config.load_pricing()["m"]["version"] == "m"
        assert integrity.model_versions()["m"] == "m"

    def test_version_under_pricing_is_rejected(self):
        # `version` no longer belongs to the pricing block — Pricing forbids extras.
        bad = self._priced_model()
        del bad["models"]["m"]["version"]
        bad["models"]["m"]["pricing"]["version"] = "m"
        with pytest.raises(ValidationError, match="version"):
            parse_registry(bad)

    def test_priced_model_without_version_is_rejected(self):
        # A benchmarkable (priced) model must carry its identity/version.
        bad = self._priced_model()
        del bad["models"]["m"]["version"]
        with pytest.raises(ValidationError, match="version"):
            parse_registry(bad)

    def test_unpriced_model_needs_no_version(self):
        # Routable-but-unscored models (the example fragments) stay versionless.
        parse_registry(_UNPRICED_REGISTRY)  # must not raise


class TestRegistrySchemaIsEnforced:
    def test_unknown_key_is_rejected_at_load(self):
        # A stale-schema file (per-model base_url) fails loudly, naming the key.
        data = {
            "providers": {"p": _provider_row()},
            "models": {"m": {"model_id": "m", "provider": "p", "base_url": "stale"}},
        }
        with pytest.raises(ValidationError, match="base_url"):
            parse_registry(data)

    def test_dangling_provider_fk_is_rejected(self):
        data = {
            "providers": {"p": _provider_row()},
            "models": {"m": {"model_id": "m", "provider": "nope"}},
        }
        with pytest.raises(ValueError, match="unknown provider"):
            parse_registry(data)


class TestRouteDerivation:
    """`route == litellm_prefix + "/" + model_id` — pinned per the design's table."""

    EXPECTED = {
        "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
        "glm-5.2": "openai/fireworks/glm-5.2",
        "qwen3.7-plus": "openai/alibaba/qwen3.7-plus",
        "gpt-5-mini": "openai/openai/gpt-5-mini",
        "kimi-k2.5": "openai/moonshot/kimi-k2.5",
        "kimi-k3": "openai/moonshot/kimi-k3",
        "claude-opus-4-6": "openai/anthropic/claude-opus-4-6",
    }

    def test_every_model_derives_its_documented_route(self):
        pool = ModelPool()
        for name, expected in self.EXPECTED.items():
            model = pool.get_model(name)
            assert model is not None, name
            assert model.route == expected, name


class TestCachingGate:
    """The benchmark must route only through models with a real cache-read discount."""

    def test_no_uncached_qwen3_max_remains(self):
        assert "qwen3-max" not in _models()

    def test_every_registry_model_has_cache_read_pricing(self):
        for name in _models():
            if name in COLLECTION_ONLY_MODELS or name in AVAILABLE_ONLY_MODELS:
                continue
            assert config.model_has_cache(name), f"{name} has no cache-read discount"

    def test_all_enabled_models_pass_the_caching_gate(self):
        config.load("benchmark/benchmark.yaml")
        assert config.models_missing_cache() == []

    def test_model_has_cache_requires_discount_below_input(self):
        assert config.model_has_cache("gpt-5-mini") is True
        assert config.model_has_cache("nonexistent-model") is False

    def test_models_missing_cache_flags_an_uncached_model(self, monkeypatch):
        pricing = dict(config.load_pricing())
        pricing["fake-nocache"] = {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}
        monkeypatch.setattr(config, "load_pricing", lambda *a, **k: pricing)
        assert config.model_has_cache("fake-nocache") is False
        assert "fake-nocache" in config.models_missing_cache(["fake-nocache", "gpt-5-mini"])


class TestCacheDiscountAuthorityIsThePricePair:
    """`cache_cost._discount` and `config.model_has_cache` resolve the SAME authority:
    `cache_read_cost_per_1m` against `input_cost_per_1m`. `supports_cache_control` is a
    ROUTING field (whether the router may send explicit breakpoints on the live wire),
    not a cost gate — a row that declares it false still banks the automatic prefix-cache
    discount its price quotes.

    This pins the reconciliation. An earlier change made `_discount` require the flag and
    split it from `model_has_cache`, so the kill gate's control model (flag false, real
    cache-read price) was priced cache-blind while the benchmark's caching gate still
    called it cache-capable.
    """

    _REGISTRY: Final = {
        "providers": {
            "p": {
                "base_url": "https://p.ai/v1",
                "api_key_env_var": "P_KEY",
                "litellm_prefix": "openai",
            }
        },
        "models": {
            # Flag FALSE, real cache-read price: automatic caching still applies.
            "autocache": {
                "model_id": "p/autocache",
                "provider": "p",
                "version": "autocache",
                "supports_cache_control": False,
                "pricing": {
                    "input_cost_per_1m": 1.0,
                    "output_cost_per_1m": 2.0,
                    "cache_read_cost_per_1m": 0.1,
                    "price_provider": "p",
                    "price_source": "https://p.ai",
                    "price_as_of": "2026-09-01",
                },
            },
            # No cache-read price: the cost model falls back, never a silent zero.
            "noread": {
                "model_id": "p/noread",
                "provider": "p",
                "version": "noread",
                "supports_cache_control": True,
                "pricing": {
                    "input_cost_per_1m": 1.0,
                    "output_cost_per_1m": 2.0,
                    "price_provider": "p",
                    "price_source": "https://p.ai",
                    "price_as_of": "2026-09-01",
                },
            },
        },
    }

    def test_flag_false_model_with_a_cache_read_price_banks_the_discount(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(self._REGISTRY, sort_keys=False))
        monkeypatch.setattr(config, "_pricing", None)
        monkeypatch.setattr(config, "_pricing_path", lambda: path)
        # The benchmark's caching gate and the cost model agree: the price pair decides.
        assert config.model_has_cache("autocache") is True
        # `shares={}` keeps the measured-token path out of it: this test is about the price.
        prices = cache_cost.cache_prices(["autocache", "noread"], shares={})
        assert prices["autocache"].discount == pytest.approx(0.9)
        assert prices["autocache"].provenance == cache_cost.MEASURED
        assert prices["autocache"].saving_fraction > 0.0
        # No cache-read price -> the assumed fallback, not a silent zero.
        assert prices["noread"].discount == pytest.approx(cache_cost.ASSUMED_CACHE_DISCOUNT)
        assert prices["noread"].provenance == cache_cost.ASSUMED


class TestCostModelConsumesCanonicalPrice:
    def test_pricing_dict_reads_canonical_fields(self):
        pd = config._pricing_dict()
        for name, info in _models().items():
            assert pd[name]["input"] == info["input_cost_per_1m"]
            assert pd[name]["output"] == info["output_cost_per_1m"]

    def test_models_matrix_maps_canonical_to_legacy_prices(self):
        matrix = config.models_matrix()
        for name, info in _models().items():
            assert matrix[name]["input_price"] == info["input_cost_per_1m"]
            assert matrix[name]["output_price"] == info["output_cost_per_1m"]

    def test_estimated_cost_uses_canonical_price(self):
        info = _models()["deepseek-v4-flash"]
        cost = integrity.estimated_cost("deepseek-v4-flash", 1_000_000, 1_000_000)
        assert round(cost, 6) == round(info["input_cost_per_1m"] + info["output_cost_per_1m"], 6)


class TestValidatorChecksConfigReferences:
    def test_validate_passes_on_current_registry(self):
        errors = config.validate()
        assert errors == [], errors

    def test_validate_flags_a_config_model_absent_from_the_registry(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("models:\n  - ghost-model\n")
        errors = config.validate(cfg)
        assert any("ghost-model" in e for e in errors)

    def test_validate_rejects_the_legacy_enabled_dict_form(self, tmp_path):
        # The old {model: {enabled: bool}} shape is no longer accepted — a listed
        # model must be a bare name in a list.
        cfg = tmp_path / "config.yaml"
        cfg.write_text("models:\n  deepseek-v4-flash:\n    enabled: true\n")
        errors = config.validate(cfg)
        assert any("list" in e.lower() for e in errors), errors


class TestEnabledModelsList:
    """`models:` in benchmark.yaml is a LIST of enabled names; in-list = enabled,
    registry-only = disabled, listed-but-unregistered = hard error."""

    def _load(self, monkeypatch, models: list[str]) -> None:
        monkeypatch.setattr(config, "_config", {"models": models})

    def test_listed_models_are_enabled(self, monkeypatch):
        self._load(monkeypatch, ["deepseek-v4-flash", "gpt-5-mini"])
        assert set(config.enabled_models()) == {"deepseek-v4-flash", "gpt-5-mini"}

    def test_registry_model_absent_from_the_list_is_disabled(self, monkeypatch):
        self._load(monkeypatch, ["deepseek-v4-flash"])
        enabled = config.enabled_models()
        assert "gpt-5-mini" not in enabled
        assert "deepseek-v4-flash" in enabled

    def test_listed_model_absent_from_the_registry_raises(self, monkeypatch):
        self._load(monkeypatch, ["deepseek-v4-flash", "ghost-model"])
        with pytest.raises(ValueError, match="ghost-model"):
            config.enabled_models()

    def test_price_sort_is_preserved(self, monkeypatch):
        # enabled_models() sorts by total list price ascending (name tie-break).
        self._load(monkeypatch, ["kimi-k3", "deepseek-v4-flash", "gpt-5-mini", "glm-5.2"])
        ordered = config.enabled_models()
        costs = [config.cost_per_1m(m) for m in ordered]
        assert costs == sorted(costs)
        assert ordered[0] == "deepseek-v4-flash"
        assert ordered[-1] == "kimi-k3"


class TestRegistryShipsWithThePackage:
    def test_default_registry_path_exists(self):
        assert default_registry_path().exists()


# ---------------------------------------------------------------------------
# D2/D4 — reasoning-arm accessors on benchmark/config.py
# ---------------------------------------------------------------------------


class TestReasoningConfigsAccessor:
    # JUDGE-ONLY models are deliberately reasoning-free: they are registered so the
    # judge-probe harness reads their pricing, but never in router.yaml's or
    # benchmark.yaml's models list, so no reasoning bracket is required — the same
    # exemption test_config.py's TestDefaultRegistryHasReasoning applies.
    JUDGE_ONLY_MODELS: Final = frozenset({"claude-sonnet-5", "gpt-5.6-terra"})
    # PROBE-ONLY models were collected in a one-off probe with NO reasoning arm declared, so
    # their committed rows carry the legacy `reasoning="default"` placeholder. Declaring a
    # bracket now would silently re-alias those measured rows to an arm they never ran
    # (`config._alias_legacy_reasoning`), so the honest state is no bracket at all. Like
    # JUDGE_ONLY_MODELS these are absent from router.yaml and benchmark.yaml's model lists.
    PROBE_ONLY_MODELS: Final = frozenset({"glm-5.3-flash"})

    def test_every_registry_model_has_a_reasoning_block(self):
        cfgs = config.reasoning_configs()
        for name in _models():
            if (
                name in COLLECTION_ONLY_MODELS
                or name in AVAILABLE_ONLY_MODELS
                or name in self.JUDGE_ONLY_MODELS
                or name in self.PROBE_ONLY_MODELS
            ):
                continue
            assert cfgs.get(name) is not None, f"{name} missing reasoning block"

    def test_deepseek_bracket_matches_adr(self):
        cfg = config.reasoning_configs()["deepseek-v4-flash"]
        assert cfg.default_arm == "high"
        assert {a.id for a in cfg.arms} == {"nothink", "high", "max"}

    def test_kimi_k3_bracket_collapses_to_one_arm(self):
        cfg = config.reasoning_configs()["kimi-k3"]
        assert cfg.default_arm == "max"
        assert [a.id for a in cfg.arms] == ["max"]


class TestDefaultArmIds:
    def test_registry_model_resolves_its_declared_default(self):
        ids = config.default_arm_ids(["deepseek-v4-flash", "gpt-5-mini"])
        assert ids == {"deepseek-v4-flash": "high", "gpt-5-mini": "medium"}

    def test_unknown_model_falls_back_to_legacy_default_literal(self):
        assert config.default_arm_ids(["not-a-real-model"]) == {"not-a-real-model": "default"}


class TestArmSamplingWeights:
    def test_returns_declared_config_weights(self):
        config.load("benchmark/benchmark.yaml")
        weights = config.arm_sampling_weights()
        assert weights and all(0.0 <= w <= 1.0 for w in weights)
        assert weights == sorted(weights, reverse=True), "weight must decrease with rank"

    def test_falls_back_to_sane_defaults_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr(config, "_config", {})
        weights = config.arm_sampling_weights()
        assert weights == list(config.DEFAULT_ARM_SAMPLING_WEIGHTS)
