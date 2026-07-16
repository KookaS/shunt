"""Tests for the first-party canonical pricing registry (data/models.json)."""

import json
from pathlib import Path

from benchmark import config
from benchmark.routing import integrity

PRICING_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark" / "routing" / "data" / "models.json"
)

REQUIRED_FIELDS = (
    "provider",
    "family",
    "tier",
    "input_cost_per_1m",
    "output_cost_per_1m",
    "price_provider",
    "price_source",
    "price_as_of",
    "access_via",
    "version",
)


def _load_raw() -> dict:
    return json.loads(PRICING_PATH.read_text())


def _models() -> dict:
    return {k: v for k, v in _load_raw().items() if isinstance(v, dict) and not k.startswith("_")}


class TestPricingParses:
    def test_file_is_valid_json_with_models(self):
        models = _models()
        assert len(models) >= 6

    def test_no_orphaned_ood176_version(self):
        for name, info in _models().items():
            assert "ood176" not in str(info.get("version", "")), name


class TestRequiredFields:
    def test_every_model_has_required_fields(self):
        for name, info in _models().items():
            for field in REQUIRED_FIELDS:
                assert field in info, f"{name} missing {field}"

    def test_canonical_prices_are_positive_numbers(self):
        for name, info in _models().items():
            for key in ("input_cost_per_1m", "output_cost_per_1m"):
                val = info[key]
                assert isinstance(val, (int, float)) and val > 0, f"{name}.{key}={val!r}"

    def test_access_via_is_valid(self):
        for name, info in _models().items():
            assert info["access_via"] in ("direct", "requesty"), name

    def test_price_source_is_a_url(self):
        for name, info in _models().items():
            assert str(info["price_source"]).startswith("http"), name

    def test_every_model_carries_a_dated_price_note(self):
        # Prices are real Requesty router quotes (2026-07-15) — each must document
        # its provenance with a non-empty price_note and a date.
        for name, info in _models().items():
            assert info.get("price_note"), f"{name} missing price_note"
            assert info.get("price_as_of"), f"{name} missing price_as_of"

    def test_cache_read_prices_are_positive_when_present(self):
        # Optional cache-pricing fields, when listed, must be positive numbers.
        for name, info in _models().items():
            for key in ("cache_read_cost_per_1m", "cache_write_cost_per_1m"):
                if key in info:
                    val = info[key]
                    assert isinstance(val, (int, float)) and val > 0, f"{name}.{key}={val!r}"

    def test_classification_metadata_is_well_formed(self):
        # Requesty-style classification: context/max-output are positive ints;
        # capabilities is a non-empty list of known lowercase tags.
        known = {"vision", "tools", "cache", "think", "web", "json"}
        for name, info in _models().items():
            for key in ("context_length", "max_output_tokens"):
                val = info[key]
                assert isinstance(val, int) and val > 0, f"{name}.{key}={val!r}"
            caps = info["capabilities"]
            assert isinstance(caps, list) and caps, f"{name}.capabilities empty"
            assert set(caps) <= known, f"{name} unknown capability: {set(caps) - known}"


class TestCachingGate:
    """The benchmark must route only through models with a real cache-read discount."""

    def test_no_uncached_qwen3_max_remains(self):
        # qwen3-max (no cache-read price) was removed; nothing should reintroduce it.
        assert "qwen3-max" not in _models()

    def test_every_registry_model_has_cache_read_pricing(self):
        # Every model shipped in the pool must have a real cache-read discount.
        for name in _models():
            assert config.model_has_cache(name), f"{name} has no cache-read discount"

    def test_all_enabled_models_pass_the_caching_gate(self):
        config.load("benchmark/config.yaml")
        assert config.models_missing_cache() == []

    def test_model_has_cache_requires_discount_below_input(self):
        # A cache-read price >= input (or absent/zero) is NOT a real discount.
        assert config.model_has_cache("gpt-5-mini") is True
        assert config.model_has_cache("nonexistent-model") is False

    def test_models_missing_cache_flags_an_uncached_model(self, monkeypatch):
        # Inject a model with supports_caching semantics but no cache-read price.
        pricing = dict(config.load_pricing())
        pricing["fake-nocache"] = {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}
        monkeypatch.setattr(config, "load_pricing", lambda *a, **k: pricing)
        assert config.model_has_cache("fake-nocache") is False
        assert "fake-nocache" in config.models_missing_cache(["fake-nocache", "gpt-5-mini"])


class TestCostModelConsumesCanonicalPrice:
    def test_pricing_dict_reads_canonical_fields(self):
        pd = config._pricing_dict()
        raw = _models()
        for name, info in raw.items():
            assert pd[name]["input"] == info["input_cost_per_1m"]
            assert pd[name]["output"] == info["output_cost_per_1m"]

    def test_models_matrix_maps_canonical_to_legacy_prices(self):
        matrix = config.models_matrix()
        raw = _models()
        for name, info in raw.items():
            assert matrix[name]["input_price"] == info["input_cost_per_1m"]
            assert matrix[name]["output_price"] == info["output_cost_per_1m"]

    def test_estimated_cost_uses_canonical_price(self):
        # 1M input + 1M output tokens should equal input+output canonical price.
        info = _models()["deepseek-v4-flash"]
        cost = integrity.estimated_cost("deepseek-v4-flash", 1_000_000, 1_000_000)
        assert round(cost, 6) == round(info["input_cost_per_1m"] + info["output_cost_per_1m"], 6)


class TestValidatorRequiresProvenance:
    def test_validate_passes_on_current_pricing(self):
        errors = config.validate()
        assert errors == [], errors

    def test_validate_flags_missing_provenance(self, tmp_path, monkeypatch):
        bad = {"m": {"tier": "cheap", "input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}}
        p = tmp_path / "models.json"
        p.write_text(json.dumps(bad))
        monkeypatch.setattr(config, "_pricing", None)
        monkeypatch.setattr(config, "_pricing_path", lambda: p)
        errors = config.validate()
        assert any("price_provider" in e for e in errors)
        assert any("access_via" in e for e in errors)
