"""The non-shipped free-model overlay registry and its five-hazard guards.

The free-model direction is reopened for CONTAINED overlay collection: free rows live in a
non-shipped overlay (`configs/free-tier/overlay.yaml`), are collectable only through
`--extra-models`, and can never enter the shipped registry, the enabled set, the live pool,
`capability_rank`, the pareto axes or the kill gate. These tests pin each guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from benchmark import config
from benchmark.routing import validate
from benchmark.runner import run_matrix

REPO: Path = Path(__file__).resolve().parents[2]
OVERLAY: Path = REPO / "configs" / "free-tier" / "overlay.yaml"

VIABLE_PROVIDERS: Final[frozenset[str]] = frozenset(
    {
        "requesty",
        "groq",
        "openrouter",
        "google_ai_studio",
        "nvidia_nim",
        "vercel_ai_gateway",
        "sambanova",
        "cloudflare_workers_ai",
        "opencode_zen",
        "kilo_gateway",
        "together",
        "explabs",
    }
)


@pytest.fixture(autouse=True)
def _overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh caches with the overlay configured (the collection path under test)."""
    monkeypatch.setattr(config, "_pricing", None)
    monkeypatch.setattr(config, "_free_registry", None)
    monkeypatch.setattr(config, "_free_registry_path_override", str(OVERLAY))


# ── (a) the shipped registry carries no collection-only free row ──────────────────────


def test_shipped_registry_holds_no_collection_only_free_row() -> None:
    from shunt.models.config import default_registry_path, load_registry

    registry = load_registry(default_registry_path())
    # W1 (2026-09-13) ships the `explabs` PROVIDER for its AVAILABLE-only priced rows, so the
    # earlier "no explabs provider at all" half is superseded. What must still never ship is a
    # `-explabs` collection-only id, and every shipped explabs row must carry a real positive
    # list price (no $0 free row).
    assert not [name for name in registry.models if name.endswith("-explabs")]
    for name, entry in registry.models.items():
        if entry.provider != "explabs":
            continue
        assert entry.pricing is not None, name
        assert entry.pricing.input_cost_per_1m > 0, name
        assert entry.pricing.output_cost_per_1m > 0, name


def test_overlay_holds_every_viable_provider() -> None:
    from shunt.models.config import load_registry

    registry = load_registry(OVERLAY)
    assert set(registry.providers) >= VIABLE_PROVIDERS
    assert registry.models


def test_overlay_rows_record_real_list_price_and_no_cache_read() -> None:
    overlay = config.free_registry()
    assert overlay
    for name, info in overlay.items():
        assert info["input_cost_per_1m"] > 0, name
        assert info["output_cost_per_1m"] > 0, name
        assert info.get("price_note"), name
        # Provenance is a real pricing-listing URL, or the documented `catalog:<provider>`
        # scheme when the paid twin is absent from models.dev and the price came from the
        # provider's own catalogue (see benchmark/routing/README.md, "Data provenance").
        assert str(info["price_source"]).startswith(("http", "catalog:")), name
        assert info.get("version"), name
        assert "cache_read_cost_per_1m" not in info, name


# ── (b) the overlay loads for --extra-models and the wire target resolves ─────────────


def test_overlay_loads_for_extra_models_and_resolves_the_wire_target() -> None:
    name = "requesty-gemma-4-31b-it"
    assert run_matrix._extra_models(name) == [name]
    config.register_collection_models([name])
    info = config.load_pricing()[name]
    assert info["provider"] == "requesty"
    assert info["route"] == "openai/google/gemma-4-31b-it"
    assert info["base_url"] == "https://router.requesty.ai/v1"
    assert info["api_key_env_var"] == "REQUESTY_API_KEY"


def test_moved_explabs_rows_resolve_from_the_overlay() -> None:
    name = "kimi-k3-explabs"
    assert run_matrix._extra_models(name) == [name]
    config.register_collection_models([name])
    info = config.load_pricing()[name]
    assert info["provider"] == "explabs"
    assert info["route"] == "openai/kimi-k3"
    assert info["version"] == "kimi-k3"


# ── (c) with no overlay the shipped behavior is unchanged ────────────────────────────


def test_no_overlay_leaves_the_shipped_view_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_free_registry", None)
    monkeypatch.setattr(config, "_free_registry_path_override", None)
    monkeypatch.delenv(config.FREE_REGISTRY_ENV, raising=False)
    assert config.free_registry() == {}
    pricing = config.load_pricing()
    assert "kimi-k3" in pricing
    assert not [name for name in pricing if name.endswith("-explabs")]


# ── (d) a free id cannot enter the enabled set / kill gate / analysis ─────────────────


@pytest.mark.parametrize("free_id", ["kimi-k3-explabs", "requesty-gemma-4-31b-it"])
def test_free_id_cannot_be_enabled(monkeypatch: pytest.MonkeyPatch, free_id: str) -> None:
    monkeypatch.setattr(config, "_config", {"models": [free_id]})
    with pytest.raises(ValueError, match="cannot enable free/collection-only"):
        config.enabled_models()


def test_validate_flags_a_free_model_in_the_enabled_list(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models:\n  - kimi-k3-explabs\n", encoding="utf-8")
    errors = config.validate(cfg)
    assert any("collection-only" in e for e in errors), errors


def test_analysis_views_exclude_overlay_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_config", {"models": ["deepseek-v4-flash", "kimi-k3"]})
    free = config.free_registry_ids()
    assert not (set(config.enabled_models()) & free)
    assert not (set(config.capability_rank().evidence) & free)
    assert not (set(config.models_matrix({})) & free)


# ── (e) the five hazards ──────────────────────────────────────────────────────────────
# 1. cache wall: overlay rows are uncached but never enabled, so the enabled-scoped wall
#    passes (and enabling one is refused before it can be reached).
def test_cache_wall_not_tripped_by_overlay_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_config", {"models": ["deepseek-v4-flash"]})
    assert config.models_missing_cache() == []


def _row(model: str, **over: object) -> dict:
    base: dict = {
        "challenge_id": "astropy__astropy-1",
        "model": model,
        "reasoning": "default",
        "pass": "False",
        "cost": "0",
        "in_tok": "100",
        "out_tok": "50",
        "calls": "7",
        "version_hash": "vh",
        "model_version": model,
        "arm_hash": "",
        "real_cost": "0",
        "estimated_cost": "0",
        "timeout_flag": "False",
        "image_digest": "",
        "computed_at": "2026-09-10T00:00:00+00:00",
        "stop_reason": "unsolved",
        "step_limit": "",
        "cost_limit": "",
        "scaffold_version": "",
        "sampling_hash": "",
        "prompt_hash": "",
    }
    base.update(over)
    return base


# 2. validate: an overlay $0 row is free-promo provenance; a non-overlay paid $0 row still
#    trips the accounting-hole negative control.
def test_validate_accepts_overlay_zero_row_but_flags_non_overlay() -> None:
    pricing = dict(config.free_registry())
    overlay_row = _row("requesty-gemma-4-31b-it")
    overlay_codes = {v.code for v in validate.validate_row(overlay_row, pricing)}
    assert validate.ACCOUNTING_HOLE not in overlay_codes

    pricing["not-an-overlay-model"] = {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}
    non_overlay_row = _row("not-an-overlay-model")
    non_overlay_codes = {v.code for v in validate.validate_row(non_overlay_row, pricing)}
    assert validate.ACCOUNTING_HOLE in non_overlay_codes


# 3. pareto / kill gate: overlay rows are absent from every analysis view.
def test_overlay_rows_are_not_analysis_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_config", {"models": ["deepseek-v4-flash", "kimi-k3"]})
    free = config.free_registry_ids()
    assert not (set(config.enabled_pricing()) & free)
    assert not (set(config.default_arm_ids()) & free)
