"""--extra-models: a collection-only extension of the run's model list.

The free-promo channel ids (``*-explabs``) are priced in the registry but NOT enabled.
``run_matrix --extra-models`` must union them into the run's collect set and nowhere
else: never into ``enabled_models()``/``models_matrix``/``capability_rank``, so no
analysis, baseline or the live pool can move because rows were collected through them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark import config
from benchmark.runner import run_matrix

OVERLAY: Path = Path(__file__).resolve().parents[2] / "configs" / "free-tier" / "overlay.yaml"

EXTRA: tuple[str, ...] = (
    "gpt-6-astra-explabs",
    "claude-fable-5.1-explabs",
    "gpt-5.6-luna-explabs",
    "qwen3.8-27b-explabs",
    "deepseek-v4-flash-explabs",
    "kimi-k3-explabs",
    "deepseek-v4-pro-explabs",
    "glm-5.3-explabs",
    "glm-5.3-flash-explabs",
)


@pytest.fixture(scope="module")
def _cfg():
    config.load("benchmark/benchmark.yaml")


@pytest.fixture(autouse=True)
def _isolated_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh pricing + overlay caches, with the non-shipped overlay configured.

    The free-promo ids moved out of the shipped registry into the overlay on 2026-09-10
    under the collection-only free-model policy, so the collection path is exercised with the
    overlay explicitly set.
    """
    monkeypatch.setattr(config, "_pricing", None)
    monkeypatch.setattr(config, "_free_registry", None)
    monkeypatch.setattr(config, "_free_registry_path_override", str(OVERLAY))


def test_extra_models_parse_and_validate(_cfg) -> None:
    parsed = run_matrix._extra_models("gpt-6-astra-explabs, deepseek-v4-flash-explabs")
    assert parsed == ["gpt-6-astra-explabs", "deepseek-v4-flash-explabs"]


def test_extra_models_reject_unpriced_id(_cfg) -> None:
    with pytest.raises(ValueError):
        run_matrix._extra_models("definitely-not-a-model")


def test_extra_models_are_priced_but_not_enabled(_cfg) -> None:
    overlay = config.free_registry()
    enabled = set(config.enabled_models())
    for m in EXTRA:
        assert m in overlay, "an extra model must be priced to be collectable"
        assert m not in enabled, "an extra model must never be enabled"
    # The capability order that decides baselines is unchanged.
    assert config.capability_rank().strongest() == "kimi-k3"


def test_extra_models_are_not_in_the_live_pool(_cfg) -> None:
    from shunt.router.policy import load_router_policy, packaged_policy_path

    live = set(load_router_policy(packaged_policy_path()).models)
    for m in EXTRA:
        assert m not in live, "an extra model must never be routable by the live proxy"


def test_collection_only_rows_are_the_uncached_overlay_namespace(_cfg) -> None:
    # The overlay is the priced-rows-without-cache-read namespace (the collection-only
    # free-model policy: no `cached_in_tok`/cache-read on a free row). The SHIPPED enabled
    # registry carries no uncached row, so the cache gate is not weakened by the move; pin
    # both halves.
    overlay = config.free_registry()
    assert all("cache_read_cost_per_1m" not in info for info in overlay.values())
    assert config.models_missing_cache() == []
    assert set(EXTRA) <= set(overlay)


def test_extra_models_absent_from_evaluated_matrix(_cfg) -> None:
    # matrix["models"] (enabled AND evaluated) must not include a not-yet-collected extra.
    mm = config.models_matrix({})
    for m in EXTRA:
        assert m not in mm


def test_classify_cells_marks_extra_cells_missing(_cfg) -> None:
    cache: dict = {}
    versions = {m: "v" for m in [*config.enabled_models(), *EXTRA]}
    status = run_matrix.classify_cells(
        ["astropy__astropy-12907"], [EXTRA[0]], cache, {}, versions, None
    )
    assert ("astropy__astropy-12907", EXTRA[0], "default") in status.missing


# --- the run_live_cells uncached-budget wall is ENABLED-scoped -------------------------


def test_live_wall_still_refuses_an_enabled_uncached_model(_cfg, monkeypatch) -> None:
    # The paid-ladder protection must stay exactly as strong: an ENABLED model without a
    # cache-read discount still refuses the whole run, before any cell. Synthesise one by
    # enabling a genuinely cache-less priced registry id.
    monkeypatch.setattr(config, "enabled_models", lambda: [EXTRA[0]])
    with pytest.raises(ValueError, match="refuses to run uncached enabled models"):
        run_matrix.run_live_cells(
            [("astropy__astropy-12907", EXTRA[0], "default")],
            {},
            {"astropy__astropy-12907": "h"},
            {EXTRA[0]: "v"},
            timeout=10,
            verbose=False,
        )


def _stub_cell_executor(monkeypatch, ran: list[list[tuple[str, str, str]]]) -> None:
    """Neutralise the executor behind the wall so only the wall itself is under test."""

    def _record(cells, *a, **k) -> list:
        ran.append(list(cells))
        return []

    monkeypatch.setattr(run_matrix, "_run_cells_serial", _record)


def test_live_wall_lets_an_extra_uncached_model_run(_cfg, monkeypatch) -> None:
    # An extra-model id is an EXPLICIT collection opt-in: cache-less and never enabled, it
    # must not trip the enabled-scoped wall (the free-promo channel reports no cache-read
    # rate and no cached tokens; free-ness is guarded by the campaign's real_cost == 0).
    monkeypatch.setattr(config, "enabled_models", lambda: [])
    ran: list[list[tuple[str, str, str]]] = []
    _stub_cell_executor(monkeypatch, ran)
    rows = run_matrix.run_live_cells(
        [("astropy__astropy-12907", EXTRA[0], "default")],
        {},
        {"astropy__astropy-12907": "h"},
        {EXTRA[0]: "v"},
        timeout=10,
        verbose=False,
    )
    assert rows == [] and ran == [[("astropy__astropy-12907", EXTRA[0], "default")]]


def test_live_wall_mixed_run_starts_with_cached_enabled_plus_extra(_cfg, monkeypatch) -> None:
    # Enabled ladder + extra channel together: only the ENABLED part of the cell set is
    # walled — a cached enabled model rides it fine, and the cache-less extra id next to it
    # is its own explicit opt-in, so the mixed run starts.
    monkeypatch.setattr(config, "enabled_models", lambda: ["gpt-5-mini"])
    ran: list[list[tuple[str, str, str]]] = []
    _stub_cell_executor(monkeypatch, ran)
    cells = [
        ("astropy__astropy-12907", "gpt-5-mini", "default"),
        ("astropy__astropy-12907", EXTRA[0], "default"),
    ]
    rows = run_matrix.run_live_cells(
        cells,
        {},
        {"astropy__astropy-12907": "h"},
        {"gpt-5-mini": "v", EXTRA[0]: "v"},
        timeout=10,
        verbose=False,
    )
    assert rows == [] and ran == [cells]


# ── runtime synthesis of arbitrary `S-explabs` catalog slugs ──────────────────────────
# Any catalog slug S not in models.yaml resolves as `S-explabs` (provider explabs, wire
# target `openai/S`, identity S) without a registry edit. See config.synthesize_collection_model.

SYNTH_ID = "brand-new-catalog-model-explabs"
SYNTH_SLUG = "brand-new-catalog-model"


def test_arbitrary_slug_synthesizes_a_collection_only_entry(_cfg) -> None:
    synth = config.synthesize_collection_model(SYNTH_ID)
    assert synth is not None
    assert synth["provider"] == "explabs"
    assert synth["route"] == f"openai/{SYNTH_SLUG}"
    assert synth["version"] == SYNTH_SLUG
    assert synth["serving_mode"] == "hosted"
    assert "cache_read_cost_per_1m" not in synth


def test_arbitrary_slug_resolves_through_extra_models_and_registration(_cfg) -> None:
    assert run_matrix._extra_models(SYNTH_ID) == [SYNTH_ID]
    config.register_collection_models([SYNTH_ID])
    info = config.load_pricing()[SYNTH_ID]
    assert info["route"] == f"openai/{SYNTH_SLUG}"
    assert info["version"] == SYNTH_SLUG


def test_synthesized_identity_feeds_model_versions(_cfg) -> None:
    # The registration runs before model_versions(), so the synthesized identity S
    # participates in the identity-skip exactly like a hand-registered extra.
    from benchmark.routing import integrity

    config.register_collection_models(["gpt-5-mini-explabs"])
    assert integrity.model_versions()["gpt-5-mini-explabs"] == "gpt-5-mini"


def test_unknown_pricing_does_not_fabricate_a_price(_cfg) -> None:
    # No catalog host rung for this slug ⇒ 0/0 with an explicit unknown note, never a
    # fabricated nonzero price.
    synth = config.synthesize_collection_model(SYNTH_ID)
    assert synth is not None
    assert synth["input_cost_per_1m"] == 0.0
    assert synth["output_cost_per_1m"] == 0.0
    assert synth["price_source"] == config.COLLECTION_UNKNOWN_SOURCE
    assert synth["price_note"] == config.COLLECTION_UNKNOWN_NOTE


def test_known_host_rung_is_reused(_cfg) -> None:
    # A direct twin's list price is the catalog host rung: reuse it, do not fabricate.
    direct = config.load_pricing()["gpt-5-mini"]
    synth = config.synthesize_collection_model("gpt-5-mini-explabs")
    assert synth is not None
    assert synth["input_cost_per_1m"] == direct["input_cost_per_1m"]
    assert synth["output_cost_per_1m"] == direct["output_cost_per_1m"]
    assert synth["price_source"] == direct["price_source"]


def test_existing_overlay_id_is_unchanged(_cfg) -> None:
    before = dict(config.free_registry()["kimi-k3-explabs"])
    assert config.synthesize_collection_model("kimi-k3-explabs") is None
    config.register_collection_models(["kimi-k3-explabs"])
    assert config.load_pricing()["kimi-k3-explabs"] == before


def test_non_explabs_id_is_not_synthesizable(_cfg) -> None:
    assert config.collection_slug("gpt-5-mini") is None
    assert config.synthesize_collection_model("gpt-5-mini") is None
    with pytest.raises(ValueError):
        run_matrix._extra_models(SYNTH_SLUG)
