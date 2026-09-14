"""The free-catalog scanner: facts only, per-channel isolation, and a shape-regression guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark.routing.scripts import refresh_price_sheet
from benchmark.routing.scripts import scan_free_models as scan

FIXTURES: Path = Path(__file__).resolve().parent / "data" / "free_catalogs"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def _listing(provider: str, listing_id: str, **over: Any) -> scan.Listing:
    base: dict[str, Any] = {
        "listing_id": listing_id,
        "provider": provider,
        "prices": None,
        "hugging_face_id": "",
        "canonical_slug": "",
        "context_length": None,
        "supports_tools": False,
        "expiration_date": None,
    }
    base.update(over)
    return scan.Listing(**base)


# ── normalisers ───────────────────────────────────────────────────────────────────────


class TestOpenAiShape:
    def test_free_zero_price_rows_are_read_with_facts(self) -> None:
        listings = scan.normalise_openai(
            fixture("openrouter.json"), "openrouter", {"free_filter": {"kind": "zero_price"}}
        )
        assert {item.listing_id for item in listings} == {
            "nex-agi/nex-n2.5-mini:free",
            "dots-studio/dots-3-note-preview:free",
        }
        first = next(item for item in listings if item.listing_id.startswith("nex"))
        assert first.context_length == 262144
        assert first.supports_tools is True
        assert first.hugging_face_id == "nex-agi/Nex-N2.5-mini"
        assert first.canonical_slug == "nex-agi/nex-n2.5-mini-20260908"
        assert first.prices == {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}
        preview = next(item for item in listings if item.listing_id.startswith("dots"))
        assert preview.expiration_date == "2026-09-30"
        assert preview.supports_tools is False

    def test_a_paid_listing_is_not_free(self) -> None:
        listings = scan.normalise_openai(
            fixture("openrouter.json"), "openrouter", {"free_filter": {"kind": "zero_price"}}
        )
        assert "openai/gpt-5-mini" not in {item.listing_id for item in listings}

    def test_requesty_per_token_prices_convert_to_per_1m(self) -> None:
        listings = scan.normalise_openai(
            fixture("requesty.json"), "requesty", {"free_filter": {"kind": "zero_price"}}
        )
        assert {item.listing_id for item in listings} == {
            "poolside/laguna-m.1",
            "nvidia/nemotron-3-super-120b-a12b",
        }
        super_120b = next(item for item in listings if "super" in item.listing_id)
        assert super_120b.context_length == 1048576
        assert super_120b.supports_tools is True

    def test_free_filters_are_data_driven(self) -> None:
        free = {"id": "x-free", "tags": ["free"]}
        zero = {"id": "x", "pricing": {"prompt": "0", "completion": "0"}}
        assert scan._passes_filter(free, "x-free", None, {"kind": "all"})
        assert scan._passes_filter(free, "x-free", None, {"kind": "id_suffix", "value": "-free"})
        assert scan._passes_filter(free, "x-free", None, {"kind": "tag", "value": "free"})
        assert scan._passes_filter(zero, "x", {"a": 0.0}, {"kind": "zero_price"})
        assert not scan._passes_filter(free, "x", None, {"kind": "id_suffix", "value": ":free"})

    def test_vercel_tag_free_selects_the_confirmed_free_chat_lanes(self) -> None:
        # The live Vercel catalogue marks free by tag, so the filter — not the `-free` suffix —
        # is what admits these. Both confirmed lanes must pass.
        payload = {
            "data": [
                {
                    "id": "inclusionai/ling-3.0-flash-fin-free",
                    "tags": ["free", "reasoning", "tool-use"],
                },
                {
                    "id": "inclusionai/ling-3.0-flash-sante-free",
                    "tags": ["free", "reasoning", "tool-use"],
                },
                {"id": "poolside/laguna-s-2.1", "tags": ["reasoning", "tool-use"]},
            ]
        }
        listings = scan.normalise_openai(
            payload, "vercel_ai_gateway", {"free_filter": {"kind": "tag", "value": "free"}}
        )
        assert {item.listing_id for item in listings} == {
            "inclusionai/ling-3.0-flash-fin-free",
            "inclusionai/ling-3.0-flash-sante-free",
        }


class TestNewShapes:
    def test_google_ai_studio_keeps_generate_content_and_drops_the_rest(self) -> None:
        listings = scan.normalise_google_ai_studio(
            fixture("google_ai_studio.json"), "google_ai_studio", {"free_filter": {"kind": "all"}}
        )
        by_id = {item.listing_id: item for item in listings}
        assert set(by_id) == {"gemini-flash-latest"}
        assert by_id["gemini-flash-latest"].context_length == 1048576
        assert by_id["gemini-flash-latest"].supports_tools is True
        assert by_id["gemini-flash-latest"].prices is None
        # embedContent (embedding-001) and generateAnswer (aqa) are not chat-capable.
        assert "embedding-001" not in by_id
        assert "aqa" not in by_id

    def test_cloudflare_reads_the_name_slug_and_drops_non_text_generation(self) -> None:
        listings = scan.normalise_cloudflare(
            fixture("cloudflare_workers_ai.json"),
            "cloudflare_workers_ai",
            {"free_filter": {"kind": "all"}},
        )
        assert {item.listing_id for item in listings} == {
            "@cf/openai/gpt-oss-120b",
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        }
        # `listing_id` is the `name` slug, never the opaque UUID the `id` field carries.
        assert all(item.listing_id.startswith("@cf/") for item in listings)
        assert all(item.supports_tools for item in listings)
        assert all(item.prices is None for item in listings)
        # The bge-m3 embeddings entry is dropped.
        assert "@cf/baai/bge-m3" not in {item.listing_id for item in listings}

    def test_cloudflare_drops_a_guard_listing_despite_a_text_generation_task(self) -> None:
        # `@cf/meta/llama-guard-3-8b` declares task "Text Generation" but is a moderation guard,
        # not a chat model. The cloudflare shape must apply the same `_is_text_chat_entry`
        # exclusion as the OpenAI/Google shapes, or it enters the snapshot/overlay/free registry.
        payload = {
            "result": [
                {
                    "id": "uuid-guard",
                    "name": "@cf/meta/llama-guard-3-8b",
                    "task": {"name": "Text Generation"},
                },
                {
                    "id": "uuid-chat",
                    "name": "@cf/openai/gpt-oss-120b",
                    "task": {"name": "Text Generation"},
                },
            ]
        }
        listings = scan.normalise_cloudflare(payload, "cloudflare_workers_ai", {})
        assert {item.listing_id for item in listings} == {"@cf/openai/gpt-oss-120b"}

    def test_google_ai_studio_drops_non_chat_modalities_and_keeps_chat(self) -> None:
        listings = scan.normalise_google_ai_studio(
            fixture("google_ai_studio_non_chat.json"),
            "google_ai_studio",
            {"free_filter": {"kind": "all"}},
        )
        assert {item.listing_id for item in listings} == {"gemini-flash-latest"}

    def test_openai_shape_drops_tts_and_image_ids_but_keeps_chat(self) -> None:
        payload = {
            "data": [
                {"id": "org/chat-model:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "org/chat-model-tts:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "org/image-gen:free", "pricing": {"prompt": "0", "completion": "0"}},
                {
                    "id": "org/compat-model:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "outputModalities": ["audio"],
                },
            ]
        }
        listings = scan.normalise_openai(
            payload, "openrouter", {"free_filter": {"kind": "zero_price"}}
        )
        assert {item.listing_id for item in listings} == {"org/chat-model:free"}

    def test_chat_filter_can_yield_empty_without_raising(self) -> None:
        cloudflare = {
            "result": [
                {"id": "uuid", "name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}}
            ]
        }
        assert scan.normalise_cloudflare(cloudflare, "cloudflare_workers_ai", {}) == []
        google = {
            "models": [{"name": "models/aqa", "supportedGenerationMethods": ["generateAnswer"]}]
        }
        assert scan.normalise_google_ai_studio(google, "google_ai_studio", {}) == []


class TestPromotionsShape:
    def test_only_free_promos_are_read_and_facts_are_recorded(self) -> None:
        listings = scan.normalise_promotions(
            fixture("experientiallabs.json"), "explabs", {"free_filter": {"kind": "promotion_free"}}
        )
        by_id = {item.listing_id: item for item in listings}
        # The paid 50%-off promo (`free: false`) is not a free listing.
        assert "qwen3.8-27b" not in by_id
        # A card-gated free deal is still FREE: the card requirement is a recorded fact.
        assert by_id["deepseek-v4-flash"].requires_payment_method is True
        assert by_id["deepseek-v4-flash"].per_org_cap_micro_usd == 5000000
        assert by_id["gpt-5.6-luna"].requires_payment_method is False
        assert by_id["gpt-5.6-luna"].expiration_date == "2026-09-30"
        # Every slug a promotion declares is a listing; slugs are read, never hardcoded.
        assert {"free-a", "free-b"} <= set(by_id)
        assert all(item.prices is None for item in listings)

    def test_the_filter_kind_is_data_driven(self) -> None:
        entry = {"free": True}
        paid = {"free": False}
        assert scan._passes_filter(entry, "m", None, {"kind": "promotion_free"})
        assert not scan._passes_filter(paid, "m", None, {"kind": "promotion_free"})

    def test_an_entry_with_no_slugs_is_skipped_without_raising(self) -> None:
        payload = {"promotions": [{"free": True, "slugs": []}, {"free": True}, "nope"]}
        spec = {"free_filter": {"kind": "promotion_free"}}
        assert scan.normalise_promotions(payload, "explabs", spec) == []

    def test_promotions_silent_on_tools_are_unknown_never_false(self) -> None:
        # The umbrella declares no tool parameters: that is UNKNOWN (the live probe gates),
        # not an explicit no-tools verdict that would refuse the lane statically.
        listings = scan.normalise_promotions(
            fixture("experientiallabs.json"), "explabs", {"free_filter": {"kind": "promotion_free"}}
        )
        assert listings
        assert all(item.supports_tools is None for item in listings)


class TestToolDeclaration:
    def test_a_parameter_list_without_tools_is_a_declared_no(self) -> None:
        assert scan._declares_tools({"supported_parameters": ["temperature"]}) is False

    def test_a_catalogue_silent_on_tools_is_unknown_not_false(self) -> None:
        assert scan._declares_tools({}) is None
        assert scan._declares_tools({"context_length": 100}) is None
        assert scan._declares_tools({"supported_parameters": []}) is None

    def test_explicit_declarations_are_preserved(self) -> None:
        assert scan._declares_tools({"supports_tool_calling": False}) is False
        assert scan._declares_tools({"supports_tool_calling": True}) is True
        assert scan._declares_tools({"supported_parameters": ["tools"]}) is True


class TestAdmissionToolGate:
    def test_an_explicit_no_tools_listing_cannot_enter_the_overlay(self) -> None:
        assert scan._is_admitted({"supports_tools": False}) is False

    def test_unknown_and_declared_tool_surfaces_are_admitted(self) -> None:
        assert scan._is_admitted({"supports_tools": None}) is True
        assert scan._is_admitted({"supports_tools": True}) is True
        # A snapshot row that predates the field is UNKNOWN, never silently refused.
        assert scan._is_admitted({}) is True


class TestMangledPayloads:
    @pytest.mark.parametrize(
        "payload",
        [None, [], {}, {"data": "not-a-list"}, {"models": 7}, {"result": "nope"}, 42, "text"],
    )
    def test_every_shape_yields_empty_and_never_raises(self, payload: Any) -> None:
        assert (
            scan.normalise(
                payload, "p", {"catalog_shape": "openai", "free_filter": {"kind": "all"}}
            )
            == []
        )
        assert scan.normalise(payload, "p", {"catalog_shape": "google_ai_studio"}) == []
        assert scan.normalise(payload, "p", {"catalog_shape": "cloudflare"}) == []
        assert scan.normalise(payload, "p", {"catalog_shape": "promotions"}) == []
        assert scan.normalise(payload, "p", {"catalog_shape": "unknown"}) == []

    def test_the_committed_mangled_fixture_is_empty(self) -> None:
        payload = fixture("mangled.json")
        assert scan.normalise(payload, "p", {"catalog_shape": "openai"}) == []

    def test_the_committed_mangled_promotions_fixture_is_empty(self) -> None:
        payload = fixture("mangled_promotions.json")
        assert scan.normalise(payload, "p", {"catalog_shape": "promotions"}) == []


# ── metadata source: models.dev (release_date / limit / open_weights / price) ──────────


def _modelsdev_spec() -> dict[str, Any]:
    return {
        "catalog_url": "https://models.dev/api.json",
        "provider_keys": {"openrouter": "openrouter", "groq": "groq"},
        "listing_aliases": {},
    }


class TestModelsdevMetadata:
    def test_index_reads_metadata_and_drops_the_cache_read_rate(self) -> None:
        index = scan.modelsdev_index(fixture("modelsdev.json"))
        entry = index[("openrouter", "google/gemma-4-31b-it:free")]
        assert entry["release_date"] == "2026-04-02"
        assert entry["open_weights"] is True
        assert entry["limit"] == {"context": 262144, "output": 32768}
        # HARD RULE 3: the cache_read rate models.dev publishes never enters the snapshot.
        assert entry["cost"] == {"input": 0.09, "output": 0.34}

    def test_join_is_scoped_to_one_publisher_and_keyed_by_the_listing_id(self) -> None:
        index = scan.modelsdev_index(fixture("modelsdev.json"))
        listings = scan.attach_metadata(
            {"openrouter": [_listing("openrouter", "google/gemma-4-31b-it:free")]},
            index,
            _modelsdev_spec(),
        )
        attached = listings["openrouter"][0]
        assert attached.release_date == "2026-04-02"
        assert attached.open_weights is True

    def test_a_publisher_with_no_declared_key_gets_no_metadata_not_a_fuzzy_match(self) -> None:
        index = scan.modelsdev_index(fixture("modelsdev.json"))
        listings = scan.attach_metadata(
            {"vercel_ai_gateway": [_listing("vercel_ai_gateway", "google/gemma-4-31b-it:free")]},
            index,
            _modelsdev_spec(),
        )
        assert listings["vercel_ai_gateway"][0].release_date is None

    def test_an_unmatched_listing_keeps_null_never_a_fabricated_date(self) -> None:
        index = scan.modelsdev_index(fixture("modelsdev.json"))
        listings = scan.attach_metadata(
            {"groq": [_listing("groq", "not-in-modelsdev")]}, index, _modelsdev_spec()
        )
        assert listings["groq"][0].release_date is None

    def test_a_curated_identity_modelsdev_pointer_is_the_alias_path(self) -> None:
        index = scan.modelsdev_index(fixture("modelsdev.json"))
        identity = scan.IdentityMap(
            entries={
                "gemma-4-31b-it": scan.IdentityEntry(
                    slug="gemma-4-31b-it",
                    hugging_face_id="google/gemma-4-31B-it",
                    aliases={"vercel_ai_gateway": ("google/gemma-4-31b-it:free",)},
                    deny=frozenset(),
                    modelsdev_provider="openrouter",
                    modelsdev_id="google/gemma-4-31b-it:free",
                )
            }
        )
        spec = {**_modelsdev_spec(), "provider_keys": {"vercel_ai_gateway": "vercel"}}
        listings = scan.attach_metadata(
            {"vercel_ai_gateway": [_listing("vercel_ai_gateway", "google/gemma-4-31b-it:free")]},
            index,
            spec,
            identity,
        )
        assert listings["vercel_ai_gateway"][0].release_date == "2026-04-02"

    @pytest.mark.parametrize("payload", [None, [], {}, {"openrouter": {"models": 7}}, 42, "text"])
    def test_a_mangled_payload_indexes_empty_and_never_raises(self, payload: Any) -> None:
        assert scan.modelsdev_index(payload) == {}

    def test_an_unreachable_source_is_named_and_leaves_the_index_empty(self) -> None:
        index, status = scan.scan_metadata_sources(
            {"catalog_url": "http://127.0.0.1:1/modelsdev.json"}
        )
        assert index == {}
        assert status["status"].startswith("UNREACHABLE")

    def test_committed_catalogs_declare_the_modelsdev_source(self) -> None:
        document = yaml.safe_load(scan.CATALOGS_PATH.read_text())
        source = document["metadata_sources"]["modelsdev"]
        assert source["catalog_url"] == "https://models.dev/api.json"
        assert source["provider_keys"]["openrouter"] == "openrouter"


def test_huggingface_price_parser_skips_free_rows_pinned() -> None:
    """The scanner must NOT reuse `_huggingface_prices`: it drops the free provider.

    This is the plan's critical negative test. `_huggingface_prices` exists to quote a
    *price* and treats a free tier as "not a price this sheet may quote"; the free scanner
    wants exactly what it discards, so it has its own path. If this assertion ever flips,
    that function has changed meaning and the scanner's decision must be revisited.
    """
    prices = refresh_price_sheet._huggingface_prices(fixture("huggingface.json"))
    assert prices["moonshotai/Kimi-K3"] == (3.0, 15.0)


# ── snapshot assembly ─────────────────────────────────────────────────────────────────


class TestShapeRegression:
    def test_a_channel_that_fell_to_zero_is_flagged(self) -> None:
        assert scan.shape_regressions({"a": 5, "b": 0}, {"a": 0, "b": 0}) == ["a"]
        assert scan.shape_regressions({"a": 0}, {"a": 0}) == []
        assert scan.shape_regressions({}, {"a": 0}) == []

    def test_guard_shape_blocks_apply_and_passes_a_clean_snapshot(self) -> None:
        blocked = {"shape_regressions": ["openrouter"]}
        with pytest.raises(scan.ShapeRegressionError, match="openrouter"):
            scan.guard_shape(blocked)
        scan.guard_shape({"shape_regressions": []})


class TestMerge:
    def test_first_seen_carries_forward_and_absent_listings_withdraw(self) -> None:
        previous = [
            {
                "provider": "p",
                "listing_id": "m",
                "first_seen": "2026-01-01",
                "last_seen": "2026-01-01",
                "withdrawn_at": None,
            },
            {
                "provider": "p",
                "listing_id": "gone",
                "first_seen": "2026-01-01",
                "last_seen": "2026-01-01",
                "withdrawn_at": None,
            },
        ]
        rows = scan.merge_listings(
            {"p": [_listing("p", "m")]}, previous, {"p": {"free_tier": {"rpd": 1}}}, "2026-09-10"
        )
        by_key = {(row["provider"], row["listing_id"]): row for row in rows}
        assert by_key[("p", "m")]["first_seen"] == "2026-01-01"
        assert by_key[("p", "m")]["last_seen"] == "2026-09-10"
        assert by_key[("p", "m")]["withdrawn_at"] is None
        assert by_key[("p", "m")]["published_limits"] == {"rpd": 1}
        assert by_key[("p", "gone")]["withdrawn_at"] == "2026-09-10"
        assert by_key[("p", "gone")]["first_seen"] == "2026-01-01"

    def test_a_fresh_listing_is_first_seen_today(self) -> None:
        rows = scan.merge_listings({"p": [_listing("p", "new")]}, [], {}, "2026-09-10")
        assert rows[0]["first_seen"] == "2026-09-10"
        assert rows[0]["last_seen"] == "2026-09-10"


class TestFreeAccessSchedulability:
    def test_free_access_false_listings_are_recorded_but_not_schedulable(self) -> None:
        specs = {
            "opencode_zen": {
                "free_tier": {"free_access": False, "access_note": "app/session-gated"}
            }
        }
        rows = scan.merge_listings(
            {"opencode_zen": [_listing("opencode_zen", "x-free")]}, [], specs, "2026-09-11"
        )
        row = rows[0]
        # Recorded for provenance, never dropped...
        assert row["listing_id"] == "x-free"
        # ...but marked non-schedulable with the named reason, so the campaign never tries it.
        assert row["schedulable"] is False
        assert "app/session-gated" in row["schedulable_reason"]

    def test_free_access_defaults_true_and_is_schedulable(self) -> None:
        specs = {"groq": {"free_tier": {"rpd": 1}}}
        rows = scan.merge_listings({"groq": [_listing("groq", "m")]}, [], specs, "2026-09-11")
        assert rows[0]["schedulable"] is True
        assert rows[0]["schedulable_reason"] is None

    def test_a_missing_reason_falls_back_to_a_named_provider_message(self) -> None:
        specs = {"p": {"free_tier": {"free_access": False}}}
        rows = scan.merge_listings({"p": [_listing("p", "m")]}, [], specs, "2026-09-11")
        assert "free_access" in rows[0]["schedulable_reason"]


def test_an_unreachable_channel_carries_its_rows_forward_unchanged() -> None:
    """A transient catalogue failure must withdraw NOTHING.

    Regression: an UNREACHABLE channel (HTML/timeout/non-JSON) returned an empty listing list,
    so every previous row was stamped `withdrawn_at` — a single network blip silently quiesced
    a whole provider. The previous rows must survive a failed scan untouched.
    """
    previous = [
        {
            "provider": "p",
            "listing_id": "m",
            "first_seen": "2026-01-01",
            "last_seen": "2026-01-01",
            "withdrawn_at": None,
        }
    ]
    channels = {"p": {"catalog_url": "https://p.example", "status": "UNREACHABLE: timeout"}}
    snapshot = scan.build_snapshot({"p": []}, channels, previous, {}, "2026-09-10")
    row = snapshot["listings"][0]
    assert row["withdrawn_at"] is None
    assert row["last_seen"] == "2026-01-01"  # unchanged, not re-stamped
    assert snapshot["shape_regressions"] == []  # a failed scan is not a shape regression


def test_a_successful_empty_scan_withdraws_an_omitted_listing() -> None:
    previous = [
        {
            "provider": "p",
            "listing_id": "gone",
            "first_seen": "2026-01-01",
            "last_seen": "2026-01-01",
            "withdrawn_at": None,
        }
    ]
    channels = {"p": {"catalog_url": "https://p.example", "status": "ok (0 free)"}}
    snapshot = scan.build_snapshot({"p": []}, channels, previous, {}, "2026-09-10")
    assert snapshot["listings"][0]["withdrawn_at"] == "2026-09-10"
    assert snapshot["shape_regressions"] == ["p"]


def test_shape_regression_blocks_the_snapshot_write(tmp_path: Path) -> None:
    """A regressed snapshot is a human-review event: it must not reach disk at all."""
    snapshot = _snapshot([_snapshot_row("openrouter", "org/new:free")])
    snapshot["shape_regressions"] = ["openrouter"]
    path = tmp_path / "latest.json"
    with pytest.raises(scan.ShapeRegressionError, match="openrouter"):
        scan.write_snapshot(snapshot, path)
    assert not path.exists()


def test_a_clean_snapshot_writes(tmp_path: Path) -> None:
    snapshot = _snapshot([_snapshot_row("openrouter", "org/new:free")])
    path = tmp_path / "latest.json"
    scan.write_snapshot(snapshot, path)
    assert json.loads(path.read_text())["scan_as_of"] == "2026-09-11"


def test_shape_regression_is_recorded_in_the_snapshot() -> None:
    previous = [
        {
            "provider": "p",
            "listing_id": "m",
            "first_seen": "x",
            "last_seen": "x",
            "withdrawn_at": None,
        }
    ]
    channels = {"p": {"catalog_url": "https://p.example", "status": "ok (0 free)"}}
    snapshot = scan.build_snapshot({"p": []}, channels, previous, {}, "2026-09-10")
    assert snapshot["shape_regressions"] == ["p"]
    assert snapshot["channels"]["p"]["count"] == 0


# ── catalog auth: a declared key is attached, an unset one names the channel ──────────


def _spec(url: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "catalog_url": url,
        "catalog_shape": "openai",
        "free_filter": {"kind": "all"},
    }
    base.update(over)
    return base


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def __call__(self, url: str, headers: dict[str, str] | None = None) -> Any:
        self.calls.append((url, headers))
        return {"data": [{"id": "m"}]}


class TestCatalogAuth:
    def test_bearer_key_is_attached_and_never_leaks_into_the_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "sk-secret")
        recorder = _Recorder()
        monkeypatch.setattr(scan, "_fetch", recorder)
        specs = {"groq": _spec("https://api.groq.com/openai/v1/models", api_key_env="GROQ_API_KEY")}
        _, channels = scan.scan_channels(specs)
        assert recorder.calls == [
            ("https://api.groq.com/openai/v1/models", {"Authorization": "Bearer sk-secret"})
        ]
        assert channels["groq"]["status"] == "ok (1 free)"
        assert "sk-secret" not in json.dumps(channels)

    def test_google_style_uses_the_x_goog_api_key_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEY", "g-key")
        recorder = _Recorder()
        monkeypatch.setattr(scan, "_fetch", recorder)
        specs = {
            "google_ai_studio": _spec(
                "https://generativelanguage.googleapis.com/v1beta/models",
                api_key_env="GOOGLE_AI_STUDIO_API_KEY",
                auth_style="google",
            )
        }
        _, channels = scan.scan_channels(specs)
        assert recorder.calls[0][1] == {"x-goog-api-key": "g-key"}
        assert channels["google_ai_studio"]["status"] == "ok (1 free)"

    def test_an_unset_key_is_named_and_no_request_is_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        recorder = _Recorder()
        monkeypatch.setattr(scan, "_fetch", recorder)
        specs = {"groq": _spec("https://api.groq.com/openai/v1/models", api_key_env="GROQ_API_KEY")}
        _, channels = scan.scan_channels(specs)
        assert recorder.calls == []
        assert channels["groq"]["status"] == "UNREACHABLE: UnsetEnv: GROQ_API_KEY"

    def test_an_empty_key_is_unset_not_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "")
        recorder = _Recorder()
        monkeypatch.setattr(scan, "_fetch", recorder)
        specs = {"groq": _spec("https://api.groq.com/openai/v1/models", api_key_env="GROQ_API_KEY")}
        _, channels = scan.scan_channels(specs)
        assert recorder.calls == []
        assert "UnsetEnv: GROQ_API_KEY" in channels["groq"]["status"]

    def test_a_keyless_provider_is_fetched_with_no_auth_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(scan, "_fetch", recorder)
        specs = {"openrouter": _spec("https://openrouter.ai/api/v1/models")}
        _, channels = scan.scan_channels(specs)
        assert recorder.calls == [("https://openrouter.ai/api/v1/models", None)]
        assert channels["openrouter"]["status"] == "ok (1 free)"

    def test_a_declared_user_agent_is_sent_with_the_default_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(scan, "_fetch", recorder)
        ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
        specs = {"together": _spec("https://api.together.xyz/v1/models", user_agent=ua)}
        _, channels = scan.scan_channels(specs)
        assert recorder.calls == [("https://api.together.xyz/v1/models", {"User-Agent": ua})]
        assert channels["together"]["status"] == "ok (1 free)"

    def test_no_declared_user_agent_leaves_the_default_fetch_ua(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(scan, "_fetch", recorder)
        specs = {"openrouter": _spec("https://openrouter.ai/api/v1/models")}
        scan.scan_channels(specs)
        # None means `_fetch` applies its own self-identifying default UA.
        assert recorder.calls == [("https://openrouter.ai/api/v1/models", None)]

    def test_a_declared_user_agent_coexists_with_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "sk-secret")
        recorder = _Recorder()
        monkeypatch.setattr(scan, "_fetch", recorder)
        specs = {
            "groq": _spec(
                "https://api.groq.com/openai/v1/models",
                api_key_env="GROQ_API_KEY",
                user_agent="browser/1.0",
            )
        }
        scan.scan_channels(specs)
        assert recorder.calls == [
            (
                "https://api.groq.com/openai/v1/models",
                {"Authorization": "Bearer sk-secret", "User-Agent": "browser/1.0"},
            )
        ]


# ── committed config ─────────────────────────────────────────────────────────────────


def test_free_catalogs_declare_shape_filter_and_provenance() -> None:
    specs = yaml.safe_load(scan.CATALOGS_PATH.read_text())["providers"]
    for provider, spec in specs.items():
        assert spec["catalog_url"].startswith("http"), provider
        assert spec["catalog_shape"] in {
            "openai",
            "google_ai_studio",
            "cloudflare",
            "promotions",
        }, provider
        assert spec["free_filter"]["kind"] in {
            "all",
            "zero_price",
            "id_suffix",
            "tag",
            "promotion_free",
        }, provider
        assert spec["verified_by"] in {"probe", "docs", "unverified"}, provider
        assert "scope" in spec["free_tier"], provider


def test_cloudflare_url_never_carries_a_literal_account_id() -> None:
    specs = yaml.safe_load(scan.CATALOGS_PATH.read_text())["providers"]
    assert "${CLOUDFLARE_ACCOUNT_ID}" in specs["cloudflare_workers_ai"]["catalog_url"]


def test_unset_env_var_is_unreachable_by_name_not_greeted_with_a_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    specs = yaml.safe_load(scan.CATALOGS_PATH.read_text())["providers"]
    channels: dict[str, Any] = {}
    listings, channels = scan.scan_channels(
        {"cloudflare_workers_ai": specs["cloudflare_workers_ai"]}
    )
    assert listings["cloudflare_workers_ai"] == []
    assert "UnsetEnv: CLOUDFLARE_ACCOUNT_ID" in channels["cloudflare_workers_ai"]["status"]


def test_committed_catalogs_declare_catalog_auth_only_where_needed() -> None:
    specs = yaml.safe_load(scan.CATALOGS_PATH.read_text())["providers"]
    assert specs["groq"]["api_key_env"] == "GROQ_API_KEY"
    assert specs["groq"].get("auth_style", "bearer") == "bearer"
    assert specs["google_ai_studio"]["api_key_env"] == "GOOGLE_AI_STUDIO_API_KEY"
    assert specs["google_ai_studio"]["auth_style"] == "google"
    assert specs["cloudflare_workers_ai"]["api_key_env"] == "CLOUDFLARE_API_TOKEN"
    for keyless in (
        "openrouter",
        "requesty",
        "nvidia_nim",
        "vercel_ai_gateway",
        "sambanova",
        "opencode_zen",
        "kilo_gateway",
        "explabs",
    ):
        assert "api_key_env" not in specs[keyless], keyless


def test_committed_catalogs_mark_vercel_free_tag_and_opencode_no_free_access() -> None:
    specs = yaml.safe_load(scan.CATALOGS_PATH.read_text())["providers"]
    assert specs["vercel_ai_gateway"]["free_filter"] == {"kind": "tag", "value": "free"}
    assert specs["opencode_zen"]["free_tier"]["free_access"] is False
    assert specs["opencode_zen"]["free_tier"]["access_note"]


def test_committed_catalogs_declare_the_explabs_promotions_adapter() -> None:
    specs = yaml.safe_load(scan.CATALOGS_PATH.read_text())["providers"]
    explabs = specs["explabs"]
    assert explabs["catalog_url"] == "https://api.experientiallabs.ai/api/models"
    assert explabs["catalog_shape"] == "promotions"
    assert explabs["free_filter"] == {"kind": "promotion_free"}
    assert explabs["verified_by"] == "probe"


# ── admission: the snapshot rows that may enter the overlay ───────────────────────────


def _snapshot_row(provider: str, listing_id: str, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "provider": provider,
        "listing_id": listing_id,
        "prices": None,
        "cost": None,
        "hugging_face_id": "",
        "canonical_slug": "",
        "context_length": None,
        "supports_tools": True,
        "expiration_date": None,
        "published_limits": {"scope": "per_account"},
        "schedulable": True,
        "schedulable_reason": None,
        "first_seen": "2026-09-11",
        "last_seen": "2026-09-11",
        "withdrawn_at": None,
    }
    row.update(over)
    return row


def _snapshot(rows: list[dict[str, Any]], channels: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": 1,
        "scan_as_of": "2026-09-11",
        "channels": channels or {"openrouter": {"catalog_url": "x", "status": "ok (1 free)"}},
        "metadata_sources": {},
        "shape_regressions": [],
        "listings": rows,
    }


class TestAdmittedIdentities:
    def test_proposal_groups_are_named_and_lone_listings_are_their_own_identity(self) -> None:
        rows = [
            _snapshot_row("openrouter", "org/alpha:free", hugging_face_id="org/alpha"),
            _snapshot_row("groq", "alpha", hugging_face_id="org/alpha"),
            _snapshot_row("openrouter", "loner/beta:free"),
        ]
        proposal = {
            "confirmed": {},
            "proposed": {"alpha": {"listings": ["openrouter:org/alpha:free", "groq:alpha"]}},
        }
        admitted = scan.admitted_identities(_snapshot(rows), proposal)
        assert admitted["alpha"] == [("openrouter", "org/alpha:free"), ("groq", "alpha")]
        assert admitted["beta"] == [("openrouter", "loner/beta:free")]

    def test_non_schedulable_free_access_false_and_withdrawn_rows_are_excluded(self) -> None:
        rows = [
            _snapshot_row("p", "no-sched", schedulable=False, schedulable_reason="nope"),
            _snapshot_row("p", "no-access", published_limits={"free_access": False}),
            _snapshot_row("p", "gone", withdrawn_at="2026-09-11"),
            _snapshot_row("p", "keep"),
        ]
        admitted = scan.admitted_identities(_snapshot(rows), {})
        assert set(admitted) == {"keep"}

    def test_a_denied_listing_is_never_admitted(self) -> None:
        rows = [_snapshot_row("openrouter", "z-ai/glm-5.3-0731:free")]
        identity = scan.IdentityMap(deny=frozenset({"z-ai/glm-5.3-0731:free"}))
        assert scan.admitted_identities(_snapshot(rows), {}, identity) == {}


def _provider() -> dict[str, str]:
    """A minimal valid provider row for an overlay that must pass `parse_registry`."""
    return {
        "base_url": "https://example.test/v1",
        "api_key_env_var": "EXAMPLE_API_KEY",
        "litellm_prefix": "openai",
    }


class TestApplyMerge:
    def _write_inputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        snapshot: dict[str, Any],
        overlay: dict[str, Any],
        *,
        identity: scan.IdentityMap | None = None,
    ) -> Path:
        snapshot_path = tmp_path / "latest.json"
        proposal_path = tmp_path / "proposal.yaml"
        overlay_path = tmp_path / "overlay.yaml"
        snapshot_path.write_text(json.dumps(snapshot))
        identity = identity or scan.IdentityMap()
        proposal = scan.propose(snapshot, identity, str(snapshot["scan_as_of"]))
        proposal_path.write_text(yaml.safe_dump(proposal, sort_keys=False))
        overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False))
        monkeypatch.setattr(scan, "SNAPSHOT_PATH", snapshot_path)
        monkeypatch.setattr(scan, "PROPOSAL_PATH", proposal_path)
        monkeypatch.setattr(scan, "OVERLAY_PATH", overlay_path)
        return overlay_path

    def test_a_newly_discovered_schedulable_listing_is_added_with_a_real_price(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [_snapshot_row("openrouter", "org/new:free", cost={"input": 1.5, "output": 6.0})]
        overlay = {"providers": {"openrouter": _provider()}}
        path = self._write_inputs(tmp_path, monkeypatch, _snapshot(rows), overlay)

        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 0

        merged = yaml.safe_load(path.read_text())
        row = merged["models"]["openrouter-new"]
        assert row["version"] == "new"
        assert row["model"] == "new"
        assert row["model_id"] == "new"
        assert row["lane"] == "org/new:free"
        pricing = row["pricing"]
        assert pricing["input_cost_per_1m"] == 1.5
        assert pricing["output_cost_per_1m"] == 6.0
        assert "cache_read_cost_per_1m" not in pricing  # HARD RULE 3
        assert pricing["price_source"] == "https://models.dev/api.json"
        assert pricing["price_as_of"] == "2026-09-11"

    def test_apply_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [_snapshot_row("openrouter", "org/new:free", cost={"input": 1.5, "output": 6.0})]
        overlay = {"providers": {"openrouter": _provider()}}
        path = self._write_inputs(tmp_path, monkeypatch, _snapshot(rows), overlay)

        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 0
        first = path.read_text()
        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 0
        assert path.read_text() == first

    def test_a_zero_price_row_is_refused_sh018_compatible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            _snapshot_row(
                "openrouter",
                "org/free:free",
                cost={"input": 0.0, "output": 0.0},
                prices={"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0},
            )
        ]
        overlay = {"providers": {"openrouter": _provider()}, "models": {}}
        path = self._write_inputs(tmp_path, monkeypatch, _snapshot(rows), overlay)

        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 0
        assert yaml.safe_load(path.read_text())["models"] == {}

    def test_a_catalogue_priced_lane_is_admitted_with_a_truthful_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # models.dev omits this lane, so the catalogue's own list price is the fallback and
        # the provenance names the catalogue, not models.dev.
        rows = [
            _snapshot_row(
                "sambanova",
                "gpt-oss-120b",
                cost=None,
                prices={"input_cost_per_1m": 0.22, "output_cost_per_1m": 0.59},
            )
        ]
        overlay = {"providers": {"sambanova": _provider()}, "models": {}}
        path = self._write_inputs(tmp_path, monkeypatch, _snapshot(rows), overlay)

        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 0
        pricing = yaml.safe_load(path.read_text())["models"]["sambanova-gpt-oss-120b"]["pricing"]
        assert pricing["input_cost_per_1m"] == 0.22
        assert pricing["price_source"] == "catalog:sambanova"
        assert "cache_read_cost_per_1m" not in pricing

    def test_a_refreshed_price_drops_any_cache_read_rate_sh018_compatible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [_snapshot_row("openrouter", "org/m:free", cost={"input": 2.0, "output": 8.0})]
        overlay = {
            "providers": {"openrouter": _provider()},
            "models": {
                "openrouter-org-m-free": {
                    "model_id": "org/m:free",
                    "provider": "openrouter",
                    "version": "org/m:free",
                    "pricing": {
                        "input_cost_per_1m": 1.0,
                        "output_cost_per_1m": 4.0,
                        "cache_read_cost_per_1m": 0.25,
                        "price_provider": "openrouter",
                        "price_source": "https://models.dev/api.json",
                        "price_as_of": "2026-09-10",
                    },
                }
            },
        }
        path = self._write_inputs(tmp_path, monkeypatch, _snapshot(rows), overlay)

        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 0
        pricing = yaml.safe_load(path.read_text())["models"]["openrouter-org-m-free"]["pricing"]
        assert pricing["input_cost_per_1m"] == 2.0
        assert "cache_read_cost_per_1m" not in pricing  # HARD RULE 3
        assert pricing["price_as_of"] == "2026-09-11"

    def test_free_access_false_row_is_excluded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            _snapshot_row(
                "opencode_zen",
                "x-free",
                schedulable=False,
                published_limits={"free_access": False, "access_note": "session-gated"},
                cost={"input": 1.0, "output": 2.0},
            )
        ]
        overlay = {"providers": {"opencode_zen": _provider()}, "models": {}}
        path = self._write_inputs(tmp_path, monkeypatch, _snapshot(rows), overlay)

        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 0
        assert yaml.safe_load(path.read_text())["models"] == {}

    def test_a_withdrawn_listing_is_retained_not_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = _snapshot([_snapshot_row("openrouter", "gone", withdrawn_at="2026-09-11")])
        overlay = {
            "providers": {"openrouter": _provider()},
            "models": {
                "openrouter-gone": {
                    "model_id": "gone",
                    "provider": "openrouter",
                    "version": "gone",
                    "pricing": {
                        "input_cost_per_1m": 1.0,
                        "output_cost_per_1m": 2.0,
                        "price_provider": "openrouter",
                        "price_source": "https://models.dev/api.json",
                        "price_as_of": "2026-09-10",
                    },
                }
            },
        }
        path = self._write_inputs(tmp_path, monkeypatch, snapshot, overlay)

        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 0
        merged = yaml.safe_load(path.read_text())
        assert "openrouter-gone" in merged["models"]
        assert scan.withdrawn_rows(merged, snapshot, {}) == ["openrouter-gone"]

    def test_shape_regression_blocks_apply_without_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = _snapshot([_snapshot_row("openrouter", "org/new:free")])
        snapshot["shape_regressions"] = ["openrouter"]
        overlay = {"providers": {"openrouter": _provider()}, "models": {}}
        path = self._write_inputs(tmp_path, monkeypatch, snapshot, overlay)
        original = path.read_text()

        assert scan.apply_snapshot(scan.IdentityMap(), "2026-09-11") == 2
        assert path.read_text() == original


# ── the refresh orchestrator: value-ordered runnable set ──────────────────────────────


def _priority_engine() -> Any:
    from benchmark.routing import collection_priority as cp

    return cp.CollectionPriority(
        channels={"hi": cp.Channel("hi", "hi"), "lo": cp.Channel("lo", "lo")},
        covered={},
        verified_total=10,
        base_importance={"hi": 10.0, "lo": 1.0},
        multimodal_eligible=lambda _model: False,
    )


class TestRefreshRunnable:
    def test_returned_set_is_priority_ordered(self) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        overlay = {
            "providers": {"p": {}},
            "models": {
                "p-lo": {"provider": "p", "model_id": "lo", "version": "lo"},
                "p-hi": {"provider": "p", "model_id": "hi", "version": "hi"},
            },
        }
        snapshot = _snapshot(
            [_snapshot_row("p", "hi"), _snapshot_row("p", "lo")],
            channels={"p": {"status": "ok (2 free)"}},
        )
        lanes = refresh.runnable_lanes(
            overlay, snapshot, engine=_priority_engine(), declared_free={"p": None}
        )
        assert [lane.version for lane in lanes] == ["hi", "lo"]
        assert lanes[0].priority > lanes[1].priority

    def test_an_explabs_promotions_listing_is_included(self) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        # A promotions row admitted by a real paid-twin price enters the overlay, so the
        # orchestrator's set carries it. `version` is the resolved identity, not the `-explabs`
        # channel name.
        snapshot = _snapshot(
            [_snapshot_row("explabs", "gpt-5.6-luna", cost={"input": 0.2, "output": 1.2})],
            channels={"explabs": {"status": "ok (1 free)"}},
        )
        admitted = scan.admitted_identities(snapshot, {})
        assert admitted == {"gpt-5.6-luna": [("explabs", "gpt-5.6-luna")]}
        overlay = {
            "providers": {"explabs": {}},
            "models": {
                "gpt-5.6-luna-explabs": {
                    "provider": "explabs",
                    "model_id": "gpt-5.6-luna",
                    "version": "gpt-5.6-luna",
                }
            },
        }
        lanes = refresh.runnable_lanes(
            overlay, snapshot, engine=_priority_engine(), declared_free={"explabs": None}
        )
        assert [lane.name for lane in lanes] == ["gpt-5.6-luna-explabs"]

    def test_a_listing_expiration_date_is_carried_onto_the_runnable_lane(self) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        overlay = {
            "providers": {"explabs": {}},
            "models": {
                "gpt-5.6-luna-explabs": {
                    "provider": "explabs",
                    "model_id": "gpt-5.6-luna",
                    "version": "gpt-5.6-luna",
                }
            },
        }
        snapshot = _snapshot(
            [_snapshot_row("explabs", "gpt-5.6-luna", expiration_date="2026-09-30")],
            channels={"explabs": {"status": "ok (1 free)"}},
        )
        lanes = refresh.runnable_lanes(
            overlay, snapshot, engine=_priority_engine(), declared_free={"explabs": None}
        )
        assert lanes[0].expires_at == "2026-09-30"
        assert lanes[0].to_row()["expires_at"] == "2026-09-30"

    def test_a_withdrawn_overlay_row_is_quiesced_out(self) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        overlay = {
            "providers": {"p": {}},
            "models": {"p-gone": {"provider": "p", "model_id": "gone", "version": "gone"}},
        }
        snapshot = _snapshot(
            [_snapshot_row("p", "gone", withdrawn_at="2026-09-11")],
            channels={"p": {"status": "ok (0 free)"}},
        )
        assert (
            refresh.runnable_lanes(
                overlay, snapshot, engine=_priority_engine(), declared_free={"p": None}
            )
            == []
        )


class TestDeclaredFreeProviderGate:
    """The scheduling-only gate: only a declared free provider may enter the runnable set."""

    def _overlay(self, provider: str, model_id: str, name: str) -> dict[str, Any]:
        return {
            "providers": {provider: {}},
            "models": {name: {"provider": provider, "model_id": model_id, "version": name}},
        }

    def test_together_lane_is_excluded_as_no_free_lane_and_kept_for_provenance(self) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        overlay = self._overlay(
            "together", "meta-models/Muse-Glimmer-30B", "together-muse-glimmer-30b"
        )
        snapshot = _snapshot([], channels={})

        assert refresh.runnable_lanes(overlay, snapshot, engine=_priority_engine()) == []
        assert refresh.lane_exclusions(overlay, snapshot) == {
            "together-muse-glimmer-30b": "no-free-lane"
        }
        # The row is not deleted: the exclusion is scheduling-only.
        assert "together-muse-glimmer-30b" in overlay["models"]

    def test_opencode_lane_is_excluded_as_free_access_false(self) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        overlay = self._overlay(
            "opencode_zen", "deepseek-v4-flash-free", "opencode-deepseek-v4-flash-free"
        )
        snapshot = _snapshot([], channels={})

        assert refresh.runnable_lanes(overlay, snapshot, engine=_priority_engine()) == []
        assert refresh.lane_exclusions(overlay, snapshot) == {
            "opencode-deepseek-v4-flash-free": "free_access:false"
        }

    @pytest.mark.parametrize(
        ("provider", "model_id"),
        [
            ("kilo_gateway", "poolside/laguna-xs-2.1:free"),
            ("vercel_ai_gateway", "inclusionai/ling-3.0-flash-fin-free"),
            ("requesty", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            ("nvidia_nim", "moonshotai/kimi-k3"),
            ("google_ai_studio", "gemini-flash-latest"),
            ("sambanova", "gpt-oss-120b"),
            ("cloudflare_workers_ai", "@cf/openai/gpt-oss-120b"),
            ("openrouter", "poolside/laguna-xs-2.1:free"),
            ("groq", "openai/gpt-oss-120b"),
            ("explabs", "gpt-5.6-luna"),
        ],
    )
    def test_a_declared_free_provider_is_retained(self, provider: str, model_id: str) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        overlay = self._overlay(provider, model_id, f"{provider}-lane")
        snapshot = _snapshot(
            [_snapshot_row(provider, model_id)],
            channels={provider: {"status": "ok (1 free)"}},
        )

        lanes = refresh.runnable_lanes(overlay, snapshot, engine=_priority_engine())
        assert [lane.name for lane in lanes] == [f"{provider}-lane"]
        assert refresh.lane_exclusions(overlay, snapshot) == {}

    def test_a_non_chat_overlay_row_is_excluded_from_the_runnable_set(self) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        overlay = {
            "providers": {"google_ai_studio": {}},
            "models": {
                "google-gemini-tts": {
                    "provider": "google_ai_studio",
                    "model_id": "gemini-2.5-flash-preview-tts",
                    "version": "gemini-tts",
                },
                "google-gemini-chat": {
                    "provider": "google_ai_studio",
                    "model_id": "gemini-flash-latest",
                    "version": "gemini-flash-latest",
                },
            },
        }
        snapshot = _snapshot(
            [
                _snapshot_row("google_ai_studio", "gemini-2.5-flash-preview-tts"),
                _snapshot_row("google_ai_studio", "gemini-flash-latest"),
            ],
            channels={"google_ai_studio": {"status": "ok (2 free)"}},
        )
        declared = {"google_ai_studio": None}
        lanes = refresh.runnable_lanes(
            overlay, snapshot, engine=_priority_engine(), declared_free=declared
        )
        assert [lane.name for lane in lanes] == ["google-gemini-chat"]
        assert refresh.lane_exclusions(overlay, snapshot, declared_free=declared) == {
            "google-gemini-tts": "non-chat"
        }

    def test_gate_keeps_the_returned_set_priority_ordered(self) -> None:
        from benchmark.routing.scripts import refresh_free_campaign as refresh

        overlay = {
            "providers": {"groq": {}},
            "models": {
                "groq-lo": {"provider": "groq", "model_id": "lo", "version": "lo"},
                "groq-hi": {"provider": "groq", "model_id": "hi", "version": "hi"},
                "together-muse-glimmer-30b": {
                    "provider": "together",
                    "model_id": "meta-models/Muse-Glimmer-30B",
                    "version": "meta/muse-glimmer-30b",
                },
            },
        }
        snapshot = _snapshot(
            [_snapshot_row("groq", "hi"), _snapshot_row("groq", "lo")],
            channels={"groq": {"status": "ok (2 free)"}},
        )
        lanes = refresh.runnable_lanes(overlay, snapshot, engine=_priority_engine())
        assert [lane.version for lane in lanes] == ["hi", "lo"]
        assert refresh.lane_exclusions(overlay, snapshot) == {
            "together-muse-glimmer-30b": "no-free-lane"
        }
