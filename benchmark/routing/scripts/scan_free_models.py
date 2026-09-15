"""Scan free-tier provider catalogues into a facts-only snapshot, and resolve identities.

python -m benchmark.routing.scripts.scan_free_models            # scan -> snapshot
python -m benchmark.routing.scripts.scan_free_models --dry-run  # print the snapshot, write nothing
python -m benchmark.routing.scripts.scan_free_models --propose  # scan, then write the proposal
python -m benchmark.routing.scripts.scan_free_models --apply    # guard + refresh overlay

REAL FACTS ONLY. Every listing, price and count this writes was fetched, this run, from a
public machine-readable endpoint. The ONE declared exception is `first_seen`, carried forward
from the previous scan because that is the appear/disappear tracker: a date, never a price. A
channel that fails is EMPTY and says why; it is never guessed. A listing absent from a scan is
recorded `withdrawn_at`, never deleted, so its lane quiesces and collected rows keep provenance.

WHICH LISTINGS ARE FREE is declared per provider in `data/free_catalogs.yaml` (`free_filter`),
never inferred from a name. This matters because a catalogue that publishes no per-model price
(Groq, NIM) is the free surface, while one that publishes prices (OpenRouter, Requesty) needs a
zero-price filter, and one that marks free by tag (Vercel) needs neither.

IDENTITY IS NOT CLAIMED IN THE SNAPSHOT. `model_identity.yaml` holds the Tier-2 curated
overrides; `--propose` writes the reviewed proposal (`confirmed` / `proposed` / `dropped`) and
`--apply` is the only path that writes a resolved `version` into the overlay, refusing any
entry absent from the proposal or whose content hash moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml

from benchmark.routing.scripts.refresh_price_sheet import _fetch
from shunt.models.config import parse_registry

_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "data"
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
CATALOGS_PATH: Final[Path] = _DATA_DIR / "free_catalogs.yaml"
SNAPSHOT_PATH: Final[Path] = _DATA_DIR / "latest_free_models.json"
IDENTITY_PATH: Final[Path] = _DATA_DIR / "model_identity.yaml"
PROPOSAL_PATH: Final[Path] = _DATA_DIR / "free_models_proposal.yaml"
OVERLAY_PATH: Final[Path] = _REPO_ROOT / "configs" / "free-tier" / "overlay.yaml"
MODELSDEV_SOURCE: Final[str] = "modelsdev"

# Every channel prices per token; the snapshot quotes per 1M, as the price sheet does.
_PER_TOKEN_TO_PER_1M: Final[float] = 1_000_000.0
_ENV_VAR: Final[re.Pattern[str]] = re.compile(r"\$\{([A-Z0-9_]+)\}")

# ── the bare-slug canonical id convention (one true id per model) ─────────────────────
# One model = one OFFICIAL bare slug: lowercase, tokens joined by `-`, NO `/`, `:`, `@`,
# no provider prefix, no `-free`/`-explabs`/`:free`/`:nitro`/`:floor`/`:extended` suffix.
# The grammar below applies mechanically to `<creator>[/-:]*<model>…`; ambiguous spellings
# the grammar cannot settle are fixed by the CURATED overrides further down.
_PROMO_SUFFIXES: Final[tuple[str, ...]] = (
    ":free",
    ":nitro",
    ":floor",
    ":extended",
    "-free",
    "-explabs",
)
# CURATED overrides: the audited ambiguous spellings, applied AFTER the mechanical grammar.
# Each key is the grammar's own output (or a raw listing id it leaves dotted); each value is
# the one canonical bare slug. `poolside/laguna-xs.2` is the same weights as `laguna-xs-2.1`;
# the size-qualified NIM/Requesty Lightning slug is the short `nemotron-3.5-lightning`; the
# SambaNova shouting-case Llama id drops its `meta-` serving prefix. Keep this list sorted and
# every entry cited; an unlisted ambiguity is a code review event, never a silent guess.
CANONICAL_OVERRIDES: Final[dict[str, str]] = {
    "laguna-xs.2": "laguna-xs-2.1",
    "laguna-xs-2-1": "laguna-xs-2.1",
    # `ling-3.0-flash-*` is dotted by the publisher but dashed in the curated identity map and
    # the committed overlay; settle the dot on the dashed bare identity so a rescan and the
    # stale reviewed proposal key ONE slug for the same weights.
    "ling-3.0-flash-fin": "ling-3-0-flash-fin",
    "ling-3.0-flash-sante": "ling-3-0-flash-sante",
    "llama-3.3-70b": "llama-3.3-70b-instruct",
    "meta-llama-3.3-70b": "llama-3.3-70b-instruct",
    "meta-llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "nemotron-3-5-lightning-30b-a3b": "nemotron-3.5-lightning",
    "nemotron-3.5-lightning-30b-a3b": "nemotron-3.5-lightning",
    "nvidia-nemotron-3.5-lightning-30b-a3b": "nemotron-3.5-lightning",
}
_FACT_FIELDS: Final[tuple[str, ...]] = (
    "listing_id",
    "provider",
    "prices",
    "hugging_face_id",
    "canonical_slug",
    "context_length",
    "supports_tools",
    "expiration_date",
)


@dataclass(frozen=True)
class Listing:
    """One free listing's own facts, before cross-scan bookkeeping is attached."""

    listing_id: str
    provider: str
    prices: dict[str, float] | None
    hugging_face_id: str
    canonical_slug: str
    context_length: int | None
    supports_tools: bool | None
    expiration_date: str | None
    # Promotion-shape facts (Experiential Labs): the umbrella's own cap on the free deal and
    # whether a payment method must be on file. Recorded as listing facts, never a price.
    per_org_cap_micro_usd: int | None = None
    requires_payment_method: bool | None = None
    # Metadata joined in from a declared metadata source (models.dev). Defaults keep the
    # normalisers free of the join: a listing with no metadata carries an explicit absence.
    release_date: str | None = None
    limit: dict[str, Any] | None = None
    open_weights: bool | None = None
    cost: dict[str, float] | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "provider": self.provider,
            "prices": self.prices,
            "hugging_face_id": self.hugging_face_id,
            "canonical_slug": self.canonical_slug,
            "context_length": self.context_length,
            "supports_tools": self.supports_tools,
            "expiration_date": self.expiration_date,
            "per_org_cap_micro_usd": self.per_org_cap_micro_usd,
            "requires_payment_method": self.requires_payment_method,
            "release_date": self.release_date,
            "limit": self.limit,
            "open_weights": self.open_weights,
            "cost": self.cost,
        }


# ── normalisers: each catalogue shape in, facts out; a mangled payload yields [] ──────


def _to_per_1m(value: Any) -> float | None:
    try:
        return float(value) * _PER_TOKEN_TO_PER_1M
    except (TypeError, ValueError):
        return None


def _openai_prices(entry: dict[str, Any]) -> dict[str, float] | None:
    """Read whichever per-token price spelling this OpenAI-shaped catalogue uses."""
    pricing = entry.get("pricing")
    if isinstance(pricing, dict):
        for in_key, out_key in (("prompt", "completion"), ("input", "output")):
            if in_key in pricing or out_key in pricing:
                return _price_pair(
                    _to_per_1m(pricing.get(in_key)), _to_per_1m(pricing.get(out_key))
                )
    return _price_pair(_to_per_1m(entry.get("input_price")), _to_per_1m(entry.get("output_price")))


def _price_pair(inp: float | None, out: float | None) -> dict[str, float] | None:
    if inp is None or out is None:
        return None
    return {"input_cost_per_1m": inp, "output_cost_per_1m": out}


def _declares_tools(entry: dict[str, Any]) -> bool | None:
    """Whether the catalogue declares tool-calling; ``None`` when it says nothing.

    A LIST present with at least one tool parameter is an explicit yes; a NON-EMPTY list that
    omits them is an explicit no (the catalogue enumerated its parameters). Absent/empty
    declarations, and catalogues that carry no tool surface at all, are UNKNOWN (``None``) —
    never a silent ``False`` — so the live tool-call probe, not this static guess, gates them.
    """
    params = entry.get("supported_parameters")
    if isinstance(params, list):
        if any(str(p).lower() in {"tools", "tool_choice"} for p in params):
            return True
        if params:
            return False
    if entry.get("supports_tool_calling") is True:
        return True
    if entry.get("supports_tool_calling") is False:
        return False
    tags = entry.get("tags")
    if isinstance(tags, list) and any("tool" in str(t).lower() for t in tags):
        return True
    caps = entry.get("capabilities")
    if isinstance(caps, dict):
        if caps.get("tools") or caps.get("function_calling"):
            return True
        if caps.get("tools") is False or caps.get("function_calling") is False:
            return False
    return None


def _context_length(entry: dict[str, Any]) -> int | None:
    for key in ("context_length", "context_window", "max_context_length"):
        value = entry.get(key)
        if isinstance(value, int):
            return value
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _is_zero_price(prices: dict[str, float] | None) -> bool:
    if prices is None:
        return False
    return not any(prices.values())


def _passes_filter(
    entry: dict[str, Any],
    listing_id: str,
    prices: dict[str, float] | None,
    free_filter: dict[str, Any],
) -> bool:
    kind = free_filter.get("kind")
    if kind == "all":
        return True
    if kind == "zero_price":
        return _is_zero_price(prices)
    if kind == "id_suffix":
        return listing_id.endswith(str(free_filter.get("value", "")))
    if kind == "tag":
        tags = entry.get("tags")
        return isinstance(tags, list) and str(free_filter.get("value", "")).lower() in {
            str(t).lower() for t in tags
        }
    if kind == "promotion_free":
        return entry.get("free") is True
    return False


# Non-text-chat listings the free campaign can never run: TTS/image/audio/embedding models,
# computer-use, deep-research, aqa and similar. They must never enter the free RUNNABLE set.
# Excluded here, at normalisation, where the catalogue declares the model's id/modality; the
# per-lane preflight is the fallback for a non-chat model that slips through.
_NON_CHAT_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "tts",
        "aqa",
        "image",
        "imagen",
        "audio",
        "speech",
        "voice",
        "veo",
        "embed",
        "embedding",
        "embeddings",
        "rerank",
        "reranker",
        "whisper",
        "moderation",
        "guard",
    }
)
_NON_CHAT_PHRASES: Final[tuple[str, ...]] = (
    "computer-use",
    "deep-research",
    "text-to-image",
    "text-to-speech",
    "image-generation",
)
_NON_CHAT_OUTPUT_MODALITIES: Final[frozenset[str]] = frozenset({"image", "audio", "video"})


def _non_chat_listing_id(listing_id: str) -> bool:
    """True when the listing id marks a non-text-chat model (``*-tts``, ``computer-use``, …)."""
    lowered = listing_id.lower()
    segments = set(re.split(r"[-/:_.]+", lowered))
    if segments & _NON_CHAT_TOKENS:
        return True
    normalized = lowered.replace("_", "-")
    return any(phrase in normalized for phrase in _NON_CHAT_PHRASES)


def _declares_non_chat_output(entry: Mapping[str, Any]) -> bool:
    """True when the catalogue's declared OUTPUT modality is not text."""
    for key in ("outputModalities", "output_modalities", "output_modality", "modalities"):
        value = entry.get(key)
        if isinstance(value, str):
            declared = {value.lower()}
        elif isinstance(value, list):
            declared = {str(item).lower() for item in value}
        else:
            continue
        # A model that outputs text ALONGSIDE an image/audio modality is still chat-capable;
        # only a text-free declaration (a pure TTS/image/audio model) is non-chat.
        if declared and "text" not in declared and _NON_CHAT_OUTPUT_MODALITIES & declared:
            return True
    return False


def _is_text_chat_entry(entry: Mapping[str, Any], listing_id: str) -> bool:
    """False for a TTS/image/audio/computer-use listing, by declared id or output modality."""
    return not (_non_chat_listing_id(listing_id) or _declares_non_chat_output(entry))


def is_non_chat_listing(listing_id: str) -> bool:
    """True when a listing id names a non-text-chat model (the campaign-runnable guard).

    Public so the overlay admission layer can exclude a stale non-chat row (``*-tts``,
    ``*-image``, ``computer-use``, ``deep-research``, ``aqa``) even before it is scheduled.
    """
    return _non_chat_listing_id(listing_id)


def normalise_openai(payload: Any, provider: str, spec: dict[str, Any]) -> list[Listing]:
    """`data[]` shape: OpenRouter/Requesty/Groq/SambaNova/NIM/Vercel/OpenCode/Kilo/Together."""
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    free_filter = spec.get("free_filter") or {}
    out: list[Listing] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        listing_id = entry.get("id")
        if not isinstance(listing_id, str) or not listing_id:
            continue
        if not _is_text_chat_entry(entry, listing_id):
            continue
        prices = _openai_prices(entry)
        if not _passes_filter(entry, listing_id, prices, free_filter):
            continue
        expires = entry.get("expiration_date")
        out.append(
            Listing(
                listing_id=listing_id,
                provider=provider,
                prices=prices,
                hugging_face_id=str(entry.get("hugging_face_id") or ""),
                canonical_slug=str(entry.get("canonical_slug") or ""),
                context_length=_context_length(entry),
                supports_tools=_declares_tools(entry),
                expiration_date=str(expires) if expires else None,
            )
        )
    return out


def normalise_google_ai_studio(payload: Any, provider: str, spec: dict[str, Any]) -> list[Listing]:
    """`models[]` shape, KEEPING ONLY `generateContent` (chat-capable) entries.

    The catalogue mixes chat models with embeddings (`embedContent`), answer generation
    (`generateAnswer`), long-running prediction and audio/music generation. Only a model whose
    `supportedGenerationMethods` includes `generateContent` is served on the chat surface the
    lane can use, so the rest are dropped here rather than admitted and failed later.
    """
    entries = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    out: list[Listing] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        methods = entry.get("supportedGenerationMethods")
        if not (isinstance(methods, list) and "generateContent" in methods):
            continue
        # generateContent is necessary but not sufficient: a TTS/image model can declare it
        # while outputting audio/image, and computer-use/deep-research declare it too.
        if not _is_text_chat_entry(entry, str(name)):
            continue
        out.append(
            Listing(
                listing_id=name.removeprefix("models/"),
                provider=provider,
                prices=None,
                hugging_face_id="",
                canonical_slug="",
                context_length=_context_length(
                    {"max_context_length": entry.get("inputTokenLimit")}
                ),
                supports_tools=True,
                expiration_date=None,
            )
        )
    return out


def normalise_cloudflare(payload: Any, provider: str, spec: dict[str, Any]) -> list[Listing]:
    """`result[]` shape from Workers AI model search, KEEPING ONLY text-generation tasks.

    In this catalogue `id` is an opaque UUID; the real model slug (`@cf/...`) the OpenAI-compatible
    `/ai/v1` chat surface serves is `name`, so `name` is the listing id. The search returns every
    task (embeddings, classification, audio, "Dumb Pipe"), but only `task.name` containing
    "text generation" (case-insensitive) is chat-capable, so the rest are dropped here.
    """
    entries = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    out: list[Listing] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        listing_id = entry.get("name")
        if not isinstance(listing_id, str) or not listing_id:
            continue
        task = entry.get("task")
        task_name = str(task.get("name", "")).lower() if isinstance(task, dict) else ""
        if "text generation" not in task_name:
            continue
        # "text generation" is necessary but not sufficient, exactly as in the OpenAI/Google
        # shapes: a guard/moderation listing (`@cf/meta/llama-guard-3-8b`) declares text
        # generation yet is not a chat model, so it must never enter the free RUNNABLE set.
        if not _is_text_chat_entry(entry, listing_id):
            continue
        out.append(
            Listing(
                listing_id=listing_id,
                provider=provider,
                prices=None,
                hugging_face_id="",
                canonical_slug="",
                context_length=None,
                supports_tools=True,
                expiration_date=None,
            )
        )
    return out


def normalise_promotions(payload: Any, provider: str, spec: dict[str, Any]) -> list[Listing]:
    """`promotions[]` shape (Experiential Labs): one listing per declared free-promo slug.

    A promotion marks its free deals with `free: true`; the paid/percentage-off promos
    (`free: false`) are dropped by the `promotion_free` filter. The umbrella's own
    `per_org_cap_micro_usd` and `requires_payment_method` are recorded as listing facts (a
    card requirement does not make a deal paid — it is a fact the admission gate reads) and
    `expiration_date` when the promotion carries one. Slugs are read from the payload each
    scan, never hardcoded; a promotion with no usable slug is skipped. A mangled payload
    yields `[]`.
    """
    entries = payload.get("promotions") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    free_filter = spec.get("free_filter") or {}
    out: list[Listing] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        slugs = entry.get("slugs")
        if not isinstance(slugs, list):
            continue
        expires = entry.get("expiration_date")
        cap = entry.get("per_org_cap_micro_usd")
        card = entry.get("requires_payment_method")
        for slug in slugs:
            if not isinstance(slug, str) or not slug:
                continue
            if not _passes_filter(entry, slug, None, free_filter):
                continue
            out.append(
                Listing(
                    listing_id=slug,
                    provider=provider,
                    prices=None,
                    hugging_face_id="",
                    canonical_slug="",
                    context_length=None,
                    supports_tools=_declares_tools(entry),
                    expiration_date=str(expires) if expires else None,
                    per_org_cap_micro_usd=cap if isinstance(cap, int) else None,
                    requires_payment_method=card if isinstance(card, bool) else None,
                )
            )
    return out


_NORMALISERS: Final[dict[str, Any]] = {
    "openai": normalise_openai,
    "google_ai_studio": normalise_google_ai_studio,
    "cloudflare": normalise_cloudflare,
    "promotions": normalise_promotions,
}


def normalise(payload: Any, provider: str, spec: dict[str, Any]) -> list[Listing]:
    """Dispatch on the declared shape; an unknown or mangled payload is empty, never raised."""
    shape = spec.get("catalog_shape")
    handler = _NORMALISERS.get(str(shape))
    if handler is None:
        return []
    return list(handler(payload, provider, spec))


# ── metadata source: models.dev (release_date / limit / open_weights / list price) ───


def modelsdev_index(payload: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Index models.dev metadata by (publisher provider key, publisher model key).

    models.dev is a METADATA source, never a joiner. The dict key it is indexed under is
    the publisher's OWN catalogue key, which is the identifier our vendor fetch already
    carries as the listing id — a publisher-issued identifier. The entry's own ``id``
    field is deliberately never read: using it to build a canonical identity or to join
    across providers would import a fuzzy match under our provenance, which the identity
    policy bans. Where a publisher's id differs from the models.dev key, the curated
    ``model_identity.yaml`` alias is the only path (see :func:`attach_metadata`).
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(payload, Mapping):
        return index
    for md_provider, provider in payload.items():
        models = provider.get("models") if isinstance(provider, Mapping) else None
        if not isinstance(models, Mapping):
            continue
        for listing_key, entry in models.items():
            if not isinstance(entry, Mapping):
                continue
            index[(str(md_provider), str(listing_key))] = {
                "release_date": entry.get("release_date"),
                "limit": entry.get("limit") if isinstance(entry.get("limit"), Mapping) else None,
                "open_weights": entry.get("open_weights"),
                "cost": _modelsdev_cost(entry.get("cost")),
            }
    return index


def _modelsdev_cost(cost: Any) -> dict[str, float] | None:
    """The published input/output list price (USD per 1M), or None; cache rates are dropped.

    models.dev is the paid twin's list price for a free listing, so it is exactly what
    HARD RULE 2 requires. Its ``cache_read`` is discarded here on purpose: HARD RULE 3
    forbids a harvested cache rate anywhere under ``configs/free-tier/`` (a shared gateway
    cache namespace would make it an artefact of other tenants' traffic).
    """
    if not isinstance(cost, Mapping):
        return None
    try:
        return {"input": float(cost["input"]), "output": float(cost["output"])}
    except (KeyError, TypeError, ValueError):
        return None


def attach_metadata(
    listings: dict[str, list[Listing]],
    index: Mapping[tuple[str, str], Mapping[str, Any]],
    spec: Mapping[str, Any],
    identity: IdentityMap | None = None,
) -> dict[str, list[Listing]]:
    """Attach models.dev metadata to listings, joined only by publisher-issued identifiers.

    The join is scoped to one publisher: a listing id fetched from the publisher's own
    catalogue is matched exactly against that publisher's models.dev catalogue key. A
    curated ``model_identity.yaml`` entry may instead declare a ``modelsdev`` id, which
    applies to every listing of that identity — the only sanctioned path for a publisher
    whose listing id differs from the models.dev key (e.g. Requesty's vendor prefixes).
    A listing with no match keeps ``release_date=None``; the admission gate then refuses
    it rather than fabricating an age.
    """
    provider_keys = spec.get("provider_keys") or {}
    listing_aliases = spec.get("listing_aliases") or {}
    identity_of = _identity_listing_lookup(identity) if identity is not None else {}
    identity_modelsdev = _identity_modelsdev_lookup(identity) if identity is not None else {}

    out: dict[str, list[Listing]] = {}
    for provider, items in listings.items():
        md_provider = str(provider_keys.get(provider) or "")
        out[provider] = [
            _attach_one(
                listing,
                provider,
                md_provider,
                index,
                listing_aliases,
                identity_of,
                identity_modelsdev,
            )
            for listing in items
        ]
    return out


def _attach_one(
    listing: Listing,
    provider: str,
    md_provider: str,
    index: Mapping[tuple[str, str], Mapping[str, Any]],
    listing_aliases: Mapping[str, Any],
    identity_of: Mapping[tuple[str, str], str],
    identity_modelsdev: Mapping[str, tuple[str, str]],
) -> Listing:
    """Resolve one listing's metadata key (curated identity first, then exact publisher id)."""
    slug = identity_of.get((provider, listing.listing_id))
    if slug is not None and slug in identity_modelsdev:
        md_provider, key = identity_modelsdev[slug]
    elif md_provider:
        key = str(listing_aliases.get(f"{provider}:{listing.listing_id}") or listing.listing_id)
    else:
        return listing
    meta = index.get((md_provider, key))
    if meta is None:
        return listing
    return replace(
        listing,
        release_date=meta.get("release_date"),
        limit=meta.get("limit"),
        open_weights=meta.get("open_weights"),
        cost=meta.get("cost"),
    )


def _identity_listing_lookup(identity: IdentityMap) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for slug, entry in identity.entries.items():
        for provider, ids in entry.aliases.items():
            for listing_id in ids:
                lookup[(provider, listing_id)] = slug
    return lookup


def _identity_modelsdev_lookup(identity: IdentityMap) -> dict[str, tuple[str, str]]:
    return {
        slug: (entry.modelsdev_provider, entry.modelsdev_id)
        for slug, entry in identity.entries.items()
        if entry.modelsdev_provider and entry.modelsdev_id
    }


def scan_metadata_sources(
    spec: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Fetch a declared metadata source. Unreachable ⇒ EMPTY index and a named status."""
    url = str(spec.get("catalog_url") or "")
    if not url:
        return {}, {"catalog_url": "", "status": "not configured"}
    resolved, missing = _resolve_url(url)
    if resolved is None:
        return {}, {"catalog_url": url, "status": f"UNREACHABLE: UnsetEnv: {missing}"}
    try:
        payload = _fetch(resolved)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {}, {"catalog_url": url, "status": f"UNREACHABLE: {type(exc).__name__}: {exc}"}
    index = modelsdev_index(payload)
    return index, {"catalog_url": url, "status": f"ok ({len(index)} model entries)"}


# ── fetch + snapshot assembly ────────────────────────────────────────────────────────


def _resolve_url(url: str) -> tuple[str | None, str]:
    """Expand `${VAR}` from the environment; a missing name is reported, never requested."""
    missing = [name for name in _ENV_VAR.findall(url) if not os.environ.get(name)]
    if missing:
        return None, missing[0]
    return _ENV_VAR.sub(lambda m: os.environ[m.group(1)], url), ""


def _auth_headers(spec: Mapping[str, Any]) -> tuple[dict[str, str] | None, str]:
    """Build catalogue auth headers from a declared key env var; an unset var is reported.

    The key VALUE is read only to build the request header — it is never logged, persisted, or
    echoed into the channel status. With no ``api_key_env`` (a keyless public channel) there are
    no extra headers, exactly as before. A declared but unset/empty variable returns its NAME so
    the caller marks the channel UNREACHABLE instead of sending an unauthenticated request.
    """
    key_env = str(spec.get("api_key_env") or "")
    if not key_env:
        return None, ""
    key = os.environ.get(key_env, "")
    if not key:
        return None, key_env
    if str(spec.get("auth_style") or "bearer") == "google":
        return {"x-goog-api-key": key}, ""
    return {"Authorization": f"Bearer {key}"}, ""


def _request_headers(spec: Mapping[str, Any]) -> tuple[dict[str, str] | None, str]:
    """Catalogue request headers: declared auth, plus a declared per-provider User-Agent.

    Some gateways reject the default self-identifying probe UA with a 403/401 and answer a
    browser-like UA, so the UA is registry data (`user_agent`) rather than a code constant.
    With none declared the headers are exactly `_auth_headers`' result and `_fetch` applies
    its own default UA; an unset key still names the channel instead of sending a request.
    """
    headers, missing = _auth_headers(spec)
    if missing:
        return None, missing
    user_agent = str(spec.get("user_agent") or "").strip()
    if not user_agent:
        return headers, ""
    return {**(headers or {}), "User-Agent": user_agent}, ""


def scan_channels(
    specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[Listing]], dict[str, dict[str, Any]]]:
    """Fetch every channel. A channel that fails is EMPTY and says so; it is never guessed."""
    listings: dict[str, list[Listing]] = {}
    channels: dict[str, dict[str, Any]] = {}
    for provider, spec in specs.items():
        catalog_url = str(spec.get("catalog_url", ""))
        url, missing = _resolve_url(catalog_url)
        if url is None:
            listings[provider] = []
            channels[provider] = _channel(catalog_url, f"UNREACHABLE: UnsetEnv: {missing}")
            continue
        headers, missing_key = _request_headers(spec)
        if missing_key:
            listings[provider] = []
            channels[provider] = _channel(catalog_url, f"UNREACHABLE: UnsetEnv: {missing_key}")
            continue
        try:
            payload = _fetch(url, headers)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            listings[provider] = []
            channels[provider] = _channel(catalog_url, f"UNREACHABLE: {type(exc).__name__}: {exc}")
            continue
        found = normalise(payload, provider, spec)
        listings[provider] = found
        channels[provider] = _channel(catalog_url, f"ok ({len(found)} free)")
    return listings, channels


def _channel(catalog_url: str, status: str) -> dict[str, Any]:
    return {"catalog_url": catalog_url, "status": status}


def _previous_listings(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return list(payload.get("listings", []))


def _channel_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("withdrawn_at"):
            continue
        provider = str(row.get("provider", ""))
        counts[provider] = counts.get(provider, 0) + 1
    return counts


def shape_regressions(previous_counts: dict[str, int], current_counts: dict[str, int]) -> list[str]:
    """A channel that fell to 0 when it previously had listings is a SHAPE_REGRESSION."""
    return sorted(
        channel
        for channel, previous in previous_counts.items()
        if previous > 0 and current_counts.get(channel, 0) == 0
    )


def merge_listings(
    listings: dict[str, list[Listing]],
    previous_rows: list[dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    scan_as_of: str,
    *,
    scanned_ok: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Attach first_seen/last_seen/withdrawn_at; carry `first_seen` forward, never a price.

    ``scanned_ok`` names the channels whose catalogue GET SUCCEEDED this run. A channel absent
    from it (an UNREACHABLE HTML/timeout/non-JSON channel) proves nothing, so its previous rows
    are CARRIED FORWARD UNCHANGED — never withdrawn on a transient failure. Only a successful
    scan that genuinely omits a listing may mark it withdrawn; ``None`` preserves the pre-existing
    behaviour of treating every listed provider as scanned.
    """
    previous = {f"{r['provider']}|{r['listing_id']}": r for r in previous_rows}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for provider, items in listings.items():
        if scanned_ok is not None and provider not in scanned_ok:
            continue  # unreachable: its previous rows are carried forward below
        limits = specs.get(provider, {}).get("free_tier") or {}
        free_access = limits.get("free_access", True) is not False
        reason = (
            None
            if free_access
            else str(
                limits.get("access_note")
                or f"{provider}: free_access is false — no free tier through the API"
            )
        )
        for listing in items:
            key = f"{provider}|{listing.listing_id}"
            seen.add(key)
            old = previous.get(key, {})
            row = listing.to_row()
            row["published_limits"] = limits
            row["schedulable"] = free_access
            row["schedulable_reason"] = reason
            row["first_seen"] = old.get("first_seen") or scan_as_of
            row["last_seen"] = scan_as_of
            row["withdrawn_at"] = None
            rows.append(row)
    for key, old in previous.items():
        if key in seen:
            continue
        if scanned_ok is not None and str(old.get("provider", "")) not in scanned_ok:
            rows.append(dict(old))  # carry forward unchanged; a failed scan withdraws nothing
            continue
        row = dict(old)
        row["withdrawn_at"] = old.get("withdrawn_at") or scan_as_of
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r["provider"]), str(r["listing_id"])))


def build_snapshot(
    listings: dict[str, list[Listing]],
    channels: dict[str, dict[str, Any]],
    previous_rows: list[dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    scan_as_of: str,
    metadata_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current_counts = {provider: len(items) for provider, items in listings.items()}
    # A regression is only claimable for a channel that was SUCCESSFULLY scanned: an
    # UNREACHABLE channel returns [] because it failed, not because the listings vanished.
    scanned = _scanned_ok({"channels": channels})
    regressions = shape_regressions(
        {p: c for p, c in _channel_counts(previous_rows).items() if p in scanned},
        {p: c for p, c in current_counts.items() if p in scanned},
    )
    for provider, info in channels.items():
        info["count"] = current_counts.get(provider, 0)
    return {
        "schema": 1,
        "scan_as_of": scan_as_of,
        "channels": channels,
        "metadata_sources": dict(metadata_sources or {}),
        "shape_regressions": regressions,
        "listings": merge_listings(listings, previous_rows, specs, scan_as_of, scanned_ok=scanned),
    }


# ── identity resolution: Tier 2 curated, Tier 0 publisher-issued, Tier 1 proposed ─────


@dataclass(frozen=True)
class IdentityEntry:
    slug: str
    hugging_face_id: str
    aliases: dict[str, tuple[str, ...]]
    deny: frozenset[str]
    # A curated models.dev pointer for a listing whose publisher id differs from the
    # models.dev key. This is the sanctioned alias path for metadata (never a fuzzy match).
    modelsdev_provider: str = ""
    modelsdev_id: str = ""


@dataclass(frozen=True)
class IdentityMap:
    entries: dict[str, IdentityEntry] = field(default_factory=dict)
    deny: frozenset[str] = frozenset()
    # version slug -> canonical version slug. The committed corpora key on `model_version`, so a
    # listing whose version is already a channel-invariant slug (and thus cannot be reached by a
    # listing-id alias) is merged here instead. See the `version_aliases:` block in
    # `model_identity.yaml`; consumed by `benchmark.routing.model_validity`.
    version_aliases: dict[str, str] = field(default_factory=dict)

    def is_denied(self, listing_id: str) -> bool:
        return listing_id in self.deny


@dataclass(frozen=True)
class Join:
    identity: str
    evidence: str
    members: tuple[tuple[str, str], ...]


def load_identity(path: Path = IDENTITY_PATH) -> IdentityMap:
    payload = yaml.safe_load(path.read_text()) or {}
    entries: dict[str, IdentityEntry] = {}
    deny: set[str] = {str(x) for x in (payload.get("deny") or [])}
    for slug, body in (payload.get("models") or {}).items():
        body = body or {}
        aliases = {
            str(provider): tuple(str(item) for item in (ids or []))
            for provider, ids in (body.get("aliases") or {}).items()
        }
        entry_deny = frozenset(str(x) for x in (body.get("deny") or []))
        deny |= set(entry_deny)
        modelsdev = body.get("modelsdev") or {}
        if not isinstance(modelsdev, Mapping):
            modelsdev = {}
        entries[str(slug)] = IdentityEntry(
            slug=str(slug),
            hugging_face_id=str(body.get("hugging_face_id") or ""),
            aliases=aliases,
            deny=entry_deny,
            modelsdev_provider=str(modelsdev.get("provider") or ""),
            modelsdev_id=str(modelsdev.get("id") or ""),
        )
    return IdentityMap(
        entries=entries,
        deny=frozenset(deny),
        version_aliases={
            str(slug): str(canonical)
            for slug, canonical in (payload.get("version_aliases") or {}).items()
        },
    )


def canonical_slug(raw: str) -> str:
    """The one true bare slug for a listing id, HF id, or legacy version string.

    Grammar: strip a leading ``@cf/`` serving prefix and any leading ``publisher/``
    namespace; strip a promo suffix (``:free``/``-free``/``-explabs``/…); lowercase; join
    ``_`` and ``:`` as ``-``; then apply the CURATED overrides for ambiguous spellings.
    Dots are PRESERVED — ``nemotron-3.5-lightning`` and ``llama-3.3-70b`` are the official
    slugs, and canonicalising them to dashes would orphan the committed corpora that key on
    them. The result carries no provider, channel, casing or promo marker.
    """
    slug = str(raw).strip()
    if slug.startswith("@cf/"):
        slug = slug[len("@cf/") :]
    if "/" in slug:
        slug = slug.rsplit("/", 1)[-1]
    slug = slug.lower()
    changed = True
    while changed:
        changed = False
        for suffix in _PROMO_SUFFIXES:
            if slug.endswith(suffix):
                slug = slug[: -len(suffix)]
                changed = True
                break
    slug = slug.replace("_", "-").replace(":", "-")
    return CANONICAL_OVERRIDES.get(slug, slug)


def tier0_joins(rows: list[dict[str, Any]]) -> list[Join]:
    """Auto-propose a join ONLY on a non-empty shared publisher-issued hugging_face_id."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        hf_id = str(row.get("hugging_face_id") or "")
        if not hf_id:
            continue
        groups.setdefault(hf_id, []).append((str(row["provider"]), str(row["listing_id"])))
    joins: list[Join] = []
    for hf_id, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        evidence = (
            f"hugging_face_id={hf_id} shared by {', '.join(sorted(lid for _, lid in members))}"
        )
        joins.append(Join(canonical_slug(hf_id), evidence, tuple(sorted(members))))
    return joins


def tier1_normalise(listing_id: str) -> str:
    """Tier-1 candidate: the bare canonical slug for a listing id (see :func:`canonical_slug`)."""
    return canonical_slug(listing_id)


def _fact_hash(row: dict[str, Any]) -> str:
    blob = json.dumps({key: row.get(key) for key in _FACT_FIELDS}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _entry_hash(
    rows: dict[tuple[str, str], dict[str, Any]], members: tuple[tuple[str, str], ...]
) -> str:
    hashes = sorted(_fact_hash(rows[key]) for key in members)
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16]


def _proposal_entry(
    identity: str,
    tier: int,
    evidence: str,
    members: tuple[tuple[str, str], ...],
    rows: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tier": tier,
        "evidence": evidence,
        "content_hash": _entry_hash(rows, members),
        "listings": [f"{provider}:{listing_id}" for provider, listing_id in members],
    }


def _member_pairs(entry: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in entry.get("listings") or []:
        provider, _, listing_id = str(item).partition(":")
        pairs.append((provider, listing_id))
    return pairs


def propose(snapshot: dict[str, Any], identity: IdentityMap, scan_as_of: str) -> dict[str, Any]:
    """Resolve the snapshot into confirmed/proposed/dropped; Tier 1 never leaves `proposed`."""
    rows = [r for r in snapshot.get("listings", []) if not r.get("withdrawn_at")]
    by_key = {(str(r["provider"]), str(r["listing_id"])): r for r in rows}
    confirmed: dict[str, Any] = {}
    claimed: set[tuple[str, str]] = set()
    for slug, entry in identity.entries.items():
        members = tuple(
            sorted(
                (provider, listing_id)
                for provider, ids in entry.aliases.items()
                for listing_id in ids
                if (provider, listing_id) in by_key
            )
        )
        if not members:
            continue
        confirmed[slug] = _proposal_entry(slug, 2, f"model_identity.yaml:{slug}", members, by_key)
        claimed.update(members)

    remaining = [
        r
        for r in rows
        if not identity.is_denied(str(r["listing_id"]))
        and (str(r["provider"]), str(r["listing_id"])) not in claimed
    ]
    proposed: dict[str, Any] = {}
    tier0_claimed: set[tuple[str, str]] = set()
    for join in tier0_joins(remaining):
        proposed[join.identity] = _proposal_entry(
            join.identity, 0, join.evidence, join.members, by_key
        )
        tier0_claimed.update(join.members)

    groups: dict[str, list[tuple[str, str]]] = {}
    for row in remaining:
        key = (str(row["provider"]), str(row["listing_id"]))
        if key in tier0_claimed:
            continue
        groups.setdefault(tier1_normalise(str(row["listing_id"])), []).append(key)
    for slug, group_members in sorted(groups.items()):
        if len(group_members) < 2 or slug in confirmed or slug in proposed:
            continue
        evidence = f"normalised slug {slug} shared by {len(group_members)} listings"
        proposed[slug] = _proposal_entry(slug, 1, evidence, tuple(sorted(group_members)), by_key)

    denied_ids = {
        str(r["listing_id"])
        for r in snapshot.get("listings", [])
        if identity.is_denied(str(r["listing_id"]))
    }
    all_rows = snapshot.get("listings", [])
    dropped = [
        {
            "listing_id": str(r["listing_id"]),
            "provider": str(r["provider"]),
            "reason": "deny" if str(r["listing_id"]) in denied_ids else "withdrawn",
            "last_seen": str(r.get("last_seen") or scan_as_of),
        }
        for r in all_rows
        if str(r["listing_id"]) in denied_ids or r.get("withdrawn_at")
    ]
    return {
        "schema": 1,
        "scan_as_of": scan_as_of,
        "confirmed": confirmed,
        "proposed": proposed,
        "dropped": dropped,
    }


class ApplyRefusedError(RuntimeError):
    """An apply that is not backed by the reviewed proposal is refused before any write."""


class ShapeRegressionError(RuntimeError):
    """A channel fell to zero listings; apply is blocked until a human sees the scan."""


def guard_shape(snapshot: dict[str, Any]) -> None:
    """SHAPE_REGRESSION blocks `--apply` and never blocks the scan itself."""
    regressed = list(snapshot.get("shape_regressions") or [])
    if regressed:
        raise ShapeRegressionError("SHAPE_REGRESSION: " + ", ".join(regressed))


def _find_entry(proposal: dict[str, Any], identity: str) -> dict[str, Any] | None:
    for section in ("confirmed", "proposed"):
        entry = (proposal.get(section) or {}).get(identity)
        if entry is not None:
            return entry
    return None


def guard_apply(proposal: dict[str, Any], current_hashes: dict[str, str]) -> None:
    """Refuse any identity absent from the reviewed proposal or whose content hash moved."""
    for identity, current in sorted(current_hashes.items()):
        entry = _find_entry(proposal, identity)
        if entry is None:
            raise ApplyRefusedError(f"{identity!r}: absent from the reviewed proposal")
        if str(entry.get("content_hash")) != str(current):
            raise ApplyRefusedError(
                f"{identity!r}: content hash moved "
                f"({entry.get('content_hash')} -> {current}); re-review the proposal"
            )


def apply_proposal(
    proposal: dict[str, Any], current_hashes: dict[str, str]
) -> dict[str, list[tuple[str, str]]]:
    """Guard, then return only the CONFIRMED identities to write into the overlay."""
    guard_apply(proposal, current_hashes)
    return {
        identity: _member_pairs(entry)
        for identity, entry in sorted((proposal.get("confirmed") or {}).items())
    }


def apply_admitted(
    proposal: dict[str, Any], current_hashes: dict[str, str]
) -> dict[str, list[tuple[str, str]]]:
    """Guard, then return CONFIRMED and PROPOSED identities — the reviewed admission set.

    `--apply` uses this rather than :func:`apply_proposal` so a newly-discovered identity the
    reviewed proposal carries (Tier 0 on a shared publisher id, Tier 1 on a normalised
    candidate) is admitted alongside the curated Tier-2 aliases. The guard is unchanged: an
    identity absent from the reviewed proposal or whose content hash moved is refused before
    any write.
    """
    guard_apply(proposal, current_hashes)
    admitted: dict[str, list[tuple[str, str]]] = {}
    for section in ("confirmed", "proposed"):
        for identity, entry in sorted((proposal.get(section) or {}).items()):
            admitted[identity] = _member_pairs(entry)
    return admitted


def admitted_identities(
    snapshot: Mapping[str, Any],
    proposal: Mapping[str, Any],
    identity: IdentityMap | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Every ADMITTED listing of the snapshot, keyed by its resolved identity (`version`).

    The discovery→campaign join. Group participants are named by the reviewed proposal (the
    deliberate cross-provider identity); a schedulable, active, non-denied listing the proposal
    does not group is its own identity (`version` = listing id), so a genuinely new singleton
    still enters the overlay. A row that is not admitted (`withdrawn_at`, `schedulable: false`,
    `free_access: false`) is dropped here, before any price is considered; the price itself is
    enforced by :func:`merge_overlay`.
    """
    active = {
        (str(row["provider"]), str(row["listing_id"])): row
        for row in snapshot.get("listings", [])
        if _is_admitted(row)
    }
    resolved: dict[str, list[tuple[str, str]]] = {}
    claimed: set[tuple[str, str]] = set()
    for section in ("confirmed", "proposed"):
        for slug, entry in sorted((proposal.get(section) or {}).items()):
            members = [pair for pair in _member_pairs(entry) if pair in active]
            if not members:
                continue
            # Fold the reviewed proposal's slug through the canonical grammar before keying the
            # identity. A proposal is regenerated on a network scan, not on every commit, so it
            # can outlive a canonicalisation fix and still carry a stale key
            # (e.g. `ling-3-0-flash-fin-free`); re-canonicalising here keeps one identity per
            # weights set instead of splitting it across the proposal's old and new spellings.
            key = canonical_slug(str(slug))
            bucket = resolved.setdefault(key, [])
            for pair in members:
                if pair not in bucket:
                    bucket.append(pair)
            claimed.update(members)
    for pair in sorted(active):
        if pair in claimed:
            continue
        if identity is not None and identity.is_denied(pair[1]):
            continue
        # A schedulable listing the reviewed proposal does not group is its own identity —
        # the bare canonical slug of its listing id, never the raw channel id.
        bare = canonical_slug(pair[1])
        resolved.setdefault(bare, [])
        if pair not in resolved[bare]:
            resolved[bare].append(pair)
    return resolved


def _positive_price(value: Any) -> dict[str, float] | None:
    """Coerce an ``{"input", "output"}`` price map, or None unless BOTH axes are > 0.

    This is HARD RULE 2's single enforcement point. ``_real_list_price`` reads it from a
    snapshot row's models.dev ``cost``; ``merge_overlay`` reads a caller-supplied ``prices``
    map through it too, so a direct call can never publish a ``$0`` input or output list
    price regardless of the arguments it is handed.
    """
    if not isinstance(value, Mapping):
        return None
    try:
        inp = float(value["input"])
        out = float(value["output"])
    except (KeyError, TypeError, ValueError):
        return None
    if inp <= 0 or out <= 0:
        return None
    return {"input": inp, "output": out}


def _catalog_price(row: Mapping[str, Any]) -> dict[str, float] | None:
    """A listing's own catalogue list price, or None unless BOTH axes are strictly positive.

    The published-price fallback for a lane models.dev does not carry (SambaNova, Requesty):
    its catalogue quotes a real per-1M list price, which is exactly the paid twin's rate the
    overlay must record. The keys differ from models.dev's ``cost`` (``input_cost_per_1m`` /
    ``output_cost_per_1m`` vs ``input`` / ``output``), so the conversion is explicit. A ``$0``
    on either axis is refused by :func:`_positive_price`, the same single enforcement point.
    """
    prices = row.get("prices")
    if not isinstance(prices, Mapping):
        return None
    return _positive_price(
        {
            "input": prices.get("input_cost_per_1m"),
            "output": prices.get("output_cost_per_1m"),
        }
    )


def _resolved_price(
    row: Mapping[str, Any], default_source: str
) -> tuple[dict[str, float] | None, str]:
    """A listing's real paid-twin list price and its source, or ``(None, default)``.

    Both axes must be strictly positive: HARD RULE 2 forbids a $0 list price on either, and
    a $0/missing price (an unpriced free lane) is NOT a price we may write — the identity is
    left out rather than inventing one. models.dev is the preferred source (the canonical paid
    listing, labelled ``default_source``); the catalogue's own list price is the fallback for
    a lane models.dev omits, labelled by its provider so provenance stays truthful.
    """
    cost = _positive_price(row.get("cost"))
    if cost is not None:
        return cost, default_source
    catalog = _catalog_price(row)
    if catalog is None:
        return None, default_source
    provider = str(row.get("provider") or "")
    return catalog, (f"catalog:{provider}" if provider else "catalog")


def _real_list_price(row: Mapping[str, Any]) -> dict[str, float] | None:
    """The numeric form of :func:`_resolved_price` (source dropped)."""
    return _resolved_price(row, "https://models.dev/api.json")[0]


def _is_admitted(row: Mapping[str, Any]) -> bool:
    """True iff a snapshot row may enter the collection overlay.

    The static half of the admission gate: the listing is active (not `withdrawn_at`),
    schedulable (the snapshot's `free_access` marker, which already names a non-free channel),
    and not explicitly `free_access: false`. A listing the catalogue EXPLICITLY declares
    tool-less (`supports_tools is False`) is refused here; an UNKNOWN tool surface (``None``,
    the catalogue says nothing) is admitted, because the live tool-call probe is its gate. A
    row with neither marker is admitted (the direct callers of :func:`merge_overlay` predate
    the snapshot's markers), so the filter only ever removes a listing that said no.
    """
    if row.get("withdrawn_at"):
        return False
    if row.get("supports_tools") is False:
        return False
    if row.get("schedulable") is False:
        return False
    limits = row.get("published_limits")
    return not (isinstance(limits, Mapping) and limits.get("free_access") is False)


def _require_price_as_of(scan_as_of: str) -> str:
    """Provenance is mandatory on a priced row; refuse to write a blank ``price_as_of``."""
    if not scan_as_of.strip():
        raise ValueError(
            "merge_overlay refuses to write a priced row without price_as_of: pass a "
            "non-empty scan_as_of (HARD RULE 2 requires provenance)"
        )
    return scan_as_of


def _slugify(identity: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", identity.lower()).strip("-")


def _new_model_name(models: Mapping[str, Any], provider: str, identity: str) -> str:
    base = f"{provider}-{_slugify(identity)}".strip("-") or "model"
    name = base
    suffix = 2
    while name in models:
        name = f"{base}-{suffix}"
        suffix += 1
    return name


def merge_overlay(
    overlay: dict[str, Any],
    resolved: dict[str, list[tuple[str, str]]],
    *,
    rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    prices: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
    scan_as_of: str = "",
    price_source: str = "https://models.dev/api.json",
) -> dict[str, Any]:
    """Stamp `version` on every resolved listing, ADD a missing admitted row, UPDATE a price.

    A new row is written only when the listing is an active, admitted snapshot row AND a real
    paid-twin list price is known (`_positive_price`, both axes > 0) — read from a validated
    ``prices`` entry, the row's models.dev ``cost``, or its catalogue ``prices``; a ``$0`` on
    either axis is refused on every path, so this helper can never emit one. A non-schedulable
    or ``free_access: false`` row is skipped before any write. An existing admitted row keeps
    its provenance but has its list price refreshed when the resolved price CHANGED (a
    no-change apply writes nothing, so the operation is idempotent). Otherwise the identity
    stays out of the overlay: no join and no price is invented. Rows never carry a
    `cache_read_cost_per_1m` (HARD RULE 3) and always carry a non-empty `price_source` /
    `price_as_of` (HARD RULE 2's provenance; a blank `scan_as_of` refuses the write). With
    neither `rows` nor `prices` this is the version-stamp-only path, exactly as before.
    """
    models: dict[str, Any] = overlay.setdefault("models", {})
    providers = overlay.get("providers") or {}
    existing_by_key: dict[tuple[str, str], Any] = {}
    for row in models.values():
        if isinstance(row, Mapping):
            key = (str(row.get("provider", "")), str(row.get("lane") or row.get("model_id", "")))
            existing_by_key[key] = row
    for identity, pairs in resolved.items():
        for provider, listing_id in pairs:
            pair = (provider, listing_id)
            explicit = _positive_price((prices or {}).get(pair))
            source = price_source
            pricing = explicit
            if rows is not None:
                row = rows.get(pair)
                if row is None or not _is_admitted(row):
                    continue
                if explicit is None:
                    pricing, source = _resolved_price(row, price_source)
            existing = existing_by_key.get(pair)
            if existing is not None:
                existing["version"] = identity
                existing["model"] = identity
                existing["model_id"] = identity
                if pricing is not None and _pricing_differs(existing, pricing):
                    _stamp_price(existing, pricing, scan_as_of, source, provider)
                continue
            if rows is None or provider not in providers or pricing is None:
                continue
            _require_price_as_of(scan_as_of)
            name = _new_model_name(models, provider, identity)
            models[name] = {
                # `lane` is the provider's RAW channel listing (the wire id); the three id
                # fields are the bare canonical identity the corpus groups on.
                "lane": listing_id,
                "model": identity,
                "model_id": identity,
                "provider": provider,
                "version": identity,
                "pricing": {
                    "input_cost_per_1m": pricing["input"],
                    "output_cost_per_1m": pricing["output"],
                    "price_provider": provider,
                    "price_source": source,
                    "price_as_of": scan_as_of,
                    "price_note": (
                        "Collection-only free lane. List price is the paid twin's published "
                        f"rate for the same weights ({pricing['input']}/"
                        f"{pricing['output']} per 1M). No cache_read rate (HARD RULE 3)."
                    ),
                },
            }
    return overlay


def _pricing_differs(existing: Mapping[str, Any], pricing: Mapping[str, float]) -> bool:
    """True when an existing row's stored input/output price moved from the resolved price."""
    block = existing.get("pricing")
    if not isinstance(block, Mapping):
        return True
    try:
        stored = (float(block["input_cost_per_1m"]), float(block["output_cost_per_1m"]))
    except (KeyError, TypeError, ValueError):
        return True
    return stored != (pricing["input"], pricing["output"])


def _stamp_price(
    existing: dict[str, Any],
    pricing: Mapping[str, float],
    scan_as_of: str,
    price_source: str,
    provider: str,
) -> None:
    """Refresh a changed list price in place, keeping HARD RULES 2 and 3 intact."""
    _require_price_as_of(scan_as_of)
    block = existing.setdefault("pricing", {})
    block["input_cost_per_1m"] = pricing["input"]
    block["output_cost_per_1m"] = pricing["output"]
    block["price_provider"] = provider
    block["price_source"] = price_source
    block["price_as_of"] = scan_as_of
    block.pop("cache_read_cost_per_1m", None)


def _scanned_ok(snapshot: Mapping[str, Any]) -> set[str]:
    """Providers whose channel was successfully scanned (an UNREACHABLE one proves nothing)."""
    return {
        str(provider)
        for provider, info in (snapshot.get("channels") or {}).items()
        if str((info or {}).get("status", "")).startswith("ok")
    }


def withdrawn_rows(
    overlay: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    active_rows: Mapping[tuple[str, str], Any],
) -> list[str]:
    """Return the overlay rows whose SCANNED channel no longer serves them — WITHOUT deleting.

    A listing absent from a scan is recorded `withdrawn_at` in the snapshot (the mark) and is
    never deleted from the overlay here: the row keeps its price/identity provenance, so a
    re-listed identity resumes with history and no collected row is ever lost. The campaign's
    runnable set is built from the active snapshot rows, so the withdrawn lane simply stops
    being scheduled. A channel that was UNREACHABLE proves nothing, so its rows are not
    reported.
    """
    scanned_ok = _scanned_ok(snapshot)
    names: list[str] = []
    models: Mapping[str, Any] = overlay.get("models") or {}
    for name, row in models.items():
        if not isinstance(row, Mapping):
            continue
        provider = str(row.get("provider", ""))
        key = (provider, str(row.get("lane") or row.get("model_id", "")))
        if provider in scanned_ok and key not in active_rows:
            names.append(str(name))
    return sorted(names)


# ── archiving a (model, provider) entry: retained for provenance, withheld from scheduling ──


def is_archived(row: Mapping[str, Any]) -> bool:
    """True iff an overlay row has been archived (and so must never be scheduled)."""
    return bool(row.get("archived"))


def archive_entry(
    overlay: dict[str, Any],
    provider: str,
    model_id: str,
    *,
    reason: str,
    archived_at: str,
) -> dict[str, Any]:
    """Mark ONE (model, provider) overlay row archived, in place, and return it.

    Archiving is the scheduling half of a retirement, not a deletion: the row keeps its price
    and identity provenance and stays in the overlay/corpus, while the campaign's runnable set
    drops it (see `refresh_free_campaign.runnable_lanes`). Scoping to ``(provider, model_id)``
    is deliberate — one channel dropping a model must not archive the same weights on another
    channel that still serves them. A non-empty *reason* and *archived_at* are required so a
    provenance row never carries an unattributed archive. Raises ``KeyError`` when nothing
    matches, so a typo cannot silently archive nothing. Idempotent.
    """
    if not reason.strip():
        raise ValueError("archive_entry requires a non-empty archive_reason")
    if not archived_at.strip():
        raise ValueError("archive_entry requires a non-empty archived_at")
    models = overlay.get("models") or {}
    for _name, row in models.items():
        if not isinstance(row, dict):
            continue
        same_provider = str(row.get("provider", "")) == provider
        channel = str(row.get("lane") or row.get("model_id", ""))
        if same_provider and channel == model_id:
            row["archived"] = True
            row["archived_at"] = archived_at
            row["archive_reason"] = reason
            return row
    raise KeyError(f"no overlay row for provider={provider!r} lane={model_id!r}")


# ── CLI ──────────────────────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(_write_yaml_body(payload))


def _overlay_header(text: str) -> str:
    """The leading comment block, kept verbatim so the HARD RULES header survives a rewrite."""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            return "".join(lines[:index])
    return text


def scan_snapshot(out: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Fetch every channel and build the facts-only snapshot; writes nothing.

    The single network step (public catalogue GETs only, no completion). Exposed so a
    long-running campaign can refresh in-process and a dry run can print the snapshot.
    """
    document = yaml.safe_load(CATALOGS_PATH.read_text()) or {}
    specs = document.get("providers") or {}
    meta_specs = document.get("metadata_sources") or {}
    identity = load_identity(IDENTITY_PATH)
    scan_as_of = datetime.now(UTC).date().isoformat()
    previous_rows = _previous_listings(out)
    listings, channels = scan_channels(specs)
    metadata_sources: dict[str, Any] = {}
    md_spec = meta_specs.get(MODELSDEV_SOURCE)
    if isinstance(md_spec, Mapping):
        index, status = scan_metadata_sources(md_spec)
        listings = attach_metadata(listings, index, md_spec, identity)
        metadata_sources[MODELSDEV_SOURCE] = status
    snapshot = build_snapshot(
        listings, channels, previous_rows, specs, scan_as_of, metadata_sources
    )
    for provider, info in sorted(channels.items()):
        print(f"{provider:22s} {info['status']}", file=sys.stderr)
    for source, info in sorted(metadata_sources.items()):
        print(f"{source:22s} {info['status']}", file=sys.stderr)
    print(f"{len(snapshot['listings'])} listings", file=sys.stderr)
    if snapshot["shape_regressions"]:
        print("SHAPE_REGRESSION: " + ", ".join(snapshot["shape_regressions"]), file=sys.stderr)
    return snapshot


def write_snapshot(snapshot: Mapping[str, Any], out: Path = SNAPSHOT_PATH) -> None:
    """Persist the snapshot JSON, refusing when a SHAPE_REGRESSION would poison the record.

    The guard belongs at the WRITE, not only at `--apply`: a snapshot that records a channel
    falling to zero is a human-review event, and persisting it first lets the next scan treat
    the regressed shape as the baseline. Blocking the write keeps the last good snapshot in
    place (and, in `refresh --write`, stops the proposal/apply that follows it).
    """
    guard_shape(dict(snapshot))
    _write_json(out, dict(snapshot))


def write_proposal(
    snapshot: Mapping[str, Any], identity: IdentityMap | None = None
) -> dict[str, Any]:
    """Derive and persist the reviewed identity proposal; returns it for in-process reuse."""
    identity = identity or load_identity(IDENTITY_PATH)
    proposal = propose(dict(snapshot), identity, str(snapshot.get("scan_as_of") or ""))
    _write_yaml(PROPOSAL_PATH, proposal)
    return proposal


def apply_snapshot(identity: IdentityMap, scan_as_of: str) -> int:
    """Guard the reviewed proposal and refresh the overlay from the written snapshot.

    Returns 0 on success and the documented refusal code (2) for SHAPE_REGRESSION, an
    unreviewed/moved proposal, or an invalid regenerated overlay — never a traceback.
    """
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    try:
        guard_shape(snapshot)
    except ShapeRegressionError as exc:
        print(f"refusing --apply: {exc}", file=sys.stderr)
        return 2
    proposal = yaml.safe_load(PROPOSAL_PATH.read_text()) or {}
    fresh = propose(snapshot, identity, scan_as_of)
    current_hashes = {key: entry["content_hash"] for key, entry in fresh["confirmed"].items()}
    current_hashes.update({key: entry["content_hash"] for key, entry in fresh["proposed"].items()})
    try:
        apply_admitted(proposal, current_hashes)
    except ApplyRefusedError as exc:
        print(f"refusing --apply: {exc}", file=sys.stderr)
        return 2
    resolved = admitted_identities(snapshot, proposal, identity)
    active_rows = {
        (str(row["provider"]), str(row["listing_id"])): row
        for row in snapshot.get("listings", [])
        if not row.get("withdrawn_at")
    }
    overlay = yaml.safe_load(OVERLAY_PATH.read_text()) or {}
    # No ``prices`` override: `merge_overlay` reads each row's own models.dev cost (preferred)
    # or catalogue price, so the recorded `price_source` names the real origin.
    merged = merge_overlay(
        overlay,
        resolved,
        rows=active_rows,
        scan_as_of=scan_as_of,
    )
    withdrawn = withdrawn_rows(merged, snapshot, active_rows)
    try:
        parse_registry(merged)
    except ValueError as exc:
        print(f"refusing --apply: regenerated overlay is invalid: {exc}", file=sys.stderr)
        return 2
    original = OVERLAY_PATH.read_text()
    OVERLAY_PATH.write_text(_overlay_header(original) + _write_yaml_body(merged))
    print(
        f"applied {len(resolved)} identities to {OVERLAY_PATH} "
        f"({len(withdrawn)} withdrawn row(s) marked, not deleted)",
        file=sys.stderr,
    )
    return 0


def _write_yaml_body(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the snapshot, write nothing")
    mode.add_argument("--propose", action="store_true", help="write free_models_proposal.yaml")
    mode.add_argument(
        "--apply", action="store_true", help="guard the proposal, refresh overlay identities"
    )
    parser.add_argument("--out", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args(argv)

    identity = load_identity(IDENTITY_PATH)
    scan_as_of = datetime.now(UTC).date().isoformat()
    if args.apply:
        return apply_snapshot(identity, scan_as_of)

    snapshot = scan_snapshot(args.out)
    if args.dry_run:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return 0
    try:
        write_snapshot(snapshot, args.out)
    except ShapeRegressionError as exc:
        print(f"refusing to write snapshot: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}", file=sys.stderr)
    if args.propose:
        write_proposal(snapshot, identity)
        print(f"wrote {PROPOSAL_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
