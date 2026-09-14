from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from shunt.models.config import (
    ModelConfig,
    Pricing,
    Provider,
    ReasoningConfig,
    Registry,
    default_registry_path,
    load_registry,
    resolve_models,
)
from shunt.models.config import (
    arm_api_params as _resolve_arm_api_params,
)

_config: dict | None = None
_pricing: dict | None = None
_free_registry: dict | None = None
_free_registry_path_override: str | None = None

# The non-shipped free-model OVERLAY registry. It holds free/discounted provider rows and
# their model ids as COLLECTION-ONLY entries, and is loaded ONLY when explicitly configured
# (`run_matrix --free-registry <path>` or the env var below). With no overlay configured,
# every code path in this module is byte-identical to before: the shipped `_pricing_path()`,
# the enabled set and the live pool never see a free row.
FREE_REGISTRY_ENV: Final[str] = "SHUNT_FREE_REGISTRY"

# Cost-weighted p(arm|model) fractions, indexed by within-model
# rank (index 0 = cheapest rank). Decreasing so a cheaper arm samples more often
# than a pricier one; overridable via `arm_sampling.weights` in benchmark.yaml.
DEFAULT_ARM_SAMPLING_WEIGHTS: Final[tuple[float, ...]] = (0.5, 0.35, 0.25)

# Legacy literal every pre-arm-hash results.csv row was written under (mirrors
# `benchmark.routing.integrity.DEFAULT_REASONING` — duplicated here, not imported,
# because integrity.py depends on this module and not vice versa).
_LEGACY_DEFAULT_REASONING: Final[str] = "default"


def load(path: str | Path | None = None) -> dict:
    global _config  # noqa: PLW0603, SH001 (module load-once config cache)
    if path is None:
        path = Path(__file__).resolve().parent / "benchmark.yaml"
    with open(path) as f:
        _config = yaml.safe_load(f)
    return _config


def get() -> dict:
    global _config  # noqa: SH001 (reads module config cache)
    if _config is None:
        return load()
    return _config


def _pricing_path() -> Path:
    """Path to the unified registry (packaged with the router, not benchmark-local)."""
    return default_registry_path()


def _flatten(model: ModelConfig, pricing: Pricing) -> dict:
    """Flatten one priced registry row into the flat dict benchmark consumers read."""
    return {
        "provider": model.provider,
        "route": model.route,
        "base_url": model.base_url,
        "api_key_env_var": model.api_key_env_var,
        # True for an anonymous free lane (Kilo Gateway's `:free` ids): the live runner must
        # admit it without a key and let the scaffold inject the harmless placeholder. Carried
        # on the flat row so `infer._model_key_optional` can resolve it per model.
        "key_optional": model.key_optional,
        # `version` is a model-identity attribute (a sibling of tier/provider), no
        # longer a pricing field; a priced model always carries one (schema-enforced).
        "version": model.version,
        # The provider's raw channel listing when it differs from the canonical `model_id`
        # (the free overlay's `lane`), and the bare canonical identity. `route` above already
        # prefers `lane`, so the wire call stays correct; they are surfaced for provenance.
        "lane": model.lane,
        "model": model.model,
        # Where the weights actually run. Carried here so the live runner can LABEL every
        # latency it records without re-parsing the registry per cell: a local batch-1
        # second and a hosted batched second are not the same measurement, and an
        # unlabelled timing is one that could later be pooled with its opposite.
        "serving_mode": model.serving_mode,
        **pricing.model_dump(exclude_none=True),
    }


def load_pricing(path: str | Path | None = None) -> dict:
    """Priced models from the registry, keyed by name. Unpriced models are absent.

    A model without a `pricing` block is routable but invisible here, so it can
    never enter a cost comparison with a fabricated price.
    """
    global _pricing  # noqa: PLW0603, SH001 (module load-once pricing cache)
    if _pricing is not None:
        return _pricing
    registry = load_registry(path if path else _pricing_path())
    _pricing = {
        name: _flatten(model, model.pricing)
        for name, model in resolve_models(registry).items()
        if model.pricing is not None
    }
    return _pricing


# ── non-shipped free-model overlay ────────────────────────────────────────────
def set_free_registry(path: str | Path | None) -> None:
    """Configure the overlay registry path (``None`` clears it) and reset the cache.

    Called by ``run_matrix`` from ``--free-registry``; tests call it directly. The overlay
    is opt-in: unset means no free row is loadable, which is the byte-identical default.
    """
    global _free_registry, _free_registry_path_override  # noqa: PLW0603, SH001
    _free_registry_path_override = str(path) if path else None
    _free_registry = None


def free_registry_path() -> Path | None:
    """The configured overlay path: the ``set_free_registry`` override, else the env var."""
    raw = _free_registry_path_override or os.environ.get(FREE_REGISTRY_ENV)
    if not raw:
        return None
    return Path(raw)


def load_free_registry() -> Registry | None:
    """The parsed overlay registry, or None when unconfigured or the file is absent."""
    path = free_registry_path()
    if path is None or not path.exists():
        return None
    return load_registry(path)


def free_registry() -> dict[str, dict]:
    """Flattened priced overlay rows keyed by name; ``{}`` when no overlay is configured.

    These rows are collection-only: they are merged into the pricing view at the run site by
    ``register_collection_models``, never by ``load_pricing`` itself.
    """
    global _free_registry  # noqa: PLW0603, SH001 (module load-once overlay cache)
    if _free_registry is not None:
        return _free_registry
    registry = load_free_registry()
    _free_registry = (
        {
            name: _flatten(model, model.pricing)
            for name, model in resolve_models(registry).items()
            if model.pricing is not None
        }
        if registry is not None
        else {}
    )
    return _free_registry


def free_registry_ids() -> set[str]:
    """Names of every priced overlay model — the collection-only free namespace."""
    return set(free_registry())


def free_registry_models() -> dict[str, ModelConfig]:
    """Overlay rows resolved to ``ModelConfig`` keyed by name; ``{}`` when unconfigured.

    The typed companion of ``free_registry()`` (the flat dict): the free-lane refusal and
    pre-flight operate on ``ModelConfig``, so callers need the resolved form.
    """
    registry = load_free_registry()
    if registry is None:
        return {}
    return resolve_models(registry)


def is_free_registry_model(name: str) -> bool:
    """True iff *name* is a collection-only overlay row (never enableable)."""
    return name in free_registry()


# ── collection-only channel synthesis ─────────────────────────────────────────
# The Experiential catalog exposes free-promo slugs the registry does not hand-enumerate.
# An `S-explabs` id that is NOT in the registry resolves at runtime to a collection-only
# row (provider explabs, wire slug S, identity S) so ANY catalog slug can be collected via
# `--extra-models` without a registry edit. It is never enabled and never reaches the
# router's live pool; identity-skip then dedupes it against a direct twin whose version is S.
COLLECTION_SUFFIX: Final[str] = "-explabs"
COLLECTION_PROVIDER: Final[str] = "explabs"
# Provenance for a synthesized row whose list price the local catalog does not carry. A $0
# collection row is honest (the promo channel is free); a fabricated nonzero price would not.
COLLECTION_UNKNOWN_SOURCE: Final[str] = "collection-only: list price unknown"
COLLECTION_UNKNOWN_NOTE: Final[str] = "collection-only, list price unknown"


def collection_slug(name: str) -> str | None:
    """The base catalog slug of a collection-only `S-explabs` id, or None when ineligible."""
    if not name.endswith(COLLECTION_SUFFIX):
        return None
    return name[: -len(COLLECTION_SUFFIX)] or None


def catalog_host_rung(slug: str) -> dict | None:
    """The catalog's host rung for *slug* from the local registry, or None if unpublished.

    The direct twin's list price is the host rung the shipped `*-explabs` rows were priced
    from; a slug with no twin (or an unpriced one) has no local host rung to quote.
    """
    return load_pricing().get(slug)


def _collection_provider() -> Provider | None:
    """The `explabs` provider row: the shipped registry first, else the non-shipped overlay.

    The collection-only free-model policy moved the provider out of the shipped registry and
    into the overlay, so the generic `S-explabs` synthesis still resolves when the overlay is
    configured.
    """
    shipped = load_registry(_pricing_path()).providers.get(COLLECTION_PROVIDER)
    if shipped is not None:
        return shipped
    overlay = load_free_registry()
    return overlay.providers.get(COLLECTION_PROVIDER) if overlay is not None else None


def synthesize_collection_model(name: str) -> dict | None:
    """Synthesize a collection-only registry row for an `S-explabs` id, or None.

    None for an ineligible id (not `-explabs`), for an id already in the registry (callers
    use that row unchanged), or when the `explabs` provider is absent. The wire slug and
    identity are both S. Price is the base slug's host rung when the local catalog publishes
    one, else 0 with an explicit unknown note — never fabricated. No `cache_read` key: the
    channel reports no cache-read rate.
    """
    slug = collection_slug(name)
    if slug is None or name in load_pricing() or name in free_registry():
        return None
    provider = _collection_provider()
    if provider is None:
        return None
    rung = catalog_host_rung(slug)
    note = str(rung["price_note"]) if rung and rung.get("price_note") else None
    if rung is None:
        note = COLLECTION_UNKNOWN_NOTE
    pricing = Pricing(
        input_cost_per_1m=float(rung["input_cost_per_1m"]) if rung else 0.0,
        output_cost_per_1m=float(rung["output_cost_per_1m"]) if rung else 0.0,
        price_provider=COLLECTION_PROVIDER,
        price_source=str(rung["price_source"]) if rung else COLLECTION_UNKNOWN_SOURCE,
        price_as_of=str(rung["price_as_of"]) if rung else "",
        price_note=note,
    )
    model = ModelConfig(
        name=name,
        model_id=slug,
        provider=COLLECTION_PROVIDER,
        version=slug,
        base_url=provider.base_url,
        api_key_env_var=provider.api_key_env_var,
        litellm_prefix=provider.litellm_prefix,
        serving_mode="hosted",
        pricing=pricing,
    )
    return _flatten(model, pricing)


def register_collection_models(names: list[str]) -> None:
    """Insert collection-only rows into the pricing view (idempotent).

    A requested id resolves from the non-shipped overlay registry when configured,
    else is synthesized as an `S-explabs` catalog slug. Called at the run site so
    every downstream reader of `load_pricing()` — route resolution, the cache gate,
    `model_versions()` — sees the collection-only row. Raises ValueError for an id that is
    neither an overlay row nor synthesizable.
    """
    pricing = load_pricing()
    overlay = free_registry()
    for name in names:
        if name in pricing:
            continue
        if name in overlay:
            pricing[name] = overlay[name]
            continue
        synthesized = synthesize_collection_model(name)
        if synthesized is None:
            raise ValueError(
                f"cannot synthesize collection model {name!r}: not a `{COLLECTION_SUFFIX}` "
                "id, not an overlay-registry row, already unpriced, or the `explabs` provider "
                "is not registered"
            )
        pricing[name] = synthesized


def resolved_models() -> dict[str, ModelConfig]:
    """All registry models resolved (name -> ModelConfig), including `reasoning`.

    Unlike `load_pricing()`, this is not filtered to priced models — the
    reasoning bracket is a routing/benchmark-arm concern, not a cost concern.
    """
    return resolve_models(load_registry(_pricing_path()))


def reasoning_configs() -> dict[str, ReasoningConfig | None]:
    """Every registry model's reasoning bracket (arms + default), keyed by name."""
    return {name: model.reasoning for name, model in resolved_models().items()}


def default_arm_ids(models: list[str] | None = None) -> dict[str, str]:
    """Map each model to its declared default reasoning-arm id."""
    # A model with no declared `reasoning` block (or not in the registry at all)
    # falls back to the legacy literal "default" placeholder — the alias every
    # legacy (pre-arm-hash) results.csv row was written under, so cached rows keep resolving.
    cfgs = reasoning_configs()
    names = models if models is not None else list(cfgs.keys())
    result: dict[str, str] = {}
    for name in names:
        cfg = cfgs.get(name)
        result[name] = cfg.default_arm if cfg is not None else _LEGACY_DEFAULT_REASONING
    return result


def arm_sampling_weights() -> list[float]:
    """Per-arm inclusion probabilities by within-model rank — each an independent
    p in [0, 1] (Bernoulli threshold), not a distribution that must sum to 1."""
    cfg = get()
    raw = cfg.get("arm_sampling", {}).get("weights")
    weights = [float(w) for w in raw] if raw else list(DEFAULT_ARM_SAMPLING_WEIGHTS)
    if any(not 0.0 <= w <= 1.0 for w in weights):
        raise ValueError(f"arm_sampling.weights must each be in [0, 1]; got {weights}")
    return weights


def arm_sampling_default_only_models() -> set[str]:
    """Models pinned to their default arm even when the sweep is on (cost control)."""
    cfg = get()
    models = cfg.get("arm_sampling", {}).get("default_only_models") or []
    return {str(m) for m in models}


def arm_api_params(model: str, arm_id: str) -> dict[str, Any]:
    """Verbatim request params for a model's reasoning arm ({} if model unregistered).

    The live executor overlays these so a sampled arm bills a DISTINCT request.
    """
    mc = resolved_models().get(model)
    return _resolve_arm_api_params(mc, arm_id) if mc is not None else {}


def arm_sampling_enabled() -> bool:
    """Gate for the multi-arm sweep — default False if unset."""
    # The live executor overlays each arm's registry API params onto the request
    # (infer._scaffold_model_kwargs), so distinct arms bill distinct requests.
    # False reproduces the default-arm-only behavior from before arm sampling existed.
    cfg = get()
    return bool(cfg.get("arm_sampling", {}).get("enabled", False))


def collect_config() -> dict:
    """The adaptive `collect` run-mode block (defaults reproduce today's full matrix)."""
    cfg = get()
    return dict(cfg.get("collect", {}))


def collect_enabled() -> bool:
    """Gate for the adaptive frontier-collection mode — default False (full matrix)."""
    return bool(collect_config().get("enabled", False))


def concordance_config() -> dict:
    """The `concordance` campaign block (the named cross-provider subset)."""
    return dict(get().get("concordance", {}) or {})


def concordance_fanout_cap() -> int:
    """How many channels of one identity the named subset may collect at a challenge (>= 1)."""
    return max(1, int(concordance_config().get("fanout_cap", 1)))


def concordance_subset_models() -> set[str]:
    """Every channel id named in the concordance subset (dedupe-exempt collection)."""
    channels: set[str] = set()
    for entry in concordance_config().get("subset", []) or []:
        channels.update(str(name) for name in entry.get("channels", []) or [])
    return channels


def concordance_subset_challenges() -> list[str]:
    """The explicit challenge ids the concordance subset is restricted to."""
    return [str(cid) for cid in concordance_config().get("challenges", []) or []]


def lanes_config() -> dict:
    """The ``lanes:`` campaign block: per-lane limits, unknown defaults, and stall timeout.

    Absent (the shipped paid config) means ``{}`` — the runner then builds no lane scheduler, so
    behaviour is byte-identical to before per-lane admission control existed.
    """
    return dict(get().get("lanes", {}) or {})


def lane_limits_config() -> dict[str, dict]:
    """Per-lane raw limit mappings from ``lanes.limits`` (lane name -> mapping)."""
    raw = lanes_config().get("limits", {}) or {}
    return {str(name): dict(limits or {}) for name, limits in raw.items()}


def lane_unknown_limits() -> dict:
    """The ``lanes.unknown_limits`` defaults (rpm/rpd) applied to a lane with no explicit entry."""
    return dict(lanes_config().get("unknown_limits", {}) or {})


def lane_stall_timeout_s() -> float:
    """The pull-loop stall budget in seconds; default 900 (bounds the blocked-lane stall case)."""
    return float(lanes_config().get("stall_timeout_s", 900.0))


_PRIORITY_DEFAULT: Final[str] = "routing/data/model_priority.yaml"


def model_priority_path() -> Path:
    """Path to the model-value priority config (importance allowlist + tunable weights)."""
    return Path(__file__).resolve().parent / _PRIORITY_DEFAULT


def load_model_priority() -> dict:
    """Parse ``benchmark/routing/data/model_priority.yaml``; ``{}`` when the file is absent.

    Data-only: the free-tier collector's value allowlist, denylist and stop knobs live here,
    never as branches in ``benchmark/routing/collection_priority.py``.
    """
    path = model_priority_path()
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


# Fields the per-lane limits registry may carry; anything else in a block is provenance
# (scope/notes/verified_*) and is not an enforceable gate. `free_access`/`access_note` are
# the access marker: `free_access: false` makes the scheduler refuse the lane (LANE_NO_FREE_ACCESS).
_LIMIT_FIELDS: Final[tuple[str, ...]] = (
    "rpm",
    "rpd",
    "tpm",
    "max_request_tokens",
    "daily_token_budget",
    "free_access",
    "access_note",
    # A time-boxed free promo's expiry, carried from the listing's `expiration_date` (or a
    # registry/`lanes.limits` override). `lane_scheduler._EXPIRED` refuses the lane once now
    # passes it, so a lapsed promo can never be scheduled.
    "expires_at",
)
_LANE_REGISTRY_DEFAULT: Final[str] = "routing/data/provider_limits.yaml"
_FREE_CATALOGS_DEFAULT: Final[str] = "routing/data/free_catalogs.yaml"
_lane_registry: dict | None = None
_free_provider_access: dict[str, str | None] | None = None


def lane_registry_path() -> Path:
    """The per-lane limits registry path from ``lanes.registry`` (config-file relative)."""
    rel = lanes_config().get("registry", _LANE_REGISTRY_DEFAULT)
    return (Path(__file__).resolve().parent / str(rel)).resolve()


def free_catalogs_path() -> Path:
    """The declared free-provider catalog (the scanner's input) the scheduler enforces against."""
    return (Path(__file__).resolve().parent / _FREE_CATALOGS_DEFAULT).resolve()


def free_provider_access() -> dict[str, str | None]:
    """Every provider in ``free_catalogs.yaml`` mapped to its free-lane status.

    ``None`` marks a DECLARED free provider (``free_tier.free_access`` not false); a string is
    the ``free_access: false`` evidence note. A provider ABSENT from this mapping is not a
    declared free provider at all, so a lane on it must be refused as ``no-free-lane`` — the
    catalogues, not the overlay, are the source of truth for what may spend.
    """
    global _free_provider_access  # noqa: PLW0603, SH001 (module load-once registry cache)
    if _free_provider_access is not None:
        return _free_provider_access
    path = free_catalogs_path()
    document = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    access: dict[str, str | None] = {}
    for provider, spec in (document.get("providers") or {}).items():
        free_tier = (spec or {}).get("free_tier") or {}
        if isinstance(free_tier, Mapping) and free_tier.get("free_access") is False:
            note = str(free_tier.get("access_note") or "").strip()
            access[str(provider)] = note or f"{provider}: free_access is false"
        else:
            access[str(provider)] = None
    _free_provider_access = access
    return access


def is_declared_free_provider(provider: str | None) -> bool:
    """True iff *provider* is a declared free lane in ``free_catalogs.yaml``."""
    declared = free_provider_access()
    return provider is not None and provider in declared and declared[provider] is None


def load_lane_registry() -> dict:
    """The parsed per-lane limits registry; ``{}`` when the declared file is absent."""
    global _lane_registry  # noqa: PLW0603, SH001 (module load-once registry cache)
    if _lane_registry is None:
        path = lane_registry_path()
        _lane_registry = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    return _lane_registry


def _declared_free_access(provider: str | None) -> dict:
    """The declared-free gate as a limits source; ``{}`` for a declared free provider.

    A provider absent from ``free_catalogs.yaml``, or one it marks ``free_access: false``,
    yields ``free_access: false`` plus the evidence note, so the existing
    ``LANE_NO_FREE_ACCESS`` limitation refuses the lane through the one registry rule. A
    PROVIDERLESS lane is refused the same way: with no provider there is no catalogue that can
    vouch for a free tier, so it must fail CLOSED rather than fall through to the scheduler's
    ``free_access=True`` default.
    """
    if not provider:
        return {
            "free_access": False,
            "access_note": "providerless lane: no provider to declare a free tier",
        }
    declared = free_provider_access()
    if provider not in declared:
        return {
            "free_access": False,
            "access_note": f"{provider}: no free lane declared in free_catalogs.yaml",
        }
    note = declared[provider]
    if note is not None:
        return {"free_access": False, "access_note": note}
    return {}


def lane_limits_from_registry(model: str, provider: str | None) -> dict:
    """Merge the registry's defaults, the declared-free gate, the provider block and overrides.

    A lane is a channel id; its provider comes from the resolved overlay row. The result holds
    only fields the scheduler enforces, so provenance keys never leak into ``LaneLimits``. The
    declared-free gate is inserted BEFORE the provider block so an explicit registry
    ``free_access`` still wins, while a non-free provider the registry never named is caught.
    """
    registry = load_lane_registry()
    block = (registry.get("providers") or {}).get(provider or "") or {}
    sources = (
        registry.get("defaults") or {},
        _declared_free_access(provider),
        block,
        (block.get("lanes") or {}).get(model) or {},
        (registry.get("lanes") or {}).get(model) or {},
    )
    resolved: dict = {}
    for source in sources:
        resolved.update({key: source[key] for key in _LIMIT_FIELDS if source.get(key) is not None})
    return resolved


def ladder_config() -> dict:
    """The `ladder` collector block (cold-start budget-reservation knob)."""
    cfg = get()
    return dict(cfg.get("ladder", {}))


def cold_start_tier_cost() -> float:
    """Per-unmeasured-tier reservation (USD); MUST be > 0 or the overspend guard is defeated."""
    # A non-positive value would reserve $0 per unmeasured tier, so a fresh run could admit an
    # unbounded concurrent first wave — reject it loudly rather than silently disarm the guard.
    value = float(ladder_config().get("cold_start_tier_cost", 1.0))
    if value <= 0:
        raise ValueError(f"ladder.cold_start_tier_cost must be > 0, got {value}")
    return value


def live_config() -> dict:
    """The `live` agent-scaffold block (step_limit / wall bounds for a paid run)."""
    cfg = get()
    return dict(cfg.get("live", {}))


def live_step_limit() -> int:
    """PRIMARY model-speed-agnostic per-cell agent-step ceiling (default 150)."""
    return int(live_config().get("step_limit", 150))


def live_cost_limit() -> float:
    """Per-cell USD cost ceiling passed to the agent scaffold (default 4.0; see benchmark.yaml)."""
    # mini-swe-agent's own AgentConfig default is 3.0 — a scaffold default that would
    # otherwise govern paid runs silently (the "undeclared default" this accessor exists to
    # end). A declared key means the cap a run obeys is recorded where the run is configured.
    # The value MOVES WITH ``step_limit``: the two are raised together, so a budget increase
    # cannot relabel step censors as cost censors under a lying label.
    return float(live_config().get("cost_limit", 4.0))


def resume_enabled() -> bool:
    """Gate for per-cell conversation resume — default False (always a fresh start)."""
    # Off by default so an existing run is byte-identical: without `resume.enabled`, a leftover
    # partial conversation on disk is ignored and the cell restarts from zero every window.
    return bool(get().get("resume", {}).get("enabled", False))


def _pricing_dict() -> dict:
    """Return pricing as {model: {input, output}} for every priced registry model."""
    pricing = load_pricing()
    result = {}
    for m, p in pricing.items():
        if not isinstance(p, dict) or m.startswith("_"):
            continue
        result[m] = {
            "input": p.get("input_cost_per_1m", 0),
            "output": p.get("output_cost_per_1m", 0),
        }
    return result


def enabled_models() -> list[str]:
    """Return enabled model names sorted by total list price ascending (name tie-break)."""
    # `models:` is a LIST of enabled names. In-list = enabled; a registry model
    # absent from the list is disabled; a listed name absent from the registry is
    # an unrecoverable config error (a listed model must exist to be routable).
    cfg = get()
    listed = cfg.get("models", [])
    pricing = load_pricing()

    # HARD WALL (collection-only free-model policy): a collection-only free row can never be
    # enabled. It lives in the non-shipped overlay (or the `-explabs` synthesis namespace) and
    # is collectable only through `--extra-models`; enabling one would leak it into
    # `capability_rank`, the pareto
    # axes, the kill gate and the live pool. Refuse before the registry lookup so the message
    # names the real reason rather than "not found".
    free_listed = [
        m for m in listed if is_free_registry_model(m) or str(m).endswith(COLLECTION_SUFFIX)
    ]
    if free_listed:
        raise ValueError(
            "benchmark.yaml cannot enable free/collection-only model(s): "
            f"{free_listed}. Free rows live in the non-shipped overlay registry and are "
            "collectable only through --extra-models; they must never enter the enabled set, "
            "the kill gate or any analysis."
        )

    unregistered = [m for m in listed if m not in pricing]
    if unregistered:
        raise ValueError(
            "benchmark/benchmark.yaml lists model(s) the benchmark cannot see "
            f"(absent from the registry, or registered without a pricing block): {unregistered}. "
            "A listed model must exist in src/shunt/config/models.yaml with pricing."
        )
    # dict.fromkeys dedupes a repeated list entry while preserving order, so a typo'd
    # duplicate can't make classify_cells enumerate (and pay for) the same cell twice.
    enabled = list(dict.fromkeys(m for m in listed if not m.startswith("_")))

    pricing_dict = _pricing_dict()

    def _sort_key(m: str) -> tuple[float, str]:
        cost = pricing_dict.get(m, {}).get("input", 0) + pricing_dict.get(m, {}).get("output", 0)
        return (cost, m)

    enabled.sort(key=_sort_key)
    return enabled


def price_bands() -> tuple[list[str], list[str], list[str]]:
    """Enabled models split into (cheap, mid, escalation) thirds by ascending price.

    Replaces the hand-assigned tiers with equal price terciles; each list is price-ordered.
    """
    ordered = enabled_models()  # price ascending
    third = max(len(ordered) // 3, 1)
    cheap = ordered[:third]
    escalation = ordered[-third:]
    mid = [m for m in ordered if m not in cheap and m not in escalation]
    return cheap, mid, escalation


def enabled_pricing() -> dict:
    """Return pricing for enabled models only."""
    pricing = _pricing_dict()
    enabled = set(enabled_models())
    return {m: p for m, p in pricing.items() if m in enabled}


def model_has_cache(model: str) -> bool:
    """True iff *model* has a real cache-read discount (not just a caching flag)."""
    info = load_pricing().get(model)
    if not isinstance(info, dict):
        return False
    cr = info.get("cache_read_cost_per_1m")
    inp = info.get("input_cost_per_1m")
    if not isinstance(cr, int | float) or cr <= 0:
        return False
    return not isinstance(inp, int | float) or cr < inp


def models_missing_cache(models: list[str] | None = None) -> list[str]:
    """Enabled (or given) benchmark models that lack a real cache-read discount."""
    names = models if models is not None else enabled_models()
    return [m for m in names if not model_has_cache(m)]


def frontier_model() -> str | None:
    """Model to use as the control baseline for kill gate comparison."""
    cfg = get()
    control = cfg.get("routing", {}).get("control_model")
    if control:
        return control
    # Fallback: most expensive enabled model
    enabled = enabled_models()
    pricing = _pricing_dict()
    if not enabled:
        return None
    return max(
        enabled,
        key=lambda m: pricing.get(m, {}).get("input", 0) + pricing.get(m, {}).get("output", 0),
    )


@dataclass(frozen=True)
class RankedModel:
    """One model's slot in the derived capability order (0 = weakest)."""

    model: str
    default_arm: str
    rank: int
    source: str  # "measured" | "price-prior"


@dataclass(frozen=True)
class ModelEvidence:
    """The measured stat behind a model's rank (reported + audited, never routed)."""

    model: str
    n: int  # real default-arm cells
    pass_rate: float  # marginal p̂
    ci_lo: float
    ci_hi: float
    rank: int
    source: str  # "measured" | "price-prior"
    price: float


@dataclass(frozen=True)
class CapabilityRank:
    """A strict total order over enabled models + the per-model evidence behind it."""

    ordered: list[RankedModel]  # weakest -> strongest
    evidence: dict[str, ModelEvidence]  # by model

    def rank_of(self, model: str) -> int | None:
        for r in self.ordered:
            if r.model == model:
                return r.rank
        return None

    def strongest(self) -> str:
        return self.ordered[-1].model


def capability_rank_config() -> dict:
    """Confidence-gate knobs for the derived rank (pinned in benchmark.yaml)."""
    cfg = get().get("capability_rank", {}) or {}
    return {
        "K": int(cfg.get("K", 20)),
        "W": float(cfg.get("W", 0.35)),
        "min_pairs": int(cfg.get("min_pairs", 8)),
        "baseline": list(cfg["baseline"]) if cfg.get("baseline") else None,
    }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Marginal pass-rate ``k/n`` with a Wilson score CI (point, lo, hi).

    Same math as ``impute.violation_ci`` — duplicated (not imported) because impute
    depends on this module, not the reverse.
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (phat, max(0.0, lo), min(1.0, hi))


def _copeland_scores(
    matrix: dict, models: list[str], min_pairs: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Copeland score + qualifying-peer count per model from co-measured disagreements.

    ``wins[(a, b)]`` = tasks where a passes and b fails; a pair contributes to the score
    only when its disagreement count (both directions) reaches ``min_pairs``.
    """
    wins: dict[tuple[str, str], int] = {}
    for cells in matrix.values():
        present = [(m, bool(cells[m].get("pass"))) for m in models if m in cells]
        for a, pa in present:
            for b, pb in present:
                if a != b and pa and not pb:
                    wins[(a, b)] = wins.get((a, b), 0) + 1
    scores: dict[str, int] = {}
    peers: dict[str, int] = {}
    for m in models:
        s = p = 0
        for b in models:
            if b == m:
                continue
            ab, ba = wins.get((m, b), 0), wins.get((b, m), 0)
            disagree = ab + ba
            if disagree < min_pairs:
                continue
            p += 1
            wr = ab / disagree
            s += 1 if wr > 0.5 else (-1 if wr < 0.5 else 0)
        scores[m] = s
        peers[m] = p
    return scores, peers


def _reversal_supported(weaker: str, stronger: str, ev: dict[str, ModelEvidence]) -> bool:
    """Is ranking ``weaker`` below ``stronger`` statistically supported vs a baseline?

    Supported when their pass-rate Wilson CIs are disjoint in that direction, or a
    source transition (price-prior <-> measured) occurs. Overlap keeps the baseline.
    """
    a, b = ev[weaker], ev[stronger]
    return a.ci_hi < b.ci_lo or a.source != b.source


def _apply_hysteresis(
    data_order: list[str], baseline: list[str], ev: dict[str, ModelEvidence]
) -> list[str]:
    """Retain the committed baseline order except where the data supports a reversal."""
    present = set(data_order)
    order = [m for m in baseline if m in present] + [m for m in data_order if m not in baseline]
    di = {m: i for i, m in enumerate(data_order)}
    changed = True
    while changed:
        changed = False
        for i in range(len(order) - 1):
            weaker, stronger = order[i], order[i + 1]
            if di[stronger] < di[weaker] and _reversal_supported(stronger, weaker, ev):
                order[i], order[i + 1] = stronger, weaker
                changed = True
    return order


def _assemble_order(
    measured: list[str],
    price_only: list[str],
    ev: dict[str, ModelEvidence],
    prices: dict[str, float],
) -> list[str]:
    """Interleave price-implied models into the measured order at their price slot."""
    order = list(measured)
    for x in sorted(price_only, key=lambda m: (prices.get(m, 0.0), m)):
        px = prices.get(x, 0.0)
        # Measured models win a price tie (placed before x); price-only models break by name.
        idx = sum(1 for m in order if ev[m].source == "measured" and prices.get(m, 0.0) <= px)
        idx += sum(
            1 for m in order if ev[m].source == "price-prior" and (prices.get(m, 0.0), m) < (px, x)
        )
        order.insert(idx, x)
    return order


def derive_capability_rank(  # noqa: PLR0913 (pure core; all inputs are explicit for testability)
    matrix: dict,
    models: list[str],
    prices: dict[str, float],
    default_arms: dict[str, str],
    knobs: dict,
    baseline: list[str] | None = None,
) -> CapabilityRank:
    """Pure derivation: Copeland over co-measured tasks, Wilson-gated measured>price prior.

    Sort key over measured models is ``(copeland, p̂, price, name)`` ascending (0 = weakest);
    a model failing the confidence gate takes its price-implied slot instead.
    """
    k_min, w_max, min_pairs = knobs["K"], knobs["W"], knobs["min_pairs"]
    scores, peers = _copeland_scores(matrix, models, min_pairs)
    ev: dict[str, ModelEvidence] = {}
    measured: list[str] = []
    price_only: list[str] = []
    for m in models:
        cells = [c[m] for c in matrix.values() if m in c]
        n = len(cells)
        passes = sum(1 for c in cells if bool(c.get("pass")))
        phat, lo, hi = _wilson_ci(passes, n)
        is_measured = n >= k_min and (hi - lo) <= w_max and peers[m] >= 2
        source = "measured" if is_measured else "price-prior"
        ev[m] = ModelEvidence(m, n, phat, lo, hi, -1, source, prices.get(m, 0.0))
        (measured if is_measured else price_only).append(m)
    measured.sort(key=lambda m: (scores[m], ev[m].pass_rate, prices.get(m, 0.0), m))
    if measured:
        order = _assemble_order(measured, price_only, ev, prices)
    else:
        order = sorted(models, key=lambda m: (prices.get(m, 0.0), m))
    if baseline:
        order = _apply_hysteresis(order, baseline, ev)
    pos = {m: i for i, m in enumerate(order)}
    ranked = [
        RankedModel(m, default_arms.get(m, _LEGACY_DEFAULT_REASONING), pos[m], ev[m].source)
        for m in order
    ]
    evidence = {
        m: ModelEvidence(m, e.n, e.pass_rate, e.ci_lo, e.ci_hi, pos[m], e.source, e.price)
        for m, e in ev.items()
    }
    return CapabilityRank(ordered=ranked, evidence=evidence)


def capability_rank(matrix: dict | None = None) -> CapabilityRank:
    """Derived per-model capability order over enabled models (weakest->strongest).

    Reads the default-arm-flattened real cache by default; pass ``matrix`` to derive
    over a supplied slice. Pure over (results, registry price, pinned knobs, baseline).
    """
    if matrix is None:
        matrix = flatten_default_arm(load_results())
    enabled = enabled_models()
    prices = {m: cost_per_1m(m) for m in enabled}
    arms = default_arm_ids(enabled)
    knobs = capability_rank_config()
    return derive_capability_rank(matrix, enabled, prices, arms, knobs, knobs["baseline"])


def impute_config() -> dict:
    """The ``impute:`` block (scoring reads ``enabled`` / ``drop_unsolvable``)."""
    return dict(get().get("impute", {}))


def cost_per_1m(model: str, pricing: dict | None = None) -> float:
    """Total cost per 1M tokens for a model."""
    if pricing is None:
        pricing = _pricing_dict()
    p = pricing.get(model, {})
    return float(p.get("input", 0) + p.get("output", 0))


def strategies() -> dict:
    cfg = get()
    return dict(cfg.get("strategies", {}))


def knn_params() -> dict:
    """Merged kNN + within-task-cascade strategy params (cascade keys override knn)."""
    strat = strategies()
    params = dict(strat.get("knn_semantic", {}))
    params.update(strat.get("knn_semantic_cascade_withintask", {}))
    return params


def gamma() -> float:
    cfg = get()
    return float(cfg.get("routing", {}).get("gamma", 0.1))


def benchmark_params() -> dict:
    cfg = get()
    return dict(cfg.get("benchmark", {}))


def sample_size() -> int:
    """Return sample_size from config (0 = all tasks)."""
    cfg = get()
    return int(cfg.get("benchmark", {}).get("sample_size", 0))


# ---------------------------------------------------------------------------
# Challenge store helpers
# ---------------------------------------------------------------------------


def challenges_path() -> Path:
    """Return path to the canonical challenges.json index."""
    cfg = get()
    rel = cfg.get("paths", {}).get("challenges", "routing/data/challenges.json")
    return Path(__file__).resolve().parent / rel


def challenge_dir(source: str = "swebench_verified") -> Path:
    """Return path to the directory containing individual challenge files.

    The ``challenge_store`` path is relative to this file's dir (benchmark/),
    matching ``challenges_path()`` and the benchmark.yaml comment.
    """
    cfg = get()
    rel = cfg.get("paths", {}).get("challenge_store", "challenges")
    return Path(__file__).resolve().parent / rel / source


def load_challenge(challenge_id: str, source: str = "swebench_verified") -> dict | None:
    """Load a single challenge file by ID. Returns None if not found."""
    path = challenge_dir(source) / f"{challenge_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def results_csv_path() -> Path:
    """Return path to the per-model outcome matrix (routing/results.csv)."""
    cfg = get()
    rel = cfg.get("paths", {}).get("results_csv", "routing/results.csv")
    return Path(__file__).resolve().parent / rel


def free_results_csv_path() -> Path:
    """Path to the physically separate free corpus declared by the free campaign config.

    Read from ``configs/free-tier/benchmark.yaml``'s ``paths.results_csv`` WITHOUT loading
    that config into the global cache: the paid instrument may include the free corpus
    alongside its own, and doing so must not swap the active campaign config underneath it.
    """
    base = Path(__file__).resolve().parent
    campaign = base.parent / "configs" / "free-tier" / "benchmark.yaml"
    rel = "routing/results_free.csv"
    if campaign.exists():
        with open(campaign) as handle:
            declared = (yaml.safe_load(handle) or {}).get("paths", {}).get("results_csv")
        if declared:
            rel = str(declared)
    return base / rel


def _bool_field(value: object) -> bool:
    return str(value or "").strip().lower() in ("true", "1", "yes")


def _arm_key(model: str, stored: str, defaults: dict[str, str]) -> str:
    """Alias a legacy ``"default"`` row to the model's declared default_arm.

    A non-"default" stored value (a real arm id from a live/simulated run) is
    never rewritten — only the legacy placeholder is aliased.
    """
    if stored != _LEGACY_DEFAULT_REASONING:
        return stored
    return defaults.get(model, _LEGACY_DEFAULT_REASONING)


# The sentinel a MISSING optional measurement reads as. Explicitly not 0.0: the whole point
# of the optional-column class is that "no record" and "measured zero" are different facts,
# and the older `float(row.get("estimated_cost") or 0.0)` idiom below — kept for the columns
# that predate the distinction and are written on every row — is the anti-pattern this
# replaces for anything new.
MISSING: Final = None


def _optional_num(
    row: dict[str, str], field: str, cast: Callable[[str], Any] = float
) -> Any:  # float | int | None
    """A MEASUREMENT-OPTIONAL column's value, or ``MISSING`` when the cell is blank.

    Never returns 0 for a blank. Aggregate the results only through
    `benchmark.routing.validate.require_measured`.
    """
    raw = str(row.get(field, "") or "").strip()
    if not raw:
        return MISSING
    try:
        return cast(raw)
    except ValueError:
        return MISSING


def row_precedence(row: dict[str, str]) -> tuple[int, str, str]:
    """Rank two committed rows that resolve to the SAME cache cell. Higher wins.

    Total and independent of file order, so the winner cannot change with row position.
    """
    # Two rows CAN legitimately name one cell: seven cells in the committed results.csv were
    # measured twice, once as a legacy `reasoning="default"` placeholder (written before the
    # arm-aware runner existed, so no `arm_hash`) and once afterwards under the explicit arm id.
    # Both are real measurements and neither may be deleted — results.csv is real measured data —
    # but they carry DIFFERENT costs, so something has to choose, explicitly and reproducibly.
    #   1. A stamped `arm_hash` wins. That row's arm identity is PROVEN by the hash; the legacy
    #      row's arm is only inferred by aliasing "default" through today's registry, which may
    #      have moved since the row was written. Prefer the measurement that names its own arm.
    #   2. Then the later `computed_at` — the more recent measurement of the same cell.
    #   3. Then the row's own contents, so the order is TOTAL: two rows indistinguishable on
    #      provenance still resolve identically whatever order the reader walks the file in.
    return (
        1 if (row.get("arm_hash") or "") else 0,
        str(row.get("computed_at") or ""),
        repr(sorted(row.items())),
    )


def reduce_reps(group: list[dict[str, str]]) -> tuple[dict[str, str], int, float]:
    """One cell's observations -> ``(canonical_row, n_reps, rep_pass_rate)`` for the scorer."""
    # REP 0 IS CANONICAL AND THE SCORING PATH NEVER SEES A REPLICATE. That is a deliberate
    # refusal to average, for two reasons that are structural rather than stylistic:
    #
    #   * `pass` is a VALIDATED BOOLEAN INVARIANT, not a rate. `validate._check_schema`
    #     enforces `pass <=> stop_reason == solved` on every row, so a cell whose `pass`
    #     became 0.67 would carry a stop_reason contradicting it and could never be written
    #     back or re-validated.
    #   * The bootstrap's RESAMPLING UNIT IS THE TASK (`metrics.bootstrap_cis` groups
    #     decisions by task id and resamples task GROUPS). Averaging reps inside a task would
    #     silently convert every published CI from a task bootstrap into a task-rep bootstrap
    #     — a different estimator, with narrower intervals — without touching a single
    #     CI-emitting line of code.
    #
    # With R=1 (the shipped default) every group is a single rep-0 row and this is the
    # IDENTITY, so today's numbers are bit-identical BY CONSTRUCTION rather than by
    # inspection. When more than one row claims rep 0 — the arm-hash migration left seven
    # such pairs — the pre-existing `row_precedence` still decides, exactly as before.
    from benchmark.routing import integrity

    zero = [r for r in group if integrity.rep_index(r) == 0]
    canonical = max(zero or group, key=row_precedence)
    n_reps = len({integrity.rep_index(r) for r in group})
    rep_pass_rate = sum(1 for r in group if _bool_field(r.get("pass", ""))) / len(group)
    return canonical, n_reps, rep_pass_rate


def replicate_config() -> dict:
    """The `replicates:` block — per-model observation depth for LIVE collection only."""
    # SPEND SAFETY: `enabled: false` is the shipped default because R>1 multiplies live cost
    # linearly, against a total new-measurement budget under $5. Read at exactly two sites
    # (`run_matrix.classify_cells` and the column-coverage report); the SCORING path must
    # never read it — depth is a collection decision, and letting an analysis consult it
    # would make a published number depend on a config knob no CSV row records.
    cfg = get()
    return dict(cfg.get("replicates", {}))


def replicate_enabled() -> bool:
    """Gate for replicate collection — default False (one observation per cell)."""
    return bool(replicate_config().get("enabled", False))


def replicate_depth(model: str) -> int:
    """How many observations of each of ``model``'s cells to collect (>= 1).

    Mirrors `arm_sampling.default_only_models`' per-model-override precedent: a `by_model`
    entry wins, else `default_r`, and the whole block collapses to 1 when disabled.
    """
    cfg = replicate_config()
    if not cfg.get("enabled", False):
        return 1
    by_model = cfg.get("by_model") or {}
    raw = by_model.get(model, cfg.get("default_r", 1))
    return max(1, int(raw))


def load_results(path: str | Path | None = None) -> dict:
    """Reconstruct the outcome cache from results.csv, keyed challenge x model x arm."""
    # A legacy reasoning="default" row aliases to its model's declared default_arm
    # (falling back to the literal "default" key for a model with no declared
    # reasoning block, or one absent from the current registry). Where that aliasing makes
    # two rows collide on one cell, `row_precedence` — not the file's row order — decides.
    import csv

    from benchmark.routing import censoring

    p = Path(path) if path else results_csv_path()
    results: dict[str, dict[str, dict[str, dict]]] = {}
    if not p.exists():
        return results
    defaults = default_arm_ids()
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    with open(p, newline="") as f:
        for raw in csv.DictReader(f):
            # ``lane`` is the channel identity; ``model`` is the bare canonical weights id.
            # The cache keys on the LANE so two channels serving one identity (a direct id and
            # its `-explabs` mirror) stay distinct cells. A pre-migration row carries no `lane`
            # and falls back to `model`, which was the channel id then.
            lane = str(raw.get("lane") or raw["model"])
            stored = str(raw.get("reasoning") or _LEGACY_DEFAULT_REASONING)
            arm = _arm_key(lane, stored, defaults)
            groups.setdefault((raw["challenge_id"], lane, arm), []).append(raw)
    for (cid, model, arm), group in groups.items():
        row, n_reps, rep_pass_rate = reduce_reps(group)
        results.setdefault(cid, {}).setdefault(model, {})[arm] = {
            "reasoning": arm,
            "pass": _bool_field(row.get("pass", "")),
            "cost": float(row.get("cost") or 0.0),
            "in_tok": int(row.get("in_tok") or 0),
            "out_tok": int(row.get("out_tok") or 0),
            "calls": int(row.get("calls") or 0),
            "version_hash": str(row.get("version_hash") or ""),
            "model_version": str(row.get("model_version") or ""),
            "arm_hash": str(row.get("arm_hash") or ""),
            "real_cost": float(row.get("real_cost") or row.get("cost") or 0.0),
            "estimated_cost": float(row.get("estimated_cost") or 0.0),
            "timeout_flag": _bool_field(row.get("timeout_flag", "")),
            "image_digest": str(row.get("image_digest") or ""),
            "computed_at": str(row.get("computed_at") or ""),
            # Always carry a resolved stop_reason: an explicit stored value, or a
            # derivation for legacy rows written before the column existed.
            "stop_reason": censoring.derive_stop_reason(
                passed=_bool_field(row.get("pass", "")),
                timeout_flag=_bool_field(row.get("timeout_flag", "")),
                stop_reason=str(row.get("stop_reason") or ""),
            ),
            # Collection-param provenance: the regime the cell was
            # collected under. Carried through so ``_is_stale`` can anchor on
            # step_limit/sampling_hash/prompt_hash; legacy rows backfilled before
            # the columns existed carry "" (grandfathered to a staleness no-op).
            "step_limit": str(row.get("step_limit") or ""),
            "cost_limit": str(row.get("cost_limit") or ""),
            "scaffold_version": str(row.get("scaffold_version") or ""),
            "sampling_hash": str(row.get("sampling_hash") or ""),
            "prompt_hash": str(row.get("prompt_hash") or ""),
            # AUDIT-ONLY, and no metric reads either. They record how many observations
            # stood behind this cell and how they split, so a reader can see replicate
            # depth without re-reading the CSV — never so a scorer can average over it.
            "n_reps": n_reps,
            "rep_pass_rate": rep_pass_rate,
        }
    return results


def _pick_default_row(
    model: str, per_arm: dict[str, dict], defaults: dict[str, str]
) -> dict | None:
    """The row strategies/coverage should see for one (challenge, model) cell."""
    # Prefers the model's declared default_arm; falls back to the sole cached arm
    # when only one is present (e.g. a partially-sampled non-default-only cell);
    # else None (no canonical single-outcome row exists for this cell yet).
    default_arm = defaults.get(model, _LEGACY_DEFAULT_REASONING)
    if default_arm in per_arm:
        return per_arm[default_arm]
    if len(per_arm) == 1:
        return next(iter(per_arm.values()))
    return None


def flatten_default_arm(results: dict) -> dict:
    """Collapse the 3-level (challenge x model x arm) cache to challenge x model."""
    # Strategy/coverage/summary consumers (oracle, kNN, cascade, ...) score ONE
    # canonical outcome per (challenge, model) cell — the reasoning-arm axis is a
    # benchmark-cache concern, not yet a strategy input (that lands with the
    # production router). This is the back-compat view
    # `load_matrix()` feeds them, picking each model's default_arm row.
    defaults = default_arm_ids()
    flat: dict[str, dict[str, dict]] = {}
    for cid, per_model in results.items():
        for model, per_arm in per_model.items():
            row = _pick_default_row(model, per_arm, defaults)
            if row is not None:
                flat.setdefault(cid, {})[model] = row
    return flat


def models_matrix(results: dict | None = None) -> dict:
    """Return {model: pricing} from the registry, optionally filtered to
    evaluated-and-enabled models.
    """
    pricing = load_pricing()
    priced = {
        m: {
            "input_price": p.get("input_cost_per_1m", 0),
            "output_price": p.get("output_cost_per_1m", 0),
        }
        for m, p in pricing.items()
        if isinstance(p, dict) and not m.startswith("_")
    }
    if results is None:
        return priced
    evaluated: set[str] = set()
    for task_results in results.values():
        evaluated.update(task_results.keys())
    # Respect the enabled list: a model not in it is excluded even if it has a
    # historical results row (defense-by-construction against silent leakage).
    enabled = set(enabled_models())
    return {m: priced[m] for m in priced if m in evaluated and m in enabled}


def load_challenges() -> dict:
    """Load and return the full challenges.json matrix."""
    return json.loads(challenges_path().read_text())


def load_matrix(path: str | Path | None = None) -> dict:
    """Load challenges.json and stitch back ``models``/``results`` from the registry
    and results.csv into the dict shape consumers expect."""
    # matrix["results"] is the challenge x model view (each model's default_arm
    # row) that strategies/coverage/summary score — load_results()'s full
    # challenge x model x arm cache is a benchmark-cache concern, flattened
    # here so the strategy layer is unaffected by the reasoning-arm axis.
    # A file passed explicitly may be SELF-CONTAINED (it carries its own
    # "results"/"models" — e.g. a hand-cut task slice). Honour those; stitching
    # over them would silently discard the caller's matrix. challenges.json has
    # neither key, so the default path still stitches from results.csv.
    p = Path(path) if path else challenges_path()
    matrix = json.loads(Path(p).read_text())
    own_results = matrix.get("results") if path is not None else None
    own_models = matrix.get("models") if path is not None else None
    if isinstance(own_results, dict) and own_results:
        if not (isinstance(own_models, dict) and own_models):
            matrix["models"] = models_matrix(own_results)
        return matrix
    results = load_results()
    matrix["results"] = flatten_default_arm(results)
    if not (isinstance(own_models, dict) and own_models):
        matrix["models"] = models_matrix(results)
    return matrix


def _ordered_tasks(tasks: list[str], seed: int) -> list[str]:
    """Canonical, diversity-first, nested run order for ``tasks`` (seeded-shuffle fallback)."""
    # Stratified (repo × difficulty) hash order so partial runs nest (sample_size 10 ⊂ 20 ⊂
    # 200); falls back to a seeded shuffle only when the manifest lacks repo metadata.
    from benchmark.runner import sampling

    try:
        manifest = load_challenges()
    except (FileNotFoundError, ValueError):
        manifest = {}
    ordered = sampling.order_from_manifest(sorted(tasks), manifest)
    if ordered is not None:
        return ordered
    import random

    shuffled = sorted(tasks)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def sample_tasks(tasks: list[str], seed: int = 42) -> list[str]:
    """First ``sample_size`` tasks in canonical nested order (0 = all tasks)."""
    ordered = _ordered_tasks(tasks, seed)
    sample = sample_size()
    if sample <= 0 or sample >= len(ordered):
        return ordered
    return ordered[:sample]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(config_path: str | Path | None = None) -> list[str]:
    """Validate benchmark.yaml against the registry. Returns list of errors (empty = valid).

    Registry *schema* (required fields, provider FK) is enforced by pydantic at
    load; this checks only benchmark.yaml's references into it.
    """
    errors: list[str] = []
    cfg = load(config_path)
    pricing = load_pricing()

    # `models:` must be a LIST of enabled names, each a priced registry model.
    models_cfg = cfg.get("models", [])
    if not isinstance(models_cfg, list):
        errors.append(
            "benchmark.yaml 'models' must be a list of model names "
            "(the legacy '{model: {enabled: bool}}' dict form is no longer supported)"
        )
    else:
        for name in models_cfg:
            if is_free_registry_model(name) or str(name).endswith(COLLECTION_SUFFIX):
                errors.append(
                    f"Model '{name}' is collection-only (free overlay) and cannot be enabled "
                    "in benchmark.yaml; collect it with --extra-models instead"
                )
            elif name not in pricing:
                errors.append(f"Model '{name}' in benchmark.yaml not found in the model registry")

    # Check strategies
    strat_cfg = cfg.get("strategies", {})
    known = {
        "oracle",
        "oracle_reward",
        "always_cheap",
        "always_frontier",
        "random",
        "knn_semantic",
        "knn_semantic_cascade",
        "knn_semantic_cascade_withintask",
        "knn_difficulty",
        "knn_difficulty_cascade",
        "difficulty_band_cascade",
        "ranker_difficulty",
        "ranker_difficulty_cascade",
        "ranker_defer_cascade",
        "price_cascade",
        "session_cascade",
        "knn_semantic_tier",
    }
    for name in strat_cfg.get("enabled", []):
        if name not in known:
            errors.append(f"Unknown strategy '{name}' in benchmark.yaml strategies.enabled")

    # Check control_model
    control = cfg.get("routing", {}).get("control_model")
    if control and control not in pricing:
        errors.append(f"control_model '{control}' not found in the model registry")

    return errors
