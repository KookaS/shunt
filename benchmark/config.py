from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import yaml

from shunt.models.config import (
    TIER_ORDER,
    ModelConfig,
    Pricing,
    ReasoningConfig,
    default_registry_path,
    load_registry,
    resolve_models,
)
from shunt.models.config import (
    arm_api_params as _resolve_arm_api_params,
)

_config: dict | None = None
_pricing: dict | None = None

# Cost-weighted p(arm|model) fractions, indexed by within-model
# rank (index 0 = cheapest rank). Decreasing so a cheaper arm samples more often
# than a pricier one; overridable via `arm_sampling.weights` in config.yaml.
DEFAULT_ARM_SAMPLING_WEIGHTS: Final[tuple[float, ...]] = (0.5, 0.35, 0.25)

# Legacy literal every pre-arm-hash results.csv row was written under (mirrors
# `benchmark.routing.integrity.DEFAULT_REASONING` — duplicated here, not imported,
# because integrity.py depends on this module and not vice versa).
_LEGACY_DEFAULT_REASONING: Final[str] = "default"


def load(path: str | Path | None = None) -> dict:
    global _config  # noqa: PLW0603, SH001 (module load-once config cache)
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
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
        "tier": model.tier,
        "provider": model.provider,
        "route": model.route,
        "base_url": model.base_url,
        "api_key_env_var": model.api_key_env_var,
        # `version` is a model-identity attribute (a sibling of tier/provider), no
        # longer a pricing field; a priced model always carries one (schema-enforced).
        "version": model.version,
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
    """Cost-weighted p(arm|model) fractions, indexed by within-model rank."""
    cfg = get()
    weights = cfg.get("arm_sampling", {}).get("weights")
    return [float(w) for w in weights] if weights else list(DEFAULT_ARM_SAMPLING_WEIGHTS)


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


def _tier_order(tier: str) -> int:
    """Canonical tier rank, derived from the registry's ``TIER_ORDER`` (single source
    of truth). Raises on an unregistered tier so a drift can't silently sort last.
    """
    ranks = {t: i for i, t in enumerate(TIER_ORDER)}
    if tier not in ranks:
        raise ValueError(f"unknown tier {tier!r}; registered tiers: {list(TIER_ORDER)}")
    return ranks[tier]


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
    """Return enabled model names sorted by tier (cheap → mid → frontier),
    then by cost ascending within each tier."""
    # `models:` is a LIST of enabled names. In-list = enabled; a registry model
    # absent from the list is disabled; a listed name absent from the registry is
    # an unrecoverable config error (a listed model must exist to be routable).
    cfg = get()
    listed = cfg.get("models", [])
    pricing = load_pricing()

    unregistered = [m for m in listed if m not in pricing]
    if unregistered:
        raise ValueError(
            "benchmark/config.yaml lists model(s) the benchmark cannot see "
            f"(absent from the registry, or registered without a pricing block): {unregistered}. "
            "A listed model must exist in src/shunt/models/default_config.yaml with pricing."
        )
    # dict.fromkeys dedupes a repeated list entry while preserving order, so a typo'd
    # duplicate can't make classify_cells enumerate (and pay for) the same cell twice.
    enabled = list(dict.fromkeys(m for m in listed if not m.startswith("_")))

    pricing_dict = _pricing_dict()

    def _sort_key(m: str) -> tuple:
        info = pricing.get(m, {})
        tier = _tier_order(info.get("tier", "cheap")) if isinstance(info, dict) else 99
        cost = pricing_dict.get(m, {}).get("input", 0) + pricing_dict.get(m, {}).get("output", 0)
        return (tier, cost)

    enabled.sort(key=_sort_key)
    return enabled


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


def cascade_order() -> list[str]:
    """Cascade order = enabled models in tier order (cheap → mid → frontier),
    cheapest-first within each tier. No manual list needed — avoids conflicts."""
    return enabled_models()


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
    """Merged kNN + kNN-cascade strategy params (cascade keys override knn)."""
    strat = strategies()
    params = dict(strat.get("knn", {}))
    params.update(strat.get("knn_cascade", {}))
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
    matching ``challenges_path()`` and the config.yaml comment.
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


def load_results(path: str | Path | None = None) -> dict:
    """Reconstruct the outcome cache from results.csv, keyed challenge x model x arm."""
    # A legacy reasoning="default" row aliases to its model's declared default_arm
    # (falling back to the literal "default" key for a model with no declared
    # reasoning block, or one absent from the current registry).
    import csv

    p = Path(path) if path else results_csv_path()
    results: dict[str, dict[str, dict[str, dict]]] = {}
    if not p.exists():
        return results
    defaults = default_arm_ids()
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            cid = row["challenge_id"]
            model = row["model"]
            stored = str(row.get("reasoning") or _LEGACY_DEFAULT_REASONING)
            arm = _arm_key(model, stored, defaults)
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
    p = Path(path) if path else challenges_path()
    matrix = json.loads(Path(p).read_text())
    results = load_results()
    matrix["results"] = flatten_default_arm(results)
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
    """Validate config.yaml against the registry. Returns list of errors (empty = valid).

    Registry *schema* (required fields, tier vocabulary, provider FK) is enforced
    by pydantic at load; this checks only config.yaml's references into it.
    """
    errors: list[str] = []
    cfg = load(config_path)
    pricing = load_pricing()

    # `models:` must be a LIST of enabled names, each a priced registry model.
    models_cfg = cfg.get("models", [])
    if not isinstance(models_cfg, list):
        errors.append(
            "config.yaml 'models' must be a list of model names "
            "(the legacy '{model: {enabled: bool}}' dict form is no longer supported)"
        )
    else:
        for name in models_cfg:
            if name not in pricing:
                errors.append(f"Model '{name}' in config.yaml not found in the model registry")

    # Check strategies
    strat_cfg = cfg.get("strategies", {})
    known = {
        "oracle",
        "oracle_reward",
        "always_cheap",
        "always_frontier",
        "random",
        "knn",
        "knn_cascade",
        "external_prior",
        "knn_blended",
    }
    for name in strat_cfg.get("enabled", []):
        if name not in known:
            errors.append(f"Unknown strategy '{name}' in config.yaml strategies.enabled")

    # Check control_model
    control = cfg.get("routing", {}).get("control_model")
    if control and control not in pricing:
        errors.append(f"control_model '{control}' not found in the model registry")

    return errors
