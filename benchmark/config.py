from __future__ import annotations

import json
from pathlib import Path

import yaml

_config: dict | None = None
_pricing: dict | None = None


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
    return Path(__file__).resolve().parent / "routing" / "data" / "models.json"


def load_pricing(path: str | Path | None = None) -> dict:
    global _pricing  # noqa: PLW0603, SH001 (module load-once pricing cache)
    if _pricing is not None:
        return _pricing
    p = Path(path) if path else _pricing_path()
    if p.exists():
        with open(p) as f:
            _pricing = json.load(f)
    else:
        _pricing = {}
    return _pricing


def _tier_order(tier: str) -> int:
    return {"cheap": 0, "mid": 1, "high": 2, "frontier": 3}.get(tier, 99)


def _pricing_dict() -> dict:
    """Return pricing as {model: {input, output}} for all models in models.json."""
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
    cfg = get()
    models_cfg = cfg.get("models", {})
    pricing = load_pricing()

    enabled = []
    for model_name in pricing:
        if not isinstance(pricing[model_name], dict) or model_name.startswith("_"):
            continue
        info = models_cfg.get(model_name, {})
        if info.get("enabled", True):
            enabled.append(model_name)

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


def load_results(path: str | Path | None = None) -> dict:
    """Reconstruct the per-model outcome cache (keyed by challenge×model) from
    routing/results.csv. ``reasoning`` defaults to ``"default"``.
    """
    import csv

    p = Path(path) if path else results_csv_path()
    results: dict[str, dict] = {}
    if not p.exists():
        return results
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            cid = row["challenge_id"]
            model = row["model"]
            results.setdefault(cid, {})[model] = {
                "reasoning": str(row.get("reasoning") or "default"),
                "pass": _bool_field(row.get("pass", "")),
                "cost": float(row.get("cost") or 0.0),
                "in_tok": int(row.get("in_tok") or 0),
                "out_tok": int(row.get("out_tok") or 0),
                "calls": int(row.get("calls") or 0),
                "version_hash": str(row.get("version_hash") or ""),
                "model_version": str(row.get("model_version") or ""),
                "real_cost": float(row.get("real_cost") or row.get("cost") or 0.0),
                "estimated_cost": float(row.get("estimated_cost") or 0.0),
                "timeout_flag": _bool_field(row.get("timeout_flag", "")),
                "image_digest": str(row.get("image_digest") or ""),
                "computed_at": str(row.get("computed_at") or ""),
            }
    return results


def models_matrix(results: dict | None = None) -> dict:
    """Return {model: pricing} from models.json, optionally filtered to
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
    # Respect `enabled: false`: a disabled model is excluded even if it has a
    # historical results row (defense-by-construction against silent leakage).
    models_cfg = get().get("models", {})
    return {
        m: priced[m]
        for m in priced
        if m in evaluated and models_cfg.get(m, {}).get("enabled", True)
    }


def load_challenges() -> dict:
    """Load and return the full challenges.json matrix."""
    return json.loads(challenges_path().read_text())


def load_matrix(path: str | Path | None = None) -> dict:
    """Load challenges.json and stitch back ``models`` (from models.json) and
    ``results`` (from routing/results.csv) into the dict shape consumers expect
    (``matrix["models"]``, ``matrix["results"]``, ``matrix["tasks"]``).
    """
    p = Path(path) if path else challenges_path()
    matrix = json.loads(Path(p).read_text())
    results = load_results()
    matrix["results"] = results
    matrix["models"] = models_matrix(results)
    return matrix


def sample_tasks(tasks: list[str], seed: int = 42) -> list[str]:
    """Subsample tasks based on config sample_size (0 = all tasks)."""
    sample = sample_size()
    if sample <= 0 or sample >= len(tasks):
        return tasks
    import random

    rng = random.Random(seed)
    shuffled = sorted(tasks)
    rng.shuffle(shuffled)
    return shuffled[:sample]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_REQUIRED_PRICING_FIELDS = (
    "tier",
    "input_cost_per_1m",
    "output_cost_per_1m",
    "price_provider",
    "price_source",
    "price_as_of",
    "access_via",
)
_VALID_ACCESS_VIA = ("direct", "requesty")


def _validate_pricing_entry(name: str, info: dict, errors: list[str]) -> None:
    """Check one models.json model has the required canonical + provenance fields."""
    for field in _REQUIRED_PRICING_FIELDS:
        if field not in info:
            errors.append(f"Model '{name}' in models.json missing '{field}'")
    access = info.get("access_via")
    if access is not None and access not in _VALID_ACCESS_VIA:
        errors.append(
            f"Model '{name}' has invalid access_via '{access}' (expected direct|requesty)"
        )


def validate(config_path: str | Path | None = None) -> list[str]:
    """Validate config.yaml against models.json. Returns list of errors (empty = valid)."""
    errors: list[str] = []
    cfg = load(config_path)
    pricing = load_pricing()

    # Check models in config exist in models.json
    models_cfg = cfg.get("models", {})
    for name in models_cfg:
        if name not in pricing:
            errors.append(f"Model '{name}' in config.yaml not found in models.json")

    # Check enabled models have required fields
    pricing_items = {
        k: v for k, v in pricing.items() if isinstance(v, dict) and not k.startswith("_")
    }
    for name, info in pricing_items.items():
        _validate_pricing_entry(name, info, errors)

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
        errors.append(f"control_model '{control}' not found in models.json")

    return errors
