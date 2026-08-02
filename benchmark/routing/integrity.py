"""Challenge content hashing and model-version helpers for the benchmark cache.

Stale detection compares a cell's stored version_hash/model_version to current.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from benchmark import config
from shunt.models.config import ModelConfig, arm_api_params

# Columns added to results.csv beyond the original 7 (pass/cost/tokens).
# ``image_digest`` (canonical manifest sha256 of the SWE-bench image the cell was
# produced with) is a staleness anchor; ``arm_hash`` (sha256 of the reasoning
# arm's resolved API params) is too — re-mapping an arm's native
# params recomputes rather than serving a stale outcome. ``computed_at`` (ISO
# timestamp) is AUDIT-ONLY and is NEVER a staleness key. ``stop_reason`` (why a cell
# stopped — see benchmark.routing.censoring) is AUDIT-ONLY and never a staleness key;
# it is appended LAST so legacy rows lacking it still parse (derived on read).
CACHE_COLUMNS: Final[tuple[str, ...]] = (
    "version_hash",
    "model_version",
    "arm_hash",
    "real_cost",
    "estimated_cost",
    "timeout_flag",
    "image_digest",
    "computed_at",
    "stop_reason",
)
# Full results.csv header, original outcome columns first for backward-compat.
# ``reasoning`` follows ``model`` and, together with them, forms the cache key:
# (challenge_id, model, reasoning). Legacy rows carry the literal
# "default" and alias-resolve to their model's declared default_arm at read time
# (`config.load_results` / `config.default_arm_ids`).
RESULTS_FIELDS: Final[tuple[str, ...]] = (
    "challenge_id",
    "model",
    "reasoning",
    "pass",
    "cost",
    "in_tok",
    "out_tok",
    "calls",
    *CACHE_COLUMNS,
)
# Default reasoning arm written for every cell until full arm support lands.
DEFAULT_REASONING: Final[str] = "default"
UNKNOWN_VERSION: Final[str] = "unknown"


# Keys that are SELECTION metadata, not execution identity — excluded from the content
# hash so correcting a label (e.g. difficulty_stratum) never stales a PAID result cell.
# The task a model actually runs (repo/base_commit/version/F2P/P2P/image_ref/
# dataset_revision) is unchanged by a relabel, so its cached outcome stays valid.
# Selection-only metadata, never part of a challenge's identity: excluded from the
# spec hash so re-stratifying difficulty can't stale cached cells. Both spellings —
# the sampling manifest reader (order_from_manifest) accepts either.
# ``problem_statement`` is excluded for a COST reason, not an identity one. It is a
# routing-only mirror of the issue text ``infer.py`` fetches from HF at run time, and that
# fetch is NOT pinned — it passes no ``revision=``; only ``build_challenges`` does — so the
# spec field is a convenience copy for routing, not the harness's source. The exclusion is
# still right on cell identity (repo/base_commit/version/F2P/P2P/image_ref/dataset_revision
# are what a model runs, and those ARE hashed); the reason it must stay excluded is that
# hashing it would stale every paid cell the day the 500 specs are backfilled with it.
_HASH_EXCLUDED_KEYS: Final[frozenset[str]] = frozenset(
    {"difficulty_stratum", "difficulty", "problem_statement"}
)


def canonical_content(challenge: dict[str, object]) -> str:
    """Canonical JSON of a challenge (sorted keys, selection-metadata excluded)."""
    hashed = {k: v for k, v in challenge.items() if k not in _HASH_EXCLUDED_KEYS}
    return json.dumps(hashed, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def hash_content(challenge: dict[str, object]) -> str:
    """SHA256 hex digest of a challenge's canonical content."""
    return hashlib.sha256(canonical_content(challenge).encode("utf-8")).hexdigest()


def challenge_hash(challenge_id: str, source: str = "swebench_verified") -> str:
    """Deterministic content hash for one challenge; empty string if absent."""
    challenge = config.load_challenge(challenge_id, source)
    if challenge is None:
        return ""
    return hash_content(challenge)


def all_hashes(source: str = "swebench_verified") -> dict[str, str]:
    """Map every challenge id in the store to its current content hash."""
    out: dict[str, str] = {}
    directory = config.challenge_dir(source)
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.json")):
        challenge = json.loads(path.read_text())
        out[path.stem] = hash_content(challenge)
    return out


# Source dir name for materialised SWE-bench Verified instance specs (see
# benchmark/runner/swebench_specs.py). Kept as a literal to avoid importing the
# runner package from routing/.
SWEBENCH_SOURCE: Final[str] = "swebench_verified"


def swebench_spec_hashes() -> dict[str, str]:
    """Content hashes for every materialised SWE-bench Verified instance spec."""
    return all_hashes(SWEBENCH_SOURCE)


def swebench_spec_hash(instance_id: str) -> str:
    """Content hash for one instance spec; empty string if not materialised."""
    return challenge_hash(instance_id, SWEBENCH_SOURCE)


def model_versions() -> dict[str, str]:
    """Map each priced model to its declared ``version`` string (from the registry)."""
    pricing = config.load_pricing()
    out: dict[str, str] = {}
    for model, info in pricing.items():
        if not isinstance(info, dict) or model.startswith("_"):
            continue
        out[model] = str(info.get("version", UNKNOWN_VERSION))
    return out


def arm_hash_value(model: ModelConfig, arm_id: str) -> str:
    """SHA256 of an arm's resolved API params — used as a staleness anchor.

    Re-mapping an arm's native request params (e.g. changing a budget) changes
    this hash, so `_is_stale` recomputes instead of serving a stale outcome.
    """
    params = arm_api_params(model, arm_id)
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def arm_hashes(models: dict[str, ModelConfig]) -> dict[str, dict[str, str]]:
    """Map each model to {arm_id: arm_hash} for all its declared arms (staleness anchor).

    A model with no declared ``reasoning`` block maps to ``{}`` (no arm axis to
    anchor — the implicit default arm carries no per-arm params to hash).
    """
    out: dict[str, dict[str, str]] = {}
    for name, model in models.items():
        if model.reasoning is None:
            out[name] = {}
            continue
        out[name] = {arm.id: arm_hash_value(model, arm.id) for arm in model.reasoning.arms}
    return out


def estimated_cost(
    model: str,
    in_tok: int,
    out_tok: int,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float:
    """Token-count cost estimate from the pricing table (USD)."""
    if pricing is None:
        pricing = config._pricing_dict()
    p = pricing.get(model, {})
    return in_tok / 1_000_000 * p.get("input", 0.0) + out_tok / 1_000_000 * p.get("output", 0.0)
