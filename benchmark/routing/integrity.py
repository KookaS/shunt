"""Challenge content hashing and model-version helpers for the benchmark cache.

Stale detection compares a cell's stored version_hash/model_version to current.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from benchmark import config

# Columns added to results.csv beyond the original 7 (pass/cost/tokens).
# ``image_digest`` (canonical manifest sha256 of the SWE-bench image the cell was
# produced with) is a staleness anchor; ``computed_at`` (ISO timestamp) is
# AUDIT-ONLY and is NEVER a staleness key.
CACHE_COLUMNS: Final[tuple[str, ...]] = (
    "version_hash",
    "model_version",
    "real_cost",
    "estimated_cost",
    "timeout_flag",
    "image_digest",
    "computed_at",
)
# Full results.csv header, original outcome columns first for backward-compat.
# ``reasoning`` follows ``model`` (default below); full reasoning-arm support is
# planned for a later release — the column is plumbed defaulted for forward-compat.
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


def canonical_content(challenge: dict[str, object]) -> str:
    """Canonical JSON of a challenge (sorted keys) — order-independent hashing."""
    return json.dumps(challenge, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


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
    """Map each priced model to its declared ``version`` string (from models.json)."""
    pricing = config.load_pricing()
    out: dict[str, str] = {}
    for model, info in pricing.items():
        if not isinstance(info, dict) or model.startswith("_"):
            continue
        out[model] = str(info.get("version", UNKNOWN_VERSION))
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
