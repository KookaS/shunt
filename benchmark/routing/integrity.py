"""Challenge content hashing and model-version helpers for the benchmark cache.

Stale detection compares a cell's stored version_hash/model_version to current.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
from pathlib import Path
from typing import Final, get_args

from benchmark import config
from shunt.models.config import ModelConfig, ServingMode, arm_api_params

# Columns added to results.csv beyond the original 7 (pass/cost/tokens).
# ``image_digest`` (canonical manifest sha256 of the SWE-bench image the cell was
# produced with) is a staleness anchor; ``arm_hash`` (sha256 of the reasoning
# arm's resolved API params) is too — re-mapping an arm's native
# params recomputes rather than serving a stale outcome. ``computed_at`` (ISO
# timestamp) is AUDIT-ONLY and is NEVER a staleness key. ``stop_reason`` (why a cell
# stopped — see benchmark.routing.censoring) is AUDIT-ONLY and never a staleness key;
# it is appended LAST so legacy rows lacking it still parse (derived on read).
# The five collection-param columns record the regime a cell was
# collected under: ``step_limit`` / ``cost_limit`` (the agent-scaffold caps in force),
# ``scaffold_version`` (installed mini-swe-agent), ``sampling_hash`` (merged request
# kwargs) and ``prompt_hash`` (scaffold system+instance templates). ``step_limit``,
# ``sampling_hash`` and ``prompt_hash`` are staleness anchors; the rest are audit-only.
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
    "step_limit",
    "cost_limit",
    "scaffold_version",
    "sampling_hash",
    "prompt_hash",
)
# ---------------------------------------------------------------------------
# Optional columns — appended AFTER ``prompt_hash`` (the same "appended LAST so legacy rows
# still parse" discipline ``stop_reason`` already documents). They are deliberately NOT in
# CACHE_COLUMNS: every member of that tuple is read as a cache/staleness field, and none of
# these are.
#
# THREE DISTINCT CLASSES, and the migration only works if they stay apart.
#
# 1. KEY. ``rep`` is the replicate index of an observation of one (challenge, model, arm)
#    cell. A legacy blank normalises to 0. That is a TAUTOLOGY — the row that exists is the
#    first observation of its cell — not an imputation, which is why it is the only new
#    column with a defined legacy value.
#
# 2. MEASUREMENT-OPTIONAL. Blank means MISSING, FOREVER, and must never acquire a default
#    anywhere: reading a blank ``wall_clock_s`` as 0.0 publishes "this cell took no time" as
#    an affirmative measured claim. Every aggregation over one of these goes through
#    ``validate.require_measured``; a consumer with nothing to aggregate OMITS the column and
#    publishes its ``n`` (the shape ``summary._context_columns`` already uses).
#
# 3. PROVENANCE-OPTIONAL (strings). Audit-only, exactly like ``computed_at`` — never a
#    staleness key, never an input to a number.
# ---------------------------------------------------------------------------
REPLICATE_COLUMN: Final[str] = "rep"
MEASUREMENT_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "wall_clock_s",
    "ttft_s",
    "latency_per_call_s",
    "cached_in_tok",
    "retry_count",
)
PROVENANCE_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "provider",
    "serving_mode",  # hosted | local
    "provider_latency_source",
)

# ---------------------------------------------------------------------------
# Vocabularies for the two provenance strings that LABEL a timing. Both exist because the
# label is what stops two incomparable populations from being silently pooled later:
#
#   * ``serving_mode`` — a batch-1 request to a local llama-server and a request to a batched
#     hosted API are different physical experiments. A latency compared across them measures
#     the serving stack, not the model. Every timing carries the mode it was taken under, and
#     `runner.infer` refuses to write a timing it cannot label.
#   * ``provider_latency_source`` — HOW the number was obtained. Client wall-clock includes
#     the network round trip and any client-side overhead; a provider-reported field does not.
#     They are different quantities under one column name, so the row says which it is.
#
# Only what this repo can actually emit is listed. A provider-reported source is a DELIBERATE
# schema edit (add the value here, and the code that reads the provider's field), not
# something a writer may invent — an unlisted value is a MALFORMED_OPTIONAL error.
# ---------------------------------------------------------------------------
LATENCY_SOURCE_CLIENT: Final[str] = "client_wall_clock"
LATENCY_SOURCES: Final[tuple[str, ...]] = (LATENCY_SOURCE_CLIENT,)
SERVING_MODES: Final[tuple[str, ...]] = get_args(ServingMode)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    *MEASUREMENT_OPTIONAL_COLUMNS,
    *PROVENANCE_OPTIONAL_COLUMNS,
)

# The columns whose drift STALES a cached cell — the single declaration of that set. It used
# to exist only as scattered branches in ``run_matrix._is_stale``, so "is this column an
# anchor?" had no answer a reader (or the coverage report) could consult. ``_is_stale``
# dispatches over this tuple, and its predicate table is asserted to cover it exactly, so
# adding an anchor here forces a deliberate decision about how it stales rather than being
# silently ignored. Everything else on a row — ``computed_at``, ``cost_limit``,
# ``scaffold_version``, ``stop_reason``, and every OPTIONAL_COLUMN — is audit-only.
STALENESS_ANCHORS: Final[tuple[str, ...]] = (
    "version_hash",
    "model_version",
    "arm_hash",
    "image_digest",
    "step_limit",
    "sampling_hash",
    "prompt_hash",
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
    REPLICATE_COLUMN,
    *OPTIONAL_COLUMNS,
)
# Default reasoning arm written for every cell until full arm support lands.
DEFAULT_REASONING: Final[str] = "default"
UNKNOWN_VERSION: Final[str] = "unknown"


def rep_index(row: dict) -> int:
    """The replicate index of a raw results row; a legacy blank normalises to 0."""
    # The only defaulting permitted anywhere in the optional-column migration, and it is a
    # tautology rather than an imputation: a row that exists IS an observation of its cell,
    # and the first observation is index 0. Normalising HERE rather than at each call site is
    # what stops a legacy blank and a freshly written "0" from keying as two different cells
    # (which `authenticity.check_duplicate_keys` would report as file-level fraud).
    raw = str(row.get(REPLICATE_COLUMN, "") or "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# RAW readers. Both live here, together, ON PURPOSE: a consumer that reads results.csv
# straight off disk bypasses `config.load_results`' replicate reducer, so it MUST state a
# replicate policy, and offering the two policies side by side makes picking one a decision
# instead of an accident.
#
# THE RULE: spend questions sum every rep; measurement questions take rep 0.
#   * A replicate really was billed, so anything reconciling against a provider invoice
#     (`cost_reconcile.load_rows`, `pipeline._real_cost`) reads `all_rows`.
#   * Anything that reports a per-cell measurement — token mixes feeding the cache-aware
#     Pareto column, the escalation cost join, the trajectory bootstrap — reads
#     `rep_zero_rows`, because rep 0 is the canonical observation the scoring path sees and
#     mixing replicates in would silently reweight cells by how many times they were re-run.
# ---------------------------------------------------------------------------
def all_rows(path: Path) -> list[dict[str, str]]:
    """Every raw row in results.csv, replicates included (the SPEND view)."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rep_zero_rows(path: Path) -> list[dict[str, str]]:
    """Only the canonical rep-0 row of each cell (the MEASUREMENT view)."""
    return [row for row in all_rows(path) if rep_index(row) == 0]


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


@functools.lru_cache(maxsize=1)
def scaffold_version() -> str:
    """The installed mini-swe-agent version; UNKNOWN_VERSION if not importable."""
    # Collection-param provenance for results rows: a ``pip install -U`` of the scaffold
    # changes the live agent; the recorded version tells which scaffold regime a row was
    # collected under.
    try:
        import minisweagent  # noqa: PLC0415
    except ImportError:
        return UNKNOWN_VERSION
    return str(getattr(minisweagent, "__version__", UNKNOWN_VERSION))


@functools.lru_cache(maxsize=1)
def scaffold_prompt_hash() -> str:
    """SHA256 of the installed scaffold's system+instance templates (prompt-drift anchor)."""
    # The prompt lives in the installed ``minisweagent`` package, so a scaffold upgrade
    # silently changes it; this hash turns that silent drift into a staleness event. Empty
    # string when unimportable — the anchor then degrades to a no-op (see run_matrix._anchor_stale).
    try:
        from minisweagent.config import builtin_config_dir, get_config_from_spec  # noqa: PLC0415
    except ImportError:
        return ""
    try:
        scaffold_cfg = get_config_from_spec(
            str(builtin_config_dir / "benchmarks" / "swebench.yaml")
        )
        agent = dict(scaffold_cfg.get("agent", {}) or {})
        rendered = str(agent.get("system_template", "")) + str(agent.get("instance_template", ""))
    except Exception:  # noqa: BLE001 (provenance is best-effort: absence means no anchor)
        return ""
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=256)
def sampling_hash(model: str, arm: str) -> str:
    """SHA256 of the merged request kwargs the scaffold would send for (model, arm) today."""
    # The merged kwargs — scaffold ``model_kwargs`` base, the routing target's ``api_base``,
    # and the arm's verbatim API params — are what ``infer._scaffold_model_kwargs`` builds for
    # a live call, minus auth secrets (a rotated key must not stale paid cells; the hash
    # anchors the CONFIG, not the credential). Empty string when unimportable/unregistered.
    try:
        from minisweagent.config import builtin_config_dir, get_config_from_spec  # noqa: PLC0415

        scaffold_cfg = get_config_from_spec(
            str(builtin_config_dir / "benchmarks" / "swebench.yaml")
        )
        base = dict((scaffold_cfg.get("model", {}) or {}).get("model_kwargs", {}) or {})
    except ImportError:
        return ""
    except Exception:  # noqa: BLE001 (provenance is best-effort: absence means no anchor)
        return ""
    info = config.load_pricing().get(model)
    if not isinstance(info, dict):
        return ""
    target: dict[str, object] = {}
    if str(info.get("route", "")).startswith("openai/"):
        target["api_base"] = str(info.get("base_url", ""))
    merged = {**base, **target, **config.arm_api_params(model, arm)}
    merged.pop("api_key", None)
    canonical = json.dumps(merged, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sampling_hash_map(models: list[str]) -> dict[str, dict[str, str]]:
    """Map each model to {arm_id: sampling_hash} for its declared arms (staleness anchor)."""
    out: dict[str, dict[str, str]] = {}
    resolved = config.resolved_models()
    for name in models:
        model = resolved.get(name)
        if model is None or model.reasoning is None:
            out[name] = {}
            continue
        out[name] = {arm.id: sampling_hash(name, arm.id) for arm in model.reasoning.arms}
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
