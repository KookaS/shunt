#!/usr/bin/env python3
"""Benchmark caching loop: find missing/stale (challenge×model) cells and
refresh artifacts. Simulated by default; --live + keys delegate to the real
SWE-bench harness executor for instances with a materialised spec.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Final, Literal, NoReturn

from benchmark import config, model_coverage
from benchmark.routing import censoring, channel, coverage, impute, integrity, validate
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier
from benchmark.routing.strategies.oracle import Oracle
from benchmark.runner import (
    campaign_scheduler,
    image_version,
    infer,
    lane_scheduler,
    swebench_multimodal_specs,
    swebench_specs,
)
from shunt.secrets import load_dotenv_file

# History log columns: the full cache row plus when the row was superseded.
HISTORY_FIELDS: Final[tuple[str, ...]] = (*integrity.RESULTS_FIELDS, "superseded_at")


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 (audit only — never a staleness key)."""
    return datetime.now(UTC).isoformat()


_KEY_ENV: Final[tuple[str, ...]] = (
    "DEEPSEEK_API_KEY",
    "REQUESTY_API_KEY",
    "XAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    # Experiential Labs' free-promo channel: the credential behind the `*-explabs`
    # collection-only ids (registered 2026-09-06). Without it a --live run whose ONLY
    # candidates are those extras would silently downgrade to simulated. It is still
    # exercised per model — litellm_model_target reads it for every explabs cell, and
    # the preflight probe refuses when it is missing.
    "EXPLABS_API_KEY",
)


@dataclass
class CellStatus:
    """Classification of every (challenge, model, reasoning-arm) cell."""

    missing: list[tuple[str, str, str]] = field(default_factory=list)
    stale: list[tuple[str, str, str]] = field(default_factory=list)
    present: int = 0
    # Additional OBSERVATIONS of cells that are already correct — one entry per rep 1..R-1
    # for a model configured with depth R>1. Deliberately NOT part of `to_run`: those cells
    # are neither missing nor stale, they run under mode="replicate", and folding them in
    # would let a replicate supersede the paid rep-0 row it is meant to sit beside.
    replicate: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def to_run(self) -> list[tuple[str, str, str]]:
        """Cells needing (re)computation: missing plus stale."""
        return self.missing + self.stale


def _has_keys() -> bool:
    return any(os.environ.get(k) for k in _KEY_ENV)


def _extra_models(raw: str | None) -> list[str]:
    """Collection-only model ids from ``--extra-models``, validated; [] when unset.

    An id is accepted when it is a priced registry model, or when the config layer can
    synthesize it as a ``-explabs`` catalog slug (see
    ``config.synthesize_collection_model``). These never reach ``enabled_models()``/
    ``models_matrix``/``capability_rank`` — the union happens at the single run site in
    ``_run_full``.
    """
    if not raw:
        return []
    ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not ids:
        return []
    pricing = config.load_pricing()
    overlay = config.free_registry()
    missing = [
        m
        for m in ids
        if m not in pricing and m not in overlay and config.synthesize_collection_model(m) is None
    ]
    if missing:
        raise ValueError(
            "--extra-models lists id(s) the benchmark cannot see (absent from the registry "
            f"and the configured free overlay, or a non-`-explabs` id with no pricing block): "
            f"{missing}. An extra model must exist in src/shunt/config/models.yaml with pricing, "
            "be a row of the overlay registry (--free-registry / SHUNT_FREE_REGISTRY), or be a "
            "`-explabs` catalog slug."
        )
    return list(dict.fromkeys(ids))


def preflight_refuses(live: bool, model: str | None = None) -> bool:
    """Live-only: probe the API with one real $0 call; True (refuse) iff the key is unusable.

    ``model`` names the probe target when the caller has a specific candidate (an
    ``--extra-models`` id); None probes the cheapest enabled model as usual.
    Simulated runs skip it entirely (offline + free). A transient blip never refuses — only a
    dead/empty key, no balance, or a down provider does, before any container is created.
    """
    if not live:
        return False
    try:
        if model is None:
            infer.preflight_api_check()
        else:
            infer.preflight_api_check(model=model)
    except (infer.ApiUnusableError, infer.MissingApiKeysError) as exc:
        print(
            f"  REFUSING --live: preflight API health check failed — {exc}. The key is "
            "invalid, out of balance, or the provider is down; no containers were started.",
            file=sys.stderr,
        )
        return True
    return False


# The process-wide scaffold breaker for a $0 run (the $0 interlock, layer 3).
# Deliberately tiny: a free lane should record real_cost==0, so any positive accumulation
# is a leak.
FREE_LANE_GLOBAL_COST_LIMIT: Final[str] = "0.05"


def _arm_free_lane_cost_breaker() -> None:
    """Turn on mini-swe-agent's process-wide cost breaker for a $0 run.

    The env var is read ONCE when ``minisweagent.models`` constructs its global stats, and
    importing this module already pulled that in — so setting the env alone would be a silent
    no-op for the running process, the same family as the ``--max-cost 0`` trap. We set the
    env for any later import AND re-apply the limit to the live stats object.
    """
    os.environ["MSWEA_GLOBAL_COST_LIMIT"] = FREE_LANE_GLOBAL_COST_LIMIT
    try:
        from minisweagent import models as mswea_models  # noqa: PLC0415
    except ImportError:  # pragma: no cover - minisweagent is a benchmark dependency
        return
    mswea_models.GLOBAL_MODEL_STATS.cost_limit = float(FREE_LANE_GLOBAL_COST_LIMIT)


def _billing_of(model: str) -> str | None:
    """The listing's declared billing entitlement (`free`/`paid`), or None when undeclared.

    Read from the shipped registry first, then the non-shipped overlay, then the collection
    synthesizer — so a `billing: free` declaration is found wherever the row lives. Never
    inferred from the `-explabs` suffix.
    """
    for row in (config.load_pricing().get(model), config.free_registry().get(model)):
        if isinstance(row, dict) and row.get("billing"):
            return str(row["billing"])
    synthesized = config.synthesize_collection_model(model)
    if isinstance(synthesized, dict) and synthesized.get("billing"):
        return str(synthesized["billing"])
    return None


def _is_free_lane(model: str, overlay_ids: set[str] | None = None) -> bool:
    """True iff *model*'s listing DECLARES `billing: free` — the entitlement, not the suffix.

    An overlay `billing: free` row on a provider with no declared free lane (Together) or an
    app-gated one (OpenCode Zen) is collection provenance, not a spendable $0 lane, so the
    provider gate still applies. The `-explabs` suffix used to stand in for a free
    declaration; it is no longer consulted at all — no fallback path for it remains in
    `benchmark/routing/scripts/backfill_channel.py`, which resolves a free listing only
    through `declared_billing`.
    """
    if _billing_of(model) != "free":
        return False
    ids = config.free_registry_ids() if overlay_ids is None else overlay_ids
    if model in ids:
        provider = config.free_registry().get(model, {}).get("provider")
        return config.is_declared_free_provider(provider)
    return True


def require_zero_cost_refusal(
    *,
    live: bool,
    enabled: list[str],
    extra: list[str],
    cost_limit: float | None = None,
    overlay_ids: set[str] | None = None,
) -> str | None:
    """Refuse a ``--require-zero-cost`` run unless EVERY model to run is a $0 free lane.

    Fail-closed so the four $0 interlocks are actually engaged: a run that touches an enabled
    paid model, an extra that is neither an overlay row nor a collection slug, or a scaffold
    cap of 0 (which DISABLES mini-swe-agent's per-cell limit rather than capping at $0) is
    refused before any container starts or any API credit moves.
    """
    if not live:
        return "--require-zero-cost requires --live (a simulated run harvests nothing)"
    if cost_limit is not None and cost_limit <= 0:
        return (
            "--require-zero-cost refuses live.cost_limit <= 0: 0 DISABLES the scaffold cap "
            "(mini-swe-agent gates on `0 < cost_limit`); set a small POSITIVE per-cell cap"
        )
    lanes = [*enabled, *extra]
    if not lanes:
        return "--require-zero-cost found no models to run"
    paid = sorted(m for m in lanes if not _is_free_lane(m, overlay_ids))
    if paid:
        return (
            f"--require-zero-cost refuses non-free model(s): {paid}. A $0 run may collect "
            "ONLY non-shipped overlay lanes (configs/free-tier/overlay.yaml via "
            "--free-registry + --extra-models)."
        )
    return None


# A lane preflight probe: the runner's seam over `infer.preflight_api_probe`, injectable so
# tests never issue a live call.
LaneProbe = Callable[[str], infer.PreflightOutcome]


@dataclass
class LanePreflightReport:
    """Per-lane preflight outcome: which lanes survive, and why the rest do not.

    Fault-tolerance is the whole point: one unusable lane is a LANE problem, not a fleet
    problem. ``refused`` (zero usable lanes) is the only condition that aborts the campaign.
    """

    usable: list[str] = field(default_factory=list)
    disabled: dict[str, str] = field(default_factory=dict)
    quarantined: dict[str, str] = field(default_factory=dict)
    transient: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        """Abort ONLY when no lane can be scheduled — a throttle still leaves a usable lane."""
        return not self.usable


def _default_lane_probe(timeout_s: float) -> LaneProbe:
    """The live probe seam: one retry-disabled, timeout-bounded completion per lane."""

    def probe(lane: str) -> infer.PreflightOutcome:
        return infer.preflight_api_probe(lane, timeout_s=timeout_s)

    return probe


def preflight_free_lanes(
    live: bool,
    lanes: list[str],
    *,
    scheduler: lane_scheduler.LaneScheduler | None = None,
    timeout_s: float | None = None,
    probe: LaneProbe | None = None,
) -> LanePreflightReport:
    """Probe every admitted free lane once; disable/quarantine the failures, keep the rest.

    A dead key or a permanently unavailable model DISABLES its lane with a named reason; a
    429/quota throttle QUARANTINES it (honouring Retry-After) so the scheduler backs off
    instead of spinning; a transient blip leaves the lane alone. A lane already disabled in
    persisted state is never re-probed (a known-dead lane must not cost a call every pass).
    The run aborts only when NO lane remains usable (``report.refused``). The result is
    recorded on ``scheduler`` so it persists with the lane state at run end.
    """
    from benchmark.runner.free_tier_smoke import free_tier_refusal  # noqa: PLC0415

    timeout = infer.DEFAULT_PREFLIGHT_TIMEOUT_S if timeout_s is None else timeout_s
    run_probe = probe or _default_lane_probe(timeout)
    overlay_models = config.free_registry_models()
    report = LanePreflightReport()
    now = time.monotonic()
    for lane in lanes:
        if scheduler is not None and scheduler.lane_state(lane).disabled_reason:
            # Persisted disable: skip the probe entirely so a dead lane is paid for once.
            report.disabled[lane] = str(scheduler.lane_state(lane).disabled_reason)
            continue
        if not live:
            report.usable.append(lane)
            continue
        model = overlay_models.get(lane)
        if model is not None:
            refusal = free_tier_refusal(model)
            if refusal is not None:
                reason = f"static $0 refusal: {refusal}"
                _disable_lane(scheduler, lane, reason)
                report.disabled[lane] = reason
                continue
        try:
            outcome = run_probe(lane)
        except Exception as exc:  # noqa: BLE001 - one lane must never abort the fleet sweep
            print(f"  lane preflight error {lane}: {exc}", file=sys.stderr)
            report.usable.append(lane)
            report.transient.append(lane)
            continue
        if outcome.disables:
            reason = f"{outcome.kind}: {outcome.reason}"
            _disable_lane(scheduler, lane, reason)
            report.disabled[lane] = reason
        elif outcome.quarantines:
            reason = f"rate_limited: {outcome.reason}"
            if scheduler is not None:
                scheduler.record_rate_limit(lane, now, retry_after=outcome.retry_after)
            report.quarantined[lane] = reason
            report.usable.append(lane)
        else:
            if outcome.kind == "transient":
                report.transient.append(lane)
            report.usable.append(lane)
    return report


def _disable_lane(scheduler: lane_scheduler.LaneScheduler | None, lane: str, reason: str) -> None:
    """Record a lane's named disable on the scheduler so it is skipped and persisted."""
    if scheduler is not None:
        scheduler.disable(lane, reason)


def _default_selected_arms(tasks: list[str], models: list[str]) -> dict[tuple[str, str], list[str]]:
    """Back-compat fallback: check every cell only at its declared default arm."""
    defaults = config.default_arm_ids(models)
    fallback = {model: [defaults.get(model, integrity.DEFAULT_REASONING)] for model in models}
    return {(cid, model): fallback[model] for cid in tasks for model in models}


def _identity_coverage(cache: dict, versions: dict[str, str]) -> dict[tuple[str, str], set[str]]:
    """(challenge, identity) -> registry ids holding a REAL cached row under that identity.

    Identity is a model ``version`` slug (``kimi-k3``); two registry ids sharing one
    (direct requesty ``kimi-k3`` and free-promo ``kimi-k3-explabs``) are the SAME model
    served from two channels. Zero-work rows (aborted-collection residue) are excluded:
    they never executed, so they cannot cover a challenge.
    """
    covered: dict[tuple[str, str], set[str]] = {}
    for cid, per_model in cache.items():
        for mid, per_arm in per_model.items():
            for row in per_arm.values():
                if impute.is_zero_work(row):
                    continue
                identity = str(row.get("model_version") or versions.get(mid, ""))
                if identity:
                    covered.setdefault((cid, identity), set()).add(mid)
    return covered


def _identity_fanout_reached(
    cid: str,
    model: str,
    versions: dict[str, str],
    coverage: dict[tuple[str, str], set[str]],
    cap: int,
) -> bool:
    """True iff ``model``'s identity already has ``cap`` or more OTHER real rows at ``cid``.

    ``cap`` is the total-channel ceiling: the default ``cap=1`` is the dedupe policy (skip as
    soon as any twin covers), and a named concordance subset raises it so up to ``cap``
    channels of one identity may be collected at a challenge. The skip excludes the model's
    own id, so an extra's stale/zero-work residue is never read as coverage.
    """
    identity = versions.get(model, "")
    if not identity:
        return False
    others = sum(1 for other in coverage.get((cid, identity), ()) if other != model)
    return others >= cap


def _covered_elsewhere(
    cid: str,
    model: str,
    versions: dict[str, str],
    coverage: dict[tuple[str, str], set[str]],
) -> bool:
    """True iff ``model``'s identity already has a real row at ``cid`` under ANOTHER id.

    The default dedupe policy — the ``cap=1`` case of :func:`_identity_fanout_reached`.
    """
    return _identity_fanout_reached(cid, model, versions, coverage, 1)


def classify_cells(
    tasks: list[str],
    models: list[str],
    cache: dict,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None = None,
    selected_arms: dict[tuple[str, str], list[str]] | None = None,
    arm_hash_map: dict[str, dict[str, str]] | None = None,
    *,
    step_limit: int | None = None,
    prompt_hash: str | None = None,
    sampling_hash_map: dict[str, dict[str, str]] | None = None,
    identity_skip_models: set[str] | None = None,
    identity_fanout_cap: int = 1,
    identity_fanout_models: set[str] | None = None,
) -> CellStatus:
    """Split (challenge, model, reasoning-arm) cells into missing / stale / present.

    ``identity_skip_models`` (collection-only extra ids) get the OWNER coverage policy:
    when a challenge already has a real row under the same model identity from another
    registry id/channel (any arm, any rep), its missing OR stale cell is dropped from
    the plan — MORE COVERAGE wins over re-running a covered challenge. Cells whose own id is
    the only coverage are never dropped (append-only; a stale/zero-work residue stays
    collectable). Every other model (enabled / direct baseline) classifies exactly as
    before — per-id presence, no identity widening.

    ``identity_fanout_cap`` is the total-channel ceiling for that policy: ``1`` (default)
    is dedupe; a named concordance subset raises it so up to ``cap`` channels of one
    identity may be collected at one challenge. ``identity_fanout_models`` SCOPES the raised
    cap to the named concordance channels: when omitted (the default) the cap applies to
    every identity-skip model, reproducing the previous behaviour; when supplied, those
    channels get ``identity_fanout_cap`` and every other skip model keeps the cap of 1.
    """
    # `cache` is 3-level: challenge_id -> model -> arm_id -> row.
    # `selected_arms` restricts each (challenge, model) to its sampled arms;
    # absent, every cell is checked only at its declared default arm
    # (config.default_arm_ids), reproducing the single-cell-per-model
    # behaviour from before arm sampling existed. Stale iff spec hash, image digest,
    # model version, the arm's own api-param hash, or a collection-param anchor
    # (step_limit / prompt_hash / sampling_hash) drifted; missing iff no row for that arm.
    arms_by_cell = selected_arms or _default_selected_arms(tasks, models)
    skip_set = identity_skip_models or set()
    fanout_set = skip_set if identity_fanout_models is None else set(identity_fanout_models)
    governed = skip_set | fanout_set
    coverage = _identity_coverage(cache, versions) if governed else {}
    status = CellStatus()
    for cid in tasks:
        for model in models:
            # None = outside the identity policy entirely (per-id presence only).
            in_fanout = model in fanout_set
            cap = None if model not in governed else (identity_fanout_cap if in_fanout else 1)
            for arm in arms_by_cell.get((cid, model), [integrity.DEFAULT_REASONING]):
                cell = cache.get(cid, {}).get(model, {}).get(arm)
                if cell is None:
                    if cap is not None and _identity_fanout_reached(
                        cid, model, versions, coverage, cap
                    ):
                        continue
                    status.missing.append((cid, model, arm))
                elif _is_stale(
                    cell,
                    cid,
                    model,
                    arm,
                    hashes,
                    versions,
                    digests,
                    arm_hash_map,
                    step_limit=step_limit,
                    prompt_hash=prompt_hash,
                    sampling_hash_map=sampling_hash_map,
                ):
                    if cap is not None and _identity_fanout_reached(
                        cid, model, versions, coverage, cap
                    ):
                        continue
                    status.stale.append((cid, model, arm))
                else:
                    status.present += 1
                # Reps 1..R-1 of every cell in the depth-R models, whatever its status: the
                # rep-0 observation is the one `to_run` (or the cache) supplies, and these are
                # the additional ones. `replicate_depth` returns 1 whenever the block is
                # disabled — the shipped default — so this is a no-op today.
                for _ in range(config.replicate_depth(model) - 1):
                    status.replicate.append((cid, model, arm))
    return status


def _anchor_stale(cell: dict, key: str, expected: object | None) -> bool:
    """Stale iff a NON-EMPTY stored anchor resolves AND differs — else never stales."""
    # The grandfather guard shared by every collection-param anchor (mirrors _arm_stale /
    # _image_stale): no expected value (None or uncomputable "") or an empty stored value
    # (a legacy row written before the column existed) degrades to a no-op instead of
    # recollecting a PAID cell. Only a stored-vs-expected mismatch on a real anchor stales.
    if expected is None:
        return False
    expected_str = str(expected)
    if not expected_str:
        return False
    stored = str(cell.get(key, ""))
    return bool(stored) and stored != expected_str


def _limit_anchor_stale(cell: dict, expected: object | None) -> bool:
    """step_limit staleness: fires ONLY for step-limit-CENSORED cells."""
    # Raising the cap changes the outcome of a cell that hit the OLD cap (it gets more
    # attempts); a cell that solved or finished naturally is regime-independent and stays
    # valid. This is the recollection-focused read of the anchor — it targets exactly the
    # censored cells the cap-raise wants recollected, not the whole paid corpus.
    if str(cell.get("stop_reason") or "") != censoring.STEP_LIMIT:
        return False
    return _anchor_stale(cell, "step_limit", expected)


def _sampling_anchor_stale(
    cell: dict, model: str, arm: str, sampling_hash_map: dict[str, dict[str, str]] | None
) -> bool:
    """sampling_hash staleness, mirroring _arm_stale's non-empty-stored guard."""
    if sampling_hash_map is None:
        return False
    expected = sampling_hash_map.get(model, {}).get(arm)
    if not expected:
        return False
    stored = str(cell.get("sampling_hash", ""))
    return bool(stored) and stored != expected


def _is_stale(
    cell: dict,
    cid: str,
    model: str,
    arm: str,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None,
    arm_hash_map: dict[str, dict[str, str]] | None = None,
    *,
    step_limit: int | None = None,
    prompt_hash: str | None = None,
    sampling_hash_map: dict[str, dict[str, str]] | None = None,
) -> bool:
    """Stale iff the row records no executed work, or spec/model/arm/image/params drifted."""
    # ZERO-WORK ROWS ARE STALE. A row with zero priced calls and $0 spend never executed —
    # it is an aborted collection's residue, not a measurement — yet every other check here
    # passes on it, so it was cached forever as though the cell had been observed. That
    # silently removed the cell from `to_run` for all time: the collector cannot tell
    # "measured" from "never happened" by presence alone.
    # Deliberately NOT `impute.is_non_observation`, which also matches CENSORED cells. A
    # censored cell DID run, burned tokens and cost real money; only its pass/fail is
    # unknown. Re-collecting one buys a likely-identical non-observation at full price, so
    # the narrower `is_zero_work` is the correct predicate here.
    if impute.is_zero_work(cell):
        return True
    # DISPATCH OVER integrity.STALENESS_ANCHORS, not over a list of branches. Each anchor
    # keeps its own predicate — they are genuinely different rules (strict equality for the
    # identity hashes; the non-empty-stored grandfather guard for the later-added ones;
    # censored-cells-only for step_limit) — but WHICH columns are anchors is now declared
    # once, in the schema module, so `_ANCHOR_CHECKS` can be asserted complete against it and
    # a new anchor cannot be added without deciding how it stales.
    return any(
        _ANCHOR_CHECKS[anchor](
            cell,
            _AnchorInputs(
                cid=cid,
                model=model,
                arm=arm,
                hashes=hashes,
                versions=versions,
                digests=digests,
                arm_hash_map=arm_hash_map,
                step_limit=step_limit,
                prompt_hash=prompt_hash,
                sampling_hash_map=sampling_hash_map,
            ),
        )
        for anchor in integrity.STALENESS_ANCHORS
    )


@dataclass(frozen=True)
class _AnchorInputs:
    """Everything the current run knows that an anchor predicate may compare a cell against."""

    cid: str
    model: str
    arm: str
    hashes: dict[str, str]
    versions: dict[str, str]
    digests: dict[str, str | None] | None
    arm_hash_map: dict[str, dict[str, str]] | None
    step_limit: int | None
    prompt_hash: str | None
    sampling_hash_map: dict[str, dict[str, str]] | None


# One predicate per staleness anchor. Keys are asserted equal to integrity.STALENESS_ANCHORS
# below, so the declaration and the behaviour cannot drift apart.
_ANCHOR_CHECKS: Final[dict[str, Callable[[dict, _AnchorInputs], bool]]] = {
    # The two identity hashes stale on ANY difference, including a stored empty — a cell with
    # no recorded spec hash is not a cell whose spec is known to match.
    "version_hash": lambda cell, ctx: cell.get("version_hash") != ctx.hashes.get(ctx.cid),
    "model_version": lambda cell, ctx: cell.get("model_version") != ctx.versions.get(ctx.model),
    "arm_hash": lambda cell, ctx: _arm_stale(cell, ctx.model, ctx.arm, ctx.arm_hash_map),
    "image_digest": lambda cell, ctx: _image_stale(cell, ctx.cid, ctx.digests),
    "step_limit": lambda cell, ctx: _limit_anchor_stale(cell, ctx.step_limit),
    "sampling_hash": lambda cell, ctx: _sampling_anchor_stale(
        cell, ctx.model, ctx.arm, ctx.sampling_hash_map
    ),
    "prompt_hash": lambda cell, ctx: _anchor_stale(cell, "prompt_hash", ctx.prompt_hash),
}
assert set(_ANCHOR_CHECKS) == set(integrity.STALENESS_ANCHORS), (
    "every declared staleness anchor needs a predicate, and nothing else may claim to be one"
)


def _arm_stale(
    cell: dict, model: str, arm: str, arm_hash_map: dict[str, dict[str, str]] | None
) -> bool:
    """Stale iff a NON-EMPTY stored arm_hash resolves AND differs — else never stales."""
    # Mirrors the digest-axis guard (_image_stale): no resolved anchor (arm_hash_map
    # absent, or the model/arm missing from it — e.g. a model with no reasoning block)
    # never marks a cell stale.
    if arm_hash_map is None:
        return False
    expected = arm_hash_map.get(model, {}).get(arm)
    if expected is None:
        return False
    # Guard on non-empty stored (like _image_stale): a legacy row (pre-arm-hash) has no
    # arm_hash column (""), so it degrades to a no-op instead of recomputing the
    # PAID cell. A genuinely new arm is MISSING (no row), not present-with-empty-hash.
    stored = str(cell.get("arm_hash", ""))
    return bool(stored) and stored != expected


def _image_stale(cell: dict, cid: str, digests: dict[str, str | None] | None) -> bool:
    """Stale iff a NON-EMPTY stored digest resolves AND differs — else never stales."""
    if digests is None:
        return False
    resolved = digests.get(cid)
    if resolved is None:
        return False
    # Single-arch assumption: the stored RepoDigest (docker inspect) and the resolved
    # manifest digest (imagetools inspect) coincide for a single-platform image. Guard
    # on non-empty stored (mirrors check_integrity.check_image_digests) so a first-live
    # cell — or a multi-arch manifest-list divergence that leaves the stored digest
    # unset — degrades to a no-op rather than recomputing the PAID cell forever.
    stored = cell.get("image_digest", "")
    return bool(stored) and resolved != stored


def _build_row(
    cid: str,
    model: str,
    outcome: dict,
    hashes: dict[str, str],
    versions: dict[str, str],
    pricing: dict[str, dict[str, float]],
    digests: dict[str, str | None] | None = None,
    arm: str = integrity.DEFAULT_REASONING,
    arm_hash_map: dict[str, dict[str, str]] | None = None,
    free_collection_models: Collection[str] | None = None,
) -> dict:
    in_tok = int(outcome.get("in_tok", 0))
    out_tok = int(outcome.get("out_tok", 0))
    real_cost = float(outcome.get("real_cost", 0.0))
    estimated_cost = integrity.estimated_cost(model, in_tok, out_tok, pricing)
    # `cost` is what every metric/strategy/kill-gate reads. Prefer the provider-returned
    # cache-aware real_cost (see infer._call_cost), and fall back to the registry's listing
    # estimate only when no measured cost exists at all (provider AND litellm both silent)
    # — otherwise those cells would cache cost=0 and score as free. NOTE: the fallback still
    # mixes the basis (measured where a cost is reported, listing-estimate otherwise) — a
    # known limitation to reconcile before running the live kill-gate.
    cost = real_cost if real_cost > 0 else estimated_cost
    row = {
        "challenge_id": cid,
        # `model` is the bare canonical weights identity; `lane` is the channel id the cell
        # actually ran on (the overlay row / registry name). Both are written, so a corpus
        # reader can group by identity while the resume key stays channel-unique.
        "model": versions.get(model, integrity.UNKNOWN_VERSION),
        "lane": model,
        # The row's arm identity comes from the caller (the (cid, model, arm) cell
        # classify_cells decided was missing/stale), not the harness outcome —
        # we label the row with the input arm, not anything inferred from the outcome.
        "reasoning": arm,
        "pass": outcome.get("pass", False),
        "cost": cost,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "calls": int(outcome.get("calls", 0)),
        "version_hash": hashes.get(cid, ""),
        "model_version": versions.get(model, integrity.UNKNOWN_VERSION),
        "arm_hash": (arm_hash_map or {}).get(model, {}).get(arm, ""),
        "real_cost": real_cost,
        "estimated_cost": estimated_cost,
        "timeout_flag": bool(outcome.get("timeout_flag", False)),
        "image_digest": _row_digest(cid, outcome, digests),
        "computed_at": str(outcome.get("computed_at") or _now_iso()),
        # Why the cell stopped (live outcomes set it; simulated/legacy outcomes derive it
        # from pass/timeout_flag so the column is always populated). See routing.censoring.
        "stop_reason": censoring.derive_stop_reason(
            passed=bool(outcome.get("pass", False)),
            timeout_flag=bool(outcome.get("timeout_flag", False)),
            stop_reason=str(outcome.get("stop_reason") or ""),
        ),
        # Collection-param provenance: the regime this cell ran under. Live
        # outcomes carry the actual caps the scaffold was given plus the scaffold-derived
        # hashes; any absent field (simulated/legacy/synthetic outcome) falls back to the
        # configured values or "" (grandfathered to a staleness no-op).
        "step_limit": str(outcome.get("step_limit") or config.live_step_limit()),
        "cost_limit": str(outcome.get("cost_limit") or config.live_cost_limit()),
        "scaffold_version": str(outcome.get("scaffold_version") or ""),
        "sampling_hash": str(outcome.get("sampling_hash") or ""),
        "prompt_hash": str(outcome.get("prompt_hash") or ""),
    }
    # Optional columns (`integrity.OPTIONAL_COLUMNS`) pass THROUGH from the outcome, and only
    # from the outcome. BLANK MEANS MISSING, FOREVER: a live cell carries what it actually
    # measured, and every other outcome path — simulated, legacy, re-derived, errored —
    # carries nothing and leaves the column blank. There is deliberately no fallback of any
    # kind here, not even to a configured value like `step_limit`'s: a default on a latency
    # column would publish "this cell took no time" as an affirmative measured claim.
    row.update({column: str(outcome.get(column) or "") for column in integrity.OPTIONAL_COLUMNS})
    # OBSERVED channel accounting — computed from the row's own evidence, independent of the
    # listing ENTITLEMENT (`billing:`). Written on every row so a corpus reader never has to
    # re-derive it; the same rule backfills historical rows (routing.scripts.backfill_channel).
    observed, channel_source = channel.observed_channel(
        model,
        real_cost,
        int(outcome.get("calls", 0)),
        str(row["computed_at"]),
        pricing=pricing,
    )
    row[integrity.CHANNEL_COLUMN] = observed
    row[integrity.CHANNEL_SOURCE_COLUMN] = channel_source
    # Write-time data-integrity wall: never silently persist a poison row. An ERROR
    # invariant (e.g. the $35 fingerprint: paid model ran but real_cost==0) aborts the
    # whole run loudly on the offending cell rather than caching a fabricated outcome.
    # `free_collection_models` is the run's admitted-free provenance: it is what lets
    # FREE_LANE_BILLED fire on a billed collection lane without treating the whole
    # `-explabs` historical namespace as free on a later corpus scan.
    validate.enforce_row(row, pricing, free_collection_models=free_collection_models)
    return row


def _row_digest(cid: str, outcome: dict, digests: dict[str, str | None] | None) -> str:
    """Prefer the digest the harness actually used (outcome); else the resolved map."""
    used = outcome.get("image_digest")
    if used:
        return str(used)
    if digests and digests.get(cid):
        return str(digests[cid])
    return ""


@dataclass(frozen=True)
class _LiveContext:
    """Read-only shared inputs for one batch of live cells (thread-safe: never mutated)."""

    hashes: dict[str, str]
    versions: dict[str, str]
    pricing: dict[str, dict[str, float]]
    digests: dict[str, str | None] | None
    arm_hash_map: dict[str, dict[str, str]] | None
    work_dir: Path
    timeout: int
    step_limit: int
    # Models this run admitted as free collection lanes. Threaded to the write-time wall so a
    # billed free lane trips FREE_LANE_BILLED on RUN PROVENANCE rather than the `-explabs`
    # suffix, which also labels legitimate paid promo history. Empty means `_build_row` was
    # reached without provenance (a direct/test call), so the wall conservatively abstains.
    free_collection_models: frozenset[str] = frozenset()


class RunAbortError(RuntimeError):
    """The whole live run is aborted for a systemic reason (message names the cause).

    Base for every run-level kill: retrying against a dead/throttled API or missing images
    only burns time and budget, so the run stops rather than marching through the remainder.
    """


class ContainerStartAbortError(RunAbortError):
    """Too many consecutive cells failed to start their container — abort fast, don't hammer."""


class _StartFailure:
    """Sentinel: a cell failed to START its container (image missing / registry throttle)."""


class _ApiUnusable:
    """Sentinel: a cell hit a SYSTEMIC API failure (dead key / no balance / provider down).

    Carries the reason for the abort message. Unlike _StartFailure/None, it aborts the run
    IMMEDIATELY (retrying a dead key is pointless) and is NEVER recorded pass=False.
    """

    def __init__(self, reason: str = "") -> None:
        self.reason = reason


class _ModelUnavailable:
    """Sentinel: this lane's model is permanently not served here (retired / plan-gated).

    A LANE fault: disable ``(model, provider)`` and continue the fleet. Never a fake pass=False
    and never a fleet-wide abort — other lanes and the same model on other providers stay live.
    """

    def __init__(self, reason: str = "") -> None:
        self.reason = reason


class _RateLimited:
    """Sentinel: a cell hit a per-lane RATE LIMIT (429 / throttle), not a model failure.

    No row is written — the cell stays MISSING and re-runs next time — and the lane is
    quarantined so the scheduler backs off instead of hammering the provider. ``retry_after``
    carries the server's own backoff floor when it published one, so the quarantine cannot
    return earlier than the provider asked.
    """

    def __init__(self, reason: str = "", retry_after: float | None = None) -> None:
        self.reason = reason
        self.retry_after = retry_after


class _PermanentPreCallFailure:
    """Sentinel: a permanent model error BEFORE any successful call (e.g. a provider 400).

    The harness records such a cell as ``pass=False, stop_reason=unsolved`` with ``calls=0``;
    as a row that is a poison invariant (an unsolved capability fail must have actually run),
    so writing it would abort the whole campaign. It is genuinely never-ran, though, so the
    correct classification is a skip: leave the cell MISSING to retry/resume next run, count
    it toward the consecutive catch-all, and keep going. Distinct from ``_StartFailure``
    (container/registry) and ``_ApiUnusable`` (systemic dead key — an immediate abort).
    """


_PERMANENT_PRE_CALL_FAILURE: Final = _PermanentPreCallFailure()


def _is_never_ran_poison(exc: validate.DataIntegrityError) -> bool:
    """True iff a poison row's ONLY fault is that it never ran (``UNSOLVED`` with ``calls=0``).

    Narrow on purpose: a row that DID run can never carry this code, so the poison-row
    guarantee — a genuinely invalid produced row still aborts the run loudly — is intact.
    """
    return {v.code for v in exc.violations} == {validate.UNSOLVED_NOT_RUN}


def _is_free_lane_billed_poison(exc: validate.DataIntegrityError) -> bool:
    """True iff a poison row's ONLY fault is that a free lane was billed (``FREE_LANE_BILLED``).

    Narrow on purpose, like ``_is_never_ran_poison``: any other poison code still aborts the
    run. This one is a LANE fault — the lane cannot be trusted to stay $0 — so it disables
    that lane and continues the fleet rather than killing every remaining lane's collection.
    """
    return bool(exc.violations) and all(v.code == validate.FREE_LANE_BILLED for v in exc.violations)


def _no_usable_free_lanes(lanes: lane_scheduler.LaneScheduler) -> bool:
    """True iff every configured lane is disabled or structurally unusable (zero remain)."""
    structural = set(lanes.structural_refusals())
    for lane in lanes.limits:
        if lane in structural:
            continue
        if lanes.lane_state(lane).disabled_reason is None:
            return False
    return True


# _run_one_cell returns this (not None) so the loop can tell a systemic container-start
# failure from a legit per-cell error (agent ran but failed grading, spec-hash, etc.).
_START_FAILURE: Final = _StartFailure()

# Substrings that mark a docker container-start / registry-throttle failure (case-folded).
# Distinct from a graded-but-unresolved outcome, which is a normal pass=False, not a raise.
_START_FAILURE_SIGNATURES: Final[tuple[str, ...]] = (
    "non-zero exit status 125",
    "rc=125",
    "429",
    "too many requests",
    "toomanyrequests",
    "starting container",
)


def _is_container_start_failure(exc: BaseException) -> bool:
    """True iff an exception looks like a docker container-start / registry-throttle failure."""
    text = str(exc).lower()
    return any(sig in text for sig in _START_FAILURE_SIGNATURES)


def _abort_start_failures(n: int) -> NoReturn:
    """Emit the kill message and raise — the run cannot make progress against missing images."""
    msg = (
        f"aborting: {n} consecutive container-start failures — images likely missing or "
        "registry throttled; pre-stage images with scripts/benchmark/prepull_swebench_images.py"
    )
    print(f"  {msg}", file=sys.stderr)
    raise ContainerStartAbortError(msg)


def _abort_api_unusable(reason: str) -> NoReturn:
    """Abort the WHOLE run on the first unambiguous API-unusable cell (retrying is futile)."""
    msg = (
        f"aborting: API unusable ({reason}) — invalid/empty key, no balance/quota, or provider "
        "down. This is NOT a model failure; refusing to record fake failures across the run."
    )
    print(f"  {msg}", file=sys.stderr)
    raise RunAbortError(msg)


def _abort_consecutive_failures(n: int) -> NoReturn:
    """Abort the whole run after N consecutive cell failures (catch-all for any systemic cause)."""
    msg = (
        f"aborting: {n} consecutive cell failures — the API or harness is systematically failing "
        "(dead key, no balance, persistent rate-limit, or docker). Not marching through the rest."
    )
    print(f"  {msg}", file=sys.stderr)
    raise RunAbortError(msg)


def _abort_no_free_lanes() -> NoReturn:
    """Abort only when billing has disabled EVERY free lane — nothing left to collect from."""
    msg = (
        "aborting: every free lane was disabled after billing a cell (FREE_LANE_BILLED); "
        "no lane remains, so the $0 campaign cannot continue"
    )
    print(f"  {msg}", file=sys.stderr)
    raise RunAbortError(msg)


class _FailureTracker:
    """Thread-safe consecutive/start-failure counters that raise the run-level aborts.

    Shared across concurrent workers so one systemic failure aborts the whole run;
    a fresh per-call instance reproduces the old single-threaded per-batch counting.
    """

    def __init__(
        self, max_start_failures: int | None, max_consecutive_failures: int | None
    ) -> None:
        self._lock = threading.Lock()
        self._start = 0
        self._consecutive = 0
        self._max_start = max_start_failures
        self._max_consecutive = max_consecutive_failures

    def record_start_failure(self) -> None:
        """A container-start failure: bumps BOTH the start and the catch-all counters."""
        with self._lock:
            self._start += 1
            self._consecutive += 1
            start, consecutive = self._start, self._consecutive
        self._maybe_abort(start, consecutive)

    def record_soft_failure(self) -> None:
        """Any other no-row failure (skip): bumps only the consecutive catch-all counter."""
        with self._lock:
            self._consecutive += 1
            consecutive = self._consecutive
        self._maybe_abort(None, consecutive)

    def record_success(self) -> None:
        """A produced row (any grade) resets both counters."""
        with self._lock:
            self._start = 0
            self._consecutive = 0

    def _maybe_abort(self, start: int | None, consecutive: int) -> None:
        if start is not None and self._max_start is not None and start >= self._max_start:
            _abort_start_failures(start)
        if self._max_consecutive is not None and consecutive >= self._max_consecutive:
            _abort_consecutive_failures(consecutive)


def _run_one_cell(
    cell: tuple[str, str, str], ctx: _LiveContext
) -> (
    dict
    | _StartFailure
    | _ApiUnusable
    | _ModelUnavailable
    | _RateLimited
    | _PermanentPreCallFailure
    | None
):
    """Run one (challenge, model, arm) cell → row, a systemic-failure sentinel, or None."""
    # _ApiUnusable → dead/no-balance API (run aborts, no fake pass=False); _ModelUnavailable →
    # model not served on this provider (lane disabled, no fake pass=False); _RateLimited →
    # per-lane 429/throttle (no row, MISSING, lane quarantined); _START_FAILURE →
    # container-start failure; None → any other per-cell error (skip). Pure/thread-safe.
    cid, model, arm = cell
    # The ENTIRE body is guarded, not just the harness call: in a worker thread an
    # unhandled raise (import, spec-hash, row-build) would propagate out of the pool's
    # result loop and discard every already-collected (paid) row. Returning a sentinel/None
    # keeps isolation total — one bad cell never costs the batch.
    try:
        outcome = infer.run_live_cell(
            cid,
            model,
            work_dir=ctx.work_dir,
            run_id=f"live-{cid}-{model}-{arm}",
            timeout=ctx.timeout,
            arm=arm,
            step_limit=ctx.step_limit,
        )
        cell_hashes = {
            **ctx.hashes,
            cid: integrity.swebench_spec_hash(cid) or ctx.hashes.get(cid, ""),
        }
        return _build_row(
            cid,
            model,
            outcome,
            cell_hashes,
            ctx.versions,
            ctx.pricing,
            ctx.digests,
            arm,
            ctx.arm_hash_map,
            ctx.free_collection_models,
        )
    except validate.DataIntegrityError as exc:
        # Poison row (e.g. paid model ran but real_cost==0). This must NOT be swallowed
        # into a per-cell skip — it aborts the whole run loudly, so corrupt data is
        # never silently persisted or marched past. Re-raise past the broad handler.
        # EXCEPT the never-ran case: a permanent model error before any call (calls=0,
        # stop_reason=unsolved) is a skip, not a corrupt measurement — classify it so one
        # bad provider/model pair cannot crash the whole campaign.
        if _is_never_ran_poison(exc):
            print(f"  permanent-pre-call {cid}:{model}:{arm} — {exc}", file=sys.stderr)
            return _PERMANENT_PRE_CALL_FAILURE
        raise
    except (infer.ApiUnusableError, infer.MissingApiKeysError) as exc:
        # Systemic: NOT a model failure. Never fabricate pass=False — abort the run.
        print(f"  api-unusable {cid}:{model}:{arm} — {exc}", file=sys.stderr)
        return _ApiUnusable(str(exc))
    except infer.ModelUnavailableError as exc:
        # This (model, provider) is not served here — a LANE fault, not a cell failure. The
        # lane is disabled and the fleet continues; recording pass=False would be fabricated.
        print(f"  model-unavailable {cid}:{model}:{arm} — {exc}", file=sys.stderr)
        return _ModelUnavailable(str(exc))
    except Exception as exc:  # noqa: BLE001 (per-cell isolation: skip one, never fabricate)
        # API-unusable is classified earlier (a 429 whose body is an insufficient-balance
        # message is a dead lane, not a throttle). A typed litellm RateLimitError or explicit
        # rate-limit text is a per-lane throttle; only then does the container-start probe run.
        if infer.is_rate_limited(exc):
            print(f"  rate-limited {cid}:{model}:{arm} — {exc}", file=sys.stderr)
            return _RateLimited(str(exc), infer.retry_after_seconds(exc))
        start_fail = _is_container_start_failure(exc)
        kind = "start-fail" if start_fail else "skip"
        print(f"  {kind} {cid}:{model}:{arm} — {exc}", file=sys.stderr)
        return _START_FAILURE if start_fail else None


def _over_budget(spent: float, max_cost: float | None) -> bool:
    """True once cumulative real_cost has reached the ceiling (None = unbounded).

    NOTE the trap this shape creates: ``--max-cost 0`` is a SILENT NO-OP, not a $0 cap —
    ``spent >= 0`` is true before the first call, so the run stops with zero cells and
    never confirms. A $0 run uses ``--require-zero-cost`` instead (plus the campaign's
    positive ``live.cost_limit``).
    """
    return max_cost is not None and spent >= max_cost


def _group_by_challenge(
    cells: list[tuple[str, str, str]],
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Group cells by challenge, preserving each challenge's first-appearance order."""
    # Challenge-major batching: a challenge split across the (missing + stale)
    # concatenation is reunited at its first appearance so its whole model×arm set is
    # grouped together — the parallel path runs each challenge atomically (a ceiling cut
    # leaves a prefix of FULLY-covered challenges), while the serial path enforces a hard
    # per-cell ceiling and may stop mid-challenge.
    groups: dict[str, list[tuple[str, str, str]]] = {}
    order: list[str] = []
    for cell in cells:
        cid = cell[0]
        if cid not in groups:
            groups[cid] = []
            order.append(cid)
        groups[cid].append(cell)
    return [(cid, groups[cid]) for cid in order]


def _row_tokens(row: dict) -> int:
    """A completed row's ACTUAL prompt + completion tokens, the charge for the token windows.

    Cached input tokens are counted conservatively (the charge only ever over-paces), so a
    provider that excludes them from its TPM is not under-charged.
    """
    return int(row.get("in_tok") or 0) + int(row.get("out_tok") or 0)


def _run_scheduled_batch(
    batch: list[tuple[str, str, str]],
    ctx: _LiveContext,
    lanes: lane_scheduler.LaneScheduler,
    tracker: _FailureTracker,
    checkpoint: Callable[[dict], None] | None,
    spent: float,
    hard: float | None,
    max_cost: float | None,
    overshoot_note: str,
) -> tuple[list[dict], float, bool]:
    """Pull-loop admission for one challenge's cells; blocked lanes leave cells MISSING.

    ``next_ready`` re-drains whenever a lane frees, so a lane that only just recovered is
    picked up at the next pull. A stall older than ``lanes.stall_timeout_s`` yields the
    challenge, leaving blocked cells MISSING for the next run to reclassify. A lane whose
    published limits can never fit a single request is refused up front (named at scheduler
    build), so its cells are dropped here without waiting out a stall they can never clear.
    """
    rows: list[dict] = []
    refused = lanes.structural_refusals()
    if refused:
        pending = [cell for cell in batch if cell[1] not in refused]
        skipped = len(batch) - len(pending)
        if skipped:
            print(
                f"  skipping {skipped} cell(s) on structurally-refused lane(s)",
                file=sys.stderr,
            )
    else:
        pending = list(batch)
    deadline = time.monotonic() + lanes.stall_timeout_s
    while pending:
        if _over_budget(spent, hard):
            print(
                f"  cost ceiling ${max_cost:g}{overshoot_note} reached"
                f" — stopping ({len(rows)} cells done)",
                file=sys.stderr,
            )
            return rows, spent, True
        cell = lanes.next_ready(pending, time.monotonic())
        if isinstance(cell, lane_scheduler.Stalled):
            if time.monotonic() >= deadline:
                print(
                    f"  lane stall exceeded {lanes.stall_timeout_s:g}s — leaving "
                    f"{len(pending)} cell(s) MISSING to re-run next time",
                    file=sys.stderr,
                )
                break
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            continue
        if not lanes.admit(cell, time.monotonic()):
            continue
        pending.remove(cell)
        try:
            row = _run_one_cell(cell, ctx)
        except validate.DataIntegrityError as exc:
            # The never-ran case is already a skip inside _run_one_cell; what reaches here is a
            # produced poison row. If its ONLY fault is that a free lane was billed, this is a
            # LANE fault: never write the poison row, disable that lane by name, and continue
            # the other lanes. Any other poison code still aborts loudly. The cell was removed
            # from ``pending`` above, so the billed cell is never re-run in this batch.
            if not _is_free_lane_billed_poison(exc):
                raise
            reason = f"{validate.FREE_LANE_BILLED}: {exc}"
            lanes.disable(cell[1], reason)
            print(f"  disabled free lane {cell[1]}: {reason}", file=sys.stderr)
            tracker.record_soft_failure()
            if _no_usable_free_lanes(lanes):
                _abort_no_free_lanes()
            continue
        if isinstance(row, _ApiUnusable):
            _abort_api_unusable(row.reason)  # immediate: retrying a dead key is pointless
        if isinstance(row, _ModelUnavailable):
            # A model not served on this provider: disable this lane, keep the fleet. The cell
            # stays MISSING (no fake pass=False); the lane must never be re-probed or re-run.
            reason = f"model unavailable: {row.reason}"
            lanes.disable(cell[1], reason)
            print(f"  disabled unavailable lane {cell[1]}: {reason}", file=sys.stderr)
            tracker.record_soft_failure()
            continue
        if isinstance(row, _RateLimited):
            lanes.record_rate_limit(cell[1], time.monotonic(), retry_after=row.retry_after)
            tracker.record_soft_failure()
            continue
        if isinstance(row, _StartFailure):
            tracker.record_start_failure()
            continue
        if isinstance(row, _PermanentPreCallFailure):
            tracker.record_soft_failure()  # no row: the cell stays MISSING to retry/resume
            continue
        if row is None:
            tracker.record_soft_failure()
            continue
        tracker.record_success()  # any produced row resets both consecutive counters
        lanes.complete(
            cell, time.monotonic(), calls=int(row.get("calls", 0) or 0), tokens=_row_tokens(row)
        )
        rows.append(row)
        spent += float(row["real_cost"])
        if checkpoint is not None:
            checkpoint(row)
    return rows, spent, False


def _run_cells_serial(
    cells: list[tuple[str, str, str]],
    ctx: _LiveContext,
    max_cost: float | None,
    checkpoint: Callable[[dict], None] | None = None,
    max_cost_overshoot: float = 0.0,
    max_start_failures: int | None = None,
    max_consecutive_failures: int | None = None,
    failures: _FailureTracker | None = None,
    lanes: lane_scheduler.LaneScheduler | None = None,
) -> list[dict]:
    """Serial (workers=1), challenge-major: no NEW challenge starts past ``max_cost``; a started
    one finishes up to ``max_cost + max_cost_overshoot`` (overshoot=0 ⇒ old hard stop). Aborts
    on the first API-unusable cell, or ``max_consecutive_failures`` of any kind (catch-all).

    With ``lanes`` supplied, each challenge's cells are admitted through the lane scheduler's
    pull loop instead of run directly; a lane that is throttled quarantines, and cells it
    blocks stay MISSING once the per-challenge stall budget expires.
    """
    # ``failures`` optionally injects a SHARED tracker (the ladder's cross-challenge counter);
    # absent, a fresh local one reproduces the original per-batch counting exactly.
    # Two ceilings decorrelate "don't start new work" from "don't waste in-flight work":
    # the boundary check on max_cost refuses a fresh challenge, while the per-cell check on
    # ``hard`` lets the current challenge run into the overshoot budget. With overshoot 0,
    # hard == max_cost and both collapse to the original per-cell hard stop.
    hard = None if max_cost is None else max_cost + max_cost_overshoot
    overshoot_note = f" (+${max_cost_overshoot:g} overshoot)" if max_cost_overshoot > 0 else ""
    rows: list[dict] = []
    spent = 0.0
    tracker = failures or _FailureTracker(max_start_failures, max_consecutive_failures)
    for _cid, batch in _group_by_challenge(cells):
        if _over_budget(spent, max_cost):
            print(
                f"  cost ceiling ${max_cost:g}{overshoot_note} reached"
                f" — stopping ({len(rows)} cells done)",
                file=sys.stderr,
            )
            return rows
        if lanes is not None:
            batch_rows, spent, stopped = _run_scheduled_batch(
                batch, ctx, lanes, tracker, checkpoint, spent, hard, max_cost, overshoot_note
            )
            rows.extend(batch_rows)
            if stopped:
                return rows
            continue
        for cell in batch:
            if _over_budget(spent, hard):
                print(
                    f"  cost ceiling ${max_cost:g}{overshoot_note} reached"
                    f" — stopping ({len(rows)} cells done)",
                    file=sys.stderr,
                )
                return rows
            row = _run_one_cell(cell, ctx)
            if isinstance(row, _ApiUnusable):
                _abort_api_unusable(row.reason)  # immediate: retrying a dead key is pointless
            if isinstance(row, _ModelUnavailable):
                # No lane object on this path (the lane-aware path returned above): skip the
                # cell and leave it MISSING; the lane preflight owns the permanent disable.
                tracker.record_soft_failure()
                continue
            if isinstance(row, _RateLimited):
                tracker.record_soft_failure()  # no row: the cell stays MISSING to re-run
                continue
            if isinstance(row, _StartFailure):
                tracker.record_start_failure()
                continue
            if isinstance(row, _PermanentPreCallFailure):
                tracker.record_soft_failure()  # no row: the cell stays MISSING to retry/resume
                continue
            if row is None:
                tracker.record_soft_failure()
                continue
            tracker.record_success()  # any produced row resets both consecutive counters
            rows.append(row)
            spent += float(row["real_cost"])
            if checkpoint is not None:
                checkpoint(row)
    return rows


def _run_challenge_batch(
    batch: list[tuple[str, str, str]], ctx: _LiveContext, pool: ThreadPoolExecutor
) -> tuple[list[dict], int, str | None]:
    """Run one challenge's cells concurrently; return (rows in input order, start-failures,
    api-unusable-reason). A non-None reason means at least one cell hit a systemic API failure.
    """
    # Barrier: as_completed drains only THIS challenge's futures before the caller
    # advances, so a challenge fully completes before the next starts. Rows are
    # re-sorted by input index → byte-identical to serial for the same cell set.
    results: dict[int, dict] = {}
    start_failures = 0
    api_unusable: str | None = None
    futures = {pool.submit(_run_one_cell, cell, ctx): i for i, cell in enumerate(batch)}
    for fut in as_completed(futures):
        row = fut.result()
        if isinstance(row, _ApiUnusable):
            api_unusable = api_unusable or row.reason
        elif isinstance(row, _ModelUnavailable):
            continue  # no row: the cell stays MISSING; the lane preflight owns the disable
        elif isinstance(row, _RateLimited):
            continue  # no row: the cell stays MISSING and is tallied with the batch's failures
        elif isinstance(row, _StartFailure):
            start_failures += 1
        elif isinstance(row, _PermanentPreCallFailure):
            continue  # no row: the cell stays MISSING and is tallied with the batch's failures
        elif row is not None:
            results[futures[fut]] = row
    return [results[i] for i in sorted(results)], start_failures, api_unusable


def _run_cells_parallel(
    cells: list[tuple[str, str, str]],
    ctx: _LiveContext,
    workers: int,
    max_cost: float | None,
    checkpoint: Callable[[dict], None] | None = None,
    max_cost_overshoot: float = 0.0,  # noqa: ARG001 (accepted for parity; see comment below)
    max_start_failures: int | None = None,
    max_consecutive_failures: int | None = None,
) -> list[dict]:
    """Challenge-major thread pool: each challenge's cells complete (up to ``workers`` at
    once) before the next starts, so a ceiling cut leaves a clean prefix of FULLY-covered
    challenges.
    """
    # ``max_cost_overshoot`` is a no-op here: the boundary check on max_cost already lets an
    # in-flight challenge complete atomically — that natural, challenge-sized overrun IS the
    # overshoot on this path (it has no separate per-cell hard ceiling to relax).
    # One pool, reused across challenges; concurrency is INTRA-challenge (its model×arm
    # cells). The ceiling is checked at challenge boundaries only — a started challenge
    # always finishes (never left partial). ``checkpoint`` and ``spent`` run HERE, on the
    # main thread in input order — worker threads (_run_one_cell) never write, so the
    # read-modify-write in merge_rows stays race-free without a lock, and the persisted
    # bytes stay order-independent (identical to serial for a completed cell set).
    # Start-failure abort is PER-BATCH here (not truly per-cell like serial): a batch that
    # produced ≥1 row resets the counter, else its start-failures accumulate toward the cap.
    rows: list[dict] = []
    spent = 0.0
    start_failures = 0
    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _cid, batch in _group_by_challenge(cells):
            if _over_budget(spent, max_cost):
                print(
                    f"  cost ceiling ${max_cost:g} reached — stopping ({len(rows)} cells done)",
                    file=sys.stderr,
                )
                break
            batch_rows, batch_start_failures, api_unusable = _run_challenge_batch(batch, ctx, pool)
            if api_unusable is not None:
                _abort_api_unusable(api_unusable)  # immediate: retrying a dead key is pointless
            for row in batch_rows:
                rows.append(row)
                spent += float(row["real_cost"])
                if checkpoint is not None:
                    checkpoint(row)
            if batch_rows:
                start_failures = 0  # a productive batch resets both consecutive counters
                consecutive_failures = 0
            else:
                # A wholly-failed batch: count every failed cell toward the catch-all.
                start_failures += batch_start_failures
                consecutive_failures += len(batch)
                if max_start_failures is not None and start_failures >= max_start_failures:
                    _abort_start_failures(start_failures)
                if (
                    max_consecutive_failures is not None
                    and consecutive_failures >= max_consecutive_failures
                ):
                    _abort_consecutive_failures(consecutive_failures)
    return rows


def _live_context(
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None,
    arm_hash_map: dict[str, dict[str, str]] | None,
    timeout: int,
    step_limit: int,
    free_collection_models: frozenset[str] = frozenset(),
) -> _LiveContext:
    """Assemble the read-only shared inputs for one batch of live cells."""
    return _LiveContext(
        hashes=hashes,
        versions=versions,
        pricing=config._pricing_dict(),
        digests=digests,
        arm_hash_map=arm_hash_map,
        work_dir=Path.cwd() / "benchmark" / "runner" / "artifacts",
        timeout=timeout,
        step_limit=step_limit,
        free_collection_models=free_collection_models,
    )


def run_live_cells(
    cells: list[tuple[str, str, str]],
    matrix: dict,  # noqa: ARG001 (kept for signature stability; spec is loaded per-cell)
    hashes: dict[str, str],
    versions: dict[str, str],
    timeout: int,
    verbose: bool,  # noqa: ARG001 (kept for signature stability; per-cell logging is unconditional)
    digests: dict[str, str | None] | None = None,
    arm_hash_map: dict[str, dict[str, str]] | None = None,
    workers: int = 1,
    max_cost: float | None = None,
    results_path: Path | None = None,
    max_cost_overshoot: float = 0.0,
    max_start_failures: int | None = None,
    max_consecutive_failures: int | None = None,
    write_lock: threading.Lock | None = None,
    failures: _FailureTracker | None = None,
    step_limit: int = infer._DEFAULT_STEP_LIMIT,
    mode: Literal["supersede", "replicate"] = "supersede",
    lanes: lane_scheduler.LaneScheduler | None = None,
    campaign: campaign_scheduler.CampaignRun | None = None,
) -> list[dict]:
    """Delegate each (challenge, model, arm) cell to the real SWE-bench harness executor.

    Honors ``workers``, ``max_cost`` (+overshoot), ``results_path`` persistence, and the
    start-failure / consecutive-failure / API-unusable run-level aborts. When ``lanes`` is
    supplied, admission is serialized through the lane scheduler's pull loop (a per-lane rate
    limiter serializes admission regardless of worker count); ``workers`` otherwise picks the
    challenge-major parallel path. When ``campaign`` is supplied (the free-campaign path), the
    plan's priority order, domination stop and per-model phase gate drive the pull loop.
    """
    # ``write_lock``/``failures`` let a concurrent caller (the ladder) serialize writes and
    # share the counters across challenge threads.
    # Structural wall (defense-in-depth): never spend uncached budget on the ENABLED (paid)
    # ladder. An uncached enabled model resends the full context every agent turn at full
    # input price — a cell that churns then fails still bills for all of it (qwen3-max billed
    # 50x its recorded cost). Scoped to enabled deliberately: `--extra-models` ids are an
    # EXPLICIT collection opt-in, and the free-promo channel they ride reports no cache-read
    # rate and no cached tokens — refusing on them would block a $0 run over a cost feature
    # the channel does not offer (free-ness is guarded separately by the campaign's
    # real_cost == 0 check on every written row). main() checks the enabled set too; this
    # backstop protects any other run_live_cells caller the same way.
    enabled = set(config.enabled_models())
    uncached = config.models_missing_cache(sorted({m for _, m, _ in cells} & enabled))
    if uncached:
        raise ValueError(
            f"benchmark refuses to run uncached enabled models (no cache-read discount): "
            f"{uncached}. Remove them from benchmark.yaml or give them "
            "cache_read_cost_per_1m in the model registry."
        )
    # The run's admitted free-lane provenance: of the models in this batch, which rode a $0
    # collection lane (`-explabs` channel or a declared-free overlay row). `_build_row` threads
    # it to the write-time wall so a billed free lane trips FREE_LANE_BILLED, while a plain
    # read of results.csv carries no provenance and never treats `-explabs` history as free.
    free_collection_models = frozenset(m for _, m, _ in cells if _is_free_lane(m))
    ctx = _live_context(
        hashes,
        versions,
        digests,
        arm_hash_map,
        timeout,
        step_limit,
        free_collection_models,
    )
    # Persist each completed cell through merge_rows (history-log supersession +
    # key-upsert + atomic write preserved); None ⇒ old in-memory-only behaviour.
    checkpoint = (
        partial(_checkpoint_row, path=results_path, lock=write_lock, mode=mode)
        if results_path is not None
        else None
    )
    if campaign is not None:
        # Free-campaign path: the priority plan, domination stop and phase gate drive the
        # pull loop. The tracker keeps the run-level abort caps that the serial path enforces.
        tracker = failures or _FailureTracker(max_start_failures, max_consecutive_failures)
        return campaign_scheduler.run_cells(campaign, ctx, tracker=tracker, checkpoint=checkpoint)
    if workers <= 1 or lanes is not None:
        return _run_cells_serial(
            cells,
            ctx,
            max_cost,
            checkpoint,
            max_cost_overshoot,
            max_start_failures,
            max_consecutive_failures,
            failures,
            lanes,
        )
    return _run_cells_parallel(
        cells,
        ctx,
        workers,
        max_cost,
        checkpoint,
        max_cost_overshoot,
        max_start_failures,
        max_consecutive_failures,
    )


def _checkpoint_row(
    row: dict,
    path: Path,
    lock: threading.Lock | None = None,
    mode: Literal["supersede", "replicate"] = "supersede",
) -> None:
    """Persist one completed row through merge_rows; ``lock`` serializes concurrent writers."""
    # The lock is what makes mode="replicate" safe under concurrency: merge_rows assigns the
    # replicate index inside its own read-modify-write, so holding the lock across that whole
    # call is what stops two writers both reading max(rep)==0 and both claiming rep 1.
    if lock is None:
        merge_rows([row], path, mode=mode)
        return
    with lock:
        merge_rows([row], path, mode=mode)


def _run_and_merge(
    cells: list[tuple[str, str, str]],
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None,
    arm_hash_map: dict[str, dict[str, str]] | None,
    *,
    timeout: int,
    workers: int,
    max_cost: float | None,
    results_path: Path,
    max_cost_overshoot: float = 0.0,
    max_start_failures: int | None = None,
    max_consecutive_failures: int | None = None,
    write_lock: threading.Lock | None = None,
    failures: _FailureTracker | None = None,
    step_limit: int = infer._DEFAULT_STEP_LIMIT,
    mode: Literal["supersede", "replicate"] = "supersede",
    lanes: lane_scheduler.LaneScheduler | None = None,
    campaign: campaign_scheduler.CampaignRun | None = None,
) -> int:
    """Run a cell list through the challenge-major executor and upsert results.csv.

    Shared by full-matrix ``main`` and the adaptive ``collect_phase``; cells are
    checkpointed as they complete and the ``models_missing_cache`` wall (enabled models only)
    still applies.
    """
    # ``write_lock``/``failures`` serialize the merge and share the counters under concurrency.
    new_rows = run_live_cells(
        cells,
        {},
        hashes,
        versions,
        timeout,
        False,
        digests,
        arm_hash_map,
        workers=workers,
        max_cost=max_cost,
        results_path=results_path,
        max_cost_overshoot=max_cost_overshoot,
        max_start_failures=max_start_failures,
        max_consecutive_failures=max_consecutive_failures,
        write_lock=write_lock,
        failures=failures,
        step_limit=step_limit,
        mode=mode,
        lanes=lanes,
        campaign=campaign,
    )
    if write_lock is not None:
        with write_lock:
            return merge_rows(new_rows, results_path, mode=mode)
    return merge_rows(new_rows, results_path, mode=mode)


def collect_phase(
    tasks: list[str],
    models: list[str],
    cache: dict,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None = None,
    *,
    live: bool = False,
    timeout: int = 600,
    workers: int = 1,
    max_cost: float | None = None,
    results_path: Path | None = None,
    max_cost_overshoot: float = 0.0,
    max_start_failures: int | None = None,
    max_consecutive_failures: int | None = None,
    write_lock: threading.Lock | None = None,
    failures: _FailureTracker | None = None,
    step_limit: int = infer._DEFAULT_STEP_LIMIT,
) -> CellStatus:
    """Classify one (tasks x models) block against the cache and, when live, run+merge it.

    The adaptive ``collect`` mode's phase primitive — reuses _arm_context/classify_cells/
    run_live_cells verbatim; simulated (live=False) only classifies (state in results.csv).
    """
    # ``write_lock``/``failures`` let the ladder run this concurrently across challenges.
    selected_arms, arm_hash_map = _arm_context(tasks, models)
    status = classify_cells(
        tasks, models, cache, hashes, versions, digests, selected_arms, arm_hash_map
    )
    if live and results_path is not None:
        # Two calls, two intents: cells the cache lacks or has staled are SUPERSEDED, extra
        # observations of correct cells are REPLICATED. merge_rows cannot infer which from
        # the rows, so the split happens here, at the only place that knows.
        for cells, mode in ((status.to_run, "supersede"), (status.replicate, "replicate")):
            if not cells:
                continue
            _run_and_merge(
                cells,
                hashes,
                versions,
                digests,
                arm_hash_map,
                timeout=timeout,
                workers=workers,
                max_cost=max_cost,
                results_path=results_path,
                max_cost_overshoot=max_cost_overshoot,
                max_start_failures=max_start_failures,
                max_consecutive_failures=max_consecutive_failures,
                write_lock=write_lock,
                failures=failures,
                step_limit=step_limit,
                mode=mode,  # type: ignore[arg-type]
            )
    return status


_RowKey = tuple[str, str, str, int]


def _run_replicates(
    status: CellStatus,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None,
    arm_hash_map: dict[str, dict[str, str]] | None,
    args: argparse.Namespace,
    step_limit: int,
    *,
    lanes: lane_scheduler.LaneScheduler | None = None,
) -> int:
    """Collect the extra observations (reps 1..R-1) under mode='replicate'; 0 when disabled."""
    if not status.replicate:
        return 0
    return _run_and_merge(
        status.replicate,
        hashes,
        versions,
        digests,
        arm_hash_map,
        timeout=args.timeout,
        workers=args.workers,
        max_cost=args.max_cost,
        results_path=config.results_csv_path(),
        max_cost_overshoot=args.max_cost_overshoot,
        max_start_failures=args.max_start_failures,
        max_consecutive_failures=args.max_consecutive_failures,
        step_limit=step_limit,
        mode="replicate",
        lanes=lanes,
    )


def _row_key(row: dict) -> _RowKey:
    """The results.csv cache key: (challenge_id, lane, reasoning, rep).

    ``lane`` is the CHANNEL identity; ``model`` is the bare canonical weights id. Keying on
    ``lane`` keeps a direct id and its `-explabs` mirror (same weights, two channels) as
    distinct cells, which keying on the canonical ``model`` would collide. A pre-migration row
    has no ``lane`` and falls back to ``model`` (which was the channel id then).
    """
    # Literal string-equality on whatever `reasoning` the row carries (no alias
    # resolution here — that is a read-time concern, config.load_results). Two
    # distinct arm values for the same (challenge, lane) are DISTINCT keys, so
    # they never collide or archive one another as history.
    #
    # `rep` is NORMALISED INSIDE THE KEY (blank -> 0), and that is load-bearing rather than
    # tidy: the 1265 committed rows carry a blank rep while a freshly written first
    # observation carries "0", so keying on the raw spelling would make one cell look like
    # two rows and `authenticity.check_duplicate_keys` would report the file as fraudulent.
    reasoning = str(row.get("reasoning") or integrity.DEFAULT_REASONING)
    lane = str(row.get("lane") or row["model"])
    return (row["challenge_id"], lane, reasoning, integrity.rep_index(row))


def _cell_key(row: dict) -> tuple[str, str, str]:
    """The (challenge, model, arm) CELL a row observes — its key minus the replicate index."""
    return _row_key(row)[:3]


def _read_raw_rows(path: Path) -> dict[_RowKey, dict]:
    rows: dict[_RowKey, dict] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[_row_key(row)] = row
    return rows


def _write_raw_rows(rows: dict[_RowKey, dict], path: Path) -> None:
    """Atomically rewrite results.csv: write a sibling temp, then os.replace onto target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: _row_key(r))
    # The temp name carries pid+thread so two writers can never share one temp file. The
    # collector holds a process-lifetime single-instance lock, but this keeps a direct or
    # test caller from truncating a sibling writer's temp even so.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(integrity.RESULTS_FIELDS))
        w.writeheader()
        for row in ordered:
            w.writerow({k: row.get(k, "") for k in integrity.RESULTS_FIELDS})
    os.replace(tmp, path)


def _history_path(results_path: Path) -> Path:
    """Append-only supersession log, gitignored under routing/artifacts/ (beside
    results.csv — NOT benchmark/artifacts/, which is not gitignored).
    """
    return results_path.parent / "artifacts" / "results_history.csv"


def _row_differs(old: dict, new: dict) -> bool:
    """True iff any cache column differs between a stored row and its replacement."""
    return any(str(old.get(k, "")) != str(new.get(k, "")) for k in integrity.RESULTS_FIELDS)


def _append_history(rows: list[dict], path: Path) -> None:
    """Append superseded rows (append-only) with a supersession timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(HISTORY_FIELDS))
        if new_file:
            w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in integrity.RESULTS_FIELDS}
            out["superseded_at"] = ts
            w.writerow(out)


def _anchors_differ(old: dict, new: dict) -> bool:
    """True iff any STALENESS anchor moved between a stored row and its replacement."""
    # Mirrors the grandfather guard `_anchor_stale`: an EMPTY stored anchor on a legacy row
    # degrades to a NO-OP rather than licensing an overwrite. A row written before a column
    # existed proves nothing about that column, and reading its blank as "different" would
    # turn every legacy cell into a free re-run permit.
    return any(
        bool(str(old.get(anchor, ""))) and str(old.get(anchor, "")) != str(new.get(anchor, ""))
        for anchor in integrity.STALENESS_ANCHORS
    )


def _next_rep(existing: dict[_RowKey, dict], cell: tuple[str, str, str]) -> int:
    """One past the highest replicate index already recorded for a cell."""
    reps = [key[3] for key in existing if key[:3] == cell]
    return 1 + max(reps) if reps else 1


def merge_rows(
    new_rows: list[dict],
    path: Path,
    history_path: Path | None = None,
    mode: Literal["supersede", "replicate"] = "supersede",
) -> int:
    """Upsert cells into results.csv (keyed by challenge×model×reasoning-arm×rep)."""
    # `mode` states the CALLER'S INTENT, because intent cannot be inferred from content.
    # `_row_differs` compares every field and a re-run always differs on `computed_at` and
    # `real_cost`, so "the row changed" is true of a legitimate supersession and of an
    # accidental clobber alike, and content can never tell the two apart.
    #
    #   * `supersede` (the default — every existing call site keeps today's behaviour
    #     byte-for-byte) overwrites only when the cell's identity actually moved
    #     (`_anchors_differ`) or the stored row records no executed work at all
    #     (`impute.is_zero_work`). If the anchors are identical and only OUTCOME columns
    #     moved it RAISES `REPLICATE_MISKEYED` instead of overwriting. That refusal is the
    #     mechanism protecting paid observations: a second observation of an unchanged cell
    #     is a replicate, and the caller has to say so.
    #   * `replicate` appends a new observation at `rep = 1 + max(existing reps)`, assigned
    #     INSIDE this read-modify-write so two writers holding the caller's lock cannot claim
    #     the same index. It never supersedes and never calls `_append_history` — nothing is
    #     being replaced, so there is nothing to archive.
    # A replaced row (same key, changed content) is moved to the append-only
    # history log — nothing is discarded. results.csv keeps only current rows.
    # Two arms of the same (challenge, model) are DISTINCT keys: a
    # new arm never collides with, or archives, a sibling arm's row.
    existing = _read_raw_rows(path)
    superseded: list[dict] = []
    for row in new_rows:
        norm = {k: row.get(k, "") for k in integrity.RESULTS_FIELDS}
        if mode == "replicate":
            norm[integrity.REPLICATE_COLUMN] = str(_next_rep(existing, _cell_key(norm)))
            existing[_row_key(norm)] = norm
            continue
        # Converge the file on ONE spelling of rep: always emit "0", never a blank.
        norm[integrity.REPLICATE_COLUMN] = str(integrity.rep_index(norm))
        key = _row_key(norm)
        stored = existing.get(key)
        if stored is not None and _row_differs(stored, norm):
            if not (_anchors_differ(stored, norm) or impute.is_zero_work(stored)):
                raise validate.DataIntegrityError(
                    [
                        validate.Violation(
                            validate.Severity.ERROR,
                            validate.REPLICATE_MISKEYED,
                            f"refusing to overwrite {key[:3]} rep={key[3]}: its staleness "
                            "anchors are unchanged, so this is a SECOND OBSERVATION of a "
                            "cell that was already paid for, not a supersession. Pass "
                            "mode='replicate' to record it as one.",
                        )
                    ],
                    norm,
                )
            superseded.append(stored)
        existing[key] = norm
    if superseded:
        _append_history(superseded, history_path or _history_path(path))
    _write_raw_rows(existing, path)
    return len(new_rows)


def _report_coverage(matrix: dict, tasks: list[str]) -> None:
    print("\nStrategy cache coverage (embedding-free strategies):")
    for strategy in (Oracle(), AlwaysCheap(), AlwaysFrontier()):
        cov = coverage.cell_coverage(strategy, matrix, tasks)
        flag = "OK" if cov.complete else f"MISSING {len(cov.missing)} cell(s)"
        print(f"  {cov.strategy:16} needs {len(cov.needed):>3} cells  [{flag}]")


def refresh_summary(matrix: dict, tasks: list[str]) -> None:
    """Write the per-strategy summary derived from the results.csv cache to reports/.

    It regenerates from results.csv, which is the sole committed source of truth. The
    summary is nonetheless the one tracked file under the otherwise-gitignored reports/
    dir, so a committed figure can be checked against the numbers it was drawn from.
    """
    from benchmark.routing import run_eval, summary

    bm = config.benchmark_params()
    strategies = run_eval.get_strategies()
    rows = summary.compute_strategy_rows(
        matrix,
        tasks,
        strategies,
        gamma=config.gamma(),
        bootstrap=bm.get("bootstrap_iterations", 1000),
        seed=bm.get("seed", 42),
    )
    out = Path(__file__).resolve().parent.parent / "routing" / "reports" / "strategy_summary.csv"
    table = summary.certified_table(rows)
    print(table.admissibility.reason)
    summary.write_summary_csv(table, out)
    print(f"Refreshed strategy summary -> {out}")


def regenerate_plots() -> None:
    """Regenerate the reports/ plots (gitignored, regenerable from results.csv) via
    report.py. Path is module-relative so it works from any cwd.
    """
    report_py = Path(__file__).resolve().parent.parent / "routing" / "report.py"
    cmd = [sys.executable, str(report_py), "--matrix", str(config.challenges_path())]
    print(f"Regenerating plots: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def _print_status(status: CellStatus, n_tasks: int, n_models: int) -> None:
    total = n_tasks * n_models
    print(
        f"Cells: {total}+ base ({n_tasks} challenges x {n_models} models; "
        f"reasoning-arm sampling adds non-default-arm cells)  "
        f"present={status.present}  missing={len(status.missing)}  stale={len(status.stale)}"
    )
    for label, cells in (("missing", status.missing), ("stale", status.stale)):
        if cells:
            sample = ", ".join(f"{c}:{m}:{a}" for c, m, a in cells[:3])
            print(f"  {label}: {len(cells)} — e.g. {sample}{' ...' if len(cells) > 3 else ''}")


def _arm_context(
    tasks: list[str], models: list[str]
) -> tuple[dict[tuple[str, str], list[str]], dict[str, dict[str, str]]]:
    """Selected arms per (challenge, model) + each model's arm-hash anchors."""
    resolved = config.resolved_models()
    arm_hash_map = integrity.arm_hashes(resolved)
    if not config.arm_sampling_enabled():
        # Gate OFF (disabled): stay on the single-default-arm-per-cell path
        # to reproduce default-arm-only behavior. The live executor now sends
        # distinct requests per arm (infer._scaffold_model_kwargs), so enabling
        # the gate generates and bills separate cells per arm.
        # arm_hash_map is still built (harmless, no extra cells result from it).
        return _default_selected_arms(tasks, models), arm_hash_map
    return _sampled_selected_arms(tasks, models, resolved), arm_hash_map


def _restrict_concordance_tasks(
    selected_arms: dict[tuple[str, str], list[str]],
    subset_models: set[str],
    subset_challenges: set[str],
) -> dict[tuple[str, str], list[str]]:
    """Blank a concordance channel's arms at challenges outside the named subset.

    The named subset is a fixed set of channels x challenges, not the whole campaign. An
    empty arm list yields no cell at that challenge (``classify_cells`` iterates the arms),
    so the subset stays its declared size while every other lane keeps the full task set.
    """
    if not subset_challenges:
        return selected_arms
    return {
        (cid, model): ([] if model in subset_models and cid not in subset_challenges else arms)
        for (cid, model), arms in selected_arms.items()
    }


def _sampled_selected_arms(
    tasks: list[str], models: list[str], resolved: dict
) -> dict[tuple[str, str], list[str]]:
    """Multi-arm p(arm|model) sweep — gated behind ``arm_sampling.enabled``.

    Models in ``default_only_models`` are pinned to their default arm (cost control on
    the expensive tiers) even while the sweep runs for everyone else.
    """
    from benchmark.runner import sampling

    weights = config.arm_sampling_weights()
    default_only = config.arm_sampling_default_only_models()
    selected: dict[tuple[str, str], list[str]] = {}
    for model in models:
        bracket = resolved[model].reasoning if model in resolved else None
        for cid in tasks:
            if bracket is None:
                selected[(cid, model)] = [integrity.DEFAULT_REASONING]
            elif model in default_only:
                selected[(cid, model)] = [bracket.default_arm]
            else:
                selected[(cid, model)] = sampling.select_arms(cid, model, bracket, weights)
    return selected


def _add_args(ap: argparse.ArgumentParser, config_path: str) -> None:
    ap.add_argument(
        "--strategy",
        choices=("cost_optimal", "full", "ladder"),
        default="cost_optimal",
        help="cost_optimal (default) = adaptive cheap-first collection (Phase A/B/C); "
        "full = exhaustive every-model x every-challenge matrix; "
        "ladder = escalate each task cheap->high until its first passing tier.",
    )
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument(
        "--tasks-file",
        default=None,
        help="ladder only: JSON list of challenge ids to target (overrides the sampled set) — "
        "e.g. only the challenges whose crossover is still unknown; cached rungs are skipped.",
    )
    ap.add_argument(
        "--live", action="store_true", help="Run uncached cells for real (needs Docker + keys)"
    )
    ap.add_argument(
        "--check-images",
        action="store_true",
        help="Resolve image digests to detect env drift (registry queries per image; "
        "NOT implied by --live — it 429s on the swebench namespace and defeats GHCR staging)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=infer._AGENT_WALL_LIMIT_S,
        help="Generous GRACEFUL per-cell wall-clock backstop (seconds), passed as the agent's own "
        "wall_time_limit_seconds; the external hard watchdog fires strictly later. The PRIMARY "
        "bound is --step-limit. Default 1800.",
    )
    ap.add_argument(
        "--step-limit",
        type=int,
        default=None,
        help="PRIMARY model-speed-agnostic per-cell bound: max agent steps (same attempts for a "
        "slow model as a fast one). Default: benchmark.yaml live.step_limit.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent live cells (I/O-bound: Docker + LLM). Each worker runs a "
        "SWE-bench container — raise it with an eye on host memory. 1 = serial.",
    )
    ap.add_argument(
        "--preflight-timeout",
        type=float,
        default=infer.DEFAULT_PREFLIGHT_TIMEOUT_S,
        help="Seconds per lane for the free-lane preflight probe. Each lane gets ONE "
        "retry-disabled attempt, so a fleet sweep cannot hang on a throttled provider. "
        f"Default {infer.DEFAULT_PREFLIGHT_TIMEOUT_S:g}s.",
    )
    ap.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Abort remaining cells once cumulative real_cost (USD) crosses this. "
        "Completed cells are kept; recommended for any paid run.",
    )
    ap.add_argument(
        "--cells",
        default=None,
        help="full only: comma-separated CID:MODEL:ARM triples to run exactly (e.g. "
        "sympy__sympy-17630:kimi-k3:max). Bypasses cache classification — an EXPLICIT "
        "re-collection of named cells (e.g. a censored-cell pilot at a new cap). Requires "
        "--live and is still bounded by --max-cost.",
    )
    ap.add_argument(
        "--extra-models",
        default=None,
        help="full only: comma-separated priced registry ids or any `S-explabs` catalog "
        "slug to COLLECT in ADDITION to the enabled list (a `-explabs` slug absent from "
        "the registry is synthesized collection-only at runtime). They are never unioned "
        "into enabled_models()/models_matrix/capability_rank, so they cannot move an "
        "analysis or a baseline — collection-only rows. A listed id must be a priced "
        "registry model or an `S-explabs` catalog slug. COVERAGE POLICY: an extra whose "
        "model identity (registry `version`) already has a real row for a challenge under "
        "ANOTHER channel id (e.g. direct requesty kimi-k3 vs kimi-k3-explabs) skips that "
        "challenge by default — prefer MORE COVERAGE over re-running a covered challenge; "
        "only --cells re-runs explicitly.",
    )
    ap.add_argument(
        "--free-registry",
        default=None,
        help="Path to the non-shipped free-model OVERLAY registry (collection-only). It is "
        "loaded on top of the shipped registry for `--extra-models` ONLY: the shipped "
        "`_pricing_path()` default, the enabled set, the live pool, the pareto axes and the "
        "kill gate never see it. Also settable via SHUNT_FREE_REGISTRY. With no overlay, "
        "behavior is byte-identical.",
    )
    ap.add_argument(
        "--require-zero-cost",
        action="store_true",
        help="Fail-closed $0 harvesting gate. Refuse unless EVERY model in the run is a "
        "non-shipped overlay free lane; pre-flight each admitted lane once (a real "
        "max_tokens=1 completion) before any container; and require real_cost==0 on every "
        "written row (validate FREE_LANE_BILLED). Corrects two traps: `--max-cost 0` is a "
        "SILENT NO-OP (`spent >= max_cost` is already true at 0, so it stops before cell 1) "
        "and `live.cost_limit: 0` DISABLES the scaffold cap (mini-swe-agent gates on "
        "`0 < cost_limit`, so only a POSITIVE limit is a cap).",
    )
    ap.add_argument(
        "--max-cost-overshoot",
        type=float,
        default=0.0,
        help="Dollars of allowed overshoot beyond --max-cost to FINISH the challenge "
        "already in progress, so a partially-collected challenge is not discarded; "
        "0 = hard stop mid-challenge.",
    )
    ap.add_argument(
        "--max-start-failures",
        type=int,
        default=5,
        help="Abort the run after this many CONSECUTIVE container-start failures (image "
        "missing / registry throttled) instead of hammering the registry with skips. "
        "Reset by any successful cell; pre-stage images with "
        "scripts/benchmark/prepull_swebench_images.py.",
    )
    ap.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="Abort the whole run after this many CONSECUTIVE cell failures of ANY cause "
        "(dead key, no balance, persistent rate-limit, docker). Reset by any successful cell. "
        "A catch-all so an unusable API can never march through hundreds of cells; the first "
        "unambiguous API-unusable (auth / no-balance) cell aborts immediately regardless.",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip writing the regenerable reports/strategy_summary.csv",
    )
    ap.add_argument("--no-plots", action="store_true", help="Skip regenerating plots")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")


def _prompt_confirm(prompt: str) -> str | None:
    """Read one confirmation line from stdin; None if non-interactive (not a TTY / EOF)."""
    if not sys.stdin.isatty():
        return None
    try:
        return input(prompt)
    except EOFError:
        return None


def _confirm_uncapped_live() -> bool:
    """Interactive gate for `full --live` with no --max-cost; abort unless the user types Y/y."""
    answer = _prompt_confirm(
        "FULL matrix --live with NO --max-cost cap can spend unbounded real money. Proceed? [y/N] "
    )
    if answer is None:
        print("  aborted: non-interactive stdin, refusing uncapped live spend.", file=sys.stderr)
        return False
    choice = answer.strip().lower()
    if choice == "y":
        return True
    if choice == "n":
        print("  aborted by user.", file=sys.stderr)
        return False
    print(f"  aborted: invalid input {answer!r} (expected y or n).", file=sys.stderr)
    return False


def _dispatch(args: argparse.Namespace) -> int:
    """Route parsed args to the cost_optimal (adaptive) or full (exhaustive) flow."""
    # `--extra-models`/`--free-registry` are a `full`-only collection path: the adaptive
    # collectors have no overlay lane to meter. Refuse rather than let the flag silently do
    # nothing (the same fail-closed stance as `--max-cost 0` being a no-op).
    if getattr(args, "require_zero_cost", False) and args.strategy != "full":
        print(
            "  REFUSING --require-zero-cost: only `--strategy full` collects --extra-models "
            "overlay lanes; cost_optimal/ladder do not support them.",
            file=sys.stderr,
        )
        return 2
    if args.strategy == "cost_optimal":
        from benchmark.runner.collect import run_collect

        return run_collect(
            args.config,
            live=args.live,
            timeout=args.timeout,
            workers=args.workers,
            max_cost=args.max_cost,
            max_cost_overshoot=args.max_cost_overshoot,
            max_start_failures=args.max_start_failures,
            max_consecutive_failures=args.max_consecutive_failures,
            check_images=args.check_images,
            step_limit=getattr(args, "step_limit", None),
        )
    if args.strategy == "ladder":
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from benchmark.runner.ladder_collect import run_ladder

        tasks = (
            json.loads(Path(args.tasks_file).read_text())
            if getattr(args, "tasks_file", None)
            else None
        )
        return run_ladder(
            args.config,
            tasks=tasks,
            live=args.live,
            timeout=args.timeout,
            workers=args.workers,
            max_cost=args.max_cost,
            max_cost_overshoot=args.max_cost_overshoot,
            max_start_failures=args.max_start_failures,
            max_consecutive_failures=args.max_consecutive_failures,
            check_images=args.check_images,
            step_limit=getattr(args, "step_limit", None),
        )
    # A `--require-zero-cost` run is contractually $0-capped (fail-closed by
    # `require_zero_cost_refusal` and the FREE_LANE_BILLED interlock), so it is NOT an
    # uncapped live spend and must run without a TTY — otherwise a background free-lane
    # campaign aborts at the prompt.
    if (
        args.live
        and args.max_cost is None
        and not getattr(args, "require_zero_cost", False)
        and not _confirm_uncapped_live()
    ):
        return 3
    return _run_full(args)


def main(config_path: str = "benchmark/benchmark.yaml") -> int:
    # Self-sufficient credentials: load a local .env (gitignored) if present so
    # `--live` works from a bare shunt checkout; real env vars still take precedence.
    load_dotenv_file()
    config.load(config_path)
    ap = argparse.ArgumentParser(
        description="Shunt benchmark runner (adaptive cost_optimal default; simulated by default)."
    )
    _add_args(ap, config_path)
    args = ap.parse_args()
    # Layer 3, scoped to the $0 gate: arm the process-wide scaffold breaker before any
    # cell runs. (The intent is to arm it "before minisweagent is first imported", but
    # importing THIS module already pulls minisweagent in via infer->scaffold_model, so the
    # helper re-applies the limit to the live stats — see _arm_free_lane_cost_breaker.)
    if getattr(args, "require_zero_cost", False):
        _arm_free_lane_cost_breaker()
    if args.config != config_path:
        config.load(args.config)
    if getattr(args, "free_registry", None):
        config.set_free_registry(args.free_registry)
    return _dispatch(args)


def _parse_cells(raw: str) -> list[tuple[str, str, str]]:
    """Parse a ``--cells`` value ('cid:model:arm[,cid:model:arm...]') into cell triples."""
    cells: list[tuple[str, str, str]] = []
    for chunk in raw.split(","):
        parts = chunk.strip().split(":", 2)
        if len(parts) != 3 or not all(parts) or ":" in parts[2]:
            raise ValueError(
                f"malformed --cells entry {chunk!r} (expected cid:model:arm, e.g. "
                "sympy__sympy-17630:kimi-k3:max)"
            )
        cells.append((parts[0], parts[1], parts[2]))
    return cells


def _apply_multimodal_gate(source: str, models: list[str]) -> tuple[list[str], dict[str, str]]:
    """Drop models below the multimodal Verified-coverage gate; (schedulable, refusals).

    A no-op for a text source: the gate only governs cells whose manifest would carry
    images. For a multimodal source a below-gate model is REFUSED here, by name, rather
    than scheduled and failed inside the harness.
    """
    if source != swebench_multimodal_specs.SOURCE:
        return models, {}
    refusals = model_coverage.multimodal_gate_refusals(models)
    return [m for m in models if m not in refusals], refusals


def _lane_reserves(cache: dict, models: list[str]) -> dict[str, int]:
    """Per-lane p90(calls) reservation from each lane's own cached history (1 when empty)."""
    history: dict[str, list[int]] = {model: [] for model in models}
    for per_model in cache.values():
        for model, per_arm in per_model.items():
            if model not in history:
                continue
            history[model].extend(int(row.get("calls", 0) or 0) for row in per_arm.values())
    return {model: lane_scheduler.p90_calls(calls) for model, calls in history.items()}


def _build_lane_scheduler(models: list[str], cache: dict) -> lane_scheduler.LaneScheduler | None:
    """Build the lane admission scheduler from the registry plus the config's ``lanes:`` block.

    ``None`` when the config declares no ``lanes:`` block, so the paid default path is
    unchanged. A lane's provider is resolved from the overlay first, then the pricing view
    (which carries registered/`-explabs` extras), and ``lane_limits_from_registry`` refuses a
    provider that is not a declared free lane. Limit precedence is registry provider defaults,
    then a registry lane override, then ``lanes.limits`` in the campaign config. The persisted
    day buckets are loaded here (before any cell) so a restart cannot reset the day's RPD, and
    a lane whose published limits can never serve a cell is named once with its structural
    reason.
    """
    if not config.lanes_config():
        return None
    unknown = config.lane_unknown_limits()
    per_lane = config.lane_limits_config()
    overlay = config.free_registry()
    pricing = config.load_pricing()
    limits: dict[str, lane_scheduler.LaneLimits] = {}
    for model in models:
        row = overlay.get(model) or pricing.get(model) or {}
        provider = row.get("provider")
        registry = config.lane_limits_from_registry(model, provider)
        limits[model] = lane_scheduler.LaneLimits.from_mapping(
            {**unknown, **registry, **per_lane.get(model, {})}
        )
    scheduler = lane_scheduler.LaneScheduler(
        limits=limits,
        reserves=_lane_reserves(cache, models),
        stall_timeout_s=config.lane_stall_timeout_s(),
        state=lane_scheduler.load_lane_state(),
    )
    for lane, reason in scheduler.structural_refusals().items():
        print(f"  refusing free lane {lane}: {reason}", file=sys.stderr)
    return scheduler


def _run_full(args: argparse.Namespace) -> int:
    """Exhaustive matrix: classify every enabled model x sampled challenge, run+merge when live."""
    matrix = config.load_matrix(config.challenges_path())
    # The configured manifest's `source` names which challenge store + spec module back
    # the run (verified default; the multimodal companion for challenges_multimodal.json).
    # Verified configs resolve to the same module + store as before, so nothing changes there.
    source = swebench_specs.manifest_source()
    spec_module = swebench_specs.spec_module_for(source)
    hashes = integrity.all_hashes(source)
    enabled = config.enabled_models()
    # --extra-models: a COLLECTION-only extension of the run's model list. Unioned here,
    # and only here — never into enabled_models()/models_matrix/capability_rank — so an
    # extra (e.g. free-promo) channel adds rows without moving any analysis or baseline.
    extra = _extra_models(getattr(args, "extra_models", None))
    # Synthesize any collection-only `-explabs` slug into the pricing view BEFORE the
    # identity map is derived, so a synthesized extra's version identity participates in
    # the identity-skip below (and route resolution sees it during the run).
    config.register_collection_models(extra)
    versions = integrity.model_versions()
    models = [*enabled, *extra]
    # Owner eligibility gate: no multimodal cell may be scheduled for a model that has not
    # completed the Verified text corpus. Text sources are a no-op; a multimodal source
    # drops a below-gate model here, by name, so it is never run.
    models, multimodal_refusals = _apply_multimodal_gate(source, models)
    for model, reason in multimodal_refusals.items():
        print(f"  REFUSING multimodal cells for {model}: {reason}")
    cache = config.load_results()
    tasks = config.sample_tasks(
        sorted(hashes.keys()), seed=config.benchmark_params().get("seed", 42)
    )
    # Resolve digests only when EXPLICITLY asked (--check-images). Each is a registry
    # `imagetools inspect` per image (~sample_size tasks) that queries Docker Hub — slow,
    # and it rate-limits (429) on the swebench namespace, which defeats GHCR pre-staging and
    # hangs the run before any cell starts. A live run no longer forces it: a failed resolve
    # returns None anyway, and on a fresh collection the digest only anchors drift for RE-runs.
    # None ⇒ image axis skipped, never stale.
    digests = (
        image_version.resolve_spec_digests(spec_module.spec_image_refs(tasks))
        if args.check_images
        else None
    )
    step_limit = args.step_limit if args.step_limit is not None else config.live_step_limit()

    selected_arms, arm_hash_map = _arm_context(tasks, models)
    # Cross-provider concordance: a NAMED subset of identities is exempt from the default
    # dedupe so one identity served by several providers yields one row per channel at the
    # same challenge. Dedupe (cap 1) stays the policy for every other extra.
    subset_models = config.concordance_subset_models()
    if subset_models:
        selected_arms = _restrict_concordance_tasks(
            selected_arms, subset_models, set(config.concordance_subset_challenges())
        )
    status = classify_cells(
        tasks,
        models,
        cache,
        hashes,
        versions,
        digests,
        selected_arms,
        arm_hash_map,
        # Collection-param anchors: the CURRENT expected values, so a changed
        # step_limit / prompt / merged-request-kwargs recomputes rather than serves a stale
        # outcome. The scaffold-derived hashes degrade to "" (no-op) if the scaffold is
        # unimportable; step_limit fires only for step-limit-censored cells.
        step_limit=step_limit,
        prompt_hash=integrity.scaffold_prompt_hash(),
        sampling_hash_map=integrity.sampling_hash_map(models),
        # Owner coverage policy: the collection-only extras skip a challenge their model
        # identity already covers under another channel id (e.g. kimi-k3-explabs skips the
        # challenges direct requesty kimi-k3 measured) — prefer MORE COVERAGE over re-running.
        # The concordance channels are excluded from that cap-1 set and governed by the raised
        # fan-out cap instead, so the named subset is the only place dedupe is lifted.
        identity_skip_models=set(extra) - subset_models,
        identity_fanout_cap=config.concordance_fanout_cap(),
        identity_fanout_models=subset_models or None,
    )
    _print_status(status, len(tasks), len(models))

    live = args.live and _has_keys()
    if args.live and not _has_keys():
        print("  --live requested but no API keys in env — staying simulated (no fabrication).")

    # Per-lane admission scheduler: built BEFORE preflight so persisted lane state is loaded
    # here (a restart cannot reset the day's RPD) and a preflight disable/quarantine is
    # recorded on the live scheduler and persisted at run end. None when the config declares
    # no `lanes:` block, so the paid default is unchanged.
    scheduler = _build_lane_scheduler(models, cache) if live else None

    # Layer 2, the $0 gate: refuse a run that is not purely overlay/free-registry BEFORE
    # any container, then probe every admitted lane once with a real max_tokens=1 completion.
    # A failing lane is disabled/quarantined FOR THAT LANE ONLY; the campaign proceeds with
    # the rest, and aborts only when no lane remains usable.
    require_zero = getattr(args, "require_zero_cost", False)
    if require_zero:
        zero_refusal = require_zero_cost_refusal(
            live=live, enabled=enabled, extra=extra, cost_limit=config.live_cost_limit()
        )
        if zero_refusal is not None:
            print(f"  REFUSING --require-zero-cost: {zero_refusal}", file=sys.stderr)
            return 2
        report = preflight_free_lanes(
            live,
            extra,
            scheduler=scheduler,
            timeout_s=getattr(args, "preflight_timeout", None),
        )
        for lane, reason in sorted(report.disabled.items()):
            print(f"  lane disabled {lane}: {reason}", file=sys.stderr)
        for lane, reason in sorted(report.quarantined.items()):
            print(f"  lane quarantined {lane}: {reason}", file=sys.stderr)
        if report.refused:
            # Persist the disables even on the terminal abort, so the next pass does not
            # re-probe a fleet already known dead.
            if scheduler is not None:
                lane_scheduler.save_lane_state(scheduler.state)
            print(
                "  REFUSING --live: preflight found NO usable free lane among "
                f"{len(extra)} admitted lane(s); no containers were started.",
                file=sys.stderr,
            )
            return 2
        # Drop disabled lanes from the runnable model set: their cells must never be planned
        # (the scheduler would refuse them anyway, then stall waiting out the batch deadline).
        disabled_lanes = set(report.disabled)
        models = [m for m in models if m not in disabled_lanes]

    # Caching gate: refuse a live run if any ENABLED model lacks a cache-read discount.
    # Uncached models are a budget bomb in agentic loops (see config.model_has_cache).
    # Scoped to `enabled` deliberately: --extra-models rows ride a $0 promotional
    # channel that reports no cache-read rate and no cached tokens — refusing on them
    # would block a free run over a cost feature the channel does not offer.
    if live:
        uncached = config.models_missing_cache(enabled)
        if uncached:
            print(
                f"  REFUSING --live: enabled models without caching {uncached} would burn "
                "uncached budget (full-context resend every turn). Disable them in "
                "benchmark.yaml or add cache_read_cost_per_1m to the model registry.",
                file=sys.stderr,
            )
            return 2

    # Preflight: one real $0 completion proves the key works BEFORE any container starts.
    # Probe the cheapest enabled model when the ladder has one; when the config enables NO
    # models and the run collects `--extra-models` only, probe the first extra instead — its
    # own credential (e.g. EXPLABS_API_KEY) is the meaningful live guard for that
    # run, and there is no enabled model to fall back on. Skipped under --require-zero-cost,
    # which already probed every admitted lane exactly once (preflight_free_lanes).
    if not require_zero:
        probe = enabled[0] if enabled else (extra[0] if extra else None)
        if preflight_refuses(live, model=probe):
            return 2

    # Free-campaign path: the priority model decides WHICH model's cell runs next, which
    # lanes are retired (duplicate / not-worth / no benchmark left) and the worker cap. The
    # plan's named reasons are printed once; its MISSING-cell set is what the executor runs.
    campaign: campaign_scheduler.CampaignRun | None = None
    if scheduler is not None:
        benchmark = (
            campaign_scheduler.MULTIMODAL_BENCHMARK
            if source == swebench_multimodal_specs.SOURCE
            else campaign_scheduler.TEXT_BENCHMARK
        )
        campaign = campaign_scheduler.build_campaign(
            models,
            status.to_run,
            benchmark=benchmark,
            lanes=scheduler,
            requested_workers=args.workers,
        )
        print(campaign_scheduler.format_plan(campaign.plan), file=sys.stderr)
    try:
        # --cells: an explicit targeted re-collection (bypasses classify's missing/stale set).
        target_cells = _parse_cells(args.cells) if getattr(args, "cells", None) else None
        if target_cells is not None:
            if not live:
                print("  --cells requires --live (an explicit re-collection of named cells).")
                return 2
            refused_cells = [c for c in target_cells if c[1] in multimodal_refusals]
            for cid, model, arm in refused_cells:
                print(
                    f"  REFUSING multimodal cell {cid}:{model}:{arm}: {multimodal_refusals[model]}"
                )
            target_cells = [c for c in target_cells if c[1] not in multimodal_refusals]
            n = _run_and_merge(
                target_cells,
                hashes,
                versions,
                digests,
                arm_hash_map,
                timeout=args.timeout,
                workers=args.workers,
                max_cost=args.max_cost,
                results_path=config.results_csv_path(),
                max_cost_overshoot=args.max_cost_overshoot,
                max_start_failures=args.max_start_failures,
                max_consecutive_failures=args.max_consecutive_failures,
                step_limit=step_limit,
                lanes=scheduler,
            )
            print(f"  live: wrote {n} cell(s) to {config.results_csv_path()}")
            matrix = config.load_matrix(config.challenges_path())
        elif status.to_run and live:
            # Cells are checkpointed as they complete; _run_and_merge returns the final
            # idempotent merge count for the summary line below.
            n = _run_and_merge(
                status.to_run,
                hashes,
                versions,
                digests,
                arm_hash_map,
                timeout=args.timeout,
                workers=campaign.plan.effective_workers if campaign is not None else args.workers,
                max_cost=args.max_cost,
                results_path=config.results_csv_path(),
                max_cost_overshoot=args.max_cost_overshoot,
                max_start_failures=args.max_start_failures,
                max_consecutive_failures=args.max_consecutive_failures,
                step_limit=step_limit,
                lanes=scheduler,
                campaign=campaign,
            )
            print(f"  live: wrote {n} cell(s) to {config.results_csv_path()}")
            n += _run_replicates(
                status, hashes, versions, digests, arm_hash_map, args, step_limit, lanes=scheduler
            )
            matrix = config.load_matrix(config.challenges_path())
        elif status.to_run:
            print(f"  simulated: would run {len(status.to_run)} cell(s); leaving them uncached.")
    finally:
        # At-least-once dispatch, exactly-once storage: persist the day buckets even when the
        # run aborts, so a restart resumes within the same day's RPD rather than resetting it.
        if scheduler is not None:
            lane_scheduler.save_lane_state(scheduler.state)

    _report_coverage(matrix, tasks)

    if not args.no_summary:
        refresh_summary(matrix, tasks)
    if not args.no_plots:
        regenerate_plots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
