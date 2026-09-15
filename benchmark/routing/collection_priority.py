"""Model-value priority and the data-driven stop rule for the free-tier campaign.

Priority is model VALUE, not novelty or cheapness: ``importance`` comes from the owner
allowlist in ``data/model_priority.yaml`` (with the measured capability rank as a fallback),
``marginal_coverage`` zeroes an identity a better channel already serves and credits the
Verified challenges no higher-priority identity reaches, and the quotient is divided by the
lane's measured cost so a free lane is "large but bounded".

The scheduler consumes four module-level functions:

    from benchmark.routing import collection_priority as cp
    if cp.duplicate_of(model) is None and cp.worth_collecting(model)[0]:
        run(cp.priority(model), cp.runnable_benchmarks(model))

Every function accepts a lane (overlay channel id) or a canonical identity, and every
decision is a pure function of committed data plus ``model_priority.yaml`` — no model calls.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from benchmark import config, model_coverage
from benchmark.routing import impute
from benchmark.routing.metrics import percentile
from shunt.models.config import Pricing, load_registry, resolve_models

TEXT_BENCHMARK: Final[str] = "swebench_verified"
MULTIMODAL_BENCHMARK: Final[str] = "swebench_multimodal"
DEFAULT_STOP_MIN_CELLS: Final[int] = 20
DEFAULT_STOP_CONFIDENCE: Final[float] = 0.95
DEFAULT_LANE_COST_EPSILON: Final[float] = 0.01
DEFAULT_IMPORTANCE: Final[float] = 0.5
DEFAULT_RECENCY_BONUS: Final[float] = 0.05
DEFAULT_RECENCY_WINDOW_DAYS: Final[int] = 90
_BOOTSTRAP_DRAWS: Final[int] = 1000
_BOOTSTRAP_SEED: Final[int] = 42
_OVERLAY_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "configs" / "free-tier" / "overlay.yaml"
)
_SNAPSHOT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[0] / "data" / "latest_free_models.json"
)
_TRUE: Final[frozenset[str]] = frozenset({"true", "1", "yes"})


@dataclass(frozen=True)
class Channel:
    """One schedulable lane: its canonical identity, provider and published list price."""

    channel: str
    version: str
    provider: str = ""
    list_cost_per_1m: float = 0.0
    release_date: str | None = None


@dataclass(frozen=True)
class PriorityWeights:
    """The tunable knobs, read from ``model_priority.yaml``'s ``weights:`` block."""

    default_importance: float = DEFAULT_IMPORTANCE
    frontier_importance: float = 1.0
    recency_bonus: float = DEFAULT_RECENCY_BONUS
    recency_window_days: int = DEFAULT_RECENCY_WINDOW_DAYS
    lane_cost_epsilon: float = DEFAULT_LANE_COST_EPSILON
    stop_min_cells: int = DEFAULT_STOP_MIN_CELLS
    stop_confidence: float = DEFAULT_STOP_CONFIDENCE

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PriorityWeights:
        """Build from a config mapping; an absent key keeps the documented default."""
        return cls(
            default_importance=float(raw.get("default_importance", DEFAULT_IMPORTANCE)),
            frontier_importance=float(raw.get("frontier_importance", 1.0)),
            recency_bonus=float(raw.get("recency_bonus", DEFAULT_RECENCY_BONUS)),
            recency_window_days=int(raw.get("recency_window_days", DEFAULT_RECENCY_WINDOW_DAYS)),
            lane_cost_epsilon=float(raw.get("lane_cost_epsilon", DEFAULT_LANE_COST_EPSILON)),
            stop_min_cells=int(raw.get("stop_min_cells", DEFAULT_STOP_MIN_CELLS)),
            stop_confidence=float(raw.get("stop_confidence", DEFAULT_STOP_CONFIDENCE)),
        )


def _is_pass(value: object) -> bool:
    """Parse a results-CSV ``pass`` cell the same way the rest of the harness does."""
    return str(value or "").strip().lower() in _TRUE


def _to_float(value: object) -> float | None:
    """A float from a CSV cell, or ``None`` when blank/unparseable (never a fabricated 0)."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def bootstrap_pass_ci(
    outcomes: Sequence[bool],
    *,
    confidence: float = DEFAULT_STOP_CONFIDENCE,
    draws: int = _BOOTSTRAP_DRAWS,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI on a pass rate, or ``None`` for zero observations.

    ``confidence`` and ``draws`` are knobs; the default is a 95% interval over 1,000
    resamples. Resampling the cells is the same task-level unit the routing metrics use.
    """
    if not outcomes:
        return None
    rng = random.Random(seed)
    n = len(outcomes)
    rates = [sum(1 for _ in range(n) if outcomes[rng.randrange(n)]) / n for _ in range(draws)]
    alpha = (1.0 - confidence) / 2.0
    return (percentile(rates, alpha), percentile(rates, 1.0 - alpha))


@dataclass(frozen=True)
class CollectionPriority:
    """The value model over provider lanes: priority, stop rule, runnable set, dedupe."""

    channels: Mapping[str, Channel]
    covered: Mapping[str, frozenset[str]]
    verified_total: int
    base_importance: Mapping[str, float] = field(default_factory=dict)
    passes: Mapping[str, Sequence[bool]] = field(default_factory=dict)
    identity_passes: Mapping[str, Sequence[bool]] | None = None
    real_cost_per_cell: Mapping[str, float] = field(default_factory=dict)
    not_worth: Mapping[str, str] = field(default_factory=dict)
    vision: frozenset[str] = frozenset()
    withdrawn: frozenset[str] = frozenset()
    weights: PriorityWeights = PriorityWeights()
    multimodal_eligible: Callable[[str], bool] = model_coverage.multimodal_eligible
    as_of: date | None = None

    # ── resolution and value ──────────────────────────────────────────────
    def _resolve(self, identity: str) -> Channel:
        """The lane record for a channel id, else the best channel for a canonical id."""
        if identity in self.channels:
            return self.channels[identity]
        best = self._best_channel(identity)
        return self.channels[best] if best is not None else Channel(identity, identity)

    def _best_channel(self, version: str) -> str | None:
        """The credited channel for an identity: highest value, cheapest, name-stable.

        A WITHDRAWN channel is skipped when a live sibling still serves the identity, so a
        withdrawn lane cannot shadow its own replacement; if every channel is withdrawn the
        full set is used (the identity then quiesces with a named reason).
        """
        matches = [c for c in self.channels.values() if c.version == version]
        if not matches:
            return None
        active = [c for c in matches if c.channel not in self.withdrawn]
        pool = active or matches
        return min((c.channel for c in pool), key=lambda n: self._order_key(self.channels[n]))

    def _order_key(self, channel: Channel) -> tuple[float, float, str]:
        """Ascending ordering; smaller is higher priority (value first, cost then name)."""
        return (-self._value(channel), channel.list_cost_per_1m, channel.channel)

    def _value(self, channel: Channel) -> float:
        """Base importance with the BOUNDED recency tie-break; 0 for a denylisted identity."""
        if channel.version in self.not_worth:
            return 0.0
        base = self.base_importance.get(channel.version, self.weights.default_importance)
        return base * _recency_factor(channel.release_date, self.as_of, self.weights)

    def importance(self, identity: str) -> float:
        """The identity's base value (allowlist/capability prior) with recency applied."""
        return self._value(self._resolve(identity))

    def lane_cost(self, identity: str) -> float:
        """Mean measured cost of one completed cell on the lane (0 for a free lane)."""
        return float(self.real_cost_per_cell.get(self._resolve(identity).channel, 0.0))

    def lane_cost_of(self, channel: Channel) -> float:
        """The internal-channel form of :meth:`lane_cost`."""
        return float(self.real_cost_per_cell.get(channel.channel, 0.0))

    # ── the four stable entry points ───────────────────────────────────────
    def priority(self, identity: str) -> float:
        """``importance * marginal_coverage / max(lane_cost, epsilon)`` for the identity."""
        channel = self._resolve(identity)
        return (
            self._value(channel)
            * self._marginal_coverage(channel)
            / max(self.lane_cost_of(channel), self.weights.lane_cost_epsilon)
        )

    def duplicate_of(self, identity: str) -> str | None:
        """The higher-priority channel that already covers this identity, else ``None``."""
        channel = self._resolve(identity)
        return self._duplicate_of(channel)

    def runnable_benchmarks(self, identity: str) -> list[str]:
        """Benchmarks still worth running for the identity: text first, then multimodal.

        A below-gate model gets the text benchmark; once its Verified text coverage is
        complete only multimodal remains, and a pure-text model then has nothing left —
        it returns ``[]`` and the collector retires it.
        """
        channel = self._resolve(identity)
        if not self.multimodal_eligible(channel.channel):
            return [TEXT_BENCHMARK]
        if channel.version in self.vision or channel.channel in self.vision:
            return [MULTIMODAL_BENCHMARK]
        return []

    def worth_collecting(self, identity: str) -> tuple[bool, str]:
        """``(keep, reason)`` under the domination stop rule; ``reason`` names every verdict."""
        channel = self._resolve(identity)
        denied = self.not_worth.get(channel.version)
        if denied is not None:
            return False, f"denylisted: {denied}"
        if channel.channel in self.withdrawn:
            return False, (
                "withdrawn: the listing is absent from the latest scan — its lane quiesces "
                "while the overlay retains the row's provenance"
            )
        remaining = self.runnable_benchmarks(channel.channel)
        if not remaining:
            return False, (
                "text coverage complete and the model is pure-text: no benchmark left to run"
            )
        outcomes = self._outcomes_for_version(channel.version)
        floor = self.weights.stop_min_cells
        if len(outcomes) < floor:
            return True, (
                f"{len(outcomes)}/{floor} completed cells — below the stop floor; keep collecting"
            )
        channel_ci = bootstrap_pass_ci(outcomes, confidence=self.weights.stop_confidence)
        dominator = self._dominator(channel, channel_ci)
        if dominator is not None and not self._adds_new_coverage(channel, dominator):
            assert channel_ci is not None
            return False, _dominated_reason(self, dominator, channel_ci)
        return True, f"not dominated after {len(outcomes)} completed cells"

    # ── internals ──────────────────────────────────────────────────────────
    def _outcomes_for_version(self, version: str) -> list[bool]:
        """Every pass outcome for an identity: its lanes aggregated, or the identity key itself."""
        if self.identity_passes is not None:
            return list(self.identity_passes.get(version, ()))
        lanes = [c for c in self.channels.values() if c.version == version]
        pooled = [o for c in lanes for o in self.passes.get(c.channel, ())]
        if not lanes:
            pooled.extend(self.passes.get(version, ()))
        return pooled

    def _version_ci(self, version: str) -> tuple[float, float] | None:
        outcomes = self._outcomes_for_version(version)
        return (
            bootstrap_pass_ci(outcomes, confidence=self.weights.stop_confidence)
            if outcomes
            else None
        )

    def _identity_value(self, version: str) -> float:
        """The identity's value: its best lane's, else the allowlist value with no recency."""
        values = [self._value(c) for c in self.channels.values() if c.version == version]
        return (
            max(values)
            if values
            else self.base_importance.get(version, self.weights.default_importance)
        )

    def _better_channels(self, channel: Channel) -> list[Channel]:
        key = self._order_key(channel)
        return [
            other
            for other in self.channels.values()
            if other.channel != channel.channel and self._order_key(other) < key
        ]

    def _marginal_coverage(self, channel: Channel) -> float:
        if self._duplicate_of(channel) is not None:
            return 0.0
        if self.verified_total <= 0:
            return 1.0
        seen = set(self.covered.get(channel.version, frozenset()))
        for other in self._better_channels(channel):
            seen |= set(self.covered.get(other.version, frozenset()))
        return max(0.0, (self.verified_total - len(seen)) / self.verified_total)

    def _duplicate_of(self, channel: Channel) -> str | None:
        best = self._best_channel(channel.version)
        return None if best is None or best == channel.channel else best

    def _dominator(self, channel: Channel, channel_ci: tuple[float, float] | None) -> str | None:
        """The identity whose pass-rate strictly dominates this lane's, or ``None``.

        Compared IDENTITY to identity, not lane to lane: a well-measured paid twin of the
        same weights and a higher-priority sibling both count as evidence that collecting
        this lane further buys nothing. BOTH sides need the evidence floor.
        """
        if channel_ci is None:
            return None
        floor = self.weights.stop_min_cells
        best: str | None = None
        best_value = 0.0
        for version in {c.version for c in self.channels.values()} | set(self.base_importance):
            if version == channel.version:
                continue
            if len(self._outcomes_for_version(version)) < floor:
                continue
            if self._identity_value(version) <= self._value(channel):
                continue
            other_ci = self._version_ci(version)
            # STRICT: a shared boundary is not dominance. Two lanes that both pass 20/20 have
            # CIs whose endpoints touch, and a non-strict test would retire one of them.
            if other_ci is None or other_ci[0] <= channel_ci[1]:
                continue
            if best is None or self._identity_value(version) > best_value:
                best, best_value = version, self._identity_value(version)
        return best

    def _adds_new_coverage(self, channel: Channel, dominator: str) -> bool:
        """Does collecting this identity still reach a challenge the dominator does not?"""
        unique = set(self.covered.get(channel.version, frozenset())) - set(
            self.covered.get(dominator, frozenset())
        )
        return bool(unique)


def _recency_factor(
    release_date: str | None, as_of: date | None, weights: PriorityWeights
) -> float:
    """A BOUNDED tie-break: +``recency_bonus`` when the release is inside the window."""
    if not release_date:
        return 1.0
    try:
        released = date.fromisoformat(str(release_date)[:10])
    except ValueError:
        return 1.0
    today = as_of or datetime.now(UTC).date()
    return (
        1.0 + weights.recency_bonus
        if 0 <= (today - released).days <= weights.recency_window_days
        else 1.0
    )


def _dominated_reason(
    model: CollectionPriority, dominator: str, channel_ci: tuple[float, float]
) -> str:
    """The named reason a dominated lane is stopped, quoting both CIs."""
    dom_ci = model._version_ci(dominator) or (0.0, 0.0)
    return (
        f"dominated by {dominator}: pass-rate CI [{dom_ci[0]:.2f}, {dom_ci[1]:.2f}] is "
        f"strictly above this lane's [{channel_ci[0]:.2f}, {channel_ci[1]:.2f}] and the lane "
        "adds no new Verified challenge coverage"
    )


# ── default engine: committed data + model_priority.yaml, built once ──────────────────────


def _list_cost(pricing: Pricing | None) -> float:
    if pricing is None:
        return 0.0
    return float(pricing.input_cost_per_1m) + float(pricing.output_cost_per_1m)


def _channel_from_flat(name: str, info: Mapping[str, object]) -> Channel:
    return Channel(
        channel=name,
        version=str(info.get("version") or name),
        provider=str(info.get("provider") or ""),
        list_cost_per_1m=(
            float(info.get("input_cost_per_1m", 0.0) or 0.0)  # type: ignore[arg-type]
            + float(info.get("output_cost_per_1m", 0.0) or 0.0)  # type: ignore[arg-type]
        ),
    )


def _overlay_channels(release_dates: Mapping[str, str]) -> dict[str, Channel]:
    """Lane records from the configured overlay, else the packaged campaign overlay file."""
    overlay = config.free_registry()
    if overlay:
        channels = {name: _channel_from_flat(name, info) for name, info in overlay.items()}
    elif _OVERLAY_PATH.exists():
        models = resolve_models(load_registry(_OVERLAY_PATH))
        channels = {
            name: Channel(
                channel=name,
                version=model.version or name,
                provider=model.provider,
                list_cost_per_1m=_list_cost(model.pricing),
            )
            for name, model in models.items()
        }
    else:
        channels = {}
    if not release_dates:
        return channels
    return {
        name: _with_release_date(channel, release_dates.get(channel.version))
        for name, channel in channels.items()
    }


def _with_release_date(channel: Channel, released: str | None) -> Channel:
    return (
        channel
        if not released
        else Channel(
            channel.channel,
            channel.version,
            channel.provider,
            channel.list_cost_per_1m,
            released,
        )
    )


def _coverage_by_version(channels: Mapping[str, Channel]) -> dict[str, frozenset[str]]:
    """Union each model's Verified coverage under its canonical identity."""
    out: dict[str, set[str]] = {}
    for model, ids in model_coverage.covered_ids_by_model().items():
        version = channels[model].version if model in channels else model
        out.setdefault(version, set()).update(ids)
    return {version: frozenset(ids) for version, ids in out.items()}


def _corpus_evidence() -> tuple[dict[str, list[bool]], dict[str, float]]:
    """Per-channel pass outcomes and mean real cost per cell, from both results CSVs."""
    passes: dict[str, list[bool]] = {}
    cost_sum: dict[str, float] = {}
    cost_n: dict[str, int] = {}
    for path in (config.results_csv_path(), config.free_results_csv_path()):
        if not path.exists():
            continue
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if impute.is_zero_work(row):
                    continue
                model = str(row.get("lane") or row.get("model") or "").strip()
                if not model:
                    continue
                passes.setdefault(model, []).append(_is_pass(row.get("pass")))
                cost = _to_float(row.get("real_cost"))
                cost_sum[model] = cost_sum.get(model, 0.0) + (cost if cost is not None else 0.0)
                cost_n[model] = cost_n.get(model, 0) + 1
    return passes, {m: cost_sum[m] / cost_n[m] for m in cost_sum}


def _base_importance(raw: Mapping[str, Any], weights: PriorityWeights) -> dict[str, float]:
    """Resolve the value allowlist, then the measured capability rank for the remainder."""
    base = {str(k): float(v) for k, v in (raw.get("importance") or {}).items()}
    for version in raw.get("frontier") or []:
        base.setdefault(str(version), weights.frontier_importance)
    for evidence in config.capability_rank().evidence.values():
        if evidence.source == "measured" and evidence.model not in base:
            base[evidence.model] = 1.0 + 9.0 * evidence.pass_rate
    return base


def _identity_passes(
    channels: Mapping[str, Channel], passes: Mapping[str, Sequence[bool]]
) -> dict[str, list[bool]]:
    """Pool every pass outcome under its canonical identity (lane + non-lane paid models)."""
    pooled: dict[str, list[bool]] = {}
    for model, outcomes in passes.items():
        version = channels[model].version if model in channels else model
        pooled.setdefault(version, []).extend(outcomes)
    return pooled


def _withdrawn_channels() -> frozenset[str]:
    """Overlay channel names whose scanned listing carries a ``withdrawn_at`` mark.

    The overlay RETAINS a withdrawn row (its price and identity provenance stay), so the
    campaign must consult the scan snapshot, not the overlay, to quiesce the lane. Withdrawal
    is per CHANNEL: one withdrawn serving channel does not retire a sibling channel that still
    serves the same identity.
    """
    if not _SNAPSHOT_PATH.exists():
        return frozenset()
    try:
        snapshot = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    withdrawn_pairs = {
        (str(row.get("provider")), str(row.get("listing_id")))
        for row in (snapshot.get("listings") or [])
        if row.get("withdrawn_at")
    }
    if not withdrawn_pairs:
        return frozenset()
    models = config.free_registry_models()
    if not models and _OVERLAY_PATH.exists():
        models = resolve_models(load_registry(_OVERLAY_PATH))
    return frozenset(
        name
        for name, model in models.items()
        if (model.provider, model.model_id or name) in withdrawn_pairs
    )


def default_engine() -> CollectionPriority:
    """Build the live priority model from the overlay, both corpora and the YAML allowlist."""
    raw = config.load_model_priority()
    weights = PriorityWeights.from_mapping(raw.get("weights") or {})
    release_dates = {str(k): str(v) for k, v in (raw.get("release_dates") or {}).items()}
    channels = _overlay_channels(release_dates)
    passes, real_cost = _corpus_evidence()
    return CollectionPriority(
        channels=channels,
        covered=_coverage_by_version(channels),
        verified_total=len(model_coverage.verified_challenge_ids()),
        base_importance=_base_importance(raw, weights),
        passes=passes,
        identity_passes=_identity_passes(channels, passes),
        real_cost_per_cell=real_cost,
        not_worth={str(k): str(v) for k, v in (raw.get("not_worth") or {}).items()},
        vision=frozenset(str(x) for x in (raw.get("vision") or [])),
        withdrawn=_withdrawn_channels(),
        weights=weights,
    )


@lru_cache(maxsize=1)
def _engine() -> CollectionPriority:
    """The process-wide default engine (one build; call :func:`clear_cache` to refresh)."""
    return default_engine()


def clear_cache() -> None:
    """Drop the cached engine so the next call rebuilds from the current files."""
    _engine.cache_clear()


def priority(identity: str) -> float:
    """Module-level :meth:`CollectionPriority.priority` over the live corpus."""
    return _engine().priority(identity)


def worth_collecting(identity: str) -> tuple[bool, str]:
    """Module-level :meth:`CollectionPriority.worth_collecting` over the live corpus."""
    return _engine().worth_collecting(identity)


def runnable_benchmarks(identity: str) -> list[str]:
    """Module-level :meth:`CollectionPriority.runnable_benchmarks` over the live corpus."""
    return _engine().runnable_benchmarks(identity)


def duplicate_of(identity: str) -> str | None:
    """Module-level :meth:`CollectionPriority.duplicate_of` over the live corpus."""
    return _engine().duplicate_of(identity)
