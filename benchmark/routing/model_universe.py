"""THE model universe: one canonical table and one canonical display name.

THE PROBLEM THIS CLOSES. Two things were forked away from the canonical identity source
(`data/model_identity.yaml` plus the committed corpora it resolves): a display canonicaliser
inside `figures/model_validity.py` and a second one inside `escalation/plots.py`. The second
emitted its own slugs (`ling-3.0-flash-fin`, `laguna-xs-2.1`) that disagreed with the registry
`version` the corpus actually keys on (`ling-3-0-flash-fin`, `poolside/laguna-xs-2.1`), and the
disagreement was baked into the escalation figure manifest. Two tables for one quantity is one
table too many, so this module owns BOTH the identity join and the display rendering, and every
figure reads it.

IDENTITY is the registry `version` slug (via `model_identity.yaml`'s Tier-2 resolution, or the
`model_version` a committed corpus row already carries). It is the join key: two channel
listings that resolve to the same slug are the SAME weights and dedupe. DISPLAY is that slug
with the serving-channel label and publisher namespace stripped — the provider is an axis, never
part of the model's name. A free-channel listing id (`kilo-step-3.7-flash-free`) is resolved to
its slug FIRST, through the committed corpus index, and only stripped as a last resort.

The inference-valid predicate is NOT re-derived here: `universe()` is a thin lens over
`benchmark.routing.model_validity.validity_census`, so the four criteria and the first-failing
reason live in exactly one place.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from typing import Final

from benchmark import config
from benchmark.routing import model_validity
from benchmark.routing.scripts.scan_free_models import canonical_slug, load_identity

# The two channel kinds, re-exported so every consumer names one source.
PAID: Final[str] = model_validity.PAID
FREE: Final[str] = model_validity.FREE

# Serving-channel prefixes that are LABELS, never part of a weights name. A publisher
# namespace (`@cf/zai-org/`, `poolside/`) and a promo suffix are handled by `canonical_slug`.
# The retired `zai-` serving-slot prefix is stripped here so the live resolver is the one place
# a `zai-glm-*` label is reduced to its bare `glm-*` identity (the migration helper
# `migrate_bare_slugs.bare_slot_text` is a one-time file rewriter, not a second live resolver).
_CHANNEL_PREFIXES: Final[tuple[str, ...]] = (
    "cloudflare_workers_ai-",
    "requesty-",
    "openrouter-",
    "kilo-",
    "google-",
    "nvidia_nim-",
    "nvidia-",
    "zai-",
)


def display_name(name: str) -> str:
    """Canonical bare slug for an identity or a raw listing/lane label.

    A lane label (`kilo-step-3.7-flash-free`, `requesty-nemotron-3.5-lightning-30b-a3b`)
    carries a channel prefix the channel id spells; stripping it and running the mechanical
    grammar (`canonical_slug`) yields the same bare slug the identity map and overlay carry.
    An already-canonical identity is a fixpoint, so this is safe to call twice.
    """
    out = name
    for prefix in _CHANNEL_PREFIXES:
        if out.startswith(prefix):
            out = out[len(prefix) :]
            break
    return canonical_slug(out)


@cache
def evidence() -> model_validity.Evidence:
    """The heavy reads, computed once per process and shared by every lens below."""
    return model_validity.gather_evidence()


@cache
def _lane_identity() -> dict[str, str]:
    """Listing/lane label -> canonical identity, from the committed corpora and registries.

    `model_validity` already resolves every committed corpus row through `model_version` and
    every registry/overlay row through its `version`; this is that one map, frozen so the
    escalation half can key a lane label (`kilo-step-3.7-flash-free`) to the same identity the
    routing half draws.
    """
    try:
        return dict(evidence().identities)
    except Exception:  # noqa: BLE001 — a plot must degrade to display-stripping, never crash
        return {}


@cache
def _alias_identity() -> dict[str, str]:
    """Provider listing id -> canonical identity, from `model_identity.yaml`'s Tier-2 aliases."""
    try:
        identity = load_identity()
    except Exception:  # noqa: BLE001 — the curated file is optional at plot time
        return {}
    out: dict[str, str] = {}
    for slug, entry in identity.entries.items():
        for listing_ids in entry.aliases.values():
            for listing in listing_ids:
                out.setdefault(str(listing), slug)
    return out


def resolve_identity(label: str) -> str:
    """Canonical identity for a lane/listing label, preferring the committed resolution.

    The committed corpus/registry map wins; an alias from `model_identity.yaml` is next; and a
    label in neither falls back to `display_name`, so the result is a bare slug with any
    serving-channel prefix (`zai-`, `kilo-`, `cloudflare_workers_ai-`, …) stripped. It is
    deliberately NOT `canonical_slug` alone: that leaves a retired `zai-glm-5.2` slot raw
    while `canonical_label` renders `glm-5.2`, and the two fork on the same label — a caller
    that membership-tests `resolve_identity(label)` against the bare universe then drops a
    model the display path would have shown.
    """
    found = _lane_identity().get(label) or _alias_identity().get(label)
    return found or display_name(label)


@dataclass(frozen=True)
class UniverseRow:
    """One canonical model's identity, evidence and inference eligibility."""

    canonical: str
    identity: str
    providers: tuple[str, ...]
    channel: str
    listings: tuple[str, ...]
    cells: int
    covered: int
    corpus: int
    capability: str
    triage: str
    valid: bool
    reason: str
    collection_only: bool

    @property
    def coverage_frac(self) -> float:
        return self.covered / self.corpus if self.corpus else 0.0


def _row(entry: model_validity.ModelValidity) -> UniverseRow:
    return UniverseRow(
        canonical=display_name(entry.model),
        identity=entry.model,
        providers=entry.providers,
        channel=entry.channel,
        listings=entry.listings,
        cells=entry.cells,
        covered=entry.covered,
        corpus=entry.corpus,
        capability=entry.capability,
        triage=entry.triage,
        valid=entry.valid,
        reason=entry.reason,
        collection_only=entry.collection_only,
    )


@cache
def _universe() -> tuple[UniverseRow, ...]:
    """The committed table, computed once per process."""
    return tuple(_row(entry) for entry in model_validity.validity_census(evidence()))


@dataclass(frozen=True)
class Performance:
    """A canonical model's measured default-arm outcomes over both committed corpora."""

    identity: str
    n: int
    passes: int
    mean_cost: float

    @property
    def pass_rate(self) -> float:
        return self.passes / self.n if self.n else 0.0


def _accumulate(rows: list[tuple[str, bool, float]]) -> dict[str, Performance]:
    counts: dict[str, list[float]] = {}
    for identity, passed, cost in rows:
        bucket = counts.setdefault(identity, [0.0, 0.0, 0.0])
        bucket[0] += 1
        bucket[1] += int(passed)
        bucket[2] += cost
    return {
        identity: Performance(identity, int(n), int(passes), cost / n if n else 0.0)
        for identity, (n, passes, cost) in counts.items()
    }


@cache
def performance() -> dict[str, Performance]:
    """Identity -> measured default-arm pass rate and mean cost, paid plus free corpus."""
    canonical = _lane_identity()
    rows: list[tuple[str, bool, float]] = []
    for cells in config.flatten_default_arm(config.load_results()).values():
        for model, cell in cells.items():
            identity = canonical.get(model, model)
            rows.append((identity, bool(cell.get("pass")), float(cell.get("real_cost") or 0.0)))
    path = config.free_results_csv_path()
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                listing = str(row.get("lane") or row.get("model") or "").strip()
                if not listing:
                    continue
                identity = canonical.get(listing, str(row.get("model_version") or listing))
                passed = str(row.get("pass") or "").strip().lower() in ("true", "1", "yes")
                cost = float(row.get("real_cost") or row.get("cost") or 0.0)
                rows.append((identity, passed, cost))
    return _accumulate(rows)


@cache
def coverage_map() -> dict[str, frozenset[str]]:
    """Identity -> the verified challenges the committed evidence covers for it."""
    return dict(evidence().coverage)


@cache
def verified_challenges() -> tuple[str, ...]:
    """Every verified challenge in the committed task set, sorted — the coverage columns."""
    from benchmark.model_coverage import verified_challenge_ids  # noqa: PLC0415

    return tuple(sorted(verified_challenge_ids()))


def universe() -> list[UniverseRow]:
    """Every model the committed evidence names, valid first (free channel last within)."""
    return list(_universe())


def valid_rows() -> list[UniverseRow]:
    """The inference-valid subset — the models the main figures may draw."""
    return [row for row in _universe() if row.valid]


def invalid_rows() -> list[UniverseRow]:
    """The inferred-invalid subset — the universe outside the inference pool."""
    return [row for row in _universe() if not row.valid]


def canonical_label(label: str) -> str:
    """Canonical display name for a lane/listing label or an already-canonical identity.

    `resolve_identity` maps a channel listing (`z-ai/glm-5.3:free`) to its bare weights
    identity (`glm-5.3`); `display_name` then strips any remaining serving prefix. This is the
    scalar of :func:`canonical_labels`, and every figure that names a model uses it so a
    channel listing can never leak a publisher prefix into a label.
    """
    return display_name(resolve_identity(label))


def canonical_labels(labels: list[str]) -> dict[str, str]:
    """Lane/listing label -> canonical display slug, resolving to identity first."""
    return {label: display_name(resolve_identity(label)) for label in labels}
