"""Which models a figure may treat as inference-valid, derived from the repo's own machinery.

THE PROBLEM THIS CLOSES. `results.csv` holds far more models than the router may serve:
superseded cheap models the triage rule dropped, unmeasured frontier slots the live pool
reserves, collection-only promo probes, and a separately-collected free channel. Every
figure that draws a model had its own ad-hoc notion of "the models", so a dominated or
never-measured model could stand beside a served one with nothing saying they were not the
same kind of thing.

THIS IS THE ONE DEFINITION, AT CANONICAL-IDENTITY GRAIN. The two corpora key their rows on a
channel listing id (`kilo-step-3.7-flash-free`, `deepseek-v4-pro-explabs`); the weights
identity is the `model_version` slug. Rows are merged by that identity, and the provider is a
LABEL, never part of the model name. A model is INFERENCE-VALID when ALL of the following
hold, each read from the machinery that already owns it:

  1. LIVE      — in the packaged live pool (`router.yaml models:`).
  2. TRIAGE    — verdict KEEP or EXCEPTION.
  3. CAPABILITY— derived rank is MEASURED, not a price prior.
  4. COVERAGE  — at least `capability_rank.K` measured default-arm cells.
  5. CHANNEL   — a PAID channel serves it (a free-only identity is never routed).

A model failing any criterion is INFERENCE-INVALID and carries the FIRST failing reason, so
the census can say WHY it is out rather than only that it is. Unmeasured live slots and
dropped benchmark-only models are different failures and are kept distinct.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.model_coverage import covered_ids_by_model, verified_challenge_ids
from benchmark.routing import triage
from benchmark.routing._live_pool import packaged_live_pool
from benchmark.routing.plot_style import RawResults

PAID: Final[str] = "paid"
FREE: Final[str] = "free"

# The triage verdicts that clear the slot; everything else is out.
_PASSING_TRIAGE: Final[frozenset[str]] = frozenset({triage.VERDICT_KEEP, triage.VERDICT_EXCEPTION})

# Criteria column order for the matrix figure — the labels are the claim, not a key.
CRITERIA: Final[tuple[str, ...]] = (
    "selected\n(live pool)",
    "triage\nKEEP",
    "capability\nmeasured",
    "coverage\n≥K cells",
)


@dataclass(frozen=True)
class ModelValidity:
    """One canonical model's inference validity and the evidence behind the verdict."""

    model: str
    channel: str
    providers: tuple[str, ...]
    listings: tuple[str, ...]
    collection_only: bool
    live: bool
    enabled: bool
    triage: str
    capability: str
    cells: int
    covered: int
    corpus: int
    valid: bool
    reason: str

    @property
    def coverage_frac(self) -> float:
        return self.covered / self.corpus if self.corpus else 0.0

    @property
    def criteria(self) -> tuple[bool, ...]:
        """The four pass/fail criteria, in `CRITERIA` order."""
        return (
            self.live,
            self.triage in _PASSING_TRIAGE,
            self.capability == "measured",
            self.cells >= cell_floor(),
        )


@dataclass(frozen=True)
class Evidence:
    """The heavy reads, computed once and shared by every row of the census."""

    cells: Mapping[str, int]
    coverage: Mapping[str, frozenset[str]]
    corpus: int
    triage: Mapping[str, str]
    capability: Mapping[str, str]
    identities: Mapping[str, str]
    providers: Mapping[str, tuple[str, ...]]
    listings: Mapping[str, tuple[str, ...]]
    paid_identities: frozenset[str]
    live: frozenset[str]
    enabled: frozenset[str]
    floor: int


@dataclass(frozen=True)
class _ChannelRow:
    """One channel listing of a canonical identity, from a corpus or a registry."""

    listing: str
    identity: str
    provider: str
    free: bool


def cell_floor() -> int:
    """The measured default-arm cell floor, pinned by benchmark.yaml `capability_rank.K`."""
    return int(config.capability_rank_config()["K"])


def _canonical(identities: Mapping[str, str], listing: str) -> str:
    """The canonical identity for a listing id; the id itself when no map entry exists."""
    return identities.get(listing, listing)


@cache
def _registry_versions() -> Mapping[str, str]:
    """Registry NAME -> the bare `version` identity it declares ({} on failure).

    Under the one-id convention a registry name IS the model's bare id, and `version` is the
    identity/staleness key (the schema: "a genuine provider model change is a NEW registry id,
    not a version bump"). The two agree for every shipped row; the map still exists because a
    free-overlay/collection name is a channel LABEL whose declared `version` is the identity,
    so `name -> version` is what keeps one weights set under one bare id across registry,
    corpus, overlay and figures.
    """
    try:
        pricing = config.load_pricing()
    except Exception:  # noqa: BLE001 — the registry is optional at plot time
        return {}
    return {
        str(name): str(meta["version"])
        for name, meta in pricing.items()
        if isinstance(meta, dict) and meta.get("version")
    }


@cache
def _version_aliases() -> Mapping[str, str]:
    """`model_version` slug -> canonical slug, from `model_identity.yaml` ({} on failure).

    A listing whose committed `model_version` is already channel-invariant cannot be merged by
    a listing-id alias; this is the curated slug-level merge (e.g. the size-qualified and short
    Nemotron-3.5-Lightning slugs are one set of published weights).
    """
    try:
        from benchmark.routing.scripts.scan_free_models import load_identity  # noqa: PLC0415

        return dict(load_identity().version_aliases)
    except Exception:  # noqa: BLE001 — the curated file is optional at plot time
        return {}


def _bare(name: str) -> str:
    """Bare canonical identity for a registry NAME, `version` slug or channel listing.

    A registry name resolves through its declared `version` first (a fixpoint for the shipped
    one-id pool), then through the curated `version_aliases:` merge; an unknown name is returned
    unchanged. A channel listing id is canonicalised by the same path, so no `zai-` serving
    prefix and no publisher namespace can leak into an identity or label.
    """
    version = _registry_versions().get(name, name)
    return _version_aliases().get(version, version)


def _resolve(listing: str, model_version: str) -> str:
    """Canonical identity: the row's `model_version` slug, never the registry/serving NAME.

    `model_version` is the bare identity the corpus/registry/overlay row already carries. A
    `version_aliases:` entry merges a same-weights legacy slug onto its canonical one; a row
    with no recorded version falls back to the registry name's declared `version` and finally
    to the listing. Provider is a label either way, never a prefix.
    """
    version = _version_aliases().get(model_version, model_version)
    return version or _bare(listing)


def _corpus_rows(path: Path, *, free: bool) -> Iterable[_ChannelRow]:
    """Channel rows from one results CSV, keyed by `lane`, resolved through `model_version`."""
    if not path.exists():
        return
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            listing = str(row.get("lane") or row.get("model") or "").strip()
            if not listing:
                continue
            yield _ChannelRow(
                listing=listing,
                identity=_resolve(listing, str(row.get("model_version") or "").strip()),
                provider=str(row.get("provider") or "").strip(),
                free=free,
            )


def _registry_rows() -> Iterable[_ChannelRow]:
    """Paid registry rows and overlay (free) rows, so a channel with no corpus row still names."""
    try:
        pricing = config.load_pricing()
    except Exception:  # noqa: BLE001 — the registry is optional at plot time
        pricing = {}
    for name, meta in pricing.items():
        if name.startswith("_") or not isinstance(meta, dict):
            continue
        yield _ChannelRow(
            listing=name,
            identity=_bare(name),
            provider=str(meta.get("provider") or ""),
            free=False,
        )
    try:
        overlay = config.free_registry()
    except Exception:  # noqa: BLE001 — the overlay is optional
        overlay = {}
    for name, meta in overlay.items():
        if not isinstance(meta, dict):
            continue
        yield _ChannelRow(
            listing=name,
            identity=_resolve(name, str(meta.get("version") or "")),
            provider=str(meta.get("provider") or ""),
            free=True,
        )


def _channel_rows() -> list[_ChannelRow]:
    """Every channel listing the committed evidence names, paid and free."""
    rows = [
        *_corpus_rows(config.results_csv_path(), free=False),
        *_corpus_rows(config.free_results_csv_path(), free=True),
        *_registry_rows(),
    ]
    # One row per (listing, identity, provider, channel kind): a shared CSV is read once, and a
    # registry twin of a corpus listing merges to the same channel instead of double-counting.
    seen: set[tuple[str, str, str, bool]] = set()
    unique: list[_ChannelRow] = []
    for row in rows:
        key = (row.listing, row.identity, row.provider, row.free)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _free_cells(identities: Mapping[str, str]) -> dict[str, int]:
    """Identity -> distinct (challenge) cells in the free corpus, so free rows show a count."""
    path = config.free_results_csv_path()
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            listing = str(row.get("lane") or row.get("model") or "").strip()
            cid = str(row.get("challenge_id") or "").strip()
            if not listing or not cid or (listing, cid) in seen:
                continue
            seen.add((listing, cid))
            identity = _canonical(identities, listing)
            counts[identity] = counts.get(identity, 0) + 1
    return counts


def _measured_cells(identities: Mapping[str, str]) -> dict[str, int]:
    """Identity -> measured default-arm cells in the paid corpus plus the free corpus."""
    counts: dict[str, int] = {}
    for cells in config.flatten_default_arm(config.load_results()).values():
        for model in cells:
            identity = _canonical(identities, model)
            counts[identity] = counts.get(identity, 0) + 1
    for identity, n in _free_cells(identities).items():
        counts[identity] = counts.get(identity, 0) + n
    return counts


def _triage_verdicts() -> dict[str, str]:
    """bare identity -> triage verdict over the committed corpus ({} on any read failure)."""
    try:
        return {_bare(row.model): row.verdict for row in triage.triage_default()}
    except Exception:  # noqa: BLE001 — the census degrades to "untriaged", never crashes
        return {}


def _capability_sources() -> dict[str, str]:
    """bare identity -> 'measured' | 'price-prior' for the enabled set ({} on read failure)."""
    try:
        rank = config.capability_rank()
    except Exception:  # noqa: BLE001 — same graceful degradation as the triage read
        return {}
    return {_bare(entry.model): entry.source for entry in rank.ordered}


def gather_evidence() -> Evidence:
    """Read every source once: coverage, cells, triage, capability, channel, pools."""
    rows = _channel_rows()
    identities: dict[str, str] = {}
    providers: dict[str, set[str]] = defaultdict(set)
    listings: dict[str, set[str]] = defaultdict(set)
    paid_identities: set[str] = set()
    free_identities: set[str] = set()
    for row in rows:
        identities.setdefault(row.listing, row.identity)
        if row.provider:
            providers[row.identity].add(row.provider)
        listings[row.identity].add(row.listing)
        (free_identities if row.free else paid_identities).add(row.identity)
    # The shipped pools are paid even when unpriced (an unmeasured frontier slot carries no
    # `load_pricing` row), so they join the paid set explicitly.
    try:
        enabled = frozenset(_bare(m) for m in config.enabled_models())
    except Exception:  # noqa: BLE001 — registry optional at plot time
        enabled = frozenset()
    live = frozenset(_bare(m) for m in packaged_live_pool())
    paid_identities |= set(enabled) | set(live)
    # A free-only identity is one no PAID channel serves; a paid identity with a free mirror
    # stays PAID, so the free overlay can never mark the shipped weights collection-only.
    free_identities -= paid_identities
    coverage: dict[str, set[str]] = defaultdict(set)
    for listing, ids in covered_ids_by_model().items():
        coverage[_canonical(identities, listing)].update(ids)
    return Evidence(
        cells=_measured_cells(identities),
        coverage={k: frozenset(v) for k, v in coverage.items()},
        corpus=len(verified_challenge_ids()),
        triage=_triage_verdicts(),
        capability=_capability_sources(),
        identities=identities,
        providers={k: tuple(sorted(v)) for k, v in providers.items()},
        listings={k: tuple(sorted(v)) for k, v in listings.items()},
        paid_identities=frozenset(paid_identities),
        live=live,
        enabled=enabled,
        floor=cell_floor(),
    )


def _all_models(ev: Evidence) -> list[str]:
    """Union of every canonical identity the committed evidence names."""
    named: set[str] = (
        set(ev.identities.values())
        | set(ev.enabled)
        | set(ev.live)
        | set(ev.cells)
        | set(ev.coverage)
    )
    return sorted(named)


def classify(model: str, ev: Evidence) -> ModelValidity:
    """One row: compute every criterion, then the verdict from the first failing one."""
    channel = PAID if model in ev.paid_identities else FREE
    cells = int(ev.cells.get(model, 0))
    covered = len(ev.coverage.get(model, ()))
    live = model in ev.live
    verdict = ev.triage.get(model, "not-triaged")
    capability = ev.capability.get(model, "not-ranked")
    collection_only = model not in ev.enabled and model not in ev.live

    triage_pass = verdict in _PASSING_TRIAGE
    cap_measured = capability == "measured"
    coverage_ok = cells >= ev.floor
    valid = live and triage_pass and cap_measured and coverage_ok and channel == PAID

    if valid:
        reason = "inference-valid: live, triage KEEP, capability measured, coverage OK"
    elif channel == FREE:
        reason = "collection-only free channel — never enabled or routed"
    elif collection_only:
        reason = "collection-only promo probe — never enabled or routed"
    elif not live:
        reason = (
            "not in the live pool — benchmark-only, triage DROP"
            if model in ev.enabled
            else "not in the live pool or benchmark — collection-only probe"
        )
    elif not triage_pass:
        reason = f"triage {verdict} — slot does not clear the frontier"
    elif not cap_measured:
        reason = "capability rank is a price prior, not a measurement"
    else:
        reason = f"coverage {cells} < K={ev.floor} measured default-arm cells"
    return ModelValidity(
        model=model,
        channel=channel,
        providers=ev.providers.get(model, ()),
        listings=ev.listings.get(model, ()),
        collection_only=collection_only,
        live=live,
        enabled=model in ev.enabled,
        triage=verdict,
        capability=capability,
        cells=cells,
        covered=covered,
        corpus=ev.corpus,
        valid=valid,
        reason=reason,
    )


def validity_census(ev: Evidence | None = None) -> list[ModelValidity]:
    """Every identity the committed evidence names, valid first (free channel last within)."""
    evidence = ev or gather_evidence()
    rows = [classify(model, evidence) for model in _all_models(evidence)]
    return sorted(rows, key=lambda r: (not r.valid, r.channel == FREE, r.model))


def inference_valid_models(ev: Evidence | None = None) -> list[str]:
    """Just the valid identities, price-ascending — the filter every model-subject figure uses.

    The enabled list is canonicalised to bare identities first (a registry slot such as
    `glm-5.2` becomes `glm-5.2`), so a model is never dropped from its own valid row.
    """
    valid = {r.model for r in validity_census(ev) if r.valid}
    try:
        enabled = [_bare(m) for m in config.enabled_models()]
    except Exception:  # noqa: BLE001 — fall back to name order rather than dropping the filter
        enabled = sorted(valid)
    return [m for m in enabled if m in valid]


def inference_valid_set(ev: Evidence | None = None) -> set[str]:
    """Set form of :func:`inference_valid_models` for membership tests."""
    return set(inference_valid_models(ev))


def filter_valid(raw: RawResults, ev: Evidence | None = None) -> RawResults:
    """Drop every channel listing whose canonical identity is outside the valid set."""
    evidence = ev or gather_evidence()
    valid = inference_valid_set(evidence)
    if not valid:
        return raw
    canonical = evidence.identities
    return {
        challenge: {
            model: arms for model, arms in by_model.items() if _canonical(canonical, model) in valid
        }
        for challenge, by_model in raw.items()
    }
