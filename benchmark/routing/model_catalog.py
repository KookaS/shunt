"""model_catalog.csv — one row per canonical weights identity, from the ONE validity census.

The catalogue is the machine-readable face of `model_validity.validity_census`: channel,
provider(s), status, first-failing reason, measured cells, verified-challenge coverage, pass
rate with a Wilson interval, mean billed cost, published size/active parameters, and the three
selection flags the inference-valid predicate reads. It is a generated report
(`benchmark/routing/reports/`), never a hand-maintained table, so a figure and this file can
never disagree about which models clear the bar.

STATUS is the taxonomy the predicate implies, evaluated in a fixed precedence so one row has
one label:

  * ``valid``        — clears every criterion.
  * ``no-evidence``  — no measured cell in either channel (the funnel's exclusion).
  * ``free-only``    — a free channel serves it and no paid one does (never routed).
  * ``insufficient`` — a paid identity below the K cell floor.
  * ``invalid:<c>``  — a paid, adequately-covered identity failing ``live`` / ``triage`` /
                       ``capability`` (coverage and channel already pass by construction).

Every value is READ from `model_validity` / `model_universe` / the registry; nothing is
re-derived or hardcoded here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from benchmark.routing import model_universe, model_validity, plot_style

if TYPE_CHECKING:
    from pathlib import Path

VALID: Final[str] = "valid"
NO_EVIDENCE: Final[str] = "no-evidence"
FREE_ONLY: Final[str] = "free-only"
INSUFFICIENT: Final[str] = "insufficient"

# The criterion names, in `ModelValidity.criteria` order, for the `invalid:<criterion>` label.
_CRITERIA_NAMES: Final[tuple[str, ...]] = (
    "live",
    "triage",
    "capability",
    "coverage",
    "channel",
)

_CSV_FIELDS: Final[tuple[str, ...]] = (
    "model",
    "channel",
    "providers",
    "status",
    "first_failing",
    "cells",
    "covered",
    "corpus",
    "pass_rate",
    "wilson_lo",
    "wilson_hi",
    "mean_cost",
    "total_params",
    "active_params",
    "serving_mode",
    "live",
    "triage",
    "capability",
)


@dataclass(frozen=True)
class CatalogRow:
    """One canonical identity's catalogue record."""

    model: str
    channel: str
    providers: tuple[str, ...]
    status: str
    first_failing: str
    cells: int
    covered: int
    corpus: int
    pass_rate: float
    wilson_lo: float
    wilson_hi: float
    mean_cost: float
    total_params: int | None
    active_params: int | None
    serving_mode: str
    live: bool
    triage: bool
    capability: bool

    @property
    def wilson(self) -> tuple[float, float]:
        return (self.wilson_lo, self.wilson_hi)


def status_of(row: model_validity.ModelValidity) -> str:
    """The one status label for a census row, by the documented precedence (never prose)."""
    if row.valid:
        return VALID
    if not row.evidenced:
        return NO_EVIDENCE
    if row.channel == model_validity.FREE:
        return FREE_ONLY
    if row.cells < model_validity.cell_floor():
        return INSUFFICIENT
    for name, passed in zip(_CRITERIA_NAMES, row.criteria, strict=True):
        if not passed:
            return f"invalid:{name}"
    return INSUFFICIENT


def _param(value: object) -> int | None:
    """A published count, or None for the literal UNDISCLOSED — never a guess."""
    return value if isinstance(value, int) else None


@dataclass(frozen=True)
class _RegistryFacts:
    """A registry row's published size and serving mode, looked up by name or version."""

    total_params: int | None
    active_params: int | None
    serving_mode: str


def _registry_map() -> dict[str, _RegistryFacts]:
    """Registry name/version -> published size + serving mode, best effort ({} on failure)."""
    try:
        from shunt.models.config import load_registry, resolve_models  # noqa: PLC0415

        resolved = resolve_models(load_registry())
    except Exception:  # noqa: BLE001 — the registry is optional at report time
        return {}
    out: dict[str, _RegistryFacts] = {}
    for name, config in resolved.items():
        size = config.size
        facts = _RegistryFacts(
            total_params=_param(size.total_params) if size else None,
            active_params=_param(size.active_params) if size else None,
            serving_mode=config.serving_mode,
        )
        out[name] = facts
        if config.version:
            out.setdefault(config.version, facts)
    return out


def catalog_rows(ev: model_validity.Evidence | None = None) -> list[CatalogRow]:
    """Every canonical identity the census names, as catalogue rows (census order preserved)."""
    evidence = ev or model_validity.gather_evidence()
    census = model_validity.validity_census(evidence)
    perf = model_universe.performance()
    registry = _registry_map()
    return [_row(entry, perf, registry) for entry in census]


def _row(
    entry: model_validity.ModelValidity,
    perf: dict[str, model_universe.Performance],
    registry: dict[str, _RegistryFacts],
) -> CatalogRow:
    """One census row plus its measured outcome, resolved to a catalogue row."""
    stats = perf.get(entry.model)
    passes = stats.passes if stats else 0
    n = stats.n if stats else 0
    lo, hi = plot_style.wilson_interval(passes, n)
    facts = registry.get(entry.model)
    return CatalogRow(
        model=entry.model,
        channel=entry.channel,
        providers=entry.providers,
        status=status_of(entry),
        first_failing=entry.reason,
        cells=entry.cells,
        covered=entry.covered,
        corpus=entry.corpus,
        pass_rate=stats.pass_rate if stats else 0.0,
        wilson_lo=lo,
        wilson_hi=hi,
        mean_cost=stats.mean_cost if stats else 0.0,
        total_params=facts.total_params if facts else None,
        active_params=facts.active_params if facts else None,
        serving_mode=facts.serving_mode if facts else "hosted",
        live=entry.live,
        # The criteria tuple is the predicate's own parts (live, triage, capability, coverage,
        # channel), so the flags index it rather than re-tests the raw verdict strings.
        triage=entry.criteria[1],
        capability=entry.criteria[2],
    )


def write_catalog(out_dir: Path, rows: list[CatalogRow]) -> Path:
    """Write the catalogue CSV into the reports directory and return its path."""
    path = out_dir / "model_catalog.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(_record(row))
    return path


def _record(row: CatalogRow) -> dict[str, object]:
    """One catalogue row as CSV cells (rates rounded to 4 dp, cost to 6 dp)."""
    return {
        "model": row.model,
        "channel": row.channel,
        "providers": ";".join(row.providers),
        "status": row.status,
        "first_failing": row.first_failing,
        "cells": row.cells,
        "covered": row.covered,
        "corpus": row.corpus,
        "pass_rate": round(row.pass_rate, 4),
        "wilson_lo": round(row.wilson_lo, 4),
        "wilson_hi": round(row.wilson_hi, 4),
        "mean_cost": round(row.mean_cost, 6),
        "total_params": row.total_params if row.total_params is not None else "UNDISCLOSED",
        "active_params": row.active_params if row.active_params is not None else "UNDISCLOSED",
        "serving_mode": row.serving_mode,
        "live": row.live,
        "triage": row.triage,
        "capability": row.capability,
    }
