"""One load of the corpus, shared by every figure `benchmark.routing.report` draws."""

# The report holds the kNN family's ONNX embedders and their per-task index in RSS at
# once and has peaked near 5 GB, so the matrix, the completion and the per-strategy
# selections are computed ONCE here and handed down. A figure that re-derives any of
# them is re-paying that peak.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark import config, plot_frame

if TYPE_CHECKING:
    from benchmark.routing.impute import ImputedMatrix
    from benchmark.routing.plot_style import RawResults

# challenge_id -> (passed, cost, imputed)
StrategyCells = dict[str, tuple[bool, float, bool]]

# The PNGs live in the published docs tree (`docs/assets/figures/routing/`) so the docs can
# link them relatively; the MANIFEST that describes them stays beside the code that writes it.
MANIFEST: Path = Path("benchmark/routing/figures.json")

# "The router" in every figure caption must be a strategy the PRODUCT can be configured with.
# It read "kNN-cascade" — the BLOCKED within-task row — so every figure that said "the router"
# described something no operator can run.
ROUTER_STRATEGY: str = "kNN"
BASELINE_STRATEGY: str = "Always-Frontier"

# The strategy `router.strategy` actually defaults to. It is NOT `ROUTER_STRATEGY`, and the gap
# is deliberate: the 5pp non-inferiority gate was pre-registered on the kNN row, and repointing a
# pre-registered arm after seeing the data would rewrite the verdict. Naming the shipped default
# EXPOSED that the pre-registered arm adjudicates a configuration no user can select — a
# pre-existing defect, not one the rename creates — so the default is published BESIDE the
# pre-registered row, explicitly marked as not pre-registered.
#
# EVERY FIGURE READS THIS NAME RATHER THAN SPELLING ONE. The kill gate's non-pre-registered row,
# the frontier's subtitle and the frontier's context bracket all resolve the default here, so
# moving the default is one edit and cannot leave a figure quoting the previous one.
DEFAULT_STRATEGY: str = "Session-Cascade"


@dataclass(frozen=True)
class RoutingContext:
    """Everything the twelve routing figures read, derived once per report run."""

    out_dir: Path
    manifest: Path
    matrix: dict[str, Any]
    completed: dict[str, Any]
    imputed: ImputedMatrix | None
    tasks: list[str]
    rows: list[dict[str, str]]
    raw: RawResults | None
    models_by_price: list[str]
    banner: str | None
    by_strategy: dict[str, tuple[StrategyCells, set[str]]]
    digest: str

    def provenance(self, generator: str) -> plot_frame.Provenance:
        return plot_frame.Provenance(
            generator=generator, data_digest=self.digest, manifest=self.manifest
        )

    def row(self, name: str) -> dict[str, str] | None:
        return next((r for r in self.rows if r.get("strategy") == name), None)

    def cells(self, name: str) -> tuple[StrategyCells, set[str]] | None:
        return self.by_strategy.get(name)

    def pass_map(self, name: str, *, measured_only: bool = False) -> dict[str, int]:
        """task -> 0/1 for one strategy, optionally restricted to fully-measured paths."""
        found = self.by_strategy.get(name)
        if not found:
            return {}
        cells, unscorable = found
        return {
            tid: int(passed)
            for tid, (passed, _cost, imputed) in cells.items()
            if tid not in unscorable and not (measured_only and imputed)
        }

    def cost_map(self, name: str, *, measured_only: bool = False) -> dict[str, float]:
        found = self.by_strategy.get(name)
        if not found:
            return {}
        cells, unscorable = found
        return {
            tid: cost
            for tid, (_passed, cost, imputed) in cells.items()
            if tid not in unscorable and not (measured_only and imputed)
        }


def corpus_digest(matrix: dict[str, Any], tasks: list[str]) -> str:
    """A stable fingerprint of the outcomes these figures were drawn from."""
    # Deliberately over the OUTCOMES rather than the file: the report completes and
    # samples the matrix in memory, so a digest of challenges.json would claim
    # provenance over data no figure actually read.
    payload = {
        tid: {
            model: [bool(cell.get("pass")), cell.get("real_cost", cell.get("cost", 0.0))]
            for model, cell in sorted(matrix.get("results", {}).get(tid, {}).items())
        }
        for tid in sorted(tasks)
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build_context(  # noqa: PLR0913 (one argument per already-computed input; see module docstring)
    out_dir: Path,
    matrix: dict[str, Any],
    completed: dict[str, Any],
    imputed: ImputedMatrix | None,
    tasks: list[str],
    rows: list[dict[str, str]],
    raw: RawResults | None,
    banner: str | None,
    by_strategy: dict[str, tuple[StrategyCells, set[str]]],
    manifest: Path = MANIFEST,
) -> RoutingContext:
    """Assemble the shared context; the price order is derived here so it is one order."""
    pricing = config.enabled_pricing()
    models_by_price = sorted(config.enabled_models(), key=lambda m: config.cost_per_1m(m, pricing))
    return RoutingContext(
        out_dir=out_dir,
        manifest=manifest,
        matrix=matrix,
        completed=completed,
        imputed=imputed,
        tasks=tasks,
        rows=rows,
        raw=raw,
        models_by_price=models_by_price,
        banner=banner,
        by_strategy=by_strategy,
        digest=corpus_digest(completed or matrix, tasks),
    )
