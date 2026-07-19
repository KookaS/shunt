#!/usr/bin/env python3
"""Adaptive frontier-collection run mode (see docs/benchmark.md)."""

# Two-phase protocol that minimises expensive frontier runs while still validly deciding
# the routing-vs-fixed-frontier kill gate:
#   Phase A — observe cheap+mid on ALL sampled tasks (cheap breadth).
#   Phase B — derive strata (pure, no spend): discriminating set D (cheap/mid disagree)
#             plus a deterministic uniform audit A of the non-discriminating remainder.
#   Phase C — run the frontier (control_model, + high tier if configured) on D ∪ A only.
# Every phase reuses run_matrix's classify_cells, challenge-major executor, --max-cost and
# per-cell checkpointing verbatim. Simulated by default; --live delegates to the real
# harness. State lives entirely in results.csv, so a killed run resumes from the cache.
# WHY each frontier cell ran (discriminating vs audit) is NOT stored in results.csv: it is
# a pure function of the cache + the pinned (audit_fraction, audit_salt, phase_a_models),
# recorded in a gitignored run manifest — so the frozen cache schema and 3-tuple key stay
# unchanged.

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmark import config
from benchmark.routing import integrity
from benchmark.routing.metrics import discriminating_set
from benchmark.runner import image_version, swebench_specs
from benchmark.runner.calibration import DEFAULT_SALT
from benchmark.runner.run_matrix import _has_keys, collect_phase
from benchmark.runner.sampling import AUDIT_SALT, in_frontier_audit
from shunt.secrets import load_dotenv_file

_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "routing" / "artifacts" / "collect_manifest.json"
)


def _tier_of(model: str) -> str:
    """Registry tier for a model ('cheap' if unknown)."""
    info = config.load_pricing().get(model)
    return info.get("tier", "cheap") if isinstance(info, dict) else "cheap"


def phase_a_models(mode: str) -> list[str]:
    """Cheap+mid models to observe on ALL tasks: one representative per tier, or all."""
    enabled = config.enabled_models()  # tier-sorted, cheapest-first within tier
    cheap_mid = [m for m in enabled if _tier_of(m) in ("cheap", "mid")]
    if mode == "full":
        return cheap_mid
    reps: dict[str, str] = {}
    for model in cheap_mid:  # first (cheapest) seen per tier wins
        reps.setdefault(_tier_of(model), model)
    return [reps[t] for t in ("cheap", "mid") if t in reps]


def frontier_models(include_high: bool) -> list[str]:
    """Phase-C models: the control_model baseline, plus the high tier when requested."""
    control = config.frontier_model()
    models = [control] if control else []
    if include_high:
        enabled = config.enabled_models()
        high = next((m for m in enabled if _tier_of(m) == "high" and m not in models), None)
        if high:
            models.append(high)
    return models


def derive_strata(
    cache: dict,
    tasks: list[str],
    models_a: list[str],
    audit_fraction: float,
    audit_salt: str,
) -> tuple[list[str], list[str]]:
    """(D, A) in canonical task order: D = cheap/mid disagree, A = uniform audit of the rest."""
    default_arms = config.default_arm_ids(models_a)
    d_set, u_set = discriminating_set(cache, tasks, models_a, default_arms)
    audit = {t for t in u_set if in_frontier_audit(t, audit_fraction, audit_salt)}
    d_ordered = [t for t in tasks if t in d_set]
    a_ordered = [t for t in tasks if t in audit]
    return d_ordered, a_ordered


def _write_manifest(
    path: Path,
    *,
    models_a: list[str],
    models_c: list[str],
    audit_fraction: float,
    audit_salt: str,
    phase_a_mode: str,
    discriminating: list[str],
    audit: list[str],
    n_tasks: int,
) -> None:
    """Persist the pinned provenance constants so discriminating-vs-audit stays derivable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "audit_salt": audit_salt,
        "audit_fraction": audit_fraction,
        "phase_a_mode": phase_a_mode,
        "phase_a_models": models_a,
        "frontier_models": models_c,
        "tier_map": {m: _tier_of(m) for m in (*models_a, *models_c)},
        "n_tasks": n_tasks,
        "n_discriminating": len(discriminating),
        "n_audit": len(audit),
        "note": "Provenance is DERIVED from results.csv + these constants; "
        "no results.csv column stores why a frontier cell ran.",
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _resolve_digests(tasks: list[str], live: bool) -> dict[str, str | None] | None:
    return (
        image_version.resolve_spec_digests(swebench_specs.spec_image_refs(tasks)) if live else None
    )


def run_collect(
    config_path: str = "benchmark/config.yaml",
    *,
    live: bool = False,
    timeout: int = 600,
    workers: int = 1,
    max_cost: float | None = None,
) -> int:
    """Drive the three-phase collection; returns a process exit code (0 ok, 2 refused)."""
    load_dotenv_file()
    config.load(config_path)
    knobs = config.collect_config()
    audit_fraction = float(knobs.get("audit_fraction", 0.20))
    audit_salt = str(knobs.get("audit_salt", AUDIT_SALT))
    phase_a_mode = str(knobs.get("phase_a_mode", "single"))
    include_high = bool(knobs.get("include_high", False))
    if audit_salt == DEFAULT_SALT:
        print(f"  REFUSING: audit_salt aliases the calibration holdout salt {DEFAULT_SALT!r}.")
        return 2

    hashes = integrity.all_hashes()
    versions = integrity.model_versions()
    tasks = config.sample_tasks(
        sorted(hashes.keys()), seed=config.benchmark_params().get("seed", 42)
    )
    models_a = phase_a_models(phase_a_mode)
    models_c = frontier_models(include_high)
    live = live and _has_keys()
    # SAFETY: never spend real money with un-pinned placeholder sizing constants — the
    # audit_fraction/margin must be pinned from live data first (docs/benchmark.md), else
    # the CI is mis-sized and the gate decision is unreliable. Simulated runs are exempt.
    if live and not bool(knobs.get("constants_pinned", False)):
        print(
            "  REFUSING --live: collect.audit_fraction / noninferiority_margin are un-pinned "
            "placeholders. Pin them from the live results.csv, then set "
            "collect.constants_pinned: true."
        )
        return 2
    if _refuse_live(live, models_a, models_c):
        return 2

    digests = _resolve_digests(tasks, live)
    results_path = config.results_csv_path()
    budget_a = knobs.get("budget_phase_a") or max_cost
    budget_c = knobs.get("budget_phase_c") or max_cost

    print(f"Phase A: observe {models_a} on {len(tasks)} tasks (mode={phase_a_mode})")
    collect_phase(
        tasks,
        models_a,
        config.load_results(),
        hashes,
        versions,
        digests,
        live=live,
        timeout=timeout,
        workers=workers,
        max_cost=budget_a,
        results_path=results_path,
    )

    cache = config.load_results()
    discriminating, audit = derive_strata(cache, tasks, models_a, audit_fraction, audit_salt)
    frontier_tasks = [t for t in tasks if t in set(discriminating) | set(audit)]
    print(
        f"Phase B: {len(discriminating)} discriminating + {len(audit)} audit "
        f"(π={audit_fraction}) → {len(frontier_tasks)} frontier tasks vs {len(tasks)} full"
    )

    print(f"Phase C: run frontier {models_c} on {len(frontier_tasks)} tasks")
    collect_phase(
        frontier_tasks,
        models_c,
        cache,
        hashes,
        versions,
        digests,
        live=live,
        timeout=timeout,
        workers=workers,
        max_cost=budget_c,
        results_path=results_path,
    )

    _write_manifest(
        _MANIFEST_PATH,
        models_a=models_a,
        models_c=models_c,
        audit_fraction=audit_fraction,
        audit_salt=audit_salt,
        phase_a_mode=phase_a_mode,
        discriminating=discriminating,
        audit=audit,
        n_tasks=len(tasks),
    )
    print(f"  manifest → {_MANIFEST_PATH}")
    if not live:
        print("  simulated: phases classified only; leaving cells uncached (no fabrication).")
    return 0


def _refuse_live(live: bool, models_a: list[str], models_c: list[str]) -> bool:
    """True (refuse) if a live run would spend uncached budget on any phase model."""
    if not live:
        return False
    uncached = config.models_missing_cache(list({*models_a, *models_c}))
    if uncached:
        print(
            f"  REFUSING --live: models without a cache-read discount {uncached} would burn "
            "uncached budget. Disable them or add cache_read_cost_per_1m to the registry."
        )
        return True
    return False


def _add_args(ap: argparse.ArgumentParser, config_path: str) -> None:
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument(
        "--live", action="store_true", help="Run uncached cells for real (needs Docker + keys)"
    )
    ap.add_argument("--timeout", type=int, default=600, help="Per-cell timeout (live mode)")
    ap.add_argument("--workers", type=int, default=1, help="Concurrent live cells (1 = serial)")
    ap.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Abort remaining cells once cumulative real_cost (USD) crosses this.",
    )


def main(config_path: str = "benchmark/config.yaml") -> int:
    print(
        "DEPRECATED: `python -m benchmark.runner.collect` is now "
        "`python -m benchmark.runner.run_matrix --strategy cost_optimal`.",
        file=sys.stderr,
    )
    ap = argparse.ArgumentParser(
        description="[deprecated alias] Shunt adaptive frontier collection (simulated by default)."
    )
    _add_args(ap, config_path)
    args = ap.parse_args()
    return run_collect(
        args.config,
        live=args.live,
        timeout=args.timeout,
        workers=args.workers,
        max_cost=args.max_cost,
    )


if __name__ == "__main__":
    raise SystemExit(main())
