"""Model triage: does a model earn a live-pool slot (routine frontier or escalation evidence)."""

# THE RULE (the committed evidence pool, reproduced mechanically). A model earns a
# live-pool slot iff it clears the Wilson-CI band of the ROUTINE frontier (marginal
# verified pass rate vs price over the incumbent live pool) OR it is a net-helpful
# ESCALATION rung (delta resolve vs the cheap base), else it is a DROP. Reference =
# frontier of the incumbent live pool, per stratum.
#
#   routine: default-arm cells only. rate = passes/n, Wilson CI (lo, hi). The frontier at
#            price p is env(p) = max routine rate over MEASURED live models priced <= p.
#            routine_keep(m) <=> hi(m) >= env(price(m)). A model cheaper than every live
#            model (or cheaper than every measured one) defines the frontier's low end -> KEEP.
#   escalation: from ladder_evidence.build_evidence(), esc_keep(m) <=> verdict == NET-HELPFUL.
#   verdict order: benchmark-disabled -> UNMEASURED-EXCEPTION; n < K -> INSUFFICIENT-DATA;
#            measured-DROP named in triage_exceptions.yaml -> EXCEPTION; routine_keep OR
#            esc_keep -> KEEP; else DROP.
# A benchmark-disabled live model cannot be measured on the committed corpus: it is exempt
# from the gate, never a violation. A named-exception model is a recorded, visible exemption
# (verdict EXCEPTION, surfaced by the gate as advisory) — measured-DROP only, never a
# silent keep.

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import yaml

from benchmark import config
from benchmark.config import _wilson_ci
from benchmark.routing._live_pool import packaged_live_pool
from benchmark.routing.scripts import ladder_evidence

VERDICT_KEEP: Final[str] = "KEEP"
VERDICT_DROP: Final[str] = "DROP"
VERDICT_EXCEPTION: Final[str] = "EXCEPTION"
VERDICT_INSUFFICIENT: Final[str] = "INSUFFICIENT-DATA"
VERDICT_UNMEASURED: Final[str] = "UNMEASURED-EXCEPTION"

# The escalation bar: the exact paired test cleared alpha on the helpful side.
_ESC_HELPFUL: Final[str] = "NET-HELPFUL"

_CSV_HEADER: Final[tuple[str, ...]] = (
    "model",
    "price_per_1m",
    "n",
    "marginal_rate",
    "ci_lo",
    "ci_hi",
    "routine_status",
    "esc_delta",
    "esc_verdict",
    "verdict",
    "exception_note",
)


@dataclass(frozen=True)
class RoutineStat:
    """Routine-stratum evidence: default-arm cells, marginal rate, Wilson CI."""

    n: int
    rate: float
    lo: float
    hi: float


@dataclass(frozen=True)
class TriageRow:
    """One model's triage verdict with the evidence behind it."""

    model: str
    price_per_1m: float
    n: int
    marginal_rate: float
    ci_lo: float
    ci_hi: float
    routine_status: str
    esc_delta: float | None
    esc_verdict: str | None
    verdict: str
    exception_note: str | None


def routine_evidence(matrix: dict, models: Sequence[str]) -> dict[str, RoutineStat]:
    """Per-model (n, marginal rate, Wilson CI) over the flattened default-arm matrix."""
    stats: dict[str, RoutineStat] = {}
    for m in models:
        cells = [c[m] for c in matrix.values() if m in c]
        n = len(cells)
        passes = sum(1 for c in cells if bool(c.get("pass")))
        phat, lo, hi = _wilson_ci(passes, n)
        stats[m] = RoutineStat(n, phat, lo, hi)
    return stats


def envelope(
    live_rates: dict[str, float], live_prices: dict[str, float]
) -> Callable[[float], float | None]:
    """Step lookup env(p): max routine rate among measured live models priced <= p.

    ``live_rates`` keys are the live models WITH a measured routine rate; ``live_prices``
    covers every live model. env(p) is None when no measured live model sits at or below p.
    """
    steps = sorted((live_prices[m], live_rates[m]) for m in live_rates if m in live_prices)
    best: list[tuple[float, float]] = []
    running = 0.0
    for price, rate in steps:
        running = max(running, rate)
        best.append((price, running))

    def env(p: float) -> float | None:
        for price, frontier in reversed(best):
            if price <= p:
                return frontier
        return None

    return env


def routine_keep(
    rate_hi: float,
    price: float,
    env: Callable[[float], float | None],
    live_prices: Sequence[float],
) -> bool:
    """True when the model's CI top clears the frontier at its price (or defines it)."""
    if not any(p <= price for p in live_prices):
        return True  # cheaper than every live model: defines the envelope's low end
    frontier = env(price)
    if frontier is None:
        return True  # no measured live model at or below this price: low end of the frontier
    return rate_hi >= frontier


def escalation_verdicts(ladder_payload: dict[str, Any] | None) -> dict[str, tuple[float, str]]:
    """Per-target (delta, verdict) from the ladder evidence payload; {} when no evidence."""
    if not ladder_payload:
        return {}
    return {
        str(row["target"]): (float(row["delta"]), str(row["verdict"]))
        for row in ladder_payload.get("targets", [])
    }


def _triage_one(
    model: str,
    stat: RoutineStat,
    price: float,
    env: Callable[[float], float | None],
    live_prices: list[float],
    esc: dict[str, tuple[float, str]],
    is_enabled: bool,
    k_min: int,
) -> TriageRow:
    """One row, following the verdict order: insufficient-data, KEEP, DROP."""
    if not is_enabled:
        return TriageRow(
            model,
            price,
            0,
            0.0,
            0.0,
            0.0,
            "N/A",
            None,
            None,
            VERDICT_UNMEASURED,
            "benchmark-disabled: not measurable on the committed corpus",
        )
    if stat.n < k_min:
        return TriageRow(
            model,
            price,
            stat.n,
            stat.rate,
            stat.lo,
            stat.hi,
            "INSUFFICIENT-DATA",
            None,
            None,
            VERDICT_INSUFFICIENT,
            f"n={stat.n} < K={k_min}: too few default-arm cells for a verdict",
        )
    routine = routine_keep(stat.hi, price, env, live_prices)
    delta, verdict = esc.get(model, (None, None))
    if routine or verdict == _ESC_HELPFUL:
        return TriageRow(
            model,
            price,
            stat.n,
            stat.rate,
            stat.lo,
            stat.hi,
            "KEEP" if routine else "DROP",
            delta,
            verdict,
            VERDICT_KEEP,
            None,
        )
    return TriageRow(
        model,
        price,
        stat.n,
        stat.rate,
        stat.lo,
        stat.hi,
        "DROP",
        delta,
        verdict,
        VERDICT_DROP,
        None,
    )


def default_exceptions_path() -> Path:
    """The committed named-exceptions file, resolved from the repo root."""
    return Path(__file__).resolve().parents[1] / "routing" / "triage_exceptions.yaml"


def load_exceptions(path: str | Path | None = None) -> dict[str, str]:
    """Read the named-exceptions YAML; {} when the file holds no exceptions."""
    p = Path(path) if path else default_exceptions_path()
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return {str(k): str(v) for k, v in (data.get("exceptions") or {}).items()}


def _apply_exceptions(
    rows: Sequence[TriageRow], live_pool: Sequence[str], exceptions: dict[str, str]
) -> list[TriageRow]:
    """Promote a measured-DROP row named in ``exceptions`` to verdict EXCEPTION."""
    pool = set(live_pool)
    out: list[TriageRow] = []
    for r in rows:
        reason = exceptions.get(r.model)
        if r.model in pool and r.verdict == VERDICT_DROP and reason is not None:
            out.append(replace(r, verdict=VERDICT_EXCEPTION, exception_note=reason))
        else:
            out.append(r)
    return out


def triage(
    live_pool: Sequence[str],
    candidates: Sequence[str],
    matrix: dict,
    prices: dict[str, float],
    ladder: dict[str, Any] | None,
    enabled: set[str],
    k_min: int,
    exceptions: dict[str, str] | None = None,
) -> list[TriageRow]:
    """Verdict every candidate (live pool union enabled) against the frontier rule."""
    missing = [m for m in candidates if m not in prices]
    if missing:
        raise ValueError(f"candidate(s) without a registry price: {missing}")
    stats = routine_evidence(matrix, candidates)
    live_rates = {m: stats[m].rate for m in live_pool if stats[m].n > 0}
    live_prices = {m: prices[m] for m in live_pool}
    env = envelope(live_rates, live_prices)
    esc = escalation_verdicts(ladder)
    rows = [
        _triage_one(
            m,
            stats[m],
            prices[m],
            env,
            list(live_prices.values()),
            esc,
            m in enabled,
            k_min,
        )
        for m in candidates
    ]
    return _apply_exceptions(rows, live_pool, exceptions or {})


def violations(live_pool: Sequence[str], rows: Sequence[TriageRow]) -> list[TriageRow]:
    """Live-pool rows whose verdict is DROP — the SH015 gate's offence list.

    Named-exception rows (verdict EXCEPTION) are excluded: they are advisory, not violations.
    """
    return [r for r in rows if r.model in live_pool and r.verdict == VERDICT_DROP]


def exceptioned(live_pool: Sequence[str], rows: Sequence[TriageRow]) -> list[TriageRow]:
    """Live-pool rows kept under a named exception — advisory, never a violation."""
    return [r for r in rows if r.model in live_pool and r.verdict == VERDICT_EXCEPTION]


def triage_default() -> list[TriageRow]:
    """Triage over the committed data: enabled benchmark models plus the packaged live pool."""
    live_pool = packaged_live_pool()
    enabled = config.enabled_models()
    candidates = list(dict.fromkeys((*live_pool, *enabled)))
    matrix = config.flatten_default_arm(config.load_results())
    prices = {m: config.cost_per_1m(m) for m in candidates}
    ladder = ladder_evidence.build_evidence()
    k_min = int(config.capability_rank_config()["K"])
    return triage(
        live_pool, candidates, matrix, prices, ladder, set(enabled), k_min, load_exceptions()
    )


def default_report_dir() -> Path:
    """The regenerable report location, alongside the other routing reports."""
    return Path(__file__).resolve().parent / "reports"


def write_csv(rows: Sequence[TriageRow], out_dir: Path) -> Path:
    """Write triage_summary.csv; returns the path written."""
    out = out_dir / "triage_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for r in rows:
            writer.writerow(
                (
                    r.model,
                    f"{r.price_per_1m:.4f}" if r.price_per_1m == r.price_per_1m else "",
                    r.n,
                    f"{r.marginal_rate:.4f}",
                    f"{r.ci_lo:.4f}",
                    f"{r.ci_hi:.4f}",
                    r.routine_status,
                    f"{r.esc_delta:.4f}" if r.esc_delta is not None else "",
                    r.esc_verdict or "",
                    r.verdict,
                    r.exception_note or "",
                )
            )
    return out


def _print_rows(rows: Sequence[TriageRow]) -> None:
    """The human-readable table the CLI prints — the report's own readable form."""
    print(
        f"{'model':<20} {'price':>7} {'n':>5} {'rate':>7} {'ci_lo':>7} {'ci_hi':>7} "
        f"{'routine':>15} {'escΔ':>8} {'esc':>16} verdict"
    )
    for r in rows:
        delta = f"{r.esc_delta:+.4f}" if r.esc_delta is not None else ""
        esc = r.esc_verdict or ""
        print(
            f"{r.model:<20} {r.price_per_1m:>7.2f} {r.n:>5} {r.marginal_rate:>7.4f} "
            f"{r.ci_lo:>7.4f} {r.ci_hi:>7.4f} {r.routine_status:>15} {delta:>8} {esc:>16} "
            f"{r.verdict}"
        )


def main(argv: list[str] | None = None) -> int:
    """Recompute triage over the committed data and write the regenerable CSV report."""
    ap = argparse.ArgumentParser(description="Triage models against the live-pool frontier")
    ap.add_argument(
        "--out-dir",
        default=str(default_report_dir()),
        help="Where triage_summary.csv is written (default: benchmark/routing/reports)",
    )
    args = ap.parse_args(argv)
    rows = triage_default()
    out = write_csv(rows, Path(args.out_dir))
    _print_rows(rows)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
