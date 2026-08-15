#!/usr/bin/env python3
"""Per-target escalation evidence: price multiple, overlap, helps/hurts, paired delta, null."""

# WHY THIS SCRIPT EXISTS. The shipped escalation ladder walks candidates in PRICE order, so
# "escalate" and "escalate to something better" are not the same claim. Whether a given rung
# helps is an empirical question about the corpus, and it was answered once by a one-off probe
# whose numbers no committed entrypoint could regenerate. This is that entrypoint: it re-derives
# the per-rung evidence from the committed results.csv every time it runs, so a rung that turns
# net-harmful (or stops being) shows up as a changed number rather than as folklore.
#
# WHAT A ROW MEANS. For one base model (the cheap tier) and one candidate target:
#   n        challenges where BOTH the base and the target have a scored default-arm outcome.
#            Only the overlap is comparable — a target measured on an easier subset would
#            otherwise look better for free.
#   helps    base failed, target resolved.       hurts    base resolved, target failed.
#   delta    target resolve rate - base resolve rate on that overlap == (helps - hurts) / n.
#            Ties (both pass, both fail) cancel, which is why the pair counts fully determine it.
#   ci95     paired percentile bootstrap over the per-challenge difference.
#   null     the EXACT paired randomization test, in closed form. Under "the two models are
#            exchangeable on this task" each discordant pair falls either way with probability
#            1/2, so delta's null distribution is (2B - d)/n for B ~ Binomial(d, 1/2) over the
#            d discordant pairs. `mcnemar_exact_p` is that test's two-sided p; `null_ci95` is the
#            same distribution's central 95%. Both are exact — no Monte Carlo error to argue
#            about, and no seed that could be chosen after seeing the answer.
#
# WHAT IT DELIBERATELY DOES NOT DO. It does not rank, choose, or wire a ladder, and it reads no
# router module: the evidence and the routing decision are kept apart on purpose. Pricing is read
# from the shipped registry through `benchmark.config`, never restated here — a hardcoded price
# is a number that goes stale silently.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import benchmark.routing
from benchmark import config
from benchmark.routing.metrics import mcnemar_exact_p

_DEFAULT_CONFIG = "benchmark/benchmark.yaml"
_ALPHA = 0.05
_EMPTY_MESSAGE = (
    "No results yet — results.csv holds no rows. "
    "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
)


@dataclass
class LadderRow:
    """One candidate escalation target scored against the cheap base model."""

    target: str
    price_per_1m: float
    price_multiple: float
    n: int
    helps: int
    hurts: int
    base_resolved: int
    target_resolved: int
    delta: float
    ci95: tuple[float, float]
    null_ci95: tuple[float, float]
    p_value: float

    @property
    def verdict(self) -> str:
        """NET-HELPFUL / NET-HARMFUL once the exact paired test clears alpha, else neutral."""
        if self.p_value >= _ALPHA or self.delta == 0:
            return "INDISTINGUISHABLE"
        return "NET-HELPFUL" if self.delta > 0 else "NET-HARMFUL"


def outcomes_by_model(results: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """Default-arm pass/fail per model, keyed by challenge — the canonical single outcome."""
    flat = config.flatten_default_arm(results)
    out: dict[str, dict[str, bool]] = {}
    for challenge_id, per_model in flat.items():
        for model, row in per_model.items():
            out.setdefault(model, {})[challenge_id] = bool(row.get("pass"))
    return out


def paired_outcomes(
    outcomes: dict[str, dict[str, bool]], base: str, target: str
) -> list[tuple[str, bool, bool]]:
    """(challenge, base_pass, target_pass) for every challenge BOTH models were run on."""
    base_rows = outcomes.get(base, {})
    target_rows = outcomes.get(target, {})
    shared = sorted(set(base_rows) & set(target_rows))
    return [(cid, base_rows[cid], target_rows[cid]) for cid in shared]


def paired_bootstrap_ci(diffs: list[int], *, seed: int, n_resamples: int) -> tuple[float, float]:
    """95% percentile CI for the mean paired difference, resampling whole challenges."""
    # The same estimator `run_eval._paired_bootstrap_ci` uses for the router-vs-frontier
    # contrast, restated because that one is private to the eval driver and reports in
    # percentage points. Challenge = row here, so a row resample IS the challenge resample.
    if not diffs:
        return (float("nan"), float("nan"))
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    tail = int(_ALPHA / 2 * n_resamples)
    return (means[tail], means[n_resamples - 1 - tail])


def exact_null_band(discordant: int, n: int) -> tuple[float, float]:
    """Central 95% of delta under the exact paired-exchangeability null (binomial, p=1/2)."""
    if n == 0:
        return (float("nan"), float("nan"))
    if discordant == 0:
        return (0.0, 0.0)
    cdf: list[float] = []
    running = 0.0
    for k in range(discordant + 1):
        running += math.comb(discordant, k) / 2.0**discordant
        cdf.append(running)
    low = next(k for k, c in enumerate(cdf) if c >= _ALPHA / 2)
    high = next(k for k, c in enumerate(cdf) if c >= 1 - _ALPHA / 2)
    return ((2 * low - discordant) / n, (2 * high - discordant) / n)


def evaluate_target(
    outcomes: dict[str, dict[str, bool]],
    base: str,
    target: str,
    pricing: dict[str, Any],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> LadderRow | None:
    """Score one target against the base, or None when the two never overlap."""
    pairs = paired_outcomes(outcomes, base, target)
    if not pairs:
        return None
    diffs = [int(t) - int(b) for _cid, b, t in pairs]
    helps = sum(1 for d in diffs if d > 0)
    hurts = sum(1 for d in diffs if d < 0)
    base_price = config.cost_per_1m(base, pricing)
    target_price = config.cost_per_1m(target, pricing)
    return LadderRow(
        target=target,
        price_per_1m=round(target_price, 4),
        price_multiple=round(target_price / base_price, 2) if base_price else float("nan"),
        n=len(pairs),
        helps=helps,
        hurts=hurts,
        base_resolved=sum(1 for _cid, b, _t in pairs if b),
        target_resolved=sum(1 for _cid, _b, t in pairs if t),
        delta=round(sum(diffs) / len(diffs), 4),
        ci95=_round2(paired_bootstrap_ci(diffs, seed=seed, n_resamples=n_resamples)),
        null_ci95=_round2(exact_null_band(helps + hurts, len(pairs))),
        p_value=mcnemar_exact_p(helps, hurts),
    )


def _round2(pair: tuple[float, float]) -> tuple[float, float]:
    """Round an interval to 4 decimals — the precision the report publishes."""
    return (round(pair[0], 4), round(pair[1], 4))


def build_evidence(
    base: str | None = None,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """The full evidence payload, or None when results.csv holds nothing comparable."""
    outcomes = outcomes_by_model(config.load_results())
    pricing = config.enabled_pricing()
    priced = [m for m in config.enabled_models() if m in outcomes]
    if not priced:
        return None
    base_model = base or priced[0]  # enabled_models() is price-ascending: cheapest first
    if base_model not in outcomes:
        raise ValueError(f"base model {base_model!r} has no scored rows in results.csv")
    scored = (
        evaluate_target(outcomes, base_model, t, pricing, n_resamples=n_resamples, seed=seed)
        for t in priced
        if t != base_model
    )
    rows = sorted((r for r in scored if r is not None), key=lambda r: r.price_multiple)
    csv_path = config.results_csv_path()
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest() if csv_path.exists() else "absent"
    return {
        "generated_from": f"results.csv@{digest}",
        "base_model": base_model,
        "base_price_per_1m": round(config.cost_per_1m(base_model, pricing), 4),
        "knobs": {"resamples": n_resamples, "seed": seed, "alpha": _ALPHA},
        "targets": [asdict(row) | {"verdict": row.verdict} for row in rows],
    }


def _print_table(payload: dict[str, Any]) -> None:
    """Print the per-rung table in price order — the ladder's own walk order."""
    print()
    print(f"Base: {payload['base_model']} (${payload['base_price_per_1m']}/1M)")
    print(
        f"{'Target':<20} {'xPrice':>7} {'n':>5} {'helps':>6} {'hurts':>6} "
        f"{'delta':>8} {'ci95':>18} {'null ci95':>18} {'p':>9}  verdict"
    )
    print("-" * 120)
    for row in payload["targets"]:
        ci = f"[{row['ci95'][0]:+.3f},{row['ci95'][1]:+.3f}]"
        nb = f"[{row['null_ci95'][0]:+.3f},{row['null_ci95'][1]:+.3f}]"
        print(
            f"{row['target']:<20} {row['price_multiple']:>7.2f} {row['n']:>5} "
            f"{row['helps']:>6} {row['hurts']:>6} {row['delta']:>+8.4f} {ci:>18} {nb:>18} "
            f"{row['p_value']:>9.3g}  {row['verdict']}"
        )


def default_output_path() -> Path:
    """The regenerable report location, alongside the other routing reports."""
    return Path(benchmark.routing.__file__).resolve().parent / "reports" / "ladder_evidence.json"


def main(config_path: str = _DEFAULT_CONFIG) -> None:
    """Re-derive the per-rung escalation evidence and write it as a regenerable JSON report."""
    config.load(config_path)
    ap = argparse.ArgumentParser(description="Per-rung escalation evidence over results.csv")
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--base", default=None, help="Base model (default: cheapest enabled)")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON report path (default: routing/reports/)")
    args = ap.parse_args()
    if args.config != config_path:
        config.load(args.config)

    payload = build_evidence(args.base, n_resamples=args.resamples, seed=args.seed)
    if payload is None or not payload["targets"]:
        print(_EMPTY_MESSAGE)
        return
    _print_table(payload)
    out = Path(args.out) if args.out else default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
