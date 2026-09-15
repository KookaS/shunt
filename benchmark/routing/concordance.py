"""Cross-provider concordance for one model identity served by several providers.

The routing registry treats a model ``version`` (the weights identity, e.g.
``openai/gpt-oss-120b``) as one stable quality unit, but a free listing is served by a
provider-specific stack whose quantisation, sampling defaults and tool-call templating
differ. This module measures whether the SAME identity, served by DIFFERENT providers,
produces the same pass rate on the same challenge: it pairs rows across providers at one
(challenge, arm), reports per-pair agreement and a paired pass-rate delta with a bootstrap
CI, and flags a provider pair whose CI excludes zero (the divergence warning).

Pure and deterministic over raw corpus rows: no model calls, no I/O beyond reading a CSV.
The verdict it emits is about the INSTRUMENT, so it clears a positive-control /
shuffled-label-null pair through the shipped numeric adjudicator before it is quoted.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.routing import impute, integrity

# Total task/challenge observations behind a provider-pair delta: two-sided 95% bootstrap.
DEFAULT_BOOTSTRAP: Final[int] = 2000
# Independent label-shuffles behind the destroyed-signal null.
DEFAULT_NULL_DRAWS: Final[int] = 200
# A confidence level, NOT a p-threshold: the paired CI decides divergence.
CHANCE_LEVEL: Final[float] = 0.0


@dataclass(frozen=True)
class Observation:
    """One real corpus row reduced to the fields concordance pairs on."""

    identity: str
    provider: str
    model: str
    challenge_id: str
    arm: str
    passed: bool


@dataclass(frozen=True)
class CellPair:
    """Two providers' outcomes for one identity at the same (challenge, arm)."""

    identity: str
    challenge_id: str
    arm: str
    provider_a: str
    provider_b: str
    model_a: str
    model_b: str
    pass_a: bool
    pass_b: bool

    @property
    def agrees(self) -> bool:
        return self.pass_a == self.pass_b


@dataclass(frozen=True)
class ProviderPair:
    """A provider pair's concordance for one identity, pooled over shared challenges."""

    identity: str
    provider_a: str
    provider_b: str
    n_paired: int
    pass_rate_a: float
    pass_rate_b: float
    delta: float
    ci_lo: float
    ci_hi: float
    n_agree: int
    agreement: float
    n_a_only: int
    n_b_only: int
    z: float
    diverges: bool


@dataclass(frozen=True)
class ConcordanceReport:
    """The measured concordance of every provider pair over the paired corpus."""

    pairs: tuple[ProviderPair, ...]
    n_cells: int

    @property
    def flagged(self) -> tuple[ProviderPair, ...]:
        return tuple(pair for pair in self.pairs if pair.diverges)

    @property
    def warnings(self) -> tuple[str, ...]:
        """The divergence warning rows: one per pair whose delta CI excludes zero."""
        return tuple(
            f"DIVERGENCE {pair.identity}: {pair.provider_a} vs {pair.provider_b} — "
            f"pass {pair.pass_rate_a:.3f} vs {pair.pass_rate_b:.3f} "
            f"(delta {pair.delta:+.3f}, 95% CI [{pair.ci_lo:+.3f}, {pair.ci_hi:+.3f}], "
            f"n={pair.n_paired})"
            for pair in self.flagged
        )


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in ("true", "1", "yes")


def load_rows(path: Path) -> list[dict[str, str]]:
    """Every raw row in *path* (empty list when the file is absent)."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def observations(rows: Iterable[dict[str, str]]) -> list[Observation]:
    """Reduce raw rows to observations, dropping zero-work residue and unlabelled rows.

    A row needs a resolved identity (``model_version``), a provider label, and a challenge;
    anything else cannot be paired and is skipped rather than defaulted.
    """
    out: list[Observation] = []
    for row in rows:
        if impute.is_zero_work(row):
            continue
        identity = str(row.get("model_version") or "").strip()
        provider = str(row.get("provider") or "").strip()
        model = str(row.get("lane") or row.get("model") or "").strip()
        challenge = str(row.get("challenge_id") or "").strip()
        if not (identity and provider and model and challenge):
            continue
        arm = str(row.get("reasoning") or integrity.DEFAULT_REASONING)
        out.append(Observation(identity, provider, model, challenge, arm, _truthy(row.get("pass"))))
    return out


def pair_observations(obs: Sequence[Observation]) -> list[CellPair]:
    """Every cross-provider pair at one (identity, challenge, arm), deterministically."""
    by_cell: dict[tuple[str, str, str], dict[str, Observation]] = defaultdict(dict)
    for item in obs:
        by_cell[(item.identity, item.challenge_id, item.arm)].setdefault(item.provider, item)
    pairs: list[CellPair] = []
    for (identity, challenge, arm), by_provider in by_cell.items():
        providers = sorted(by_provider)
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                a, b = by_provider[providers[i]], by_provider[providers[j]]
                pairs.append(
                    CellPair(
                        identity,
                        challenge,
                        arm,
                        providers[i],
                        providers[j],
                        a.model,
                        b.model,
                        a.passed,
                        b.passed,
                    )
                )
    return pairs


def _paired_delta_ci(pairs: Sequence[CellPair], *, seed: int, n_boot: int) -> tuple[float, float]:
    """95% percentile CI on ``p_a - p_b`` by resampling the paired challenge observations."""
    n = len(pairs)
    diffs = [int(pair.pass_a) - int(pair.pass_b) for pair in pairs]
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def _summarise(
    identity: str,
    provider_a: str,
    provider_b: str,
    pairs: Sequence[CellPair],
    *,
    seed: int,
    n_boot: int,
) -> ProviderPair:
    order = sorted(pairs, key=lambda pair: (pair.challenge_id, pair.arm))
    n = len(order)
    passed_a = sum(1 for pair in order if pair.pass_a)
    passed_b = sum(1 for pair in order if pair.pass_b)
    n_agree = sum(1 for pair in order if pair.agrees)
    a_only = sum(1 for pair in order if pair.pass_a and not pair.pass_b)
    b_only = sum(1 for pair in order if pair.pass_b and not pair.pass_a)
    discordant = a_only + b_only
    z = (a_only - b_only) / math.sqrt(discordant) if discordant else 0.0
    ci_lo, ci_hi = _paired_delta_ci(order, seed=seed, n_boot=n_boot)
    return ProviderPair(
        identity=identity,
        provider_a=provider_a,
        provider_b=provider_b,
        n_paired=n,
        pass_rate_a=passed_a / n,
        pass_rate_b=passed_b / n,
        delta=(passed_a - passed_b) / n,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_agree=n_agree,
        agreement=n_agree / n,
        n_a_only=a_only,
        n_b_only=b_only,
        z=z,
        diverges=(ci_lo > 0.0 or ci_hi < 0.0),
    )


def provider_pairs(
    cell_pairs: Sequence[CellPair], *, seed: int = 42, n_boot: int = DEFAULT_BOOTSTRAP
) -> list[ProviderPair]:
    """Pool cell pairs per (identity, provider pair) and summarise each."""
    grouped: dict[tuple[str, str, str], list[CellPair]] = defaultdict(list)
    for pair in cell_pairs:
        grouped[(pair.identity, pair.provider_a, pair.provider_b)].append(pair)
    out: list[ProviderPair] = []
    for offset, ((identity, a, b), group) in enumerate(sorted(grouped.items())):
        out.append(_summarise(identity, a, b, group, seed=seed + offset, n_boot=n_boot))
    return out


def concordance_report(
    rows: Sequence[dict[str, str]], *, seed: int = 42, n_boot: int = DEFAULT_BOOTSTRAP
) -> ConcordanceReport:
    """Pair the corpus across providers and report per-pair agreement and divergence."""
    obs = observations(rows)
    cell_pairs = pair_observations(obs)
    return ConcordanceReport(
        pairs=tuple(provider_pairs(cell_pairs, seed=seed, n_boot=n_boot)),
        n_cells=len(cell_pairs),
    )


def _z_squared(cell_pairs: Sequence[CellPair]) -> float:
    grouped: dict[tuple[str, str, str], list[CellPair]] = defaultdict(list)
    for pair in cell_pairs:
        grouped[(pair.identity, pair.provider_a, pair.provider_b)].append(pair)
    total = 0.0
    for group in grouped.values():
        a_only = sum(1 for pair in group if pair.pass_a and not pair.pass_b)
        b_only = sum(1 for pair in group if pair.pass_b and not pair.pass_a)
        discordant = a_only + b_only
        if discordant:
            total += (a_only - b_only) ** 2 / discordant
    return total


def concordance_statistic(rows: Sequence[dict[str, str]]) -> float:
    """A scalar cross-provider signal: the sum of squared per-pair McNemar z's.

    Under a provider-blind null each z is standard normal, so the statistic is
    chi-square-shaped with one degree of freedom per provider pair; a real serving
    difference pushes its pairs' z away from zero and the sum up. No bootstrap here — the
    controls call this many times, and the CI is a reporting concern.
    """
    return _z_squared(pair_observations(observations(rows)))


def _shuffle_pass(rows: Sequence[dict[str, str]], *, seed: int) -> list[dict[str, str]]:
    """Permute pass outcomes among providers within each (identity, challenge, arm).

    Preserves each challenge's pass multiset and each identity's provider set, destroying
    only the provider-to-outcome link — the destroyed-signal null for this instrument.
    """
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (
            str(row.get("model_version") or ""),
            str(row.get("challenge_id") or ""),
            str(row.get("reasoning") or integrity.DEFAULT_REASONING),
        )
        groups[key].append(index)
    out = [dict(row) for row in rows]
    rng = random.Random(seed)
    for indices in groups.values():
        values = [rows[index].get("pass") for index in indices]
        rng.shuffle(values)
        for index, value in zip(indices, values, strict=True):
            out[index]["pass"] = "" if value is None else str(value)
    return out


def _null_band(draws: Sequence[float], centre: float) -> float:
    """97.5th-percentile spread of the null draws around their own centre (half-band)."""
    ordered = sorted(abs(draw - centre) for draw in draws)
    if not ordered:
        return 0.0
    return float(ordered[int(0.975 * (len(ordered) - 1))])


def instrument_control(
    planted: Sequence[dict[str, str]], *, n_draws: int = DEFAULT_NULL_DRAWS, seed: int = 0
) -> AdmissibilityResult:
    """Positive control + shuffled-label null for the concordance instrument.

    ``planted`` is a control fixture carrying a KNOWN cross-provider effect (a control, never
    measurement data). The positive leg must recover it; the null leg shuffles the pass
    outcomes among providers and must collapse to chance. Adjudicated by the shipped gate.
    """
    positive = concordance_statistic(planted)
    draws = [
        concordance_statistic(_shuffle_pass(planted, seed=seed + 1 + draw))
        for draw in range(n_draws)
    ]
    shuffled = statistics.median(draws)
    return admissibility_verdict(
        positive, shuffled, chance_level=CHANCE_LEVEL, chance_band=_null_band(draws, shuffled)
    )


def planted_control_corpus(
    *, n_identities: int = 3, n_challenges: int = 20
) -> list[dict[str, str]]:
    """A deterministic control corpus with a planted provider effect (never written)."""
    rows: list[dict[str, str]] = []
    weak_passes = 4
    for identity_index in range(n_identities):
        identity = f"control/model-{identity_index}"
        for challenge_index in range(n_challenges):
            for provider, passed in (
                ("provider-a", True),
                ("provider-b", challenge_index < weak_passes),
            ):
                rows.append(
                    {
                        "challenge_id": f"ctl-{challenge_index:02d}",
                        "model": f"{identity}:{provider}",
                        "model_version": identity,
                        "provider": provider,
                        "reasoning": integrity.DEFAULT_REASONING,
                        "pass": str(passed),
                        "calls": "5",
                        "real_cost": "0.0",
                    }
                )
    return rows


def _print_report(report: ConcordanceReport) -> None:
    print(
        f"Cross-provider concordance: {len(report.pairs)} provider pair(s) over "
        f"{report.n_cells} paired cell(s)"
    )
    for pair in report.pairs:
        flag = " DIVERGES" if pair.diverges else ""
        print(
            f"  {pair.identity:32} {pair.provider_a:16} vs {pair.provider_b:16} "
            f"n={pair.n_paired:<3} agree={pair.agreement:.2f} "
            f"delta={pair.delta:+.2f} CI=[{pair.ci_lo:+.2f},{pair.ci_hi:+.2f}]{flag}"
        )
    for warning in report.warnings:
        print(f"  {warning}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="configs/free-tier/benchmark.yaml",
        help="Campaign config whose paths.results_csv names the corpus",
    )
    ap.add_argument("--results", default=None, help="Corpus CSV (default: the config's path)")
    ap.add_argument("--draws", type=int, default=DEFAULT_NULL_DRAWS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    config.load(args.config)
    path = Path(args.results) if args.results else config.results_csv_path()
    rows = load_rows(path)
    if not rows:
        print(f"No corpus rows at {path} — nothing to measure yet.")
        return 0

    report = concordance_report(rows, seed=args.seed)
    _print_report(report)
    verdict = instrument_control(planted_control_corpus(), n_draws=args.draws, seed=args.seed)
    print(f"\n{verdict.headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
