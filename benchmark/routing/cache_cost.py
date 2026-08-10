"""The one cache-aware cost model: a repeat of the same model banks its cache discount.

Shared by the kill gate and the routing summary so a published cost cannot be cache-aware
in one table and cache-blind in the other.
"""

# WHY THIS IS NOT A DETAIL. Every cascade re-serves the SAME model on consecutive attempts by
# construction, and providers price a repeated prefix at the cache-read rate. A raw sum over
# attempts therefore charges the second attempt as if the first had never run. On this corpus the
# gap between what the provider billed (`real_cost`) and what naive per-call pricing predicts
# (`estimated_cost`) is 4.8x, so "naive total cost" is not a conservative reading of the bill —
# it is a different quantity, and ranking strategies on it ranks them on cache blindness.
#
# WHAT IS MEASURED HERE AND WHAT IS ASSUMED. The per-model DISCOUNT is read off the registry
# (`cache_read_cost_per_1m` / `input_cost_per_1m`) and the per-model INPUT SHARE is derived from
# the measured token mix in results.csv priced at the registry's rates. Only the HIT RATE is
# assumed, and it carries the `assumed` flag wherever it is reported.

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from benchmark import config

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

MEASURED: Final[str] = "measured"
ASSUMED: Final[str] = "assumed"

# No run in this corpus records a per-turn cache-hit ratio, so the hit rate is the one genuinely
# invented number left in the cache-aware criterion. It stays a named constant with an ``assumed``
# flag rather than a literal buried in a default argument.
ASSUMED_CACHE_HIT_RATE: Final[float] = 0.9
# Fallback discount for a model whose registry entry carries no cache-read price. Deliberately
# conservative-for-the-router: it neither invents a discount the provider does not offer nor
# assumes the worst.
ASSUMED_CACHE_DISCOUNT: Final[float] = 0.5
# Fallback input share for a model with no measured tokens in results.csv. Agentic coding is
# input-dominated everywhere it has been measured here (0.86-0.97), so the fallback sits at the
# bottom of that measured range rather than at a price-implied guess.
ASSUMED_INPUT_SHARE: Final[float] = 0.86


@dataclass(frozen=True)
class CachePrice:
    """One model's cache economics, and whether the corpus measured them or we assumed."""

    model: str
    input_share: float
    discount: float
    hit_rate: float
    provenance: str
    share_provenance: str

    @property
    def saving_fraction(self) -> float:
        """Share of a turn's cost a cache hit removes, at this model's input share."""
        return self.input_share * self.hit_rate * self.discount


def _token_totals(results_csv: Path) -> dict[str, tuple[int, int]]:
    """Per model, ``(total in_tok, total out_tok)`` over every measured row."""
    totals: dict[str, list[int]] = {}
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            bucket = totals.setdefault(row["model"], [0, 0])
            bucket[0] += int(row.get("in_tok") or 0)
            bucket[1] += int(row.get("out_tok") or 0)
    return {m: (i, o) for m, (i, o) in totals.items()}


def measured_input_shares(results_csv: Path | None = None) -> dict[str, float]:
    """Per model, the COST-weighted share of spend that is input, from measured tokens.

    ``in_tok * input_price / (in_tok * input_price + out_tok * output_price)``.
    """
    # This replaces a price-ratio proxy (`input_price / (input_price + output_price)`), which is
    # a statement about the PRICE LIST, not about what the corpus actually sent. The two differ by
    # 3-8x here — the proxy reads 0.11-0.33 where the measured mix reads 0.86-0.97 — and always in
    # the direction that understates caching, because output tokens are priced high and this
    # workload barely emits any (deepseek: 356.9M in vs 4.8M out).
    pricing = config.load_pricing()
    shares: dict[str, float] = {}
    for model, (in_tok, out_tok) in _token_totals(
        results_csv if results_csv is not None else config.results_csv_path()
    ).items():
        info = pricing.get(model)
        if not isinstance(info, dict):
            continue
        in_spend = in_tok * float(info.get("input_cost_per_1m") or 0.0)
        out_spend = out_tok * float(info.get("output_cost_per_1m") or 0.0)
        if in_spend + out_spend > 0:
            shares[model] = in_spend / (in_spend + out_spend)
    return shares


def _discount(info: object) -> tuple[float, str]:
    """A model's cache-read discount from the registry, or the assumed fallback."""
    if isinstance(info, dict):
        read = info.get("cache_read_cost_per_1m")
        inp = info.get("input_cost_per_1m")
        if (
            isinstance(read, int | float)
            and isinstance(inp, int | float)
            and inp > 0
            and 0 <= read < inp
        ):
            return (1.0 - read / inp, MEASURED)
    return (ASSUMED_CACHE_DISCOUNT, ASSUMED)


def cache_prices(
    models: Iterable[str], shares: Mapping[str, float] | None = None
) -> dict[str, CachePrice]:
    """Per-model cache economics: registry discount x measured input share, each flagged."""
    pricing = config.load_pricing()
    measured = dict(shares) if shares is not None else measured_input_shares()
    out: dict[str, CachePrice] = {}
    for model in models:
        discount, provenance = _discount(pricing.get(model))
        share = measured.get(model)
        out[model] = CachePrice(
            model=model,
            input_share=ASSUMED_INPUT_SHARE if share is None else share,
            discount=discount,
            hit_rate=ASSUMED_CACHE_HIT_RATE,
            provenance=provenance,
            share_provenance=ASSUMED if share is None else MEASURED,
        )
    return out


def cache_aware_total(
    attempts: Sequence[tuple[str, float]], prices: Mapping[str, CachePrice]
) -> float:
    """Total cost of a billed attempt sequence once a repeat of the same model banks its
    discount."""
    # Adjacency is the whole model: the discount applies when the PREVIOUS billed attempt served
    # the same model, because that is when the prefix is still warm. A caller that reorders the
    # sequence changes the answer, which is why no bootstrap resamples this statistic.
    savings = 0.0
    previous: str | None = None
    for model, cost in attempts:
        price = prices.get(model)
        if previous is not None and model == previous and price is not None:
            savings += cost * price.saving_fraction
        previous = model
    return sum(cost for _model, cost in attempts) - savings
