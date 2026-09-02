"""Repriced cost: what a measured row would cost at TODAY's cheapest listed price.

The plot-only cost basis; `real_cost` stays the immutable audit record.
"""

# WHY THIS EXISTS. Two costs already live on every results.csv row and mean different things.
# `real_cost` is what the provider billed, cache included -- a historical FACT about a run that
# happened, and the only cost a verdict may rest on. `estimated_cost` is the registry's list
# price times the row's tokens, and the registry's `pricing` block deliberately means *the
# price in force when the run happened* (its `price_as_of` says so).
#
# Neither answers the question a router's cost axis is asked: what would this cost a user who
# shops around TODAY. Prices fall, providers appear, and a July cell plotted beside an August
# one is compared at two different price levels with nothing on the canvas saying so. This
# module answers it from a dated, committed price sheet (`data/price_sheet.json`, refreshed by
# `scripts/refresh_price_sheet.py`), and is the ONLY reader of that sheet.
#
# THREE RULES, and they are the whole contract:
#
# 1. PLOTS ONLY. Nothing here may reach a kill gate, a pre-registered verdict, or
#    `strategy_summary.csv`. A price refresh that could flip a recorded result would make the
#    verdict a statement about a price sheet rather than about an experiment.
# 2. NAIVE ONLY. A repriced cost is `in_tok * input + out_tok * output`. It feeds the naive
#    cost axes; the cache-aware axis keeps historical `real_cost`, priced at run time. Legacy
#    rows carry no `cached_in_tok`, so a cache-aware repricing would not merely be unavailable
#    -- it would be invented.
# 3. MISSING IS MISSING. A model the sheet does not price has NO repriced cost. It is never
#    imputed, never zero, and an aggregate over it is refused by `validate.require_measured`
#    rather than silently completed -- the same read-side rule the optional columns live under.

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from benchmark.routing import validate

_DATA_DIR: Final[Path] = Path(__file__).resolve().parent / "data"
SHEET_PATH: Final[Path] = _DATA_DIR / "price_sheet.json"
CHANNELS_PATH: Final[Path] = _DATA_DIR / "price_channels.yaml"

# How a model's many live quotes collapse to one canonical price. Ranking on the SUM of the
# two per-1M prices rather than on a blended rate is deliberate: a blend needs a token mix,
# the mix differs per corpus, and a canonical price that moved when the corpus moved would be
# unreadable. It is also the rule `shunt.models.config` already ranks the shipped ladder by,
# so one price order governs both. Ties break on the lower input price, then lexicographically
# on `channel:id` — a total order, so a refresh that changes nothing changes no bytes.
RANK_RULE: Final[str] = (
    "min(input_cost_per_1m + output_cost_per_1m); ties -> lower input_cost_per_1m -> "
    "lexicographic channel:id"
)

# Every quote the sheet carries names one of these. `native` is declared and unpopulated: no
# native provider publishes a machine-readable price endpoint, and this sheet admits only
# prices a script fetched (see price_channels.yaml).
CHANNELS: Final[tuple[str, ...]] = ("openrouter", "requesty", "huggingface", "native")


@dataclass(frozen=True)
class Quote:
    """One channel's current listed price for one model, per 1M tokens."""

    channel: str
    listing_id: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    source: str
    as_of: str

    @property
    def rank_key(self) -> tuple[float, float, str]:
        return (
            self.input_cost_per_1m + self.output_cost_per_1m,
            self.input_cost_per_1m,
            f"{self.channel}:{self.listing_id}",
        )


@dataclass(frozen=True)
class PriceSheet:
    """A dated snapshot of the cheapest listed price per model, plus every quote behind it."""

    as_of: str
    digest: str
    canonical: dict[str, Quote]
    quotes: dict[str, tuple[Quote, ...]]

    def price(self, model: str) -> Quote | None:
        """This model's canonical (cheapest-today) quote, or None when the sheet prices it not."""
        return self.canonical.get(model)


def _quote(raw: dict[str, Any]) -> Quote:
    return Quote(
        channel=str(raw["channel"]),
        listing_id=str(raw["id"]),
        input_cost_per_1m=float(raw["input_cost_per_1m"]),
        output_cost_per_1m=float(raw["output_cost_per_1m"]),
        source=str(raw["source"]),
        as_of=str(raw["as_of"]),
    )


def sheet_digest(path: Path | None = None) -> str:
    """16-hex content fingerprint of the sheet — the `data_digest` shape figures already use."""
    target = SHEET_PATH if path is None else path
    if not target.exists():
        return ""
    return hashlib.sha256(target.read_bytes()).hexdigest()[:16]


@functools.lru_cache(maxsize=4)
def load_sheet(path: Path | None = None) -> PriceSheet:
    """Read the committed price sheet. An absent sheet prices nothing; it is not an error."""
    # Absent is a legitimate state, not a crash: a checkout that has never run the refresh
    # still has to draw its figures, and it draws them on recorded cost with the basis said
    # out loud (`basis()`), which is exactly what the sheet's absence means.
    target = SHEET_PATH if path is None else path
    if not target.exists():
        return PriceSheet(as_of="", digest="", canonical={}, quotes={})
    payload = json.loads(target.read_text())
    canonical: dict[str, Quote] = {}
    quotes: dict[str, tuple[Quote, ...]] = {}
    for model, entry in sorted(payload.get("models", {}).items()):
        quotes[model] = tuple(_quote(q) for q in entry.get("quotes", []))
        chosen = entry.get("canonical")
        if chosen is not None:
            canonical[model] = _quote(chosen)
    return PriceSheet(
        as_of=str(payload.get("as_of", "")),
        digest=sheet_digest(target),
        canonical=canonical,
        quotes=quotes,
    )


def naive_cost(model: str, in_tok: float, out_tok: float) -> float | None:
    """`in_tok * input + out_tok * output` at today's cheapest price, or None if unpriced."""
    quote = load_sheet().price(model)
    if quote is None:
        return None
    return (
        in_tok / 1_000_000 * quote.input_cost_per_1m
        + out_tok / 1_000_000 * quote.output_cost_per_1m
    )


def row_naive_cost(row: dict) -> float | None:
    """One results.csv row repriced. None means the sheet does not price this row's model."""
    model = str(row.get("model", ""))
    if not model:
        return None
    in_tok = float(row.get("in_tok") or 0.0)
    out_tok = float(row.get("out_tok") or 0.0)
    return naive_cost(model, in_tok, out_tok)


def total_naive_cost(rows: list[dict], consumer: str) -> float | None:
    """Repriced total over `rows`, or None when ANY row is unpriced — never a partial sum."""
    # All-or-nothing on purpose. A total that quietly skipped its unpriced rows would be a
    # smaller number wearing the same axis label, and the strategies that use an unpriced model
    # are exactly the ones it would flatter.
    try:
        priced = validate.require_measured(
            [row_naive_cost(r) for r in rows], "repriced_cost", consumer
        )
    except validate.DataIntegrityError:
        return None
    return sum(priced)


def axis_basis(*, repriced: bool) -> str:
    """The one line a naive cost axis must say about WHICH prices drew it."""
    # A cost axis without this is a dollar amount with no date on it, which is the whole
    # failure: two cells measured a month apart are plotted side by side as if one price level
    # produced both.
    if not repriced:
        return "cost as billed when each run happened"
    return f"cost repriced at the cheapest listed price as of {load_sheet().as_of}"


def provenance_stamp() -> dict[str, str] | None:
    """The sheet's identity for `figures.json`, or None when no sheet priced the figure."""
    sheet = load_sheet()
    if not sheet.as_of:
        return None
    return {"as_of": sheet.as_of, "digest": sheet.digest, "rank_rule": RANK_RULE}
