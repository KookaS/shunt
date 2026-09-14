"""The OBSERVED per-row channel rule, shared by the CSV writer and the backfill script.

The listing `billing` field is the ENTITLEMENT (what a channel is); `channel` is what a row's
own committed evidence says happened. Keeping one implementation means a freshly written row and
a backfilled historical row cannot disagree, and `channel_source` names the exact rule that fired
so a reader can audit the call. See `docs/benchmark-design.md`.
"""

from __future__ import annotations

from typing import Final

from benchmark.routing import validate

PAID: Final[str] = "paid"
FREE: Final[str] = "free"

# `channel_source` vocabulary — one value per rule, in precedence order.
SOURCE_REAL_COST: Final[str] = "real_cost"
SOURCE_UNOBSERVED: Final[str] = "unobserved"
SOURCE_FREE_WINDOW: Final[str] = "declared_free_window"
SOURCE_FREE_FILE: Final[str] = "results_free_file"
SOURCE_BILLING_FREE: Final[str] = "overlay_billing_free"
SOURCE_NONE: Final[str] = "no_evidence"

SOURCES: Final[tuple[str, ...]] = (
    SOURCE_REAL_COST,
    SOURCE_UNOBSERVED,
    SOURCE_FREE_WINDOW,
    SOURCE_FREE_FILE,
    SOURCE_BILLING_FREE,
    SOURCE_NONE,
)


def listing_billing(model: str, pricing: dict | None = None) -> str | None:
    """The listing's declared billing entitlement (`free`/`paid`), or None when undeclared."""
    return validate.declared_billing(model, pricing)


def observed_channel(
    lane: str,
    real_cost: float,
    calls: int,
    computed_at: str,
    *,
    in_free_file: bool = False,
    pricing: dict | None = None,
) -> tuple[str, str]:
    """The observed `(channel, channel_source)` for one row, by the pre-registered rule.

    Precedence: a positive real cost is PAID; zero calls is UNOBSERVED (blank); a declared free
    window, the free corpus file, or a `billing: free` listing make it FREE; anything else is
    blank with `no_evidence`. There is no longer an `-explabs` suffix fallback: the shipped
    `explabs` provider means `declared_billing` resolves every `-explabs` lane through the
    collection synthesizer, so a suffix branch could only fire on a lane whose listing no
    longer resolves — and a blank reading is the honest answer there, not a free claim.
    """
    if real_cost > 0:
        return PAID, SOURCE_REAL_COST
    if calls == 0:
        return "", SOURCE_UNOBSERVED
    if validate.in_free_window(lane, computed_at):
        return FREE, SOURCE_FREE_WINDOW
    if in_free_file:
        return FREE, SOURCE_FREE_FILE
    if listing_billing(lane, pricing) == FREE:
        return FREE, SOURCE_BILLING_FREE
    return "", SOURCE_NONE
