"""Pricing and escalation-target helpers shared by the cascade strategies.

Price-Cascade is the like-for-like zero-ML baseline for kNN-semantic-cascade, so the two must
escalate to the SAME model or their rows are not comparable. One helper, one semantics.
"""

from __future__ import annotations

from . import BilledAttempt


def billed_attempt(model: str, outcome: dict) -> BilledAttempt:
    """One attempt's billing record — its cost plus the cell's measured token counts."""
    # An imputed or never-measured cell carries no token columns at all, so these read 0, and
    # `context_cost.token_complete_tasks` then drops the whole task rather than inventing a
    # context size for it. One constructor for every cascade, so the three rows cannot disagree
    # about what a billed attempt records.
    return BilledAttempt(
        model=model,
        cost=float(outcome.get("cost", 0.0)),
        in_tok=int(outcome.get("in_tok") or 0),
        out_tok=int(outcome.get("out_tok") or 0),
        calls=int(outcome.get("calls") or 0),
    )


# Only reachable on a matrix that prices nothing at all — the cheap end of the shipped
# ladder, so a degraded cascade still degrades downwards.
_LAST_RESORT_MODEL = "deepseek-v4-flash"


def model_pricing(matrix: dict) -> dict[str, float]:
    """``{model: total price per 1M tokens}`` for the models the matrix actually prices."""
    return {
        model: meta.get("input_price", 0.0) + meta.get("output_price", 0.0)
        for model, meta in matrix.get("models", {}).items()
        if "input_price" in meta or "output_price" in meta
    }


def measured_models_by_price(matrix: dict) -> list[str]:
    """Models the benchmark priced AND measured, cheapest first (name breaks price ties).

    The coverage bar: a model with no cell anywhere in the matrix has no evidence behind it,
    so no cascade routes to it.
    """
    pricing = model_pricing(matrix)
    measured: set[str] = set()
    for task_results in matrix.get("results", {}).values():
        measured.update(task_results)
    return sorted((m for m in pricing if m in measured), key=lambda m: (pricing[m], m))


def frontier_model(matrix: dict) -> str | None:
    """The escalation target of last resort: the priciest MEASURED model, or ``None``."""
    by_price = measured_models_by_price(matrix)
    return by_price[-1] if by_price else None


def cheapest_priced_model(matrix: dict) -> str:
    """Cheapest priced model — where a cascade degrades when it has no ladder to walk."""
    pricing = model_pricing(matrix)
    return min(pricing, key=lambda m: (pricing[m], m)) if pricing else _LAST_RESORT_MODEL
