"""The alpha cost model: what an escalation costs once the next model is handed the context."""

# WHY THIS IS NOT `cache_cost.py`. That module prices ONE model re-serving a prefix it has
# already seen — a cache HIT, discounted at `cache_read_cost_per_1m`. This module prices the
# opposite event: a DIFFERENT model receiving a prefix it has never seen. That is a cache MISS by
# construction (new model, new prefix), so the carried tokens are billed at the full input rate
# and never at the cache-read rate. The two modules share exactly one thing, `config.load_pricing`,
# and must not be merged: `cache_aware_total` is consumed by the kill gate, and folding a
# surcharge into it would move a pre-registered verdict.
#
# WHAT THE OFFLINE BENCHMARK MEASURES vs WHAT INFERENCE RUNS. Offline, each rung of a cascade
# starts from a fresh tree and a fresh context, so the replayed cost is the alpha = 0 end. Live,
# the router never rewrites `messages`, so the escalated model is resent the whole prior
# conversation by the CLI, uncached — the alpha = 1.0 end. Everything between is a share of the
# prior context carried forward. NOTHING HERE IS A MEASUREMENT OF LIVE COST: it is a cost model
# over measured tokens and real registry prices, and it asserts no pass rate.
#
# THE FORMULA.
#
#     C(alpha) = sum_i [ alpha * t_{i-1} * p_i.input + billed_i ]
#
# `billed_i` is the attempt's own MEASURED cost, not a token-priced reconstruction of it. That is
# the one deliberate departure from a pure token formula, and it buys the identity that makes the
# whole model auditable: C(0) is the published `TotalCost` exactly, so the surcharge is the only
# thing alpha can move. Reconstructing `billed_i` from tokens instead would make C(0) a fourth
# cost estimate that silently disagrees with the three already published.
#
# `t_{i-1}` is the context the PREVIOUS attempt ended holding, and `t_{-1}` is zero — the first
# attempt of a task carries nothing.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from benchmark import config
from benchmark.routing.strategies import BilledAttempt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Alphas the published bracket ticks at. THREE, not two, and the middle pair is a BAND rather
# than a tick: `context_transfer: summary` hands the next model a summarised prefix, and how far
# a summariser compresses is not a constant — it moves with the conversation and with the
# summariser. One tick at 0.1 asserted a precision the model does not have. 0.1 and 0.3 bracket
# that uncertainty; 1.0 is `context_transfer: full`, which is exact because nothing is dropped.
BRACKET_ALPHAS: Final[tuple[float, float, float]] = (0.1, 0.3, 1.0)
_PER_TOKEN: Final[float] = 1e-6


class TokenIncompleteError(RuntimeError):
    """Raised when a context cost is asked for over an attempt with no measured tokens."""


class EmptyBracketError(RuntimeError):
    """Raised when a surcharge ratio is asked for over a bracket that rests on no tasks."""


def estimated_final_context(in_tok: int, calls: int) -> float | None:
    """Tokens the context held at the end of a task, from its measured input total and calls."""
    # DERIVATION. An agentic task issues `calls` requests, and each one resends the conversation
    # so far, so the prefix grows from roughly nothing to its final size `t`. Under LINEAR growth
    # the sizes are t/calls, 2t/calls, ..., t, whose sum is the measured `in_tok`:
    #     in_tok = t * (calls + 1) / 2  ~=  t * calls / 2   for calls >> 1
    # hence t ~= 2 * in_tok / calls. The approximation is the whole assumption, it is one-sided in
    # an unknown direction (tool output and file reads do not arrive at a constant rate), and it is
    # published as a limitation on every figure that draws it.
    #
    # `calls <= 0` is not a degenerate zero, it is the ABSENCE of a measurement — 15 rows of
    # results.csv carry `in_tok == 0 and calls == 0`, and an imputed cell carries no tokens at all.
    # Returning None makes that absence propagate to a raise rather than to a plausible 0.0.
    if calls <= 0:
        return None
    return 2.0 * float(in_tok) / float(calls)


def is_token_complete(attempt: BilledAttempt) -> bool:
    """Whether this attempt's cell carries the measured tokens the model needs."""
    return attempt.in_tok > 0 and attempt.calls > 0


def token_complete_tasks(attempts: Mapping[str, Sequence[BilledAttempt]]) -> set[str]:
    """The task ids whose EVERY realized attempt has a measured, token-bearing cell."""
    # PATH-dependent, not coverage-dependent, and the distinction is load-bearing. A task needs
    # measured tokens only on the models its realized path actually BILLED — one to three of six
    # for a cascade — so this is NOT the set of tasks measured on every model, and it is far
    # larger than that set. It is recomputed per strategy, because two strategies walk different
    # cells over the same corpus.
    return {tid for tid, seq in attempts.items() if seq and all(is_token_complete(a) for a in seq)}


def input_rates(models: Sequence[str] | None = None) -> dict[str, float]:
    """Per-model input price in dollars per TOKEN, from the registry's per-1M list price."""
    pricing = config.load_pricing()
    names = sorted(pricing) if models is None else models
    rates: dict[str, float] = {}
    for name in names:
        info = pricing.get(name)
        if isinstance(info, dict):
            rates[name] = float(info.get("input_cost_per_1m") or 0.0) * _PER_TOKEN
    return rates


def context_cost_total(
    attempts: Sequence[BilledAttempt],
    alpha: float,
    rates: Mapping[str, float],
) -> float:
    """One task's cost once `alpha` of each attempt's ending context is carried to the next."""
    # RAISE, DO NOT DROP. A token-less attempt has no estimable context, and quietly charging it
    # zero surcharge would publish a bracket that is narrower than the truth precisely on the
    # cells nobody measured — the failure mode of every imputation that reports as a measurement.
    # Callers select the token-complete subset FIRST (`token_complete_tasks`) and are told how
    # many tasks that left.
    for attempt in attempts:
        if not is_token_complete(attempt):
            raise TokenIncompleteError(
                f"attempt on {attempt.model!r} carries no measured tokens "
                f"(in_tok={attempt.in_tok}, calls={attempt.calls}); the context cost model "
                "refuses to impute a context size"
            )
    total = 0.0
    carried = 0.0
    for attempt in attempts:
        total += attempt.cost + alpha * carried * rates.get(attempt.model, 0.0)
        carried = estimated_final_context(attempt.in_tok, attempt.calls) or 0.0
    return total


@dataclass(frozen=True)
class ContextBracket:
    """A cost model's totals at each alpha, and the task subset they were computed on."""

    alphas: tuple[float, ...]
    totals: tuple[float, ...]
    baseline: float
    tasks: tuple[str, ...]

    @property
    def n_tasks(self) -> int:
        """How many tasks the totals rest on — published beside every bracket number."""
        return len(self.tasks)

    @property
    def publishable(self) -> bool:
        """Whether a surcharge factor exists at all — false on an empty or zero-baseline subset."""
        return self.n_tasks > 0 and self.baseline > 0

    def ratio(self, alpha: float) -> float:
        """Surcharge factor at `alpha`. Raises when there is no subset to compute it on."""
        # RAISE, DO NOT RETURN 1.0. The old default was a fabricated measurement: a corpus with
        # no token columns yields an EMPTY subset, and 1.0 publishes
        # `context_cost_alpha_* == cache_total` — the affirmative claim "carrying context costs
        # nothing" — beside `context_cost_n: 0`, with every coverage gate green. Absence of a
        # measurement is not a measurement of zero, so the only honest options are to omit the
        # columns (what `summary._context_columns` does, guarded on `publishable`) or to fail
        # loudly here. Both are wired, so a new caller cannot re-introduce the fabrication by
        # forgetting the guard.
        if not self.publishable:
            raise EmptyBracketError(
                f"context-cost bracket has no publishable ratio: n_tasks={self.n_tasks}, "
                f"baseline={self.baseline}. No token-complete task entered it, so there is no "
                "surcharge factor to quote — omit the columns rather than defaulting them."
            )
        return self.totals[self.alphas.index(alpha)] / self.baseline


def context_bracket(
    attempts: Mapping[str, Sequence[BilledAttempt]],
    alpha: float | Sequence[float] = BRACKET_ALPHAS,
    rates: Mapping[str, float] | None = None,
) -> ContextBracket:
    """Total cost at each alpha over the token-complete subset of `attempts`."""
    # `alpha` takes a sequence so a future ladder can tick at more than two points without
    # re-deriving the walk; a bare float is the one-tick case.
    alphas = (float(alpha),) if isinstance(alpha, int | float) else tuple(float(a) for a in alpha)
    priced = input_rates() if rates is None else rates
    keep = sorted(token_complete_tasks(attempts))
    totals = tuple(
        sum(context_cost_total(attempts[tid], a, priced) for tid in keep) for a in alphas
    )
    baseline = sum(context_cost_total(attempts[tid], 0.0, priced) for tid in keep)
    return ContextBracket(alphas=alphas, totals=totals, baseline=baseline, tasks=tuple(keep))
