"""Join escalation trajectories to the routing benchmark's BILLED cost, or fail loudly."""

# WHY A JOIN AT ALL. The escalation corpus records outcomes, not money: nine StepView fields —
# `real_cost`, `model`, `reasoning_effort`, `rank_index`, `effort_index`, `test_passed`,
# `test_total`, `subgoal_progress`, `replay_rc` — are populated on no step in it, so there is no
# per-step cost to sum and no per-step model to group by. What every trajectory DOES carry is its
# header: `instance_id` is the challenge and `trajectory_id` spells `<instance>__<model>__<effort>`,
# so the arm is on the header. The routing benchmark ran those same (challenge, model, reasoning)
# cells and recorded the provider's own `real_cost` for each, which makes that triple the join key
# and `benchmark/routing/results.csv` the cost source.
#
# WHY IT MUST BE TOTAL. A partial join produces a cost for a subset of sessions while every number
# built on it reads as if it covered the corpus, and nothing in the payload says otherwise. So a
# miss raises instead: `CostJoinError` names the sessions it could not price.

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from benchmark import config
from benchmark.escalation import session_eval
from benchmark.routing import cache_cost

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from benchmark.escalation.schema import Trajectory

# The two currencies a pre-alpha cost claim may be quoted in, and they are NOT interchangeable.
# `naive` sums the provider's billed cost per attempt as if no attempt reused a warm prefix, which
# overstates every arm that repeats one model — the cheap retry above all. `cache_aware` applies
# the shared cache model (`benchmark.routing.cache_cost`), whose discount and input share are
# measured and whose hit rate is assumed. Both are reported; neither is reported alone.
NAIVE: Final[str] = "naive"
CACHE_AWARE: Final[str] = "cache_aware"

_ID_SEGMENTS: Final[int] = 3


class CostJoinError(RuntimeError):
    """A trajectory carried no priced row in the routing benchmark's results."""


@dataclass(frozen=True)
class JoinSummary:
    """How complete the join was — published so the cost numbers carry their own coverage."""

    n_sessions: int
    n_joined: int
    n_rows: int
    n_rows_priced: int

    @property
    def join_rate(self) -> float:
        return 0.0 if self.n_sessions == 0 else self.n_joined / self.n_sessions

    def to_dict(self) -> dict[str, object]:
        return {
            "n_sessions": self.n_sessions,
            "n_joined": self.n_joined,
            "join_rate": round(self.join_rate, 4),
            "n_result_rows": self.n_rows,
            "n_result_rows_with_real_cost": self.n_rows_priced,
        }


def arm_of(traj: Trajectory) -> tuple[str, str, str]:
    """The join key off the HEADER: (challenge_id, model, reasoning)."""
    parts = traj.header.trajectory_id.split("__")
    if len(parts) < _ID_SEGMENTS:
        raise CostJoinError(
            f"trajectory id {traj.header.trajectory_id!r} does not spell "
            f"<instance>__<model>__<effort>, so it carries no model/reasoning to join on"
        )
    challenge = traj.header.instance_id or "__".join(parts[:-2])
    return (challenge, parts[-2], parts[-1])


def _priced_rows(results_csv: Path) -> tuple[dict[tuple[str, str, str], float], int, int]:
    """(challenge, model, reasoning) -> billed real_cost, plus the row counts behind it."""
    if not results_csv.exists():
        raise CostJoinError(f"no routing results at {results_csv}: nothing to price against")
    prices: dict[tuple[str, str, str], float] = {}
    n_rows = 0
    with results_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            raw = row.get("real_cost")
            if not raw:
                continue
            prices[(row["challenge_id"], row["model"], row["reasoning"])] = float(raw)
    return (prices, n_rows, len(prices))


def join(
    trajectories: Sequence[Trajectory], results_csv: Path | None = None
) -> tuple[dict[str, tuple[str, float]], JoinSummary]:
    """Every trajectory's (model, billed cost), or raise naming the ones it could not price."""
    prices, n_rows, n_priced = _priced_rows(
        results_csv if results_csv is not None else config.results_csv_path()
    )
    joined: dict[str, tuple[str, float]] = {}
    missing: list[str] = []
    for traj in trajectories:
        challenge, model, reasoning = arm_of(traj)
        cost = prices.get((challenge, model, reasoning))
        if cost is None:
            missing.append(f"{challenge}/{model}/{reasoning}")
            continue
        joined[traj.header.trajectory_id] = (model, cost)
    if missing:
        head = ", ".join(sorted(missing)[:5])
        more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        raise CostJoinError(
            f"{len(missing)} of {len(trajectories)} escalation sessions have no priced row in "
            f"the routing results: {head}{more}. A partial join would publish a cost number for "
            f"a subset while implying the whole corpus, so nothing is priced until all are."
        )
    return (
        joined,
        JoinSummary(
            n_sessions=len(trajectories),
            n_joined=len(joined),
            n_rows=n_rows,
            n_rows_priced=n_priced,
        ),
    )


def _cache_aware(
    prices: Mapping[str, cache_cost.CachePrice],
) -> Callable[[Sequence[tuple[str, float]]], float]:
    """The cache-aware currency, bound to the per-model prices the corpus's models resolve to."""

    def total(attempts: Sequence[tuple[str, float]]) -> float:
        return cache_cost.cache_aware_total(attempts, prices)

    return total


def session_costs(
    trajectories: Sequence[Trajectory], results_csv: Path | None = None
) -> tuple[session_eval.SessionCosts, JoinSummary]:
    """The billed per-session costs and both currencies, ready for `session_cadence`."""
    joined, summary = join(trajectories, results_csv)
    prices = cache_cost.cache_prices({model for model, _cost in joined.values()})
    return (
        session_eval.SessionCosts(
            session=joined,
            currencies={
                NAIVE: lambda attempts: sum(cost for _model, cost in attempts),
                CACHE_AWARE: _cache_aware(prices),
            },
        ),
        summary,
    )


def cache_price_provenance(
    trajectories: Sequence[Trajectory], results_csv: Path | None = None
) -> dict[str, dict[str, object]]:
    """Per model, whether the cache discount and input share were measured or assumed."""
    joined, _summary = join(trajectories, results_csv)
    prices = cache_cost.cache_prices({model for model, _cost in joined.values()})
    return {
        model: {
            "discount": round(price.discount, 4),
            "discount_provenance": price.provenance,
            "input_share": round(price.input_share, 4),
            "input_share_provenance": price.share_provenance,
            "hit_rate": price.hit_rate,
            "hit_rate_provenance": cache_cost.ASSUMED,
        }
        for model, price in sorted(prices.items())
    }


__all__ = [
    "CACHE_AWARE",
    "NAIVE",
    "CostJoinError",
    "JoinSummary",
    "arm_of",
    "cache_price_provenance",
    "join",
    "session_costs",
]
