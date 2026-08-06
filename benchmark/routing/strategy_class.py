"""Which strategies can run live, which are bounds, which are controls."""

# The `live` set is DERIVED from the product's own `LIVE_STRATEGIES`, never restated
# here. A benchmark keeping its own idea of deployable is how a strategy the router
# rejects at boot ends up drawn as a Pareto-optimal deployable point.

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final

from shunt.router.policy import LIVE_STRATEGIES


class StrategyClass(Enum):
    """A strategy's epistemic role — what a reader may conclude from its numbers."""

    LIVE = "live"
    BOUND = "bound"
    CONTROL = "control"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Classification:
    """Why a strategy is not live, and (for BLOCKED) what would make it live."""

    cls: StrategyClass
    reason: str
    path_to_live: str | None = None


# Benchmark code reports by display name (`Strategy.name`); the product configures by
# id (`router.strategy`). Every cross-check between the two has to go through this map,
# so it is the only place the two vocabularies meet.
DISPLAY_TO_ID: Final[Mapping[str, str]] = MappingProxyType(
    {
        "kNN": "knn",
        "Always-Cheap": "always_cheap",
        "Always-Frontier": "always_frontier",
        "Oracle": "oracle",
        "Oracle-reward": "oracle_reward",
        "Random": "random",
        "kNN-cascade": "knn_cascade",
        "Price-Cascade": "price_cascade",
        "Tier-Classifier": "tier_classifier",
    }
)

_CASCADE_BLOCKER: Final[str] = (
    "needs a verified outcome per attempt mid-session, which is more than one decision "
    "per session and breaks cache-safety"
)
# UNMEASURED AND CAPPED — not merely "a mechanism exists". The live ladder is the
# cache-safe analogue (one decision per session, reading only closed prior sessions), but
# it cannot reach the cascade's operating point, so quoting the cascade's numbers as a
# preview of what escalation would buy is wrong by roughly a factor of five.
_CASCADE_PATH: Final[str] = (
    "the live escalation ladder (src/shunt/router/escalation.py, effort_then_rank) is the "
    "cache-safe analogue — one decision per session, over closed prior sessions only — but "
    "it escalates base_pick + 1 rung and then resets its ladder (escalation.py:378-383), so "
    "it CANNOT walk up to the frontier and cannot reach the cascade's operating point; it "
    "also ships disabled and no benchmark strategy models it, so this path is unmeasured"
)

# Everything that is not live, and why. A name absent from both this table and
# LIVE_STRATEGIES is an error, not a default — see `classify`.
_NON_LIVE: Final[Mapping[str, Classification]] = MappingProxyType(
    {
        "Oracle": Classification(
            StrategyClass.BOUND,
            "reads the query task's own verified pass/fail (strategies/oracle.py:22)",
        ),
        "Oracle-reward": Classification(
            StrategyClass.BOUND,
            "reads the query task's own verified pass/fail (strategies/oracle.py:54)",
        ),
        "Arm-oracle": Classification(
            StrategyClass.BOUND,
            "picks the best REALIZED (model, arm) for the query task (report.py:315)",
        ),
        "Random": Classification(
            StrategyClass.CONTROL,
            "draws its candidate set from the query task's own results row "
            "(strategies/fixed.py:91); a live random router would sample the model pool",
        ),
        "Arm-bandit": Classification(
            StrategyClass.BLOCKED,
            "its option set comes from the query task's own sampled (model, arm) rows "
            "(report.py:354), and it is documented as illustrative, not a Strategy",
            "the live exploration layer (src/shunt/router/exploration.py) already "
            "implements cost-aware Thompson sampling over the neighbourhood",
        ),
        "kNN-cascade": Classification(StrategyClass.BLOCKED, _CASCADE_BLOCKER, _CASCADE_PATH),
        "Price-Cascade": Classification(StrategyClass.BLOCKED, _CASCADE_BLOCKER, _CASCADE_PATH),
        "Tier-Classifier": Classification(
            StrategyClass.BLOCKED,
            "its model order comes from an offline-fit capability rank derived from the "
            "outcome matrix (config.py:487), which the live path cannot compute",
            "swap the rank source to the live CapabilityRankResolver "
            "(src/shunt/router/capability_rank.py); nothing else blocks it",
        ),
    }
)


def known_names() -> frozenset[str]:
    """Every display name this module can classify."""
    return frozenset(DISPLAY_TO_ID) | frozenset(_NON_LIVE)


def classify(name: str) -> Classification:
    """The class of one strategy by display name; raises on an unknown name."""
    # Fail closed. Defaulting an unrecognised strategy to "deployable" is exactly the bug
    # this module exists to remove, so a new strategy must be classified to be plotted.
    if DISPLAY_TO_ID.get(name) in LIVE_STRATEGIES:
        return Classification(StrategyClass.LIVE, "runs in inference today")
    found = _NON_LIVE.get(name)
    if found is None:
        raise KeyError(
            f"strategy {name!r} has no class; add it to benchmark/routing/strategy_class.py "
            f"(known: {', '.join(sorted(known_names()))})"
        )
    return found


def is_live(name: str) -> bool:
    """True iff `router.strategy` may name this strategy in production."""
    return classify(name).cls is StrategyClass.LIVE


def blocker(name: str) -> str | None:
    """Why this strategy cannot run live, or None when it can."""
    found = classify(name)
    return None if found.cls is StrategyClass.LIVE else found.reason


def live_names(names: Iterable[str]) -> list[str]:
    """The subset of `names` that can run live, order preserved."""
    return [n for n in names if is_live(n)]


def live_rows(
    rows: Sequence[Mapping[str, object]], key: str = "strategy"
) -> list[Mapping[str, object]]:
    """The subset of summary rows whose strategy can run live, order preserved."""
    return [r for r in rows if is_live(str(r[key]))]
