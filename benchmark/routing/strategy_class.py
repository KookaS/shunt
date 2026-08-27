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
    # The config surface the measured MECHANISM already runs under in production, or None
    # when nothing equivalent ships. BLOCKED covers two situations that a reader must not
    # be shown as one: a strategy that cannot be run at all, and one whose mechanism runs
    # in every default install while only its NAME is unselectable. Reporting the second as
    # "not deployable" is the mirror image of reporting a blocked strategy as deployable —
    # the defect this module exists to prevent, pointed the other way. Kept as a FIELD
    # rather than a fifth StrategyClass on purpose: every consumer switches on `cls`, and a
    # new member would fall through cost_quality_frontier's `_encode` to the live-circle
    # default, silently drawing an undeployable strategy as deployable. A field cannot.
    shipped_as: str | None = None


# Benchmark code reports by display name (`Strategy.name`); the product configures by
# id (`router.strategy`). Every cross-check between the two has to go through this map,
# so it is the only place the two vocabularies meet. The pre-rename ids `knn`/`knn_cascade`
# are handled by the product's own migration aliases, not restated here.
DISPLAY_TO_ID: Final[Mapping[str, str]] = MappingProxyType(
    {
        "kNN-semantic": "knn_semantic",
        "kNN-semantic-cascade": "knn_semantic_cascade",
        "kNN-difficulty": "knn_difficulty",
        "kNN-difficulty-cascade": "knn_difficulty_cascade",
        "Difficulty-Band-cascade": "difficulty_band_cascade",
        "Always-Cheap": "always_cheap",
        "Always-Frontier": "always_frontier",
        "Oracle": "oracle",
        "Oracle-reward": "oracle_reward",
        "Random": "random",
        "kNN-semantic-cascade (within-task)": "knn_semantic_cascade_withintask",
        "Price-Cascade": "price_cascade",
        "Session-Cascade": "session_cascade",
        "kNN-semantic-tier": "knn_semantic_tier",
    }
)

_CASCADE_BLOCKER: Final[str] = (
    "needs a verified outcome per attempt mid-session, which is more than one decision "
    "per session and breaks cache-safety"
)
# TAKEN, AND MEASURED — this is no longer an assertion about an unbuilt mechanism. The route
# is `router.strategy: session_cascade`, a live id since 2026-08-21, and Session-Cascade is
# what it costs. State the SHORTFALL with the route, so nobody reads "there is a path" as
# "the blocked row's number is available": paying one decision per session instead of one per
# attempt costs +$1.37 [+0.82, +2.00] paired against Price-Cascade, and is not distinguishable
# from kNN-semantic-cascade (within-task) (-$1.23 [-3.42, +0.73]). That is the price of the
# cache-safety spine, and it is the whole of what these two rows now measure.
_CASCADE_PATH: Final[str] = (
    "TAKEN: the same ladder, paced one decision per session, is selectable as "
    "`router.strategy: session_cascade` (src/shunt/router/policy.py LIVE_STRATEGIES) — "
    "always_cheap plus escalation.enabled, over closed prior sessions only, walking to the "
    "frontier because the engine's per-task rank floor persists the climbed rung across "
    "sessions (engine.py `_lift_to_rank_floor`). Session-Cascade "
    "(strategies/session_cascade.py) measures what that cadence costs: +$1.37 paired against "
    "Price-Cascade, indistinguishable from kNN-semantic-cascade (within-task). What stays "
    "blocked is "
    "the WITHIN-TASK cadence itself, and permanently: it is excluded by design, not "
    "pending. These two rows "
    "are kept as the comparator that prices session cadence, not as unbuilt work"
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
        "kNN-semantic": Classification(
            StrategyClass.CONTROL,
            "the selection rule with the escalation ladder removed. No `router.strategy` value "
            "produces it; it is kept as the contrast that isolates what the ladder buys",
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
        "kNN-semantic-cascade (within-task)": Classification(
            StrategyClass.BLOCKED, _CASCADE_BLOCKER, _CASCADE_PATH
        ),
        "Price-Cascade": Classification(StrategyClass.BLOCKED, _CASCADE_BLOCKER, _CASCADE_PATH),
        "kNN-semantic-tier": Classification(
            StrategyClass.BLOCKED,
            "its model order comes from an offline-fit capability rank derived from the "
            "outcome matrix (config.py:487), which the live path cannot compute",
            "swap the rank source to the live CapabilityRankResolver "
            "(src/shunt/router/capability_rank.py); nothing else blocks it",
        ),
        "kNN-difficulty": Classification(
            StrategyClass.CONTROL,
            "the judge-difficulty selection rule with the escalation ladder removed. No "
            "`router.strategy` value produces it; it is kept as the contrast that isolates "
            "what the ladder buys on the difficulty axis",
        ),
        "kNN-difficulty-cascade": Classification(
            StrategyClass.BLOCKED,
            "needs a judge call per task at inference: a judge provider + key, a task-boundary "
            "difficulty label, and a difficulty index over labelled history — none of which "
            "the live path has",
            "docs/routing.md §difficulty-routing lists exactly what to add (judge config, "
            "per-task label at the task boundary, difficulty index build, cold-start fallback "
            "to session_cascade); nothing else blocks it",
        ),
        "Difficulty-Band-cascade": Classification(
            StrategyClass.BLOCKED,
            "the same judge dependency as kNN-difficulty-cascade — a per-task difficulty "
            "label is required before the band rule can fire",
            "docs/routing.md §difficulty-routing; identical path to kNN-difficulty-cascade",
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


def shipped_mechanism(name: str) -> str | None:
    """The config surface a non-live row's mechanism already runs under, or None."""
    # `is_live` answers "may router.strategy name it"; this answers the different question
    # "does the thing it measures run today". Session-Cascade used to be the bearer of that
    # split — False on the first, not-None on the second — and a figure conflating them told
    # readers a shipped, default-on mechanism could not be run. It is now LIVE on both counts
    # (`router.strategy: session_cascade`), so NO row sets `shipped_as` today and this returns
    # None everywhere. The seam is kept, not deleted: the defect it guards is a class, not one
    # row, and the next mechanism that ships under a config surface before it earns a strategy
    # id needs it. If that never happens, delete the field, this function and the hollow-marker
    # branch in cost_quality_frontier together — not one of the three.
    return classify(name).shipped_as


def live_names(names: Iterable[str]) -> list[str]:
    """The subset of `names` that can run live, order preserved."""
    return [n for n in names if is_live(n)]


def live_rows(
    rows: Sequence[Mapping[str, object]], key: str = "strategy"
) -> list[Mapping[str, object]]:
    """The subset of summary rows whose strategy can run live, order preserved."""
    return [r for r in rows if is_live(str(r[key]))]
