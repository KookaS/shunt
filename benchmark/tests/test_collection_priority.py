"""Collection priority and the data-driven stop rule — pure, synthetic, no model calls.

Every case builds a `CollectionPriority` over a tiny hand-made corpus, so the four public
functions are exercised without touching the committed results or any provider.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from benchmark.routing import collection_priority as cp

AS_OF = date(2026, 9, 11)


def _engine(
    channels: list[cp.Channel],
    *,
    covered: dict[str, list[str]] | None = None,
    passes: dict[str, list[bool]] | None = None,
    costs: dict[str, float] | None = None,
    importance: dict[str, float] | None = None,
    not_worth: dict[str, str] | None = None,
    vision: set[str] | None = None,
    withdrawn: set[str] | None = None,
    eligible: Callable[[str], bool] | None = None,
    total: int = 10,
    weights: cp.PriorityWeights | None = None,
) -> cp.CollectionPriority:
    return cp.CollectionPriority(
        channels={c.channel: c for c in channels},
        covered={k: frozenset(v) for k, v in (covered or {}).items()},
        verified_total=total,
        base_importance=importance or {},
        passes=passes or {},
        real_cost_per_cell=costs or {},
        not_worth=not_worth or {},
        vision=frozenset(vision or ()),
        withdrawn=frozenset(withdrawn or ()),
        weights=weights or cp.PriorityWeights(),
        multimodal_eligible=eligible or (lambda _model: False),
        as_of=AS_OF,
    )


def test_frontier_beats_obscure() -> None:
    """An explicitly-important model outranks an unnamed one of equal cost and coverage."""
    engine = _engine(
        [cp.Channel("front", "kimi-k3"), cp.Channel("obscure", "some-unknown-model")],
        importance={"kimi-k3": 10.0},
    )

    assert engine.priority("front") > engine.priority("obscure") > 0.0
    assert engine.importance("front") == 10.0
    assert engine.importance("obscure") == cp.DEFAULT_IMPORTANCE


def test_a_dominated_model_is_stopped_but_an_improving_one_is_not() -> None:
    """After N cells a shadowed lane is retired; a strong one is kept."""
    front = cp.Channel("front", "front")
    dom = cp.Channel("dom", "dom")
    improving = cp.Channel("improving", "improving")
    engine = _engine(
        [front, dom, improving],
        covered={"dom": [f"c{i}" for i in range(10)], "front": [f"c{i}" for i in range(10)]},
        passes={
            "front": [True] * 20,
            "dom": [True] * 4 + [False] * 16,
            "improving": [True] * 18 + [False] * 2,
        },
        costs={"front": 0.1, "dom": 0.2, "improving": 0.2},
        importance={"front": 9.0, "dom": 1.0, "improving": 1.0},
    )

    keep_dom, reason_dom = engine.worth_collecting("dom")
    assert keep_dom is False
    assert "dominated by front" in reason_dom
    assert "no new Verified challenge coverage" in reason_dom

    keep_imp, _reason_imp = engine.worth_collecting("improving")
    assert keep_imp is True


def test_below_the_stop_floor_the_domination_rule_does_not_fire() -> None:
    """Fewer than N completed cells keeps a lane alive even when it looks weak."""
    engine = _engine(
        [cp.Channel("front", "front"), cp.Channel("dom", "dom")],
        passes={"front": [True] * 20, "dom": [False] * 5},
        importance={"front": 9.0, "dom": 1.0},
    )

    keep, reason = engine.worth_collecting("dom")
    assert keep is True
    assert "stop floor" in reason


def test_a_thin_dominator_never_retires_a_measured_lane() -> None:
    """A high-priority lane below the evidence floor does not dominate anyone."""
    engine = _engine(
        [cp.Channel("thin", "thin"), cp.Channel("measured", "measured")],
        passes={"thin": [True] * 5, "measured": [False] * 20},
        importance={"thin": 9.0, "measured": 1.0},
    )

    keep, _reason = engine.worth_collecting("measured")
    assert keep is True


def test_the_credited_channel_beats_its_duplicates() -> None:
    """One identity served by two channels credits the cheaper one; the other is a duplicate."""
    engine = _engine(
        [
            cp.Channel("alpha", "shared", list_cost_per_1m=0.1),
            cp.Channel("beta", "shared", list_cost_per_1m=0.2),
        ],
        importance={"shared": 5.0},
    )

    assert engine.duplicate_of("beta") == "alpha"
    assert engine.duplicate_of("alpha") is None
    assert engine.priority("alpha") > 0.0
    assert engine.priority("beta") == 0.0


def test_pure_text_with_only_multimodal_left_retires() -> None:
    """Text first: below the gate -> text; text complete + vision -> multimodal; else nothing."""
    below_gate = _engine(
        [cp.Channel("t", "t")],
        importance={"t": 1.0},
        eligible=lambda _m: False,
    )
    assert below_gate.runnable_benchmarks("t") == [cp.TEXT_BENCHMARK]

    complete_text = _engine(
        [cp.Channel("t", "t")],
        importance={"t": 1.0},
        eligible=lambda _m: True,
    )
    assert complete_text.runnable_benchmarks("t") == []

    vision = _engine(
        [cp.Channel("t", "t")],
        importance={"t": 1.0},
        vision={"t"},
        eligible=lambda _m: True,
    )
    assert vision.runnable_benchmarks("t") == [cp.MULTIMODAL_BENCHMARK]


def test_release_date_is_only_a_bounded_tiebreak() -> None:
    """An old model freshly listed cannot outrank a frontier model, but breaks equal ties."""
    old = cp.Channel("front", "front", release_date="2024-01-01")
    fresh_obscure = cp.Channel("fresh", "fresh", release_date="2026-09-01")
    engine = _engine([old, fresh_obscure], importance={"front": 10.0})

    assert engine.priority("front") > engine.priority("fresh")
    # Equal importance: the recent listing wins the tie-break, and by no more than the bonus.
    tie = _engine(
        [
            cp.Channel("old", "old", release_date="2024-01-01"),
            cp.Channel("recent", "recent", release_date="2026-09-01"),
        ],
        importance={"old": 1.0, "recent": 1.0},
    )
    assert tie.priority("recent") > tie.priority("old")
    assert tie.priority("recent") <= tie.priority("old") * (1.0 + cp.DEFAULT_RECENCY_BONUS)


def test_the_denylist_stops_a_model_with_its_reason() -> None:
    engine = _engine(
        [cp.Channel("gpt-5-mini", "gpt-5-mini")],
        not_worth={"gpt-5-mini": "dominated; adds no coverage"},
    )

    keep, reason = engine.worth_collecting("gpt-5-mini")
    assert keep is False
    assert "dominated; adds no coverage" in reason


def test_unknown_identity_resolves_to_itself_without_error() -> None:
    engine = _engine([cp.Channel("known", "known")], importance={"known": 1.0})

    assert engine.duplicate_of("mystery") is None
    assert engine.priority("mystery") > 0.0


def test_a_withdrawn_channel_is_not_scheduled() -> None:
    """A withdrawn listing quiesces even though the overlay retains its row."""
    engine = _engine(
        [cp.Channel("gone", "gone")],
        importance={"gone": 9.0},
        withdrawn={"gone"},
    )

    keep, reason = engine.worth_collecting("gone")
    assert keep is False
    assert "withdrawn" in reason


def test_withdrawal_is_per_channel_not_per_identity() -> None:
    """One withdrawn serving channel does not retire a sibling channel of the same identity."""
    engine = _engine(
        [
            cp.Channel("gone-a", "shared"),
            cp.Channel("live-b", "shared"),
        ],
        importance={"shared": 5.0},
        withdrawn={"gone-a"},
    )
    # The withdrawn channel is out; the live sibling is not shadowed by it.
    assert engine.worth_collecting("gone-a")[0] is False
    assert "withdrawn" in engine.worth_collecting("gone-a")[1]
    assert engine.duplicate_of("live-b") is None
    assert engine.worth_collecting("live-b")[0] is True
