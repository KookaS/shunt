"""ladder_rungs.png must derive its policy panel, and must never print an impossible p."""

from __future__ import annotations

from typing import Any

from benchmark.routing.figures import ladder_rungs


def _row(target: str, delta: float, p_value: float, verdict: str) -> dict[str, Any]:
    return {
        "target": target,
        "price_multiple": 3.8,
        "n": 100,
        "helps": 10,
        "hurts": 2,
        "delta": delta,
        "ci95": [delta - 0.05, delta + 0.05],
        "null_ci95": [-0.04, 0.04],
        "p_value": p_value,
        "verdict": verdict,
    }


def test_the_exact_tail_is_never_printed_as_zero() -> None:
    # `mcnemar_exact_p` underflows to 0.0 on the strongest rung. A figure that prints "p = 0"
    # states an impossible thing, so the tail renders as an inequality instead.
    assert ladder_rungs.p_text(0.0).startswith("<")
    assert ladder_rungs.p_text(2.3e-7).startswith("<")
    assert ladder_rungs.p_text(0.5078) == "0.51"


def test_the_visit_path_is_stepped_from_the_shipped_shortlist_not_hardcoded() -> None:
    # Six models (the shipped pool's size): the shortlist walks the cheapest ranks one at a
    # time and then jumps to the top rank, skipping everything between.
    visits, shortlist = ladder_rungs.shipped_walk(6)
    assert shortlist > 0
    assert visits[-1] == 5
    assert visits == tuple(range(1, shortlist)) + (5,)
    # A pool smaller than the shortlist has nothing to jump over — every rank is a rung.
    assert ladder_rungs.shipped_walk(3)[0] == (1, 2)
    # And a pool with only the base model has no rung at all.
    assert ladder_rungs.shipped_walk(1)[0] == ()


def test_a_skipped_rung_is_tagged_from_the_walk_and_named_in_the_annotations() -> None:
    payload = {
        "base_model": "cheap-base",
        "targets": [
            _row("t1", 0.03, 0.51, "INDISTINGUISHABLE"),
            _row("t2", -0.17, 0.0, "NET-HARMFUL"),
            _row("t3", -0.02, 0.81, "INDISTINGUISHABLE"),
            _row("t4", 0.15, 0.001, "NET-HELPFUL"),
            _row("t5", 0.24, 0.0, "NET-HELPFUL"),
        ],
    }
    rows = ladder_rungs.rungs(payload)
    visited = [r.target for r in rows if r.visited]
    assert visited == ["t1", "t2", "t5"]
    ann = ladder_rungs._annotations(rows, payload, 3)
    # The cheapest net-helpful target the shortlist steps over must be NAMED, not left for the
    # reader to work out from the markers — that is the whole finding.
    assert any("t4" in note and "jumps over" in note for note in ann.notes)
    assert any("skipped: t3" in fact and "t4" in fact for fact in ann.subtitle_facts)
    assert dict(ann.counts)["visited_rungs"] == 3
