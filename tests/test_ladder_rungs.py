"""ladder_rungs.png must derive its policy panel, and must never print an impossible p."""

from __future__ import annotations

from typing import Any

from benchmark.routing._live_pool import packaged_live_pool
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
    rows = ladder_rungs.rungs(payload, live_pool=["cheap-base", "t1", "t2", "t3", "t4", "t5"])
    visited = [r.target for r in rows if r.visited]
    assert visited == ["t1", "t2", "t5"]
    assert all(r.live for r in rows)
    ann = ladder_rungs._annotations(rows, payload, 3)
    # The cheapest net-helpful target the shortlist steps over must be NAMED, not left for the
    # reader to work out from the markers — that is the whole finding.
    assert any("t4" in note and "jumps over" in note for note in ann.notes)
    assert any("skipped: t3" in fact and "t4" in fact for fact in ann.subtitle_facts)
    assert dict(ann.counts)["visited_rungs"] == 3


def test_a_benchmark_target_outside_the_live_pool_is_not_live_not_skipped() -> None:
    payload = {
        "base_model": "cheap-base",
        "targets": [
            _row("t1", 0.03, 0.51, "INDISTINGUISHABLE"),
            _row("t2", 0.24, 0.0, "NET-HELPFUL"),
        ],
    }
    # t1 is not in the live pool: it must be tagged NOT-LIVE (measured, never served) and must
    # not be counted as visited or skipped — the ladder cannot reach a model the router does
    # not route to.
    rows = ladder_rungs.rungs(payload, live_pool=["cheap-base", "t2"])
    assert [r.target for r in rows if not r.live] == ["t1"]
    assert [r.target for r in rows if r.live] == ["t2"]
    assert [r.target for r in rows if r.visited] == ["t2"]
    ann = ladder_rungs._annotations(rows, payload, 3)
    assert any("not live" in fact and "t1" in fact for fact in ann.subtitle_facts)
    assert dict(ann.counts)["visited_rungs"] == 1


def test_the_shipped_live_pool_excludes_the_dominated_models() -> None:
    # The derivation must read the packaged router.yaml models list, so a pool edit moves the
    # drawing. The shipped pool now holds the evidence-backed rungs and the frontier tail —
    # the strictly-dominated models are registry-only, not live.
    live = packaged_live_pool()
    assert "deepseek-v4-flash" in live
    assert "zai-glm-5.2" in live
    assert "kimi-k3" in live
    assert "qwen3.7-plus" not in live
    assert "gpt-5-mini" not in live
    assert "kimi-k2.5" not in live


class _DrawRecorder:
    """Records the drawing calls `_draw_steps` makes, so geometry is asserted, not rendered."""

    def __init__(self) -> None:
        self.steps: list[tuple[float, float]] = []
        self.jumps: list[tuple[float, float]] = []
        self.labels: list[str] = []

    def plot(self, xs: list[float], ys: list[float], **kwargs: Any) -> None:  # noqa: ARG002
        self.steps.append((ys[0], ys[1]))

    def annotate(
        self, _text: str, xy: tuple[float, float], xytext: tuple[float, float], **kwargs: Any
    ) -> None:  # noqa: ARG002
        self.jumps.append((xytext[1], xy[1]))

    def text(self, x: float, y: float, s: str, **kwargs: Any) -> None:  # noqa: ARG002
        self.labels.append(s)


def test_the_walk_is_drawn_on_live_ranks_and_the_jump_passes_only_live_rows() -> None:
    # The shipped configuration: the three dominated targets draw BELOW zai while its live
    # rank is 1 (adjacent to the base at 0), and kimi-k3 is the row the jump passes over.
    # The base -> zai segment must be a STEP (adjacent live ranks) — never a jump over the
    # NOT-LIVE rows — and the jump to the top rank arcs up over kimi-k3, not over them.
    payload = {
        "base_model": "deepseek-v4-flash",
        "targets": [
            _row("qwen3.7-plus", -0.05, 0.51, "INDISTINGUISHABLE"),
            _row("gpt-5-mini", -0.17, 0.0, "NET-HARMFUL"),
            _row("kimi-k2.5", -0.02, 0.81, "INDISTINGUISHABLE"),
            _row("zai-glm-5.2", 0.15, 0.001, "NET-HELPFUL"),
            _row("kimi-k3", 0.24, 0.0, "NET-HELPFUL"),
        ],
    }
    live = [
        "deepseek-v4-flash",
        "zai-glm-5.2",
        "gemini-3.1-pro",
        "kimi-k3",
        "gpt-5.6-sol",
        "claude-opus-4-8",
        "claude-fable-5",
    ]
    rows = ladder_rungs.rungs(payload, live_pool=live)
    rec = _DrawRecorder()
    ladder_rungs._draw_steps(rec, rows, len(live) - 1)  # type: ignore[arg-type]
    # Base (rank 0) -> zai (rank 1) are adjacent live ranks: a STEP, not a jump over the
    # three NOT-LIVE rows the benchmark still draws below zai.
    assert rec.steps == [(-1.0, 3.0)]
    # The single jump is shortlist -> top rank, arcing from zai over kimi-k3 to just past
    # the top row (row 4 of 5) — it never passes over a NOT-LIVE row.
    assert rec.jumps == [(3.0, 4.5)]
    assert rec.labels == ["jump to top rank"]


def test_a_jump_landing_on_a_drawn_top_row_is_not_extended() -> None:
    # When the top-rank model is itself a benchmark row, the jump ends there — there is no
    # invisible top rung beyond the last drawn one to arc over.
    payload = {
        "base_model": "cheap-base",
        "targets": [
            _row("t1", 0.03, 0.51, "INDISTINGUISHABLE"),
            _row("t5", 0.24, 0.0, "NET-HELPFUL"),
        ],
    }
    live = ["cheap-base", "t1", "t2", "t3", "t4", "t5"]  # 6 models -> top rank 5
    rows = ladder_rungs.rungs(payload, live_pool=live)
    rec = _DrawRecorder()
    ladder_rungs._draw_steps(rec, rows, len(live) - 1)  # type: ignore[arg-type]
    # base -> t1 step, then the jump t1 -> t5 lands on the drawn top row: no extension.
    assert rec.steps == [(-1.0, 0.0)]
    assert rec.jumps == [(0.0, 1.0)]
    assert rec.labels == ["jump to top rank"]
