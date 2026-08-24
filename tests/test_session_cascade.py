"""Session-Cascade: the live escalation ladder replayed at session cadence, and its R0 gate."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import numpy as np
import pytest

from benchmark.routing import session_cascade_control as control
from benchmark.routing import summary
from benchmark.routing.strategies.session_cascade import ArmLadder, SessionCascadeStrategy

_PRICES = (1.0, 2.0, 4.0)
_MODELS = ("cheap", "mid", "dear")
_COSTS: Final[Mapping[str, float]] = MappingProxyType({"cheap": 0.01, "mid": 0.05, "dear": 0.50})


def _matrix(passes: dict[str, set[str]]) -> dict:
    """A complete 3-rung matrix; ``passes[task]`` names the models that solve it."""
    return {
        "tasks": {tid: {"description": tid} for tid in passes},
        "results": {
            tid: {m: {"pass": m in solved, "cost": _COSTS[m], "calls": 1} for m in _MODELS}
            for tid, solved in passes.items()
        },
        "models": {
            m: {"input_price": p, "output_price": 0.0}
            for m, p in zip(_MODELS, _PRICES, strict=True)
        },
    }


def _flat(**kwargs) -> SessionCascadeStrategy:
    """A rank-only replay over models with no reasoning ladder (arm axis out of scope)."""
    kwargs.setdefault("ladder", "rank_only")
    return SessionCascadeStrategy(arm_results={}, arm_ladders={}, **kwargs)


class TestLadderSemantics:
    def test_a_cheap_pass_ends_the_ladder_after_one_session(self):
        s = _flat()
        assert s.select("t", {}, _matrix({"t": {"cheap", "mid", "dear"}})) == "cheap"
        assert s.session_path == [("cheap", "")]
        assert s.cascade_total_cost == pytest.approx(_COSTS["cheap"])

    def test_each_rung_is_attempted_escalate_after_n_times_before_climbing(self):
        s = _flat(escalate_after_n=2)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        # The recurrence counter needs two verified failures per rung, and every session bills.
        assert s.cascade_tried_models == ["cheap", "cheap", "mid", "mid", "dear"]
        assert s.session_hops == 3

    def test_a_lower_threshold_climbs_on_the_first_recurrence(self):
        s = _flat(escalate_after_n=1)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        assert s.cascade_tried_models == ["cheap", "mid", "dear"]

    def test_the_climbed_rank_persists_across_sessions(self):
        # The rank floor is what makes this a cascade rather than an oscillation: after the
        # step to `mid` the NEXT session must re-serve mid, never fall back to cheap.
        s = _flat(escalate_after_n=2)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        assert s.cascade_tried_models[2:4] == ["mid", "mid"]

    def test_the_ladder_stops_at_the_ceiling_instead_of_spinning(self):
        s = _flat(escalate_after_n=2)
        s.select("t", {}, _matrix({"t": set()}))  # nothing solves it
        assert s.cascade_tried_models == ["cheap", "cheap", "mid", "mid", "dear", "dear"]

    def test_every_attempt_is_billed(self):
        s = _flat(escalate_after_n=2)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        expected = 2 * _COSTS["cheap"] + 2 * _COSTS["mid"] + _COSTS["dear"]
        assert s.cascade_total_cost == pytest.approx(expected)

    def test_an_unmeasured_cell_makes_the_path_unscorable(self):
        # A second task keeps `mid` in the price ladder — the gap is in THIS task's row only.
        m = _matrix({"t": {"dear"}, "u": {"mid"}})
        del m["results"]["t"]["mid"]
        s = _flat(escalate_after_n=2)
        s.select("t", {}, m)
        assert s.cascade_scorable is False
        assert s.session_unmeasured > 0


class TestRankShortlist:
    """The rank rung's SHAPE, shared with the live engine via `escalation.next_rung_rank`."""

    def test_shortlist_zero_walks_every_rank(self):
        # The pre-shortlist behaviour, pinned: with the jump disabled the replay buys `mid` on the
        # way to `dear`, which is the ladder this module modelled before the knob existed.
        s = _flat(escalate_after_n=2, rank_shortlist=0)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        assert s.cascade_tried_models == ["cheap", "cheap", "mid", "mid", "dear"]

    def test_a_shortlist_narrower_than_the_pool_jumps_to_the_top_rank(self):
        # The whole point of the knob: leaving the shortlist skips the intermediate rung entirely,
        # so `mid` is never bought and the ladder saturates in fewer sessions.
        s = _flat(escalate_after_n=2, rank_shortlist=1)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        assert s.cascade_tried_models == ["cheap", "cheap", "dear"]
        assert s.session_hops == 2

    def test_a_shortlist_wider_than_the_pool_is_the_every_rank_walk(self):
        # `next_rung_rank`'s `max` guard: the target must stay strictly above the current rung even
        # when the pool cannot reach the shortlist, or the ladder would stall at its own ceiling.
        s = _flat(escalate_after_n=2, rank_shortlist=9)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        assert s.cascade_tried_models == ["cheap", "cheap", "mid", "mid", "dear"]


class TestEffortRung:
    """`effort_then_rank` steps the reasoning arm first — same model, so cache-safe."""

    _LADDERS = {
        "cheap": ArmLadder(arms=("lo", "hi"), default_arm="lo"),
        "mid": ArmLadder(arms=("lo",), default_arm="lo"),
        "dear": ArmLadder(arms=("lo",), default_arm="lo"),
    }

    def _strategy(self, arm_results: dict) -> SessionCascadeStrategy:
        return SessionCascadeStrategy(
            ladder="effort_then_rank",
            escalate_after_n=2,
            arm_results=arm_results,
            arm_ladders=self._LADDERS,
        )

    def test_the_effort_rung_precedes_the_rank_rung(self):
        arms = {"t": {"cheap": {"hi": {"pass": False, "cost": 0.02}}}}
        s = self._strategy(arms)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        assert s.session_path[:3] == [("cheap", "lo"), ("cheap", "lo"), ("cheap", "hi")]

    def test_an_unmeasured_higher_arm_is_a_coverage_gap_not_a_free_fail(self):
        # Nothing measured the `hi` arm, so the replay cannot say what it would have cost.
        s = self._strategy({})
        s.select("t", {}, _matrix({"t": {"dear"}}))
        assert s.cascade_scorable is False

    def test_a_terminal_non_default_arm_is_unscorable(self):
        # `summary.evaluate` reads pass/fail from the DEFAULT-arm cell, a different measurement
        # from the one the replay observed, so such a row must never be scored.
        ladders = {"cheap": ArmLadder(arms=("lo", "hi"), default_arm="lo")}
        arms = {"t": {"cheap": {"hi": {"pass": True, "cost": 0.02}}}}
        s = SessionCascadeStrategy(
            ladder="effort_then_rank", escalate_after_n=2, arm_results=arms, arm_ladders=ladders
        )
        m = _matrix({"t": set()})
        m["results"]["t"] = {"cheap": m["results"]["t"]["cheap"]}
        m["models"] = {"cheap": {"input_price": 1.0, "output_price": 0.0}}
        s.select("t", {}, m)
        assert s.session_path[-1] == ("cheap", "hi")
        assert s.cascade_scorable is False


class TestScoringIntegration:
    def test_summary_evaluate_bills_the_whole_session_sequence(self):
        m = _matrix({"a": {"cheap"}, "b": {"dear"}})
        decisions, unscorable = summary.evaluate(_flat(escalate_after_n=2), m, ["a", "b"])
        assert not unscorable
        costs = {tid: cost for tid, _model, _passed, cost in decisions}
        assert costs["a"] == pytest.approx(_COSTS["cheap"])
        assert costs["b"] == pytest.approx(2 * _COSTS["cheap"] + 2 * _COSTS["mid"] + _COSTS["dear"])
        assert all(passed for _t, _m, passed, _c in decisions)


class TestAdmissibilityGate:
    """R0: the assembled replay recovers a planted ladder depth and collapses on a shuffled one."""

    def test_the_instrument_is_admissible(self):
        verdict = control.run_control(n_perm=40, seed=0)
        assert verdict.admissible, verdict.reason
        assert verdict.numbers["hop_depths_observed"] == [1, 2, 3, 4]

    def test_the_control_asserts_the_ladder_was_actually_walked(self):
        # The teeth: a replay that reaches only rung 1 raises rather than scoring quietly.
        with pytest.raises(RuntimeError, match="never exercised the full ladder"):
            control._assert_ladder_is_exercised([1, 1, 1], None)

    def test_the_hop_requirement_follows_the_shortlist_rather_than_the_pool(self):
        # A shortlist SKIPS rungs by design, so the assertion must be derived from the shortlist
        # geometry, not from "one hop per model". Hardcoding 1..N would fail an honest shortlist-2
        # replay (which visits ranks 0,1,3 and can never occupy 4 rungs) and so would have been
        # quietly widened to make a number reportable.
        assert control.rung_sequence(0) == [0, 1, 2, 3]
        assert control.rung_sequence(2) == [0, 1, 3]
        assert control._expected_hop_depths(0) == {1, 2, 3, 4}
        assert control._expected_hop_depths(2) == {1, 2, 3}
        control._assert_ladder_is_exercised([1, 2, 3], 2)  # legal under a shortlist of 2
        with pytest.raises(RuntimeError, match="never exercised the full ladder"):
            control._assert_ladder_is_exercised([1, 2, 3], 0)  # not legal without one

    def _mutant_score(self, monkeypatch, cls) -> tuple[float, set[int]]:
        monkeypatch.setattr(
            control,
            "_strategy",
            lambda n, shortlist, initial_rung=None: cls(
                ladder="rank_only",
                escalate_after_n=n,
                arm_results={},
                arm_ladders={},
                rank_shortlist=shortlist,
            ),
        )
        depths = np.repeat(np.arange(len(control.CONTROL_MODELS)), 8)
        costs, hops = control.replay_costs(control.build_control_matrix(depths), 2)
        return control._correlation(costs, depths), set(hops)

    def test_a_frozen_ladder_scores_at_chance(self, monkeypatch):
        # Mutation control #1: an engine that never escalates bills every escalation-requiring
        # task the same, so cost carries no depth information. Scored over the WHOLE corpus this
        # mutant read r=+0.77 and the gate certified it — which is why the statistic conditions
        # on depth >= 1.
        class _Frozen(SessionCascadeStrategy):
            def _apply(self, directive, runner, model, arm, rank_floor, rungs, ladders):  # noqa: PLR0913
                return (model, arm, rank_floor, arm)

        score, hops = self._mutant_score(monkeypatch, _Frozen)
        assert hops == {1}
        assert score == pytest.approx(0.0)

    def test_a_ladder_with_no_rank_floor_is_degraded(self, monkeypatch):
        # Mutation control #2: the bug the live engine carried before the per-task rank floor —
        # the climbed rung is not re-served, so the ladder oscillates and never reaches the top.
        class _NoFloor(SessionCascadeStrategy):
            def _apply(self, directive, runner, model, arm, rank_floor, rungs, ladders):  # noqa: PLR0913
                served, next_arm, floor, held = super()._apply(
                    directive, runner, model, arm, rank_floor, rungs, ladders
                )
                return (served, next_arm, 0, held)

        _score, hops = self._mutant_score(monkeypatch, _NoFloor)
        assert max(hops) < len(control.CONTROL_MODELS)  # never walks the whole ladder
        # The hop assertion fires before any score is adjudicated — the strongest failure mode.
        with pytest.raises(RuntimeError, match="never exercised the full ladder"):
            control.run_control(n_perm=5, seed=0)
