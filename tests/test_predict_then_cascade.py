"""Predict-then-cascade: the binary gate's cheap-direct branch and its no-regression guarantee.

The gate is one decision per session boundary (cache-safe); the ladder path must be
byte-for-byte Session-Cascade.
"""

from __future__ import annotations

from typing import Final

import pytest

from benchmark.routing import summary
from benchmark.routing.strategies import BilledAttempt
from benchmark.routing.strategies.predict_then_cascade import (
    FractionalGate,
    PerfectCheapGate,
    PredictThenCascadeStrategy,
)
from benchmark.routing.strategies.session_cascade import SessionCascadeStrategy

_PRICES = (1.0, 2.0, 4.0)
_MODELS = ("cheap", "mid", "dear")
_COSTS: Final = {"cheap": 0.01, "mid": 0.05, "dear": 0.50}


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


def _flat(gate: FractionalGate | PerfectCheapGate, **kwargs) -> PredictThenCascadeStrategy:
    """A rank-only replay over models with no reasoning ladder (arm axis out of scope)."""
    kwargs.setdefault("ladder", "rank_only")
    return PredictThenCascadeStrategy(gate=gate, arm_results={}, arm_ladders={}, **kwargs)


def _ladder_baseline(**kwargs) -> SessionCascadeStrategy:
    """The Session-Cascade row f=0 must reproduce — the same replay, no gate."""
    kwargs.setdefault("ladder", "rank_only")
    return SessionCascadeStrategy(arm_results={}, arm_ladders={}, **kwargs)


class TestDegeneration:
    """A chance-level gate MUST degenerate to Session-Cascade (f=0) / Always-Cheap (f=1)."""

    def test_f0_reproduces_session_cascade_per_task(self):
        m = _matrix({"a": {"cheap"}, "b": {"dear"}, "c": {"mid"}, "d": set()})
        base = _ladder_baseline()
        gated = _flat(FractionalGate(0.0))
        for tid in m["results"]:
            base.select(tid, {}, m)
            gated.select(tid, {}, m)
            assert gated.cascade_tried_models == base.cascade_tried_models
            assert gated.cascade_total_cost == pytest.approx(base.cascade_total_cost)
            assert gated.cascade_scorable is base.cascade_scorable

    def test_f0_reproduces_session_cascade_under_summary_evaluate(self):
        m = _matrix({"a": {"cheap"}, "b": {"dear"}, "c": {"mid"}})
        base_dec, base_un = summary.evaluate(_ladder_baseline(), m, ["a", "b", "c"])
        gate_dec, gate_un = summary.evaluate(_flat(FractionalGate(0.0)), m, ["a", "b", "c"])
        assert gate_un == base_un
        assert gate_dec == base_dec

    def test_f0_enters_the_ladder(self):
        # Nothing but the frontier solves it, so f=0 must climb rung by rung.
        s = _flat(FractionalGate(0.0), escalate_after_n=2)
        s.select("t", {}, _matrix({"t": {"dear"}}))
        assert s.cascade_tried_models == ["cheap", "cheap", "mid", "mid", "dear"]
        assert s.session_hops == 3

    def test_f1_is_always_cheap_single_shot(self):
        # A task the ladder would escalate is served cheap once and never climbs.
        m = _matrix({"t": {"mid", "dear"}})
        s = _flat(FractionalGate(1.0))
        assert s.select("t", {}, m) == "cheap"
        assert s.session_path == [("cheap", "")]
        assert s.session_hops == 1
        assert s.cascade_total_cost == pytest.approx(_COSTS["cheap"])
        assert s.cascade_attempts == [
            BilledAttempt(model="cheap", cost=_COSTS["cheap"], in_tok=0, out_tok=0, calls=1)
        ]

    def test_f1_is_always_cheap_under_summary_evaluate(self):
        m = _matrix({"a": {"cheap"}, "b": {"dear"}})
        decisions, unscorable = summary.evaluate(_flat(FractionalGate(1.0)), m, ["a", "b"])
        assert not unscorable
        costs = {tid: cost for tid, _model, _passed, cost in decisions}
        assert costs["a"] == pytest.approx(_COSTS["cheap"])
        assert costs["b"] == pytest.approx(_COSTS["cheap"])  # b fails at $0.01, never escalates
        passed = {tid: p for tid, _model, p, _cost in decisions}
        assert passed == {"a": True, "b": False}


class TestGateSeam:
    """The seam a real difficulty score plugs into, and its score-free placeholders."""

    def test_fractional_gate_is_deterministic_across_instances(self):
        g1, g2 = FractionalGate(0.5), FractionalGate(0.5)
        ids = [f"task-{i}" for i in range(200)]
        assert [g1.decides_cheap(t, {}) for t in ids] == [g2.decides_cheap(t, {}) for t in ids]

    def test_fractional_gate_sends_f_share_of_tasks(self):
        ids = [f"task-{i}" for i in range(2000)]
        frac = 0.3
        share = sum(1 for t in ids if FractionalGate(frac).decides_cheap(t, {})) / len(ids)
        assert share == pytest.approx(frac, abs=0.04)

    def test_fractional_gate_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            FractionalGate(-0.1)
        with pytest.raises(ValueError):
            FractionalGate(1.1)

    def test_fractional_gate_endpoints_are_constant(self):
        assert FractionalGate(0.0).decides_cheap("t", {}) is False
        assert FractionalGate(1.0).decides_cheap("t", {}) is True

    def test_perfect_gate_sends_exactly_its_cheap_set(self):
        g = PerfectCheapGate({"a", "c"})
        assert g.decides_cheap("a", {}) is True
        assert g.decides_cheap("b", {}) is False
        assert g.decides_cheap("c", {}) is True

    def test_the_gate_seam_sees_task_meta(self):
        # A real score gate will compute a difficulty score from task_meta; the seam passes it.
        seen: list[str] = []

        class RecordingGate(PerfectCheapGate):
            def decides_cheap(self, task_id: str, task_meta: dict) -> bool:
                seen.append(str(task_meta.get("description")))
                return super().decides_cheap(task_id, task_meta)

        s = _flat(RecordingGate({"t"}))
        s.select("t", {"description": "resolve bug"}, _matrix({"t": {"cheap"}}))
        assert seen == ["resolve bug"]


class TestScoringConventions:
    """The cheap-direct branch honours the ladder's unscorable / censoring conventions."""

    def test_an_unmeasured_cheap_cell_makes_the_path_unscorable(self):
        # A second task keeps `cheap` in the price ladder — the gap is in THIS task's row only.
        m = _matrix({"t": {"mid", "dear"}, "u": {"cheap"}})
        del m["results"]["t"]["cheap"]
        s = _flat(FractionalGate(1.0))
        s.select("t", {}, m)
        assert s.cascade_scorable is False
        assert s.session_unmeasured == 1

    def test_a_censored_cheap_cell_is_not_scored_as_fail_at_zero(self):
        # A resource-limit stop has an unknown true outcome — summary excludes it rather than
        # crediting a cheap fail@$0, exactly as it does for a ladder row's censored cell.
        m = _matrix({"t": {"mid", "dear"}})
        m["results"]["t"]["cheap"] = {
            "pass": False,
            "cost": 0.3,
            "calls": 1,
            "timeout_flag": True,
        }
        _decisions, unscorable = summary.evaluate(_flat(FractionalGate(1.0)), m, ["t"])
        assert "t" in unscorable

    def test_trace_respects_the_gate(self):
        m = _matrix({"t": {"mid", "dear"}})
        assert len(_flat(PerfectCheapGate({"t"})).trace("t", m).path) == 1
        assert len(_flat(PerfectCheapGate(set())).trace("t", m).path) > 1

    def test_trace_on_a_degenerate_matrix_matches_the_base_contract(self):
        # An empty rungs list (no priced-and-measured model) must not IndexError in the
        # cheap-direct branch: the base class reports the empty trace, and the gate must
        # not change what a degenerate replay reports.
        m = {"tasks": {"t": {}}, "results": {"t": {}}, "models": {}}
        gated = _flat(FractionalGate(1.0))
        base = _ladder_baseline()
        assert gated.trace("t", m) == base.trace("t", m) == _flat(FractionalGate(0.0)).trace("t", m)
