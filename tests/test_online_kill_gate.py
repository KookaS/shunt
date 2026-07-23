"""Statistical-property tests for the served online kill-gate.

The load-bearing test is peeking-safety (false-PASS within alpha over all looks); the
rest pin the cost guard, reachability, and the underpowered verdict.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

from benchmark.routing import online_kill_gate as gate

_NORMAL = NormalDist()


# ---------------------------------------------------------------------------
# Stream generators
# ---------------------------------------------------------------------------


def _paired(r: int, b: int, rc: float = 0.5, bc: float = 1.0, known: bool = True):
    return gate.PairedOutcome(
        router_pass=r, baseline_pass=b, router_cost=rc, baseline_cost=bc, cost_known=known
    )


def _null_stream(p_b: float, p_c: float, length: int, seed: int, *, rc: float, bc: float):
    """Paired stream with E[X] = p_b - p_c; costs fixed (rc router, bc baseline)."""
    rng = random.Random(seed)
    out = []
    for _ in range(length):
        u = rng.random()
        if u < p_b:
            r, b = 1, 0
        elif u < p_b + p_c:
            r, b = 0, 1
        else:
            r, b = 1, 1
        out.append(_paired(r, b, rc=rc, bc=bc))
    return out


# ---------------------------------------------------------------------------
# Peeking-safety — the load-bearing property
# ---------------------------------------------------------------------------


class TestPeekingSafety:
    def _anytime_ever_pass(self, seeds: int, length: int, margin: float) -> float:
        # Null boundary: E[X] = 0.10 - 0.15 = -margin. Cost always router-cheaper so the
        # cost guard never blocks — a PASS is then purely a false quality decision.
        ever = 0
        for seed in range(seeds):
            state = gate.OnlineGateState()
            passed = False
            for obs in _null_stream(0.10, 0.15, length, seed, rc=0.5, bc=1.0):
                state = gate.update_online_gate(state, obs, margin=margin)
                if gate.decide_online_verdict(state)[0] == gate.PASS:
                    passed = True
                    break
            ever += 1 if passed else 0
        return ever / seeds

    def _fixed_alpha_ever_reject(self, seeds: int, length: int, margin: float, alpha: float):
        # A naive nightly fixed-alpha paired z-test, re-run at every look. Same null.
        z_crit = _NORMAL.inv_cdf(1.0 - alpha)
        ever = 0
        for seed in range(seeds):
            xs: list[float] = []
            rejected = False
            for obs in _null_stream(0.10, 0.15, length, seed, rc=0.5, bc=1.0):
                xs.append(float(obs.router_pass - obs.baseline_pass))
                n = len(xs)
                if n < 2:
                    continue
                mean = sum(xs) / n
                var = sum((x - mean) ** 2 for x in xs) / (n - 1)
                se = math.sqrt(var / n)
                if se > 0 and (mean - (-margin)) / se > z_crit:
                    rejected = True
                    break
            ever += 1 if rejected else 0
        return ever / seeds

    def test_anytime_false_pass_stays_within_budget(self):
        rate = self._anytime_ever_pass(seeds=1500, length=80, margin=0.05)
        assert rate <= 0.075, rate

    def test_fixed_alpha_nightly_test_inflates_type_one(self):
        # The contrast that motivates the whole design: the same null, looked at nightly
        # with a fixed-alpha test, rejects far above alpha; the anytime monitor does not.
        anytime = self._anytime_ever_pass(seeds=800, length=80, margin=0.05)
        fixed = self._fixed_alpha_ever_reject(seeds=800, length=80, margin=0.05, alpha=0.05)
        assert fixed > anytime
        assert fixed >= 0.15, fixed


# ---------------------------------------------------------------------------
# Cost guard + verdict semantics
# ---------------------------------------------------------------------------


class TestCostGuardAndVerdict:
    def test_pass_impossible_while_not_cheaper_even_with_great_quality(self):
        # Router aces quality (always pass, baseline always fail) but costs 2x -> ratio>=1.
        stream = [_paired(1, 0, rc=2.0, bc=1.0) for _ in range(60)]
        state = gate.run_online_gate(stream)
        verdict, _ = gate.decide_online_verdict(state)
        assert state.quality is not None and state.quality.decided
        assert state.quality.direction == "router_wins"
        assert state.cache_aware_ratio >= 1.0
        assert verdict != gate.PASS
        assert verdict == gate.FAIL

    def test_passes_when_non_inferior_and_cheaper(self):
        stream = [_paired(1, 0, rc=0.5, bc=1.0) for _ in range(60)]
        state = gate.run_online_gate(stream)
        verdict, _ = gate.decide_online_verdict(state)
        assert verdict == gate.PASS
        assert state.cache_aware_ratio < 1.0

    def test_fail_when_quality_credibly_worse(self):
        # Router always fails, baseline always passes -> quality credibly below margin.
        stream = [_paired(0, 1, rc=0.5, bc=1.0) for _ in range(60)]
        state = gate.run_online_gate(stream)
        verdict, msg = gate.decide_online_verdict(state)
        assert verdict == gate.FAIL
        assert "quality" in msg

    def test_fail_when_cost_credibly_not_cheaper(self):
        # Quality concordant (both pass); router strictly more expensive every session ->
        # the cost sign-sequence credibly decides 'not cheaper' -> FAIL.
        stream = [_paired(1, 1, rc=2.0, bc=1.0) for _ in range(60)]
        state = gate.run_online_gate(stream)
        verdict, msg = gate.decide_online_verdict(state)
        assert verdict == gate.FAIL
        assert "cheaper" in msg

    def test_aggregate_cheaper_router_is_not_failed_by_per_session_sign(self):
        # Regression: a router that is aggregate-cheaper (the real cost criterion) but
        # pricier on the MAJORITY of individual sessions must NOT be FAILed by the
        # per-session cost-sign sequence. Here 9 sessions are slightly router-pricier and
        # 1 is hugely router-cheaper -> aggregate ratio << 1 while the sign says 'not
        # cheaper'. Quality is a clear router win. Expected: PASS, never a cost FAIL.
        stream = [_paired(1, 0, rc=1.1, bc=1.0) for _ in range(9)]
        stream.append(_paired(1, 0, rc=0.1, bc=100.0))
        state = gate.run_online_gate(stream * 4)  # repeat to let both sequences decide
        assert state.cache_aware_ratio < 1.0
        verdict, _ = gate.decide_online_verdict(state)
        assert verdict == gate.PASS

    def test_undecided_stream_continues(self):
        stream = [_paired(1, 0), _paired(0, 1), _paired(1, 1), _paired(0, 1)]
        state = gate.run_online_gate(stream)
        verdict, _ = gate.decide_online_verdict(state, available_n=1000, reachable=True)
        assert verdict == gate.CONTINUE

    def test_ratio_is_inf_without_cost_data(self):
        stream = [_paired(1, 0, known=False) for _ in range(10)]
        state = gate.run_online_gate(stream)
        assert math.isinf(state.cache_aware_ratio)
        # Quality decided router_wins but no cost evidence -> CONTINUE, never PASS/FAIL.
        assert gate.decide_online_verdict(state)[0] == gate.CONTINUE


# ---------------------------------------------------------------------------
# Reachability and the underpowered verdict
# ---------------------------------------------------------------------------


class TestReachability:
    def test_solo_volume_with_honest_effect_is_underpowered(self):
        r = gate.reachability(effect=0.054, daily_pairs=10, window_days=30)
        assert r.available_n == 300
        assert not r.reachable
        assert r.required_n > r.available_n

    def test_high_volume_and_effect_is_reachable(self):
        r = gate.reachability(effect=0.15, daily_pairs=40, window_days=30)
        assert r.reachable
        assert r.required_n <= r.available_n

    def test_larger_effect_needs_fewer_samples(self):
        small = gate.reachability(effect=0.05)
        big = gate.reachability(effect=0.20)
        assert big.required_n < small.required_n

    def test_zero_gap_is_unreachable(self):
        # effect = -margin (true mean sits ON the boundary) -> gap 0 -> never separable.
        r = gate.reachability(effect=-0.05, margin=0.05)
        assert not r.reachable
        assert not math.isfinite(r.n_fixed)

    def test_verdict_reports_underpowered_when_window_exhausted_and_unreachable(self):
        stream = [_paired(1, 0), _paired(0, 1)]  # 2 obs, undecided
        state = gate.run_online_gate(stream)
        verdict, _ = gate.decide_online_verdict(state, available_n=2, reachable=False)
        assert verdict == gate.UNDERPOWERED
        # Same undecided state, but reachable -> CONTINUE, never a fabricated verdict.
        assert gate.decide_online_verdict(state, available_n=2, reachable=True)[0] == gate.CONTINUE

    def test_format_reachability_names_status(self):
        assert "UNDERPOWERED" in gate.format_reachability(
            gate.reachability(effect=0.054, daily_pairs=10, window_days=30)
        )


# ---------------------------------------------------------------------------
# Decorrelated fixed-N confirmation + store adapter
# ---------------------------------------------------------------------------


class TestConfirmationAndAdapter:
    def test_fixed_n_confirmation_agrees_on_a_clear_win(self):
        stream = [_paired(1, 0) for _ in range(40)] + [_paired(1, 1) for _ in range(40)]
        res = gate.confirm_fixed_n(stream, margin=0.05)
        assert res.decision == "non_inferior"

    def test_final_verdict_requires_the_cross_check_to_confirm_a_pass(self):
        # A clear router win + cheaper -> monitor PASS, and the fixed-N test also confirms.
        winning = [_paired(1, 0, rc=0.5, bc=1.0) for _ in range(60)]
        assert gate.final_verdict(winning)[0] == gate.PASS

    def test_final_verdict_downgrades_a_pass_the_cross_check_rejects(self, monkeypatch):
        # A PASS the fixed-N estimator does not confirm must be held at CONTINUE. The
        # corner is near-impossible to hit naturally (both estimators agree on a decided
        # win), so we force a non-confirming cross-check to prove the composition gates.
        from benchmark.routing.frontier_estimate import McNemarResult

        winning = [_paired(1, 0, rc=0.5, bc=1.0) for _ in range(60)]
        assert gate.decide_online_verdict(gate.run_online_gate(winning))[0] == gate.PASS

        monkeypatch.setattr(
            gate,
            "confirm_fixed_n",
            lambda *a, **k: McNemarResult(0, 0, 0.0, 1.0, "inconclusive"),
        )
        assert gate.final_verdict(winning)[0] == gate.CONTINUE
        # With confirmation disabled, the monitor PASS stands.
        assert gate.final_verdict(winning, confirm=False)[0] == gate.PASS

    def test_read_paired_outcomes_pairs_arms_by_key(self):
        class FakeStore:
            def labeled_outcome_rows(self, *, tier2_only: bool = False):
                return [
                    {"session_id": "s_r", "tier2_outcome": "success"},
                    {"session_id": "s_b", "tier2_outcome": "failure"},
                    {"session_id": "lonely", "tier2_outcome": "success"},
                ]

            def get_session(self, session_id: str):
                prov = {
                    "s_r": '{"arm": "router", "pair_key": "p1"}',
                    "s_b": '{"arm": "baseline", "pair_key": "p1"}',
                    "lonely": '{"arm": "router", "pair_key": "p2"}',
                }
                cost = {"s_r": 0.4, "s_b": 1.0, "lonely": 0.5}
                return {
                    "decision_provenance": prov[session_id],
                    "cost": cost[session_id],
                    "cost_known": 1,
                }

        pairs = gate.read_paired_outcomes(FakeStore())
        assert len(pairs) == 1  # only p1 is a complete pair; p2 has no baseline leg
        (p,) = pairs
        assert p.router_pass == 1 and p.baseline_pass == 0
        assert p.router_cost == 0.4 and p.baseline_cost == 1.0

    def test_read_paired_outcomes_empty_without_tags(self):
        class UntaggedStore:
            def labeled_outcome_rows(self, *, tier2_only: bool = False):
                return [{"session_id": "s1", "tier2_outcome": "success"}]

            def get_session(self, session_id: str):
                return {"decision_provenance": None, "cost": 1.0, "cost_known": 1}

        assert gate.read_paired_outcomes(UntaggedStore()) == []
