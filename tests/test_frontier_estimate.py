"""Deterministic statistical validation of the sparse frontier estimators."""

# Every test uses synthetic matrices with a KNOWN ground-truth Q_F / violation rate /
# paired difference, seeded, no live calls. A statistical bug here is a wrong kill-gate
# decision, so the estimators are checked for unbiasedness, CI coverage, and Type-I
# control against numbers they never observe.

from __future__ import annotations

import random
from statistics import NormalDist

from benchmark.routing.frontier_estimate import (
    frontier_violation_rate,
    mcnemar_noninferiority,
    ppi_frontier_quality,
    update_confidence_sequence,
)

_NORMAL = NormalDist()


# ---------------------------------------------------------------------------
# Synthetic-matrix fixture (the oracle): constructs the FULL frontier row so every
# estimator can be checked against a population number it does not observe.
# ---------------------------------------------------------------------------


def _make_population(
    n: int, q_f: float, p_agree: float, seed: int
) -> tuple[dict[str, float], dict[str, int]]:
    """(covariate g over all N, frontier outcome y over all N) with tunable corr(g, y)."""
    rng = random.Random(seed)
    covariate: dict[str, float] = {}
    outcome: dict[str, int] = {}
    for i in range(n):
        tid = f"t{i}"
        y = 1 if rng.random() < q_f else 0
        g = y if rng.random() < p_agree else 1 - y  # corr(g, y) rises with p_agree
        covariate[tid] = float(g)
        outcome[tid] = y
    return covariate, outcome


def _label(
    covariate: dict[str, float],
    outcome: dict[str, int],
    n_disc: int,
    audit_fraction: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Draw the labeled set: first n_disc tasks at pi=1 (D), rest audited at pi=f (U)."""
    rng = random.Random(seed * 7919 + 1)
    ids = list(covariate)
    labeled_outcome: dict[str, float] = {}
    labeled_prob: dict[str, float] = {}
    for idx, tid in enumerate(ids):
        if idx < n_disc:
            labeled_outcome[tid] = float(outcome[tid])
            labeled_prob[tid] = 1.0
        elif rng.random() < audit_fraction:
            labeled_outcome[tid] = float(outcome[tid])
            labeled_prob[tid] = audit_fraction
    return labeled_outcome, labeled_prob


class TestPPICoverage:
    def _coverage(self, p_agree: float, seeds: int = 1000) -> tuple[float, float]:
        """Return (CI coverage of true Q_F, mean signed bias) over many masks."""
        covered = 0
        bias = 0.0
        for seed in range(seeds):
            cov, out = _make_population(n=400, q_f=0.7, p_agree=p_agree, seed=seed)
            truth = sum(out.values()) / len(out)
            ly, lp = _label(cov, out, n_disc=40, audit_fraction=0.3, seed=seed)
            est = ppi_frontier_quality(cov, ly, lp, alpha=0.05)
            if est.ci_lo <= truth <= est.ci_hi:
                covered += 1
            bias += est.point - truth
        return covered / seeds, bias / seeds

    def test_ci_covers_true_q_f_at_nominal_rate(self):
        # Informative covariate: nominal 95% coverage, near-zero bias.
        coverage, bias = self._coverage(p_agree=0.85)
        assert 0.92 <= coverage <= 0.975, coverage
        assert abs(bias) < 0.01, bias

    def test_unbiased_even_with_a_useless_covariate(self):
        # Doubly-robust guarantee: a covariate uncorrelated with y (p_agree=0.5) still
        # yields valid coverage and unbiasedness — only efficiency (CI width) suffers.
        coverage, bias = self._coverage(p_agree=0.5)
        assert coverage >= 0.92, coverage
        assert abs(bias) < 0.01, bias

    def test_lambda_zero_recovers_horvitz_thompson_mean(self):
        # Zero-covariate ⇒ weighted Var(g)=0 ⇒ lam falls back to 0 ⇒ point == the
        # design-based (HT) labeled mean, the provable safety floor.
        cov = {f"t{i}": 0.0 for i in range(100)}  # no covariate signal
        ly = {f"t{i}": float(i % 2) for i in range(0, 40)}  # D, pi=1
        lp = {t: 1.0 for t in ly}
        for i in range(40, 100, 3):  # audit at pi=0.5
            ly[f"t{i}"] = float(i % 2)
            lp[f"t{i}"] = 0.5
        est = ppi_frontier_quality(cov, ly, lp)
        ht = sum(ly[t] / lp[t] for t in ly) / len(cov)
        assert est.lam == 0.0
        assert abs(est.point - ht) < 1e-9

    def test_full_coverage_is_exact(self):
        # Every task labeled at pi=1 ⇒ zero variance ⇒ CI collapses to the true mean.
        cov, out = _make_population(n=50, q_f=0.6, p_agree=0.8, seed=3)
        ly = {t: float(out[t]) for t in cov}
        lp = {t: 1.0 for t in cov}
        est = ppi_frontier_quality(cov, ly, lp)
        truth = sum(out.values()) / len(out)
        assert abs(est.point - truth) < 1e-9
        assert abs(est.ci_hi - est.ci_lo) < 1e-9

    def test_clip_false_preserves_unbounded_cost_estimand(self):
        # The cost total C_F can exceed 1; clip=False must NOT clamp it to [0,1]
        # (clip=True, the pass-rate default, would silently corrupt a cost estimate).
        cov = {f"t{i}": 5.0 for i in range(20)}  # per-task cost ~5, total mean far above 1
        ly = {t: 5.0 for t in cov}
        lp = {t: 1.0 for t in cov}
        assert ppi_frontier_quality(cov, ly, lp, clip=False).point > 1.0
        assert ppi_frontier_quality(cov, ly, lp, clip=True).point == 1.0


class TestViolationRate:
    def test_recovers_injected_violation_rate(self):
        # Inject frontier failures on cheap-passed tasks at a known rate; the audit
        # estimate's Wilson CI should cover it.
        rng = random.Random(11)
        v_true = 0.2
        covariate: dict[str, float] = {}
        outcome: dict[str, float] = {}
        audit_ids: list[str] = []
        for i in range(600):
            tid = f"t{i}"
            covariate[tid] = 1.0  # cheap passed
            outcome[tid] = 0.0 if rng.random() < v_true else 1.0  # frontier fails at v_true
            audit_ids.append(tid)
        est = frontier_violation_rate(outcome, covariate, audit_ids)
        assert est.ci_lo <= v_true <= est.ci_hi
        assert abs(est.point - v_true) < 0.05

    def test_no_cheap_passed_tasks_returns_empty(self):
        est = frontier_violation_rate({}, {"t0": 0.0}, ["t0"])
        assert est.n_labeled == 0


class TestMcNemar:
    def test_reduces_to_classic_mcnemar_z_at_zero_margin(self):
        router = {f"t{i}": 1 for i in range(12)}
        base = {f"t{i}": 0 for i in range(12)}
        for i in range(12, 15):  # 3 c-type (router fail, baseline pass)
            router[f"t{i}"] = 0
            base[f"t{i}"] = 1
        for i in range(15, 50):  # concordant filler
            router[f"t{i}"] = 1
            base[f"t{i}"] = 1
        res = mcnemar_noninferiority(router, base, margin=0.0)
        assert res.b == 12 and res.c == 3
        expected_z = (12 - 3) / (15**0.5)
        assert abs(res.stat - expected_z) < 1e-9

    def test_equality_p_matches_exact_binomial(self):
        # At margin 0 the two-sided p from the score z should track the exact conditional
        # McNemar (scipy binomtest on discordants) at moderate discordant counts.
        try:
            from scipy.stats import binomtest
        except ImportError:  # pragma: no cover
            return
        router = {f"t{i}": 1 for i in range(30)}
        base = {f"t{i}": 0 for i in range(30)}
        for i in range(30, 40):
            router[f"t{i}"] = 0
            base[f"t{i}"] = 1
        for i in range(40, 100):
            router[f"t{i}"] = 1
            base[f"t{i}"] = 1
        res = mcnemar_noninferiority(router, base, margin=0.0)
        two_sided = 2.0 * (1.0 - _NORMAL.cdf(abs(res.stat)))
        exact = binomtest(10, 40, 0.5, alternative="two-sided").pvalue
        assert abs(two_sided - exact) < 0.02

    def test_no_discordants_is_inconclusive(self):
        router = {f"t{i}": 1 for i in range(20)}
        base = {f"t{i}": 1 for i in range(20)}
        res = mcnemar_noninferiority(router, base, margin=0.05)
        assert res.decision == "inconclusive"
        assert res.b == 0 and res.c == 0

    def _simulate_decision(self, p_b: float, p_c: float, margin: float, seeds: int, n: int) -> dict:
        counts = {"non_inferior": 0, "inferior": 0, "inconclusive": 0}
        for seed in range(seeds):
            rng = random.Random(seed)
            router: dict[str, int] = {}
            base: dict[str, int] = {}
            for i in range(n):
                tid = f"t{i}"
                u = rng.random()
                if u < p_b:  # router pass, baseline fail
                    router[tid], base[tid] = 1, 0
                elif u < p_b + p_c:  # router fail, baseline pass
                    router[tid], base[tid] = 0, 1
                else:  # concordant (both pass)
                    router[tid], base[tid] = 1, 1
            counts[mcnemar_noninferiority(router, base, margin=margin).decision] += 1
        return {k: v / seeds for k, v in counts.items()}

    def test_type_one_at_the_noninferiority_boundary(self):
        # True paired diff = -margin exactly (H0 boundary). P(declare non_inferior) ≤ α.
        rates = self._simulate_decision(p_b=0.10, p_c=0.15, margin=0.05, seeds=1500, n=60)
        assert rates["non_inferior"] <= 0.085, rates

    def test_power_when_router_clearly_wins(self):
        rates = self._simulate_decision(p_b=0.25, p_c=0.05, margin=0.05, seeds=800, n=60)
        assert rates["non_inferior"] >= 0.8, rates

    def test_detects_a_clearly_inferior_router(self):
        rates = self._simulate_decision(p_b=0.05, p_c=0.35, margin=0.05, seeds=800, n=60)
        assert rates["inferior"] >= 0.7, rates


class TestConfidenceSequence:
    def _run_stream(
        self, p_b: float, p_c: float, margin: float, length: int, seed: int
    ) -> str | None:
        rng = random.Random(seed)
        state = None
        for _ in range(length):
            u = rng.random()
            if u < p_b:
                r, b = 1, 0
            elif u < p_b + p_c:
                r, b = 0, 1
            else:
                r, b = 1, 1
            state = update_confidence_sequence(state, r, b, margin=margin)
            if state.decided:
                return state.direction
        return None

    def test_anytime_type_one_under_the_null(self):
        # True mean X = p_b - p_c = -margin (null). Across all stopping times, the
        # fraction that EVER declares router_wins must stay within the α budget.
        margin = 0.05
        wins = 0
        seeds = 2000
        for seed in range(seeds):
            if (
                self._run_stream(p_b=0.10, p_c=0.15, margin=margin, length=80, seed=seed)
                == "router_wins"
            ):
                wins += 1
        assert wins / seeds <= 0.075, wins / seeds

    def test_power_when_edge_is_real(self):
        # Mean X = +0.25, well above -margin: should decide router_wins on most streams.
        decided_wins = 0
        seeds = 500
        for seed in range(seeds):
            if (
                self._run_stream(p_b=0.30, p_c=0.05, margin=0.05, length=120, seed=seed)
                == "router_wins"
            ):
                decided_wins += 1
        assert decided_wins / seeds >= 0.8, decided_wins / seeds

    def test_e_value_and_ci_are_populated(self):
        state = None
        for _i in range(10):
            state = update_confidence_sequence(state, 1, 0, margin=0.05)
        assert state is not None
        assert state.e_value > 0.0
        assert state.ci_lo <= state.ci_hi
        assert state.n == 10
