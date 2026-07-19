"""Partial-coverage estimators for the sparse fixed-frontier kill-gate."""

# Three pure estimands, each independently unit-testable against synthetic matrices with
# a known ground-truth Q_F (see tests/test_frontier_estimate.py). No model is ever run
# here — they read the already-collected outcome matrix; docs/benchmark.md explains how
# they decide the gate on partial coverage:
#   1. ppi_frontier_quality      — PPI++/AIPW estimate of frontier pass-rate/cost Q_F over
#      the whole population from a partial known-probability sample, cheap+mid as covariate.
#   2. mcnemar_noninferiority    — paired non-inferiority test on the discriminating set.
#   3. update_confidence_sequence — anytime-valid betting sequence on the paired difference,
#      so Phase C stops the moment the gate is decided either way.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

_NORMAL = NormalDist()


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


# ---------------------------------------------------------------------------
# 5a. PPI++ / AIPW estimator of Q_F (and C_F, same function) with a valid CI.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a two-sided CI and the labeled-sample size."""

    point: float
    ci_lo: float
    ci_hi: float
    n_labeled: int
    lam: float = 0.0


def _tuned_lambda(
    labeled_outcome: dict[str, float],
    covariate: dict[str, float],
    labeled_prob: dict[str, float],
) -> float:
    """Variance-minimising control-variate coefficient, clipped to [0, 1]."""
    # Weighted regression-through-origin of frontier outcome on the covariate over the
    # labeled set (weight (1-pi)/pi; fully-sampled D tasks carry zero weight and zero
    # variance). Falls back to 0 (plain design-based mean) when the weighted Var is 0, so
    # the estimator is provably never worse than the labeled Horvitz-Thompson mean.
    num = 0.0
    den = 0.0
    for tid, y in labeled_outcome.items():
        pi = labeled_prob[tid]
        weight = (1.0 - pi) / pi if pi < 1.0 else 0.0
        g = covariate[tid]
        num += weight * g * y
        den += weight * g * g
    if den <= 0.0:
        return 0.0
    return min(1.0, max(0.0, num / den))


def ppi_frontier_quality(
    covariate: dict[str, float],
    labeled_outcome: dict[str, float],
    labeled_prob: dict[str, float],
    alpha: float = 0.05,
    *,
    clip: bool = True,
) -> Estimate:
    """PPI++/AIPW estimate of a frontier population mean (pass-rate Q_F or cost C_F)."""
    # ``covariate`` (cheap+mid proxy g_i) is observed for ALL N tasks; ``labeled_outcome``
    # (frontier y_i) and its KNOWN sampling probability ``labeled_prob`` (pi_i = 1.0 for
    # discriminating tasks, = audit_fraction for the uniform audit) only on the labeled
    # set L. Because pi is known the estimate is UNBIASED for any covariate — a poor proxy
    # only widens the CI (doubly-robust guarantee); the lam=0 fallback equals the
    # design-based Horvitz-Thompson labeled mean.
    n = len(covariate)
    if n == 0:
        raise ValueError("covariate must cover the full task population (N > 0)")
    for tid, pi in labeled_prob.items():
        if not 0.0 < pi <= 1.0:
            raise ValueError(f"labeled_prob[{tid}] must be in (0, 1], got {pi}")
    lam = _tuned_lambda(labeled_outcome, covariate, labeled_prob)
    imputed = sum(lam * g for g in covariate.values()) / n
    rectifier = (
        sum((labeled_outcome[i] - lam * covariate[i]) / labeled_prob[i] for i in labeled_outcome)
        / n
    )
    point = imputed + rectifier
    var = 0.0
    for tid, y in labeled_outcome.items():
        pi = labeled_prob[tid]
        resid = y - lam * covariate[tid]
        var += (1.0 - pi) / (pi * pi) * resid * resid
    var /= n * n
    half = _NORMAL.inv_cdf(1.0 - alpha / 2.0) * math.sqrt(var)
    lo, hi = point - half, point + half
    # Q_F is a pass-rate in [0,1] → clip. The unbounded cost total C_F (Σ c_i, may exceed
    # 1) must pass clip=False so the estimate/CI are not silently corrupted.
    if clip:
        point, lo, hi = _clip01(point), _clip01(lo), _clip01(hi)
    return Estimate(point, lo, hi, len(labeled_outcome), lam)


def _wilson(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (small-n honest)."""
    if n == 0:
        return (0.0, 0.0)
    z = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (_clip01(center - margin), _clip01(center + margin))


def frontier_violation_rate(
    labeled_outcome: dict[str, float],
    covariate: dict[str, float],
    audit_ids: list[str],
    cheap_pass_threshold: float = 0.5,
    alpha: float = 0.05,
) -> Estimate:
    """Non-monotonicity violation rate: frontier fails where the cheap proxy passed."""
    # Estimated only on the UNIFORM audit stratum (known pi), among tasks whose covariate
    # indicates the cheaper tier passed (>= threshold); Wilson CI. This is the assumption
    # other routers leave undisclosed — Shunt measures it.
    fails = 0
    n = 0
    for tid in audit_ids:
        if covariate.get(tid, 0.0) >= cheap_pass_threshold:
            n += 1
            if labeled_outcome.get(tid, 0.0) < 0.5:
                fails += 1
    if n == 0:
        return Estimate(0.0, 0.0, 0.0, 0, 0.0)
    lo, hi = _wilson(fails, n, alpha)
    return Estimate(fails / n, lo, hi, n, 0.0)


# ---------------------------------------------------------------------------
# 5b. Paired McNemar non-inferiority on the discriminating set.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    """Discordant counts and the non-inferiority decision."""

    b: int
    c: int
    stat: float
    p_value: float
    decision: str


def _constrained_p21(b: int, c: int, m: int, delta0: float) -> float:
    """Constrained MLE of the c-cell probability under H0: paired diff = delta0."""
    # Solves the score equation b/(delta0+psi) + c/psi - 2m/(1-delta0-2psi) = 0 by
    # bisection over the valid open interval; clamps to a bound when no interior root
    # exists (e.g. m == 0). Used by the Tango score statistic below.
    n = b + c + m
    lo = max(0.0, -delta0) + 1e-9
    hi = (1.0 - delta0) / 2.0 - 1e-9
    if lo >= hi:
        return max(lo, min(hi, (b + c) / (2.0 * n) if n else lo))

    def score(psi: float) -> float:
        return b / (delta0 + psi) + c / psi - 2.0 * m / (1.0 - delta0 - 2.0 * psi)

    f_lo, f_hi = score(lo), score(hi)
    if f_lo * f_hi > 0:  # no sign change → constrained optimum at a boundary
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _tango_z(b: int, c: int, n: int, delta0: float) -> float:
    """Tango score statistic for H0: paired difference (p_router - p_baseline) = delta0.

    Reduces to the classic McNemar z = (b-c)/sqrt(b+c) at delta0 = 0.
    """
    psi = _constrained_p21(b, c, n - b - c, delta0)
    var = n * (2.0 * psi + delta0 * (1.0 - delta0))
    if var <= 0.0:
        return 0.0
    return ((b - c) - n * delta0) / math.sqrt(var)


def mcnemar_noninferiority(
    router_pass: dict[str, int],
    baseline_pass: dict[str, int],
    margin: float = 0.05,
    alpha: float = 0.05,
) -> McNemarResult:
    """Test H0: Q_router <= Q_F - margin vs H1: Q_router > Q_F - margin (paired)."""
    # Depends only on the discordant tasks b (router pass, baseline fail) and c (router
    # fail, baseline pass) via the small-sample Tango score at the boundary delta0 =
    # -margin. ``non_inferior`` when the router is credibly within the margin,
    # ``inferior`` when credibly beyond it, else ``inconclusive`` (incl. no discordants).
    b = sum(1 for t in router_pass if router_pass[t] and not baseline_pass.get(t, 0))
    c = sum(1 for t in router_pass if not router_pass[t] and baseline_pass.get(t, 0))
    n = len(router_pass)
    if b + c == 0 or n == 0:
        return McNemarResult(b, c, 0.0, 1.0, "inconclusive")
    z = _tango_z(b, c, n, -margin)
    p_value = 1.0 - _NORMAL.cdf(z)  # one-sided: evidence router beats the -margin bound
    z_crit = _NORMAL.inv_cdf(1.0 - alpha)
    if z > z_crit:
        decision = "non_inferior"
    elif z < -z_crit:
        decision = "inferior"
    else:
        decision = "inconclusive"
    return McNemarResult(b, c, z, p_value, decision)


# ---------------------------------------------------------------------------
# 5c. Anytime-valid betting confidence sequence on the paired difference.
# ---------------------------------------------------------------------------

_GRID_POINTS = 99
_BET_GAIN = 1.5  # predictable-bet gain; validity holds for ANY value, this tunes power.


@dataclass(frozen=True)
class SeqState:
    """Accumulated betting confidence sequence over the paired difference X = r - b.

    The public fields (e_value/decided/direction/ci_lo/ci_hi) are the gate signal; the
    remaining fields carry the O(1)-update running state (per-grid hedged capital).
    """

    e_value: float
    decided: bool
    direction: str | None
    ci_lo: float
    ci_hi: float
    n: int = 0
    sum_x: float = 0.0
    grid: tuple[float, ...] = field(default_factory=tuple)
    cap_plus: tuple[float, ...] = field(default_factory=tuple)
    cap_minus: tuple[float, ...] = field(default_factory=tuple)


def _make_grid(margin: float) -> tuple[float, ...]:
    """Candidate paired-mean grid on (-1, 1), always including the boundary -margin."""
    step = 1.96 / (_GRID_POINTS - 1)
    pts = {round(-0.98 + i * step, 6) for i in range(_GRID_POINTS)}
    pts.add(-margin)
    return tuple(sorted(pts))


def _bet(mu_hat: float, m: float) -> tuple[float, float]:
    """Predictable (lambda_plus, lambda_minus) for candidate mean m, truncated safe.

    Magnitude follows the running mean estimate toward m; truncated at half the maximal
    capital-preserving bet (Waudby-Smith--Ramdas c=0.5) so 1 + lambda*(x-m) >= 0.
    """
    raw = _BET_GAIN * (mu_hat - m)
    lam_plus = min(max(raw, 0.0), 0.5 / (1.0 + m))
    lam_minus = max(min(raw, 0.0), -0.5 / (1.0 - m))
    return lam_plus, lam_minus


def update_confidence_sequence(
    prev: SeqState | None,
    router_pass_i: int,
    baseline_pass_i: int,
    margin: float,
    alpha: float = 0.05,
) -> SeqState:
    """Fold one newly-observed paired task into the anytime-valid confidence sequence."""
    # Hedged betting capital per candidate mean; the gate is decided the instant the
    # capital against H0: mean = -margin crosses 1/alpha (Ville's inequality bounds the
    # ever-crossing probability by alpha under the null, for any bet). ``direction`` is
    # the side the running mean sits on.
    x = float(router_pass_i - baseline_pass_i)  # in {-1, 0, 1}
    if prev is None:
        grid = _make_grid(margin)
        cap_plus = tuple(1.0 for _ in grid)
        cap_minus = tuple(1.0 for _ in grid)
        n, sum_x = 0, 0.0
    else:
        grid, cap_plus, cap_minus = prev.grid, prev.cap_plus, prev.cap_minus
        n, sum_x = prev.n, prev.sum_x
    mu_hat = sum_x / n if n > 0 else 0.0  # predictable: uses data through t-1 only

    new_plus, new_minus, hedged = [], [], []
    for m, kp, km in zip(grid, cap_plus, cap_minus, strict=True):
        lam_plus, lam_minus = _bet(mu_hat, m)
        kp2 = kp * (1.0 + lam_plus * (x - m))
        km2 = km * (1.0 + lam_minus * (x - m))
        new_plus.append(kp2)
        new_minus.append(km2)
        hedged.append(0.5 * (kp2 + km2))

    threshold = 1.0 / alpha
    live = [grid[i] for i, h in enumerate(hedged) if h < threshold]
    new_n = n + 1
    new_mean = (sum_x + x) / new_n
    ci_lo = min(live) if live else new_mean
    ci_hi = max(live) if live else new_mean
    margin_idx = grid.index(-margin)
    e_value = hedged[margin_idx]
    decided = e_value >= threshold
    direction = None
    if decided:
        direction = "router_wins" if new_mean > -margin else "no_win"
    return SeqState(
        e_value=e_value,
        decided=decided,
        direction=direction,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n=new_n,
        sum_x=sum_x + x,
        grid=grid,
        cap_plus=tuple(new_plus),
        cap_minus=tuple(new_minus),
    )
