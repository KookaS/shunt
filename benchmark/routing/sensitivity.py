#!/usr/bin/env python3
"""How much routing signal would this corpus have to carry before the null test could see it?

Sweeps a planted signal downward until the published test stops detecting it, and reports that
floor as a minimum detectable effect with an interval.
"""

# WHY A PASSED POSITIVE CONTROL IS NOT ENOUGH, AND WHY THIS MODULE EXISTS.
# `instrument_control` plants a signal that is lexically explicit and vastly larger than any
# plausible real routing signal, so clearing it licenses exactly one claim: the committed
# `NULL RESULT` is a null on an instrument PROVEN TO DETECT SOMETHING — at the strength probed.
# Whether that null means "there is nothing there" or "there is something, below our floor" is a
# different question, and only a sensitivity sweep answers it. A negative result quoted without
# its floor is the instrument reporting its own blind spot as a finding.
#
# WHAT IS VARIED, AND WHY THAT IS THE RIGHT KNOB. Strength is `rho`: the fraction of tasks whose
# outcome row is aligned with a direction in the REAL embedding space, the rest re-assigned at
# random. It is the right knob because it is the only one that maps onto the question "could a
# real routing signal have been detected". A real signal is not a louder word in the prompt — it
# is a partial regularity: SOME tasks have text that predicts whether the cheap model suffices,
# and the rest do not. `rho` is exactly that fraction, and it converts to a unit that is
# comparable across this whole workstream: a perfect reader of the planted signal separates
# cheap-sufficient from escalation-needed tasks at an AUROC of one half plus half of rho, so an
# MDE of rho = 0.8 is an MDE of AUROC 0.90 — directly comparable to the escalation
# detector's measured 0.576 and to published SOTA near 0.62. Lexical salience was rejected as the
# knob: it varies how loudly a signal is written, not how much of it exists, and a corpus whose
# real texts already resolve through a ~106-character label has no salience axis a real signal
# would move along. Embedding separability is not assumed either — it is swept as a SECOND axis
# (`GEOMETRIES`), because a floor measured under one geometry is a floor for that geometry only.
#
# WHAT IS HELD FIXED, AND WHY EVERY ONE OF THESE MATTERS.
#   * The corpus is the REAL one, at its real size and real repository mix. A floor computed on a
#     synthetic corpus is a floor for that corpus.
#   * The outcome ROWS are the real ones, re-assigned rather than invented. This is what keeps
#     the sweep comparable to the published null: the permutation band is a function of the row
#     MULTISET alone, and re-assignment leaves it untouched, so the bar the planted signal must
#     clear is the same null the real analysis quoted — same rows, same statistic, same
#     percentile (asserted in tests/test_sensitivity.py, TestTheNullIsTheOneThePublished-
#     AnalysisUses).
#   * The planted class is balanced WITHIN each repository, so repository identity carries no
#     information about it — the same discipline, and the same reason, as
#     `instrument_control`'s orthogonality construction. Without it the sweep would measure how
#     well the front end recovers repository names, which nobody is asking.
#   * Both splits are reported. The published split is ungrouped and leaks (a held-out task's own
#     repository siblings sit in its index); the repo-grouped split is what a deployed router
#     faces. The GAP between the two floors is itself the finding.

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.routing.scripts import knn_nulls

# Detection is "observed above the null band's 97.5th percentile", so a signal-free corpus trips
# it this often by construction. It is the chance level the two control legs are adjudicated on.
NOMINAL_FALSE_POSITIVE_RATE: Final[float] = 0.025

# The conventional power floor. An effect below the rho that reaches it is one this corpus cannot
# be relied on to see, which is the whole point of reporting a floor rather than a point.
DEFAULT_TARGET_POWER: Final[float] = 0.80

SPLITS: Final[tuple[str, ...]] = ("ungrouped", "repo-grouped")
RULES: Final[tuple[str, ...]] = ("fixed-k", "best-over-k")
GEOMETRIES: Final[tuple[str, ...]] = ("linear", "coherent")

_DEFAULT_RHO_GRID: Final[tuple[float, ...]] = tuple(round(0.05 * i, 2) for i in range(21))
_Z95: Final[float] = 1.959963984540054


@dataclass(frozen=True)
class Corpus:
    """The real routing corpus as the sweep needs it: geometry, repositories, outcome rows."""

    sims: np.ndarray
    emb: np.ndarray
    repos: tuple[str, ...]
    pass_mat: np.ndarray

    @property
    def n(self) -> int:
        return int(self.pass_mat.shape[0])

    @property
    def repo_index(self) -> dict[str, np.ndarray]:
        arr = np.array(self.repos)
        return {r: np.flatnonzero(arr == r) for r in sorted(set(self.repos))}

    @property
    def n_escalation_rows(self) -> int:
        """Rows the cheapest model fails — the class a router would have to escalate."""
        return int((self.pass_mat[:, 0] == 0).sum())


@dataclass(frozen=True)
class SweepSpec:
    """One fully-specified sensitivity question. Every field changes what the floor means."""

    k: int
    threshold: float
    ks: tuple[int, ...]
    split: str = "ungrouped"
    rule: str = "fixed-k"
    geometry: str = "linear"
    rho_grid: tuple[float, ...] = _DEFAULT_RHO_GRID
    n_trials: int = 200
    n_perm: int = knn_nulls.DEFAULT_PERMUTATIONS
    target_power: float = DEFAULT_TARGET_POWER
    seed: int = 0

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}; got {self.split!r}")
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}; got {self.rule!r}")
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}; got {self.geometry!r}")
        if self.k not in self.ks:
            raise ValueError(f"the fixed-k rule's k={self.k} must be one of ks={self.ks}")


@dataclass(frozen=True)
class PowerPoint:
    """Detection rate at one planted strength, with the binomial interval that earned it."""

    rho: float
    auroc: float
    power: float
    ci_lo: float
    ci_hi: float
    n_trials: int


@dataclass(frozen=True)
class MinimumDetectableEffect:
    """The floor: the smallest planted effect this corpus and pipeline reliably flag."""

    # REQUIRED and FIRST, with no default, for the same reason `knn_nulls.TransferCurve` and
    # `summary.StrategyTable` carry one. An MDE read off a pipeline that cannot detect even a
    # MAXIMAL planted signal is not a floor, it is a description of a dead instrument — and the
    # two ends of the swept curve ARE a positive control and a destroyed-signal null, so the
    # shared adjudicator settles it rather than a second, local rule.
    admissibility: AdmissibilityResult
    spec: SweepSpec
    n_tasks: int
    curve: tuple[PowerPoint, ...]
    rho: float | None
    rho_lo: float | None
    rho_hi: float | None
    numbers: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def auroc_of(rho: float | None) -> float | None:
        """A perfect reader of a signal planted at ``rho`` separates the classes at this AUROC."""
        return None if rho is None else 0.5 + rho / 2.0

    @property
    def headline(self) -> str:
        """One line an emitted verdict can carry verbatim."""
        s = self.spec
        where = f"{s.split} / {s.rule} (k={s.k}) / {s.geometry} geometry, n={self.n_tasks}"
        if self.rho is None:
            top = self.curve[-1]
            return (
                f"MDE UNATTAINABLE [{where}]: even a maximal planted signal (rho="
                f"{top.rho:.2f}, AUROC {top.auroc:.3f}) is detected only "
                f"{top.power:.0%} of the time, below the {s.target_power:.0%} power floor. "
                f"A NULL RESULT from this configuration is a coverage-gap, not a falsification."
            )
        return (
            f"MDE [{where}]: rho={self.rho:.2f} "
            f"[{_fmt(self.rho_lo)}, {_fmt(self.rho_hi)}], i.e. a text-derived predictor of "
            f"AUROC {self.auroc_of(self.rho):.3f} "
            f"[{_fmt(self.auroc_of(self.rho_lo), 3)}, {_fmt(self.auroc_of(self.rho_hi), 3)}] "
            f"is the weakest routing signal detected at {s.target_power:.0%} power."
        )


def _fmt(value: float | None, digits: int = 2) -> str:
    return "unattained" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# The planted signal: repo-orthogonal by construction, strength-controlled by rho.
# ---------------------------------------------------------------------------


def _repo_quota(corpus: Corpus) -> dict[str, int]:
    """Per-repo class-E counts by largest remainder, summing EXACTLY to the real class size."""
    # Rounding each repo independently loses or gains a task, which would change the row multiset
    # and silently move the null band away from the published one.
    n_e = corpus.n_escalation_rows
    index = corpus.repo_index
    exact = {r: len(idx) * n_e / corpus.n for r, idx in index.items()}
    take = {r: int(math.floor(v)) for r, v in exact.items()}
    short = n_e - sum(take.values())
    for repo in sorted(exact, key=lambda r: -(exact[r] % 1.0))[:short]:
        take[repo] += 1
    return take


def _linear_classes(corpus: Corpus, quota: dict[str, int]) -> np.ndarray:
    """Class from the leading direction of the repo-residualised embeddings — a plain tell."""
    resid = np.array(corpus.emb, dtype=float, copy=True)
    for idx in corpus.repo_index.values():
        resid[idx] -= resid[idx].mean(axis=0)
    direction = np.linalg.svd(resid, full_matrices=False)[2][0]
    score = corpus.emb @ direction
    classes = np.zeros(corpus.n, dtype=int)
    for repo, idx in corpus.repo_index.items():
        classes[idx[np.argsort(-score[idx])][: quota[repo]]] = 1
    return classes


def _coherent_classes(corpus: Corpus, quota: dict[str, int], seed: int) -> np.ndarray:
    """The tightest repo-balanced cluster in the REAL similarity graph — best-case separability."""
    # Grown greedily by mean similarity to the set so far. This is the friendliest geometry a kNN
    # rule could be handed, so the floor measured under it is a LOWER BOUND on the floor: a real
    # signal shaped less conveniently needs to be stronger, never weaker.
    rng = np.random.default_rng(seed)
    room = dict(quota)
    repos = np.array(corpus.repos)
    eligible = np.flatnonzero([room[r] > 0 for r in repos])
    classes = np.zeros(corpus.n, dtype=int)
    start = int(rng.choice(eligible))
    classes[start] = 1
    room[repos[start]] -= 1
    for _ in range(sum(room.values())):
        affinity = corpus.sims[:, np.flatnonzero(classes == 1)].mean(axis=1)
        affinity[classes == 1] = -np.inf
        for repo, left in room.items():
            if left == 0:
                affinity[corpus.repo_index[repo]] = -np.inf
        nxt = int(np.argmax(affinity))
        classes[nxt] = 1
        room[repos[nxt]] -= 1
    return classes


def planted_classes(corpus: Corpus, geometry: str, seed: int = 0) -> np.ndarray:
    """The target class per task: 1 = "the cheapest model fails here", balanced within each repo."""
    quota = _repo_quota(corpus)
    if geometry == "linear":
        return _linear_classes(corpus, quota)
    return _coherent_classes(corpus, quota, seed)


def plant(corpus: Corpus, classes: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Re-assign the REAL outcome rows so a ``rho`` fraction of tasks follow ``classes``."""
    # Re-assignment, never fabrication: the returned matrix is a permutation of `corpus.pass_mat`'s
    # rows, so every model's marginal pass rate and the joint structure of a row survive exactly.
    # rho=0 is a uniform permutation (a draw from the published null); rho=1 is perfect alignment.
    n, n_e = corpus.n, corpus.n_escalation_rows
    rows_e = corpus.pass_mat[corpus.pass_mat[:, 0] == 0]
    rows_c = corpus.pass_mat[corpus.pass_mat[:, 0] == 1]
    assigned = np.full(n, -1)
    keep = rng.permutation(n)[: int(round(rho * n))]
    assigned[keep] = classes[keep]
    free = np.flatnonzero(assigned < 0)
    remaining = n_e - int((assigned == 1).sum())
    assigned[free] = rng.permutation(
        np.array([1] * remaining + [0] * (len(free) - remaining), dtype=int)
    )
    out = np.empty_like(corpus.pass_mat)
    out[assigned == 1] = rows_e[rng.permutation(n_e)]
    out[assigned == 0] = rows_c[rng.permutation(n - n_e)]
    return out


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUROC with tie correction — the portable unit the floor is reported in."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    for value in np.unique(score):
        tied = score == value
        ranks[tied] = ranks[tied].mean()
    pos = float(label.sum())
    neg = float(len(label) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


# ---------------------------------------------------------------------------
# The statistic — the published one, under either split.
# ---------------------------------------------------------------------------


def routed_scores(
    corpus: Corpus, pass_mat: np.ndarray, ks: Sequence[int], threshold: float, split: str
) -> np.ndarray:
    """Routed pass rate at each k: leave-one-TASK-out (published) or leave-one-REPO-out."""
    if split == "ungrouped":
        idx = np.arange(corpus.n)
        return np.array(
            [knn_nulls.routed_pass_rate(corpus.sims, pass_mat, idx, idx, k, threshold) for k in ks]
        )
    repos = np.array(corpus.repos)
    out = np.zeros(len(ks), dtype=float)
    for repo, idx in corpus.repo_index.items():
        outside = np.flatnonzero(repos != repo)
        for j, k in enumerate(ks):
            rate = knn_nulls.routed_pass_rate(corpus.sims, pass_mat, idx, outside, k, threshold)
            out[j] += rate * len(idx)
    return out / corpus.n


def _rule_scalar(scores: np.ndarray, spec: SweepSpec) -> float:
    return float(scores[spec.ks.index(spec.k)] if spec.rule == "fixed-k" else scores.max())


def detection_threshold(corpus: Corpus, spec: SweepSpec) -> float:
    """The bar the observation must clear: the null band's 97.5th percentile, computed ONCE.

    The same bar serves every ``rho``: the null depends on the row MULTISET, which planting
    preserves — so it is the published analysis's own null, not a sweep-local one.
    """
    # Rows are canonicalised before permuting so the bar is EXACTLY, not merely distributionally,
    # invariant to which rho produced the matrix in hand. Without it a planted matrix (a
    # permutation of the same rows) composes with the same RNG stream to a different realised
    # sample, and a sweep whose bar drifts with its own treatment is not measuring a floor.
    # `test_the_canonicalised_bar_agrees_with_the_published_permutation_null` ties this back to
    # `knn_nulls`' own draws, which the figures quote.
    base = corpus.pass_mat[np.lexsort(corpus.pass_mat.T[::-1])]
    rng = np.random.default_rng(spec.seed)
    draws = np.array(
        [
            _rule_scalar(
                routed_scores(
                    corpus,
                    base[rng.permutation(corpus.n)],
                    spec.ks,
                    spec.threshold,
                    spec.split,
                ),
                spec,
            )
            for _ in range(spec.n_perm)
        ]
    )
    return knn_nulls.band_of(draws).hi


# ---------------------------------------------------------------------------
# Power curve -> floor.
# ---------------------------------------------------------------------------


def _wilson(hits: int, n: int) -> tuple[float, float]:
    """Wilson score interval — behaves at the 0 and 1 ends, where a power curve lives."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1.0 + _Z95**2 / n
    centre = (p + _Z95**2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / n + _Z95**2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def power_curve(corpus: Corpus, spec: SweepSpec, bar: float) -> tuple[PowerPoint, ...]:
    """Detection rate against ``bar`` at every planted strength in the grid."""
    classes = planted_classes(corpus, spec.geometry, spec.seed)
    points: list[PowerPoint] = []
    for rho in spec.rho_grid:
        hits = 0
        aucs: list[float] = []
        for trial in range(spec.n_trials):
            rng = np.random.default_rng((spec.seed + 1) * 1_000_003 + trial)
            planted = plant(corpus, classes, rho, rng)
            aucs.append(auroc(classes.astype(float), (planted[:, 0] == 0).astype(int)))
            scores = routed_scores(corpus, planted, spec.ks, spec.threshold, spec.split)
            hits += int(_rule_scalar(scores, spec) > bar)
        lo, hi = _wilson(hits, spec.n_trials)
        points.append(
            PowerPoint(
                rho=rho,
                auroc=float(np.mean(aucs)),
                power=hits / spec.n_trials,
                ci_lo=lo,
                ci_hi=hi,
                n_trials=spec.n_trials,
            )
        )
    return tuple(points)


def _crossing(xs: Sequence[float], ys: Sequence[float], target: float) -> float | None:
    """First x where a monotone-ish y curve reaches ``target``, linearly interpolated."""
    for i in range(1, len(ys)):
        if ys[i] >= target:
            if ys[i] == ys[i - 1]:
                return float(xs[i])
            frac = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1]))
    return None


def minimum_detectable_effect(corpus: Corpus, spec: SweepSpec) -> MinimumDetectableEffect:
    """Sweep the planted strength down and report the floor, its interval, and its validity."""
    bar = detection_threshold(corpus, spec)
    curve = power_curve(corpus, spec, bar)
    rhos = [p.rho for p in curve]
    point = _crossing(rhos, [p.power for p in curve], spec.target_power)
    # The optimistic end uses the interval's UPPER edge (the smallest rho at which target power
    # cannot be ruled out) and the conservative end its LOWER edge (the smallest rho at which
    # target power is established). Both come from the same binomial uncertainty the trials buy.
    lo = _crossing(rhos, [p.ci_hi for p in curve], spec.target_power)
    hi = _crossing(rhos, [p.ci_lo for p in curve], spec.target_power)
    band = (
        _wilson(round(NOMINAL_FALSE_POSITIVE_RATE * spec.n_trials), spec.n_trials)[1]
        - NOMINAL_FALSE_POSITIVE_RATE
    )
    verdict = admissibility_verdict(
        curve[-1].power,
        curve[0].power,
        chance_level=NOMINAL_FALSE_POSITIVE_RATE,
        chance_band=band,
    )
    return MinimumDetectableEffect(
        admissibility=verdict,
        spec=spec,
        n_tasks=corpus.n,
        curve=curve,
        rho=point,
        rho_lo=lo,
        rho_hi=hi,
        numbers={
            "detection_bar": bar,
            "power_at_max_rho": curve[-1].power,
            "false_positive_rate": curve[0].power,
            "auroc": float("nan") if point is None else 0.5 + point / 2.0,
        },
    )


# ---------------------------------------------------------------------------
# CLI — prints; writes nothing. No committed artifact is derived from this.
# ---------------------------------------------------------------------------


def load_corpus(config_path: str = "benchmark/benchmark.yaml", matrix: str | None = None) -> Corpus:
    """The real scored matrix, embedded by the shipped embedder the figures use."""
    from benchmark import config
    from benchmark.routing import summary
    from benchmark.routing.scripts import viz_knn

    config.load(config_path)
    loaded = summary.load_scored_matrix(Path(matrix) if matrix else config.challenges_path())
    results = loaded.get("results", {})
    if not results:
        raise RuntimeError("results.csv holds no rows; collect the matrix before sweeping it.")
    task_ids = sorted(results)
    models = config.enabled_models() or list(loaded.get("models", {}).keys())
    pricing = config.enabled_pricing()
    by_price = sorted(models, key=lambda m: config.cost_per_1m(m, pricing))
    emb = viz_knn.build_task_embeddings(loaded, task_ids)
    pass_mat = np.array(
        [
            [1.0 if results[t].get(m, {}).get("pass", False) else 0.0 for m in by_price]
            for t in task_ids
        ]
    )
    return Corpus(
        sims=emb @ emb.T,
        emb=emb,
        repos=tuple(knn_nulls.repo_of(t) for t in task_ids),
        pass_mat=pass_mat,
    )


def _print(mde: MinimumDetectableEffect) -> None:
    print(mde.headline)
    print(f"  {mde.admissibility.reason}")
    print("  rho   AUROC   power  95% CI")
    for p in mde.curve:
        print(f"  {p.rho:.2f}  {p.auroc:.3f}  {p.power:.3f}  [{p.ci_lo:.3f}, {p.ci_hi:.3f}]")


def main() -> int:
    """Sweep every (split, rule, geometry) combination and print the floors."""
    from benchmark import config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="benchmark/benchmark.yaml")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--permutations", type=int, default=knn_nulls.DEFAULT_PERMUTATIONS)
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    ap.add_argument("--rule", choices=[*RULES, "all"], default="all")
    ap.add_argument("--geometry", choices=[*GEOMETRIES, "all"], default="all")
    args = ap.parse_args()

    corpus = load_corpus(args.config, args.matrix)
    params = config.knn_params()
    k = int(params.get("k", 20))
    threshold = float(params.get("success_rate_threshold", 0.6))
    candidates = (2, 5, 10, 20, 40, 80, corpus.n - 1, k)
    ks = tuple(sorted({j for j in candidates if 1 <= j <= corpus.n - 1}))
    print(
        f"corpus: {corpus.n} tasks, {len(set(corpus.repos))} repos, "
        f"{corpus.n_escalation_rows} rows the cheapest model fails; k={k} threshold={threshold}"
    )
    splits = SPLITS if args.split == "all" else (args.split,)
    rules = RULES if args.rule == "all" else (args.rule,)
    geometries = GEOMETRIES if args.geometry == "all" else (args.geometry,)
    for geometry in geometries:
        for split in splits:
            for rule in rules:
                spec = SweepSpec(
                    k=k,
                    threshold=threshold,
                    ks=ks,
                    split=split,
                    rule=rule,
                    geometry=geometry,
                    n_trials=args.trials,
                    n_perm=args.permutations,
                )
                print()
                _print(minimum_detectable_effect(corpus, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
