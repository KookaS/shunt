"""The session-cascade instrument's two-sided validity control — run BEFORE quoting its numbers.

Plants a known ladder depth per task, asks the ASSEMBLED replay to spend in proportion to it,
then destroys the depth signal and asks the same replay to collapse to chance.
"""

# WHAT THE INSTRUMENT IS, AND WHERE THE CONTROL ENTERS. The session-cascade instrument is
#
#     matrix["results"]  ->  SessionCascadeStrategy._run_task  ->  EscalationRunner.decide
#         ->  concrete effort/rank application  ->  per-task billed cost
#
# and the verdict it emits ("session cadence costs N x a within-task cascade") is read off the
# last stage. This control enters at `matrix["results"]`, the FRONT of that chain, and never
# touches an intermediate: everything from `measured_models_by_price` through the shared
# `EscalationRunner` to the billing loop is the code the published row runs.
#
# WHY THE STATISTIC IS COST-VS-DEPTH AND NOT PASS RATE. A cascade walks its whole ladder, so on
# any corpus where some rung solves the task it eventually passes: pass rate is ~1.0 whatever the
# labels say, and its shuffled-label null is ~1.0 too, which would certify a frozen ladder. The
# quantity that actually depends on the escalation machinery working is WHERE the ladder stops,
# and cost is the published projection of that. So the score is the correlation between each
# task's replayed cost and the rung the corpus planted as its solution.
#
# AND WHY IT IS SCORED ONLY ON THE TASKS THAT REQUIRE AN ESCALATION. Depth-0 tasks are solved by
# the base pick, so an instrument that never escalates at all still separates them from the rest
# and still scores well above chance — measured, not assumed: a frozen-ladder mutant scored
# r=+0.77 on the whole corpus, and an earlier version of this gate certified it. The statistic
# therefore conditions on planted depth >= 1, the subset where the verdict's claim actually lives.
# There a ladder that never climbs bills every task the same, and so does one that jumps straight
# to the top: both collapse to r=0 and FAIL the positive leg. Only a ladder that climbs rung by
# rung, stops on a verified pass, and re-serves its floor across sessions tracks the planted
# depth. Depth-0 tasks stay in the CORPUS — they exercise the pass-retires-the-ladder rule and
# the hop-depth assertion below — they are just not part of the score.
#
# WHY THE NULL IS NOT VACUOUS. Permuting the depth labels reassigns WHICH task each solution rung
# belongs to while leaving the marginal distribution of rungs untouched, so total cost is nearly
# invariant — the null destroys only the per-task alignment, which is exactly the signal the
# positive leg claims. A replay that scored high here would be reading the label off something
# other than the corpus it was handed.
#
# A FAILURE HERE IMPLICATES: the ladder order, the recurrence counter, the effort/rank
# application, the rank floor's persistence, the pass-retires-the-ladder rule and the billing. It
# does NOT implicate the real corpus (whether SWE-bench tasks HAVE a ladder structure is the
# question the instrument exists to ask), the imputation layer, or the arm-level coverage gap.

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Final

import numpy as np

from benchmark.admissibility import AdmissibilityResult, admissibility_verdict
from benchmark.routing.scripts import knn_nulls
from benchmark.routing.strategies import Strategy
from benchmark.routing.strategies.knn_session_cascade import kNNSessionCascadeStrategy
from benchmark.routing.strategies.session_cascade import SessionCascadeStrategy
from shunt.router.escalation import EscalationConfig, next_rung_rank

# A correlation's value under no signal. Analytic, not borrowed from the pipeline's own null:
# centring the band on the permutation MEAN would let a leak move the observation and the band
# together and still read "at chance".
CHANCE_LEVEL: Final[float] = 0.0

# The planted ladder: four price-ascending arms, geometric so a rung's cost is unmistakable.
# Deliberately NOT registry models — the strategy's ladder comes from `matrix["models"]` pricing,
# so borrowing real names would couple the control to pricing config it does not test, and the
# empty arm ladders keep the effort rung (a separate, coverage-starved question) out of the gate.
CONTROL_MODELS: Final[tuple[str, ...]] = ("ctl-a", "ctl-b", "ctl-c", "ctl-d")
_PRICES: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0)
_COSTS: Final[tuple[float, ...]] = (0.01, 0.04, 0.16, 0.64)

# Tasks per planted depth. Balanced across depths so no depth dominates the correlation.
_PER_DEPTH: Final[int] = 40


# The ladders this control actually certifies. `rank_only` and nothing else: the planted corpus
# gives every model ONE implicit arm, so the effort rung is never exercised and never scored here.
CERTIFIED_LADDERS: Final[frozenset[str]] = frozenset({"rank_only"})


def assert_ladder_quotable(ladder: str) -> None:
    """Refuse to produce a Session-Cascade row at a ladder this control does not certify."""
    # WHY A BLOCK AND NOT AN EXTENDED CONTROL. The control's corpus is synthetic, so planting arm
    # ladders and certifying `effort_then_rank` MECHANICALLY is easy — and would certify the wrong
    # thing. The mechanism is not what fails: the CORPUS is. results.csv carries a >=3-arm
    # measurement for 54 of 770 (task, model) cells, so an `effort_then_rank` replay on real data
    # censors most of the suite and leaves a survivor set that is ~100% pass at a fraction of the
    # cost — a coverage artefact that would plot as a Pareto-dominant point. A green synthetic
    # control would license quoting exactly that artefact, which is the failure mode the
    # instrument-validity rule exists to stop. So the honest gate is the block, and it lifts when
    # per-arm coverage exists, not when the control is made cleverer.
    if ladder not in CERTIFIED_LADDERS:
        raise RuntimeError(
            f"session-cascade ladder {ladder!r} has no positive control: this gate certifies only "
            f"{sorted(CERTIFIED_LADDERS)}. Its planted corpus gives each model a single arm, so "
            f"the effort rung is never exercised, and the real corpus measures >=3 reasoning arms "
            f"on too few cells to score it without selection bias. No number computed at this "
            f"ladder may be published. Re-pin the ladder, or land per-arm coverage first."
        )


def build_control_matrix(depths: np.ndarray) -> dict:
    """A complete matrix whose task *i* is solved by every rung at or above ``depths[i]``.

    Monotone by construction — the same ladder axiom the real corpus is completed under.
    """
    results = {
        f"ctl-task-{i:04d}": {
            model: {"pass": bool(rank >= int(depth)), "cost": _COSTS[rank], "calls": 1}
            for rank, model in enumerate(CONTROL_MODELS)
        }
        for i, depth in enumerate(depths)
    }
    return {
        "tasks": {tid: {"description": tid} for tid in results},
        "results": results,
        "models": {
            m: {"input_price": _PRICES[i], "output_price": 0.0}
            for i, m in enumerate(CONTROL_MODELS)
        },
    }


class _FixedRungPicker(Strategy):
    """A selector planted on a KNOWN rung — the seeded leg's stand-in for the real kNN pick."""

    def __init__(self, rung: int) -> None:
        self._rung = rung

    @property
    def name(self) -> str:
        """Never plotted; the seeded leg only ever reads this strategy's billed cost."""
        return "control-fixed-rung"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        """The planted opening rung, identical for every task — the floor under test."""
        del task_id, task_meta, matrix
        return CONTROL_MODELS[self._rung]


def _strategy(
    escalate_after_n: int, rank_shortlist: int | None, initial_rung: int | None = None
) -> SessionCascadeStrategy:
    """The replay under test, with both config sources injected so it reads no repo state."""
    # `arm_ladders={}` is not a stub of the effort rung: with no ladder every model is served on
    # its one implicit arm, which is what a rank-only ladder does, and the effort rung is scored
    # nowhere in this gate because the corpus cannot support it (see the module header).
    # `initial_rung` selects the SEEDED leg: the same ladder, opened by a planted pick instead of
    # at the cheapest rung. Only the picker is replaced — every billing rule below it is shared.
    common = {
        "ladder": "rank_only",
        "escalate_after_n": escalate_after_n,
        "arm_results": {},
        "arm_ladders": {},
        "rank_shortlist": rank_shortlist,
    }
    if initial_rung is None:
        return SessionCascadeStrategy(**common)  # type: ignore[arg-type]
    return kNNSessionCascadeStrategy(picker=_FixedRungPicker(initial_rung), **common)  # type: ignore[arg-type]


def rung_sequence(rank_shortlist: int | None) -> list[int]:
    """The ranks this shortlist actually visits, climbing from the base rung to the ceiling."""
    # Derived from the SHARED rule rather than assumed, because the shortlist is exactly what makes
    # "the replay visits every rank" false: at shortlist 2 over four rungs the ladder goes 0,1,3 and
    # rank 2 is never bought. The hop-depth assertion below is scored against this, so it still
    # fails loudly on a frozen ladder while not failing on a ladder that is correctly jumping.
    shortlist = EscalationConfig().rank_shortlist if rank_shortlist is None else rank_shortlist
    top = len(CONTROL_MODELS) - 1
    visited = [0]
    while visited[-1] < top:
        visited.append(min(next_rung_rank(visited[-1], top, shortlist), top))
    return visited


def _expected_hop_depths(rank_shortlist: int | None) -> set[int]:
    """Hop counts a working replay must produce, one per planted depth in the control corpus."""
    # A task planted at depth d is solved by the first visited rung >= d, so its hop count is that
    # rung's 1-based position. Every planted depth contributes one, and the set is what the replay
    # must cover — no more (a shortlist skips rungs) and no less (a frozen ladder covers one).
    visited = rung_sequence(rank_shortlist)
    return {
        next(i for i, rank in enumerate(visited) if rank >= depth) + 1
        for depth in range(len(CONTROL_MODELS))
    }


def replay_costs(
    matrix: dict,
    escalate_after_n: int,
    rank_shortlist: int | None = None,
    initial_rung: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Per-task replayed cost and hop depth, in task order — the assembled instrument's output."""
    strategy = _strategy(escalate_after_n, rank_shortlist, initial_rung)
    costs: list[float] = []
    hops: list[int] = []
    for tid in matrix["results"]:
        trace = strategy.trace(tid, matrix)
        costs.append(trace.cost)
        hops.append(trace.hops)
    return np.asarray(costs, dtype=float), hops


def _correlation(costs: np.ndarray, depths: np.ndarray, min_depth: int = 1) -> float:
    """Pearson r on the escalation-requiring subset (depth >= min_depth), else 0.0 if flat."""
    # The subset is chosen from the PLANTED depths, never from the replay's own output, so the
    # positive and null legs are scored over exactly the same task indices. The seeded leg raises
    # `min_depth` to its floor: below the floor the router never routes, so those tasks carry no
    # depth signal by construction and scoring them would credit the instrument for a constant.
    keep = depths >= min_depth
    x = np.asarray(costs, dtype=float)[keep]
    y = depths.astype(float)[keep]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _assert_ladder_is_exercised(hops: list[int], rank_shortlist: int | None) -> None:
    """Fail loudly if the replay never actually escalated — the bug class this gate exists for."""
    # A state-machine test on this same subsystem once passed against a deliberately reintroduced
    # bug because its sequences never assembled the precondition the assertion needed. So the
    # control asserts the PATH, not just the score: the planted corpus has a task at every depth,
    # therefore a working replay must occupy every hop count the shortlist geometry predicts.
    depths_reached = set(hops)
    expected = _expected_hop_depths(rank_shortlist)
    if not expected <= depths_reached:
        raise RuntimeError(
            f"the session-cascade control never exercised the full ladder: hop depths observed "
            f"{sorted(depths_reached)}, expected every depth in {sorted(expected)} for rung "
            f"sequence {rung_sequence(rank_shortlist)}. The replay is not escalating, so no "
            f"verdict from it may be trusted."
        )


def run_control(
    *,
    escalate_after_n: int = 2,
    n_perm: int = knn_nulls.DEFAULT_PERMUTATIONS,
    seed: int = 0,
    rank_shortlist: int | None = None,
) -> AdmissibilityResult:
    """Run both legs through the assembled replay and adjudicate.

    ``escalate_after_n`` and ``rank_shortlist`` MUST be the values the certified row used.
    """
    depths = np.repeat(np.arange(len(CONTROL_MODELS)), _PER_DEPTH)
    positive_costs, hops = replay_costs(
        build_control_matrix(depths), escalate_after_n, rank_shortlist
    )
    _assert_ladder_is_exercised(hops, rank_shortlist)
    positive = _correlation(positive_costs, depths)

    # The band is the spread of the SAME statistic when the depth labels are permuted: the
    # empirical answer to "how far from 0 does this replay land when the corpus says nothing
    # about which task is hard?".
    rng = np.random.default_rng(seed + 1)
    draws = np.array(
        [
            _correlation(
                replay_costs(
                    build_control_matrix(rng.permutation(depths)), escalate_after_n, rank_shortlist
                )[0],
                depths,
            )
            for _ in range(n_perm)
        ]
    )
    band = knn_nulls.band_of(draws)
    half_width = (band.hi - band.lo) / 2.0

    # The destroyed-signal leg reads the MEDIAN of a few fresh draws from a disjoint seed. This
    # softens nothing — the boundary is still chance ± the SINGLE-draw band — it only stops an
    # honest instrument reading INADMISSIBLE once in twenty runs on sampling noise alone.
    shuffle_rng = np.random.default_rng(seed + 10_007)
    shuffled = float(
        np.median(
            [
                _correlation(
                    replay_costs(
                        build_control_matrix(shuffle_rng.permutation(depths)),
                        escalate_after_n,
                        rank_shortlist,
                    )[0],
                    depths,
                )
                for _ in range(5)
            ]
        )
    )

    verdict = admissibility_verdict(
        positive, shuffled, chance_level=CHANCE_LEVEL, chance_band=half_width
    )
    return replace(
        verdict,
        numbers={
            **verdict.numbers,
            "n_tasks": int(depths.size),
            "escalate_after_n": escalate_after_n,
            "rank_shortlist": (
                EscalationConfig().rank_shortlist if rank_shortlist is None else rank_shortlist
            ),
            "n_perm": n_perm,
            "rungs": list(CONTROL_MODELS),
            "rung_sequence": rung_sequence(rank_shortlist),
            "hop_depths_observed": sorted(set(hops)),
            "hop_depths_required": sorted(_expected_hop_depths(rank_shortlist)),
            "null_mean": band.mean,
            "null_sd": band.sd,
            "null_lo": band.lo,
            "null_hi": band.hi,
        },
    )


# The rung the SEEDED leg plants its pick on. 1, not 0: it must be above the bottom (or the leg
# is the base control again) and below the top (or nothing can escalate above it). With four rungs
# that leaves exactly one task depth BELOW the floor, which is the population leg (b) scores.
SEEDED_FLOOR: Final[int] = 1


def _floor_billing_failure(costs: np.ndarray, depths: np.ndarray, floor: int) -> str | None:
    """The reason the seeded floor is not honoured, or ``None`` when it is.

    Returns rather than raises: this is the ONE constraint the seeded leg carries that the base
    leg cannot, so it has to reach the leg's ``admissible`` verdict, not bypass it as a crash.
    """
    # THE FAILURE MODE UNIQUE TO THIS ROW. A router that opens on the kNN pick never routes BELOW
    # that pick, so a task the cheapest rung would have solved is still paid for at the picked
    # rung. An implementation that quietly clamped the floor back to 0 would look healthier than
    # the shipped default is, and the correlation leg above could not see it: those tasks are
    # outside its subset by construction. So this is checked directly, on the exact cost.
    #
    # AND WHY THE VERDICT CANNOT REST ON THE CORRELATION ALONE. Measured, not assumed: seeding the
    # floor shifts every SCORED task's cost by the same constant (-0.02 here, because the scored
    # subset is depth >= floor and the floor only changes which rung opens, not which rung solves),
    # and Pearson r is shift-invariant — so the seeded leg's `positive_score` is bit-identical to
    # the base leg's (0.94491118252307 at the shipped knobs). A leg whose only statistic cannot
    # move independently of another leg certifies nothing on its own. This check is what makes the
    # seeded leg falsifiable separately: it reads the below-floor population, which the
    # correlation excludes by construction and which the base leg bills differently.
    below = np.asarray(costs, dtype=float)[depths < floor]
    if below.size == 0:
        return (
            f"seeded control planted floor {floor} with no task below it, so nothing constrains it"
        )
    expected = _COSTS[floor]
    if not bool(np.allclose(below, expected)):
        return (
            f"the seeded session-cascade control billed a below-floor task at something other "
            f"than its floor: expected every depth<{floor} task to cost {expected} (rung "
            f"{CONTROL_MODELS[floor]}), observed {sorted(set(below.tolist()))}. The seeded rank "
            f"floor is not being honoured, so the shipped-default row understates its own cost."
        )
    return None


def run_seeded_control(
    *,
    escalate_after_n: int = 2,
    n_perm: int = knn_nulls.DEFAULT_PERMUTATIONS,
    seed: int = 0,
    rank_shortlist: int | None = None,
    floor: int = SEEDED_FLOOR,
) -> AdmissibilityResult:
    """The kNN-seeded leg: the same replay opened on a planted rung instead of the cheapest."""
    depths = np.repeat(np.arange(len(CONTROL_MODELS)), _PER_DEPTH)
    costs, hops = replay_costs(
        build_control_matrix(depths), escalate_after_n, rank_shortlist, initial_rung=floor
    )
    floor_failure = _floor_billing_failure(costs, depths, floor)
    positive = _correlation(costs, depths, min_depth=floor)

    rng = np.random.default_rng(seed + 2)
    draws = np.array(
        [
            _correlation(
                replay_costs(
                    build_control_matrix(rng.permutation(depths)),
                    escalate_after_n,
                    rank_shortlist,
                    initial_rung=floor,
                )[0],
                depths,
                min_depth=floor,
            )
            for _ in range(n_perm)
        ]
    )
    band = knn_nulls.band_of(draws)
    shuffle_rng = np.random.default_rng(seed + 20_011)
    shuffled = float(
        np.median(
            [
                _correlation(
                    replay_costs(
                        build_control_matrix(shuffle_rng.permutation(depths)),
                        escalate_after_n,
                        rank_shortlist,
                        initial_rung=floor,
                    )[0],
                    depths,
                    min_depth=floor,
                )
                for _ in range(5)
            ]
        )
    )
    verdict = admissibility_verdict(
        positive, shuffled, chance_level=CHANCE_LEVEL, chance_band=(band.hi - band.lo) / 2.0
    )
    if floor_failure is not None:
        # The floor is a precondition of the row, not a footnote: fail the leg, do not crash the
        # process, so `main()`'s `all(v.admissible ...)` sees an INADMISSIBLE seeded leg beside an
        # unaffected base leg — which is what "this row is not quotable" has to look like.
        verdict = replace(
            verdict,
            admissible=False,
            reason=f"INADMISSIBLE: {floor_failure}",
        )
    return replace(
        verdict,
        numbers={
            **verdict.numbers,
            "n_tasks": int(depths.size),
            "seeded_floor": floor,
            "floor_billing": "honoured" if floor_failure is None else "VIOLATED",
            "floor_rung": CONTROL_MODELS[floor],
            "below_floor_billed": _COSTS[floor],
            "n_scored": int((depths >= floor).sum()),
            "escalate_after_n": escalate_after_n,
            "n_perm": n_perm,
            "hop_depths_observed": sorted(set(hops)),
            "null_lo": band.lo,
            "null_hi": band.hi,
        },
    )


def main() -> int:
    """Run the gate at the configured escalation policy and print the verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="benchmark/benchmark.yaml")
    ap.add_argument("--escalate-after-n", type=int, default=None)
    ap.add_argument("--rank-shortlist", type=int, default=None)
    ap.add_argument("--permutations", type=int, default=knn_nulls.DEFAULT_PERMUTATIONS)
    args = ap.parse_args()

    from benchmark import config

    config.load(args.config)
    knobs = config.strategies().get("session_cascade", {})
    n = (
        args.escalate_after_n
        if args.escalate_after_n is not None
        else int(knobs.get("escalate_after_n", 2))
    )
    shortlist = (
        args.rank_shortlist if args.rank_shortlist is not None else knobs.get("rank_shortlist")
    )
    kwargs = {
        "escalate_after_n": n,
        "n_perm": args.permutations,
        "rank_shortlist": None if shortlist is None else int(shortlist),
    }
    # BOTH session-cadence rows are gated here, because they are one instrument opened at two
    # rungs: `Session-Cascade` starts at the cheapest model, `kNN-semantic-cascade` at the
    # kNN pick. A
    # verdict may be quoted for a row only if ITS leg is admissible — and note what "its leg"
    # buys: the two legs' correlation statistic is shift-invariant and therefore IDENTICAL by
    # construction, so the seeded leg's independent content is its floor-billing check alone
    # (`floor_billing` in its numbers). Read the seeded row as "the shared ladder recovers depth,
    # AND the seeded floor is billed", never as a second, independent recovery of the signal.
    legs = {
        "session-cascade": run_control(**kwargs),
        "knn-cascade (seeded floor)": run_seeded_control(**kwargs),
    }
    for label, verdict in legs.items():
        print(f"[{label}] {verdict.reason}")
        for key, value in verdict.numbers.items():
            print(f"  {key}: {value}")
    return 0 if all(v.admissible for v in legs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
