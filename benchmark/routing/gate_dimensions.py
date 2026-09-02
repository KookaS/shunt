"""The pre-registered MULTI-DIMENSIONAL make-or-break kill gate."""

# A hard quality gate, then tolerance-aware Pareto dominance over the four distinct things a
# user pays, against TWO baselines — fixed-frontier-with-caching AND a zero-ML constant policy
# running the same handoff ladder.

# WHY THIS MODULE EXISTS, AND WHAT IT DELIBERATELY DOES NOT DO.
#
# The old gate judged on exactly two things: paired quality non-inferiority at a 5pp margin,
# and an aggregate cache-aware cost ratio < 1 (`benchmark/runner/kill_gate.py:462
# decide_verdict`, `benchmark/routing/online_kill_gate.py:135 decide_online_verdict`). Two
# defects follow from that shape and neither is fixable by moving a threshold.
#
#   (1) IT CANNOT SEE THE AXIS THE ROUTER WINS OR LOSES ON. `summary._dimension_columns`
#       publishes `sessions_mean`, `sessions_p95`, `cost_cv` and the call/token totals into the
#       tracked `benchmark/routing/reports/strategy_summary.csv`, and NO verdict reads them.
#       Measured on that table the shipped default burns 2.217 sessions per task against 1.0 for
#       a single-shot arm, with a p95 tail of 7. A gate blind to that records a router that
#       bought its saving with the user's afternoon as a success, and a router that saved the
#       afternoon at equal money as a failure. Both directions are wrong, and the second is the
#       one that would have been quietly recorded here.
#
#   (2) IT HAS ONLY ONE BASELINE, AND NOT THE DANGEROUS ONE. Scrouting (arXiv:2608.04804) ran
#       the ablation on 266 SWE-bench Pro tasks: its router scored 59.77% at $0.230/solve, and
#       ALWAYS routing to one cheap strong model with the SAME handoff got 159 solves at
#       $0.227/solve. The authors' conclusion — "the handoff rather than the routing decision
#       carries the result" — is the cheapest possible falsifier of this entire project, and the
#       old gate could not express it. The constant-policy arm below is that falsifier.
#
# WHAT IT DOES NOT DO. It never reprices anything. Every number it reads is the cost RECORDED
# at collection time, straight out of the tracked strategy table; a price-sheet refresh must
# never be able to flip a pre-registered verdict, so this module imports no pricing module and
# takes no price argument. It also does not fix the separately-recorded adjudicability defect
# — imputed cells and the coverage floor are upstream of it. It INHERITS that defect instead:
# the coverage precondition below reads the existing gate's verdict artifact and refuses to
# emit PASS or FAIL while the floor is tripped.

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmark.routing.metrics import Axis

# Verdict labels — deliberately the same three words the offline gate uses.
PASS: Final[str] = "PASS"
FAIL: Final[str] = "FAIL"
UNTESTED: Final[str] = "UNTESTED"

#: Non-inferiority margin on the quality axis, in PERCENTAGE POINTS. The same 0.05 the paired
#: online gate registered (`online_kill_gate.DEFAULT_MARGIN`), restated on the table's scale.
QUALITY_COLUMN: Final[str] = "AvgPerf%"
QUALITY_MARGIN_PP: Final[float] = 5.0

#: ONE relative tolerance for EVERY operational axis. Not a per-axis table: a per-axis
#: tolerance is one free knob per axis, and every knob is a place to launder a failure into a
#: pass after seeing the data. 5% mirrors the quality margin's spirit — a difference smaller
#: than this is called a tie in BOTH directions, so the tolerance can neither manufacture a win
#: nor excuse a loss.
OPERATIONAL_TOLERANCE: Final[float] = 0.05


@dataclass(frozen=True)
class GateAxis:
    """One operational axis of the gate: what it measures and why the user pays for it."""

    axis: Axis
    why: str


#: THE PRE-REGISTERED OPERATIONAL AXES — four, one per distinct thing the user pays.
#:
#: `TotalCalls` and `TotalOutTok` are published beside these and are deliberately NOT axes:
#: provider-side work is already priced into `TotalCost_cacheaware`, so admitting them would
#: count the same failure twice and make the conjunction below arbitrarily harder. `cost_p95`
#: is likewise omitted — it is the money tail, and `cost_cv` already carries dispersion.
OPERATIONAL_AXES: Final[tuple[GateAxis, ...]] = (
    GateAxis(Axis("TotalCost_cacheaware", "min"), "money, as recorded at collection time"),
    GateAxis(Axis("sessions_mean", "min"), "sessions the user waits through, on average"),
    GateAxis(Axis("sessions_p95", "min"), "the tail of that wait — the afternoon it can burn"),
    GateAxis(Axis("cost_cv", "min"), "how predictable the bill is"),
)

#: The arm under test. `kNN-semantic-cascade` is the nearest SELECTABLE kNN configuration —
#: the one an operator can actually turn on — not the single-shot `kNN-semantic` row the old
#: offline gate scores, which no `router.strategy` value serves.
DEFAULT_ROUTER_ARM: Final[str] = "kNN-semantic-cascade"

#: The two baselines. The gate must clear BOTH; passing one and losing the other is a FAIL.
#:   Always-Frontier  — fixed-frontier-with-caching, the original pre-registered baseline.
#:   Price-Cascade    — the zero-ML constant policy: cheapest-first, same handoff ladder, no
#:                      routing decision anywhere in it. Scrouting's falsifier.
DEFAULT_BASELINE_ARMS: Final[tuple[str, ...]] = ("Always-Frontier", "Price-Cascade")

#: Read as "the router is <verdict> than this baseline on this axis".
BETTER: Final[str] = "better"
TIE: Final[str] = "tie"
WORSE: Final[str] = "worse"


@dataclass(frozen=True)
class AxisComparison:
    """The router against one baseline on one axis, adjudicated against the tolerance."""

    column: str
    direction: str
    router: float
    baseline: float
    relative_delta: float
    result: str


@dataclass(frozen=True)
class BaselineComparison:
    """The router against one baseline: quality gate plus every operational axis."""

    baseline: str
    quality_router: float
    quality_baseline: float
    quality_delta_pp: float
    quality_non_inferior: bool
    axes: tuple[AxisComparison, ...]

    @property
    def worse_axes(self) -> tuple[str, ...]:
        """Axes on which the router is worse than this baseline beyond the tolerance."""
        return tuple(a.column for a in self.axes if a.result == WORSE)

    @property
    def better_axes(self) -> tuple[str, ...]:
        """Axes on which the router is better than this baseline beyond the tolerance."""
        return tuple(a.column for a in self.axes if a.result == BETTER)

    @property
    def dominates(self) -> bool:
        """Tolerance-aware Pareto dominance: nowhere worse, and somewhere strictly better."""
        # A router that merely TIES a zero-ML constant policy on every axis has earned
        # nothing, and the strict-win requirement is what says so. It is also what stops the
        # tolerance from becoming a free pass: widening it turns wins into ties too.
        return not self.worse_axes and bool(self.better_axes)

    @property
    def passes(self) -> bool:
        """This baseline is cleared: quality non-inferior AND tolerance-aware dominance."""
        return self.quality_non_inferior and self.dominates


@dataclass(frozen=True)
class GateVerdict:
    """The gate's answer, with every comparison that produced it."""

    verdict: str
    reason: str
    router: str
    comparisons: tuple[BaselineComparison, ...]
    blockers: tuple[str, ...] = ()
    #: The verdict this gate WOULD emit if every blocker were cleared. Published only
    #: alongside UNTESTED, and never substitutable for `verdict`: it exists so a reader can
    #: see WHICH way an unadjudicable gate is leaning without anyone having to re-derive it
    #: from the axis table by eye. Quoting it as the verdict is the failure it must not enable.
    provisional: str = ""
    provisional_reason: str = ""


def compare_axis(axis: Axis, router: float, baseline: float, tolerance: float) -> AxisComparison:
    """Adjudicate one axis into better / tie / worse against a RELATIVE tolerance."""
    # A zero baseline has no relative scale, so the comparison falls back to the sign of the
    # raw difference. It is never smoothed to "tie": a baseline that costs nothing and a
    # router that costs something is a real loss, and reporting it as a tie is how a
    # divide-by-zero becomes a pass.
    if baseline == 0.0:
        # No relative scale exists, so the SIGN of the raw difference decides, and the reported
        # relative delta carries that same sign. It used to be an unsigned +inf, which printed a
        # genuine improvement (a negative cost against a zero baseline) as `(+inf%)` in the
        # report — the machine verdict was right and the line a human reads was backwards.
        # `raw` is the SIGNED INFINITY, not the raw dollar difference: `tolerance` is a
        # fraction, so comparing an absolute difference against it mixes units and would call a
        # $0.01 router against a $0 baseline a tie.
        relative = 0.0 if router == 0.0 else math.copysign(float("inf"), router - baseline)
        raw = relative
    else:
        relative = (router - baseline) / abs(baseline)
        raw = relative
    if axis.direction == "min":
        better, worse = raw < -tolerance, raw > tolerance
    else:
        better, worse = raw > tolerance, raw < -tolerance
    return AxisComparison(
        column=axis.column,
        direction=axis.direction,
        router=router,
        baseline=baseline,
        relative_delta=relative,
        result=BETTER if better else WORSE if worse else TIE,
    )


def compare_to_baseline(
    router_row: dict[str, float],
    baseline_row: dict[str, float],
    baseline_name: str,
    *,
    axes: tuple[GateAxis, ...] = OPERATIONAL_AXES,
    tolerance: float = OPERATIONAL_TOLERANCE,
    quality_margin_pp: float = QUALITY_MARGIN_PP,
) -> BaselineComparison:
    """Run the quality gate and every operational axis against one baseline."""
    q_router = router_row[QUALITY_COLUMN]
    q_baseline = baseline_row[QUALITY_COLUMN]
    delta_pp = q_router - q_baseline
    return BaselineComparison(
        baseline=baseline_name,
        quality_router=q_router,
        quality_baseline=q_baseline,
        quality_delta_pp=delta_pp,
        # NON-INFERIORITY, not superiority: the router may be up to `quality_margin_pp` worse.
        # This margin is NEVER widened to let an operational win through — see the ADR. A
        # router inferior beyond it FAILS no matter what the other four axes say.
        quality_non_inferior=delta_pp >= -quality_margin_pp,
        axes=tuple(
            compare_axis(g.axis, router_row[g.axis.column], baseline_row[g.axis.column], tolerance)
            for g in axes
        ),
    )


def adjudicate(
    rows: dict[str, dict[str, float]],
    *,
    router: str = DEFAULT_ROUTER_ARM,
    baselines: tuple[str, ...] = DEFAULT_BASELINE_ARMS,
    axes: tuple[GateAxis, ...] = OPERATIONAL_AXES,
    tolerance: float = OPERATIONAL_TOLERANCE,
    quality_margin_pp: float = QUALITY_MARGIN_PP,
    blockers: tuple[str, ...] = (),
) -> GateVerdict:
    """The pre-registered verdict: PASS only when EVERY baseline is cleared."""
    # ``blockers`` are adjudicability failures found upstream (coverage floor tripped, instrument
    # inadmissible). Any blocker forces UNTESTED — never PASS, and never FAIL either: a gate that
    # cannot be adjudicated has no verdict, it has a gap.
    missing = _missing_values(rows, router, baselines, axes)
    all_blockers = tuple(blockers) + missing
    comparisons = (
        ()
        if missing
        else tuple(
            compare_to_baseline(
                rows[router],
                rows[b],
                b,
                axes=axes,
                tolerance=tolerance,
                quality_margin_pp=quality_margin_pp,
            )
            for b in baselines
        )
    )
    decided = _decide(router, comparisons)
    if all_blockers:
        return GateVerdict(
            verdict=UNTESTED,
            reason="not adjudicable: " + "; ".join(all_blockers),
            router=router,
            comparisons=comparisons,
            blockers=all_blockers,
            provisional=decided.verdict if comparisons else "",
            provisional_reason=decided.reason if comparisons else "",
        )
    return decided


def _decide(router: str, comparisons: tuple[BaselineComparison, ...]) -> GateVerdict:
    """PASS iff EVERY baseline is cleared. Never called with an unadjudicable input."""
    failed = [c for c in comparisons if not c.passes]
    if not failed:
        return GateVerdict(
            verdict=PASS,
            reason=(
                "quality non-inferior and tolerance-aware Pareto dominance over "
                + ", ".join(c.baseline for c in comparisons)
            ),
            router=router,
            comparisons=comparisons,
        )
    return GateVerdict(
        verdict=FAIL,
        reason="; ".join(_failure_reason(c) for c in failed),
        router=router,
        comparisons=comparisons,
    )


def _failure_reason(c: BaselineComparison) -> str:
    """One clause naming exactly why this baseline was not cleared."""
    if not c.quality_non_inferior:
        return (
            f"vs {c.baseline}: quality inferior by {-c.quality_delta_pp:.2f}pp, beyond the "
            f"{QUALITY_MARGIN_PP:.0f}pp margin"
        )
    if c.worse_axes:
        return f"vs {c.baseline}: worse on {', '.join(c.worse_axes)}"
    return f"vs {c.baseline}: no axis strictly better — ties a policy that does no routing"


def _missing_values(
    rows: dict[str, dict[str, float]],
    router: str,
    baselines: tuple[str, ...],
    axes: tuple[GateAxis, ...],
) -> tuple[str, ...]:
    """Arms or axis values the table does not carry — each one blocks adjudication."""
    # A missing value is NEVER defaulted to 0. On a min-axis a zero is un-dominatable by
    # construction, so a default would certify "measured nothing" as optimal — the same
    # refusal `metrics.pareto_front` makes by EXCLUDING rows with a None.
    problems: list[str] = []
    for arm in (router, *baselines):
        if arm not in rows:
            problems.append(f"arm {arm!r} is absent from the strategy table")
            continue
        for column in (QUALITY_COLUMN, *(g.axis.column for g in axes)):
            value = rows[arm].get(column)
            # `is None` is not enough on a dict built by a caller other than `load_arm_rows`
            # (a test, a control, a future consumer): a NaN reaches every comparison as False
            # and would read as a tie on an axis that was never measured.
            if value is None or not math.isfinite(value):
                problems.append(f"arm {arm!r} carries no usable value for {column!r}")
    return tuple(problems)


# ---------------------------------------------------------------------------
# Loading the recorded arms
# ---------------------------------------------------------------------------

SUMMARY_PATH: Final[Path] = Path("benchmark/routing/reports/strategy_summary.csv")
OFFLINE_VERDICT_PATH: Final[Path] = Path("benchmark/runner/kill_gate_verdict.json")
VERDICT_PATH: Final[Path] = Path("benchmark/runner/multidim_kill_gate_verdict.json")

_NEEDED_COLUMNS: Final[tuple[str, ...]] = (
    QUALITY_COLUMN,
    *(g.axis.column for g in OPERATIONAL_AXES),
)


def load_arm_rows(path: Path = SUMMARY_PATH) -> dict[str, dict[str, float]]:
    """Read the tracked strategy table into ``{arm: {column: value}}`` — recorded, not repriced."""
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            name = raw.get("strategy")
            if not name:
                continue
            rows[name] = {c: v for c in _NEEDED_COLUMNS if (v := _num(raw.get(c))) is not None}
    return rows


def _num(value: str | None) -> float | None:
    """A recorded numeric cell, or ``None`` when blank/absent/unparseable/non-finite."""
    # NON-FINITE IS MISSING, NOT A VALUE. `float("nan")` parses happily, and every `<`/`>` on a
    # NaN is False — so a NaN cell would sail through `compare_axis` as a TIE and let a router
    # PASS on an axis nobody measured. That is precisely the silent-accept this module refuses
    # for a blank cell, so the two are refused the same way. An infinity is rejected for the
    # mirror reason: it is un-dominatable in one direction and dominates in the other.
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def admissibility_blockers(
    path: Path = SUMMARY_PATH,
    arms: tuple[str, ...] = (DEFAULT_ROUTER_ARM, *DEFAULT_BASELINE_ARMS),
) -> tuple[str, ...]:
    """Arms whose own instrument-validity column does not say ADMISSIBLE."""
    # `summary` already stamps every row with the routing instrument's two-sided control
    # verdict. A gate quoting a row whose instrument never cleared its own positive control
    # would be publishing a verdict on an uncertified measurement.
    flags: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            if raw.get("strategy"):
                flags[raw["strategy"]] = (raw.get("instrument_admissible") or "").strip()
    return tuple(
        f"arm {arm!r} instrument_admissible={flags.get(arm, '<absent>')!r}"
        for arm in arms
        if flags.get(arm) != "True"
    )


@dataclass(frozen=True)
class CoverageCensus:
    """The offline gate's coverage precondition, RE-DERIVED from the census it summarises."""

    floor: float
    control: float
    router: float
    tripped: bool
    problems: tuple[str, ...]


def _census_number(value: object) -> float | None:
    """A census field that is a real number, or ``None`` — a bool is never a count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def coverage_census(payload: dict[str, object]) -> CoverageCensus:
    """Re-derive coverage from ``n`` and the measured-cell counts; never trust the stored flag."""
    # THE STORED BOOLEAN IS NOT AN INPUT. `coverage.tripped` used to be the whole precondition,
    # and it is one hand-editable character in a tracked JSON that no job regenerates: flipping
    # it moves this gate UNTESTED -> FAIL. So it is recomputed here from the two counts and the
    # floor — the quantities it claims to summarise — and the stored copies (`tripped`,
    # `control_coverage`, `router_coverage`) are checked AGAINST that derivation rather than
    # read. A hand edit to any of them is therefore a loud blocker, not a silent verdict move.
    coverage = payload.get("coverage")
    n = _census_number(payload.get("n"))
    if not isinstance(coverage, dict) or n is None or n <= 0:
        return CoverageCensus(
            0.0, 0.0, 0.0, True, ("the offline verdict carries no coverage census (n / coverage)",)
        )
    floor = _census_number(coverage.get("floor"))
    control_measured = _census_number(coverage.get("control_measured"))
    router_measured = _census_number(coverage.get("router_measured"))
    if floor is None or control_measured is None or router_measured is None:
        return CoverageCensus(
            0.0,
            0.0,
            0.0,
            True,
            ("the offline verdict's coverage census is incomplete (floor / measured counts)",),
        )
    control = control_measured / n
    router = router_measured / n
    tripped = min(control, router) < floor
    problems: list[str] = []
    for key, derived in (("control_coverage", control), ("router_coverage", router)):
        stored = _census_number(coverage.get(key))
        if stored is None or not math.isclose(stored, derived, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(
                f"offline verdict {key}={coverage.get(key)!r} disagrees with the census it "
                f"summarises ({derived!r} from {n:.0f} tasks) — the artifact does not match "
                f"its own numbers"
            )
    if bool(coverage.get("tripped")) != tripped:
        problems.append(
            f"offline verdict coverage.tripped={coverage.get('tripped')!r} contradicts its own "
            f"census (floor {floor}, control {control!r}, router {router!r}) — the stored flag "
            f"has been edited away from the data it summarises"
        )
    return CoverageCensus(floor, control, router, tripped, tuple(problems))


def coverage_blockers(path: Path = OFFLINE_VERDICT_PATH) -> tuple[str, ...]:
    """The pre-registered coverage floor, RE-DERIVED from the existing gate's verdict artifact.

    This gate does NOT re-adjudicate coverage and does not fix the imputation problem behind
    it — it inherits that gate's census, so the two can never disagree.
    """
    if not path.exists():
        return (f"no offline kill-gate verdict at {path} — coverage is unknown",)
    census = coverage_census(json.loads(path.read_text(encoding="utf-8")))
    tripped = (
        (
            "coverage floor tripped in the offline gate "
            f"(floor {census.floor}, control {census.control}, router {census.router})",
        )
        if census.tripped
        else ()
    )
    return census.problems + tripped


# ---------------------------------------------------------------------------
# The tracked verdict artifact
# ---------------------------------------------------------------------------


def verdict_payload(verdict: GateVerdict) -> dict[str, object]:
    """The deterministic, diffable record — a pure function of the recorded inputs."""
    return {
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "router_arm": verdict.router,
        "blockers": list(verdict.blockers),
        "provisional_verdict": verdict.provisional,
        "provisional_reason": verdict.provisional_reason,
        "quality_margin_pp": QUALITY_MARGIN_PP,
        "operational_tolerance": OPERATIONAL_TOLERANCE,
        "axes": [
            {"column": g.axis.column, "direction": g.axis.direction, "why": g.why}
            for g in OPERATIONAL_AXES
        ],
        "comparisons": [
            {
                "baseline": c.baseline,
                "quality_router": c.quality_router,
                "quality_baseline": c.quality_baseline,
                "quality_delta_pp": round(c.quality_delta_pp, 4),
                "quality_non_inferior": c.quality_non_inferior,
                "dominates": c.dominates,
                "passes": c.passes,
                "axes": [
                    {
                        "column": a.column,
                        "router": a.router,
                        "baseline": a.baseline,
                        "relative_delta": round(a.relative_delta, 6),
                        "result": a.result,
                    }
                    for a in c.axes
                ],
            }
            for c in verdict.comparisons
        ],
    }


def write_verdict(payload: dict[str, object], path: Path = VERDICT_PATH) -> None:
    """Serialize deterministically: sorted keys, trailing newline, no wall-clock state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rederive_verdict(
    summary: Path = SUMMARY_PATH,
    offline: Path = OFFLINE_VERDICT_PATH,
    *,
    router: str = DEFAULT_ROUTER_ARM,
    baselines: tuple[str, ...] = DEFAULT_BASELINE_ARMS,
    axes: tuple[GateAxis, ...] = OPERATIONAL_AXES,
) -> GateVerdict:
    """The whole gate over a recorded corpus. ``main``, the control and SH016 all share it."""
    # ONE assembly of the chain, used by the CLI and by `verdict_integrity_problems`. If the
    # integrity gate re-derived the verdict by its own route the two could drift, and a gate
    # that recomputes a DIFFERENT thing from the artifact it checks certifies nothing.
    blockers = coverage_blockers(offline) + admissibility_blockers(summary, (router, *baselines))
    return adjudicate(
        load_arm_rows(summary), router=router, baselines=baselines, axes=axes, blockers=blockers
    )


def _clip(value: object, limit: int = 160) -> str:
    """A field's value for a diff message, truncated — the whole axis table helps nobody."""
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "... (truncated)"


def verdict_integrity_problems(
    verdict: Path = VERDICT_PATH,
    summary: Path = SUMMARY_PATH,
    offline: Path = OFFLINE_VERDICT_PATH,
    *,
    router: str = DEFAULT_ROUTER_ARM,
    baselines: tuple[str, ...] = DEFAULT_BASELINE_ARMS,
) -> tuple[str, ...]:
    """Every field of the TRACKED verdict that disagrees with the corpus it claims to summarise."""
    # WHY A TRACKED VERDICT NEEDS ITS OWN GATE. Both kill-gate verdict artifacts are committed
    # files that no hook, job or pipeline stage regenerates, and the suite only ever asserted
    # the headline word. Editing `provisional_verdict` to PASS and fabricating a cost win in the
    # axis table therefore left the whole suite green — a falsified verdict is exactly the
    # artifact a reader quotes. The record is a PURE FUNCTION of the committed inputs
    # (`verdict_payload(adjudicate(load_arm_rows(...)))`, no clock, no path, no price sheet), so
    # the check is to recompute it and diff. Same shape as SH015's triage recomputation.
    if not summary.exists():
        return (f"no recorded strategy table at {summary} — the verdict cannot be re-derived",)
    if not verdict.exists():
        return (f"no tracked verdict artifact at {verdict}",)
    try:
        committed = json.loads(verdict.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (f"{verdict} is not readable JSON: {exc}",)
    if not isinstance(committed, dict):
        return (f"{verdict} is not a verdict record",)
    derived = verdict_payload(
        rederive_verdict(summary, offline, router=router, baselines=baselines)
    )
    problems = tuple(
        f"{verdict}: {key!r} says {_clip(committed.get(key))}, the committed corpus derives "
        f"{_clip(derived[key])}"
        for key in sorted(derived)
        if committed.get(key) != derived[key]
    )
    extra = tuple(
        f"{verdict}: {key!r} is not a field this gate emits — the artifact was hand-edited"
        for key in sorted(set(committed) - set(derived))
    )
    return problems + extra


def format_verdict(verdict: GateVerdict) -> str:
    """The human-readable gate report."""
    lines = [
        "=" * 78,
        f"MULTI-DIMENSIONAL KILL GATE — {verdict.router}",
        "=" * 78,
        f"Quality margin      : {QUALITY_MARGIN_PP:.0f}pp non-inferiority on {QUALITY_COLUMN}",
        f"Operational tolerance: {OPERATIONAL_TOLERANCE:.0%} relative, every axis",
        "",
    ]
    for c in verdict.comparisons:
        lines.append(f"vs {c.baseline}")
        lines.append(
            f"  quality      {c.quality_router:>10.2f} vs {c.quality_baseline:>10.2f}  "
            f"({c.quality_delta_pp:+.2f}pp) -> "
            f"{'non-inferior' if c.quality_non_inferior else 'INFERIOR'}"
        )
        for a in c.axes:
            lines.append(
                f"  {a.column:<22} {a.router:>12.4f} vs {a.baseline:>12.4f}  "
                f"({a.relative_delta:+.1%}) -> {a.result}"
            )
        lines.append(f"  => {'cleared' if c.passes else 'NOT cleared'}")
        lines.append("")
    lines.append(f"VERDICT: {verdict.verdict} — {verdict.reason}")
    if verdict.provisional:
        lines.append(
            f"  (provisional, NOT a verdict — what the gate would say once the blockers "
            f"above are cleared: {verdict.provisional} — {verdict.provisional_reason})"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    """CLI: adjudicate the recorded strategy table and write the tracked verdict artifact."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=str(SUMMARY_PATH), help="Recorded strategy table")
    ap.add_argument("--router", default=DEFAULT_ROUTER_ARM, help="Arm under test")
    ap.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="Baseline arm (repeatable). Default: the two pre-registered baselines.",
    )
    ap.add_argument("--verdict", default=str(VERDICT_PATH), help="Verdict artifact path")
    ap.add_argument(
        "--offline-verdict",
        default=str(OFFLINE_VERDICT_PATH),
        help="Offline gate verdict, read for the pre-registered coverage floor",
    )
    args = ap.parse_args()

    baselines = tuple(args.baseline) if args.baseline else DEFAULT_BASELINE_ARMS
    verdict = rederive_verdict(
        Path(args.summary),
        Path(args.offline_verdict),
        router=args.router,
        baselines=baselines,
    )
    print(format_verdict(verdict))
    write_verdict(verdict_payload(verdict), Path(args.verdict))
    print(f"Verdict artifact written to {args.verdict}")
    # EXIT 0 MEANS PASS AND NOTHING ELSE. An automated consumer reads 0 as "the gate cleared",
    # so UNTESTED shares FAIL's non-zero code rather than looking green — a gate that could not
    # be adjudicated has not been passed. Same convention as the older offline gate.
    raise SystemExit(0 if verdict.verdict == PASS else 1)


if __name__ == "__main__":
    main()
