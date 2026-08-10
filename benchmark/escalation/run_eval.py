"""End-to-end escalation eval: policy replay at trajectory level + a causal prefix risk model."""

# TWO INDEPENDENT BLOCKS, because they answer different questions and only one of them is a model:
#
#   POLICY   — replay the SHIPPED recurrence policy over each trajectory and ask whether firing
#              predicts terminal failure. One row per trajectory (not per prefix), `n_escalated`
#              per cell (not summed over the sweep), Wilson intervals, permutation null.
#   PREFIX   — fit a continuous risk score from prefix-only features at fixed decision depths,
#              grouped-CV by challenge, and report skill INCREMENTAL to the router's t=0 prior.
#
# The status gate is the permutation null, not `auprc > prevalence`. The old strict-inequality gate
# passed a +0.0008 excess against a null sd of 0.00055 and printed `status: OK` on a pure null.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib

from benchmark import plot_frame
from benchmark.escalation import (
    datasets,
    deployability,
    features,
    metrics,
    plots,
    policy_eval,
    prefix_eval,
    replay,
)
from benchmark.escalation.authenticity import errors, verify_trajectory

# The class is imported directly because `EvalReport.deployability` is the field name a reader
# looks for in the JSON, and a field cannot be annotated with the module it shadows.
from benchmark.escalation.deployability import Cadence, CountingMode, Deployability
from benchmark.escalation.ope import (
    PolicyValueEstimate,
    estimate_policy_value,
    rows_from_records,
)
from benchmark.escalation.schema import Trajectory, load_jsonl
from benchmark.escalation.session_eval import SessionCadenceReport, session_cadence
from benchmark.plot_frame import Annotations
from shunt.router.escalation import EscalationConfig
from shunt.router.policy import load_router_policy, packaged_policy_path

matplotlib.use("Agg")

if TYPE_CHECKING:
    from benchmark.escalation.features import ModelCoverage
    from benchmark.escalation.policy_eval import PolicyCell
    from benchmark.escalation.prefix_eval import DepthReport

logger = logging.getLogger(__name__)

_STATUS_OK = "OK"
# A cell that clears its null on a corpus that is NOT a deployable estimate is a real signal
# measured at a cadence production never runs. The badge must say that, not "OK": `status` is
# read before `reason`, so the scope note must live in the value itself.
_STATUS_OK_OFFLINE_ONLY = "OK_OFFLINE_ONLY"
_STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
_STATUS_NO_SKILL = "NO_SKILL"
# A corpus that fails its own Layer-1 authenticity recompute is not a corpus with a weak result;
# it is a corpus whose numbers describe something other than what the harness recorded. It gates
# ahead of every skill verdict.
_STATUS_UNVERIFIED = "AUTHENTICITY_FAILED"


def _shipped_escalation() -> tuple[int, int, str]:
    """The escalation knobs the product actually ships, read from the PACKAGED router.yaml."""
    # Read, never restated. A literal here is a second source of truth that drifts silently: this
    # module claimed `escalate_after_n=2` was "the shipped configuration" while saying nothing
    # about `stale_window`, and the swept grid pinned 5 against the shipped 10. The packaged path
    # is used explicitly rather than `load_router_policy()`, whose lookup would prefer a local
    # $SHUNT_CONFIG_DIR override — the claim is about what SHIPS, not what this box runs.
    escalation = load_router_policy(packaged_policy_path()).escalation
    return escalation.escalate_after_n, escalation.stale_window, escalation.ladder


# The configuration the product ships; the sweep must report it even when another cell scores
# better, so a quiet default change can never hide inside an "argmax" line.
SHIPPED_ESCALATE_AFTER_N, SHIPPED_STALE_WINDOW, SHIPPED_LADDER = _shipped_escalation()

# THE CANONICAL OPERATING POINT, and the only per-cell point any figure reads.
#
# Every knob the product sets is held at its shipped value; exactly one eval-only thing varies —
# how the recurrence counter treats the reproduction phase. That is what makes it comparable: the
# as-shipped and edit-gated rows of the same (n, stale_window) differ in ONE way, so the gap
# between them is attributable. The figures used to read three different points — the shipped
# default on the outcome bars, an argmax over 60 swept cells on the confusion and null figures,
# and a continuous score at a third window on the ROC — so no two of them described one policy,
# and the argmax was presented as an operating point without ever saying it was selected.
CANONICAL_COUNTING: Final[str] = "edit_gated"


@dataclass(frozen=True)
class RecurrenceScores:
    """The continuous recurrence score per stamped run, plus every axis a figure strata on."""

    plain: tuple[float, ...]
    edit: tuple[float, ...]
    labels: tuple[bool, ...]
    groups: tuple[str, ...]
    models: tuple[str, ...]


@dataclass(frozen=True)
class PrefixAdmission:
    """Who the prefix risk model is fitted on, and how that differs from the corpus."""

    # Published at the TOP LEVEL because the prefix half's NO_SKILL verdict is quoted about "the
    # corpus" and is not measured on it: the anti-leak margin admits under half the stamped runs,
    # and the admitted half fails materially more often than the whole.

    depth: int
    n_stamped: int
    n_excluded_too_short: int
    n_excluded_by_margin: int
    n_admitted: int
    admitted_base_rate: float
    corpus_base_rate: float

    @property
    def admitted_share(self) -> float:
        return self.n_admitted / self.n_stamped if self.n_stamped else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "n_stamped": self.n_stamped,
            "n_excluded_too_short": self.n_excluded_too_short,
            "n_excluded_by_margin": self.n_excluded_by_margin,
            "n_admitted": self.n_admitted,
            "admitted_share": round(self.admitted_share, 4),
            "admitted_base_rate": round(self.admitted_base_rate, 4),
            "corpus_base_rate": round(self.corpus_base_rate, 4),
        }


@dataclass(frozen=True)
class EvalReport:
    """The eval outcome: corpus census, policy sweep, prefix risk model, and the skill verdict."""

    status: str
    reason: str
    n_trajectories: int
    n_stamped: int
    n_multistep: int
    authenticity_errors: int
    policy_cells: list[PolicyCell]
    # REQUIRED, deliberately not defaulted: a skill verdict says whether a signal is there, and
    # this says whether that signal is one production could act on. A default would let a report
    # be built — and its numbers quoted — without anyone stating which of the two it is.
    deployability: Deployability
    # The SAME recurrence sweep, replayed with `count_from_first_edit=True`: failures before the
    # agent's first edit are treated as the reproduction phase (the target bug at t=0), not as
    # escalation evidence. This is the variant that actually discriminates on this corpus — the
    # ungated shipped cell fires on every run and reads the base rate, while the edit-gated cells
    # at n=2..10 carry AUROC ~0.71-0.73 with a run-length baseline near 0.57 — so it is reported
    # as its OWN family (own family-wise null) rather than folded into the shipped one. It is an
    # EVAL-ONLY policy: the live router has no per-step action stream to gate on, so it describes
    # what the recurrence mechanism could do if a per-step detector ran in production.
    policy_cells_edit_gated: list[PolicyCell] = field(default_factory=list)
    # The SAME gate re-run for the edit-gated family. `deployability` above describes the shipped
    # counter; this one adds the counting mismatch, so the canonical cell — the point every
    # per-cell figure reads — carries its own scope statement rather than borrowing the shipped
    # counter's. Optional so a hand-built report stays constructible; `evaluate` always sets it.
    canonical_deployability: Deployability | None = None
    depth_reports: list[DepthReport] = field(default_factory=list)
    coverage: list[ModelCoverage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # The session-cadence escalation value: the corpus's multiple (model, arm) sessions per
    # instance read as repeated attempts at one task, so "cheap session failed -> escalate the
    # next session to a frontier model" is measurable at the cadence production actually runs.
    # Observational (parallel arms, adaptive coverage), and small-n; reported as context for the
    # per-step sweep, never as a gate. None when the corpus cannot support it.
    session_cadence: SessionCadenceReport | None = None
    # The per-run continuous recurrence scores behind `escalation_decision.png`, aligned
    # index-for-index over the STAMPED trajectories. Not serialized — it is figure input, not a
    # reported statistic (the AUROCs computed off it are the reportable numbers, and those DO
    # reach `reports/metrics.json`). The groups ride along so the figure's detection floor can use
    # the SAME challenge-block permutation the policy cells use (`policy_eval.null_roc_band`), and
    # the models so the stratified AUROC can rank inside a model.
    recurrence_scores: RecurrenceScores | None = None
    # AUROC of both recurrence scores at the shipped stale_window and at an effectively unbounded
    # one. Published because the headline gap is quoted at a FIXED window: as-shipped reaches
    # 0.728 at stale_window=1000 against 0.601 at the shipped 10, so a figure that let the window
    # move between its two curves would credit the counting change with a window change.
    window_sensitivity: dict[str, float] = field(default_factory=dict)
    # Per-model split of the canonical cell's two arms — `corpus_and_coverage.png`'s panel B.
    canonical_model_arms: list[plots.ModelArm] = field(default_factory=list)
    # The recurrence score's detection floor: the scalar family-wise AUROC null and the shaded
    # 2.5-97.5% band drawn on `escalation_decision.png`. Figure input, not serialized.
    recurrence_null: metrics.NullResult | None = None
    recurrence_null_band: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = (
        None
    )
    # The off-policy escalation-value estimate, present only when a real exploration log was
    # supplied. None means "not asked", NOT "no effect" — the two must stay distinguishable.
    policy_value: PolicyValueEstimate | None = None
    # The prefix half's scope, published beside its verdict rather than buried in a depth row.
    prefix_admission: PrefixAdmission | None = None
    # How many label shuffles every null on this report was estimated from.
    n_permutations: int = metrics.MIN_PERMUTATIONS

    @property
    def headline_cell(self) -> PolicyCell | None:
        """The shipped configuration — the one the product actually runs, never an argmax."""
        # Matched on ALL THREE knobs. Matching `escalate_after_n` alone made the claim false in
        # general: the swept grid pins stale_window=5 where the product ships 10, and the two are
        # not interchangeable — `_in_window` admits at most `stale_window` events, so at n=20 with
        # stale_window=10 the policy can NEVER fire. That they scored alike on this corpus is a
        # property of this corpus, not of the knobs. There is no fallback to `policy_cells[0]`
        # either: labelling an arbitrary cell "shipped" is worse than reporting none, and
        # `evaluate` guarantees the shipped point is in the grid so the None branch means a
        # hand-built report, not a routine run.
        return next((c for c in self.policy_cells if _is_shipped(c)), None)

    @property
    def best_skilled_cell(self) -> PolicyCell | None:
        """Strongest skilled cell across BOTH families, ranked by AUROC (the detection claim)."""
        # Each family carries its own maxT null (computed inside `policy_eval.evaluate`), so a
        # cell's `has_skill` is already multiplicity-corrected within its family. Reading the best
        # across two families is a mild selection over two corrected tests; the p-values on this
        # corpus are at the draw floor (0.005), so it survives trivially, and the reason string
        # names the family. AUROC first, precision second: the verdict is a DETECTION claim, and
        # ranking by precision alone would headline the 69-run n=30 cell over the 435-run n=2 cell
        # that actually separates runs. The status gate routes through the same helper so the
        # report and the reason cannot disagree about which cell is best.
        return _best_skilled_cell([*self.policy_cells, *self.policy_cells_edit_gated])

    @property
    def best_depth(self) -> DepthReport | None:
        """The depth with the largest incremental AUROC among those the design could evaluate.

        A rank-deficient depth is excluded: its incremental is arithmetic (rank 3 of 4 on the
        rebuilt corpus), and the report falls back to the plain max only when no depth ranks full.
        """
        estimable = [d for d in self.depth_reports if d.design_full_rank]
        pool = estimable or self.depth_reports
        return max(pool, key=lambda d: d.incremental_auroc, default=None)

    def to_dict(self) -> dict[str, object]:
        cell = self.headline_cell
        canonical = canonical_cell(self)
        return {
            "status": self.status,
            "reason": self.reason,
            "n_permutations": self.n_permutations,
            # Beside `status`, not buried: `status` says whether a signal was found, this says
            # whether the thing that found it is a policy production could run.
            "deployability": self.deployability.to_dict(),
            "n_trajectories": self.n_trajectories,
            "n_stamped": self.n_stamped,
            "n_multistep": self.n_multistep,
            "authenticity_errors": self.authenticity_errors,
            "headline_policy_cell": None if cell is None else cell.to_dict(),
            # THE point every figure reads: the shipped knobs with edit-gated counting. Published
            # beside the shipped row so the two can be compared without recomputing either.
            "canonical_counting": CANONICAL_COUNTING,
            "canonical_policy_cell": None if canonical is None else canonical.to_dict(),
            # The canonical cell's OWN scope statement. Without it a reader comparing the two
            # cells sees one `deployability` block and reasonably assumes it covers both, when the
            # edit-gated one carries a mismatch the shipped counter does not have.
            "canonical_deployability": (
                None
                if self.canonical_deployability is None
                else self.canonical_deployability.to_dict()
            ),
            "prefix_admission": (
                None if self.prefix_admission is None else self.prefix_admission.to_dict()
            ),
            "window_sensitivity": {k: round(v, 4) for k, v in self.window_sensitivity.items()},
            "policy_cells": [c.to_dict() for c in self.policy_cells],
            # The reproduction-phase-gated family, reported as its own sweep. Its cells carry
            # their own family-wise null; `best_skilled_cell` reads across both.
            "policy_cells_edit_gated": [c.to_dict() for c in self.policy_cells_edit_gated],
            "prefix_model": [d.to_dict() for d in self.depth_reports],
            "capture_coverage": [c.to_dict() for c in self.coverage],
            "notes": self.notes,
            "session_cadence": (
                None if self.session_cadence is None else self.session_cadence.to_dict()
            ),
            "escalation_policy_value": (
                None if self.policy_value is None else asdict(self.policy_value)
            ),
        }


def _is_shipped(cell: PolicyCell) -> bool:
    """Whether a swept cell IS the shipped configuration, on every knob the product sets."""
    return (
        cell.escalate_after_n == SHIPPED_ESCALATE_AFTER_N
        and cell.stale_window == SHIPPED_STALE_WINDOW
        and cell.ladder == SHIPPED_LADDER
    )


def shipped_grid_point() -> replay.GridPoint:
    """The swept cell that IS the shipped configuration."""
    return replay.GridPoint(SHIPPED_ESCALATE_AFTER_N, SHIPPED_STALE_WINDOW, SHIPPED_LADDER)


def _with_shipped(grid: Sequence[replay.GridPoint]) -> list[replay.GridPoint]:
    """The requested grid, guaranteed to contain the shipped configuration."""
    # This is what makes the mismatch impossible rather than merely detected: if the registered
    # grid pins knobs that differ from what ships, the shipped cell is APPENDED and measured too.
    # The alternative — reporting no headline — hides the shipped default from its own report.
    point = shipped_grid_point()
    return list(grid) if point in grid else [*grid, point]


def _status(
    report_cells: Sequence[PolicyCell],
    depths: Sequence[DepthReport],
    authenticity_errors: int = 0,
    deployability: Deployability | None = None,
    edit_gated_cells: Sequence[PolicyCell] = (),
    canonical_deployability: Deployability | None = None,
) -> tuple[str, str]:
    """Gate on Layer-1 authenticity first, then on the permutation nulls (policy AND prefix)."""
    if authenticity_errors:
        # This was computed and then never read: a corpus with hash or derived-field mismatches
        # still printed OK/NO_SKILL, so the one gate that says "these rows are not what was
        # recorded" could not affect the verdict it was collected for.
        return (
            _STATUS_UNVERIFIED,
            f"{authenticity_errors} Layer-1 authenticity error(s) on the corpus — recomputed "
            "content hashes or derived fields disagree with what the files declare. No skill "
            "verdict is issued off rows that fail their own integrity recompute.",
        )
    if not report_cells and not depths:
        return _STATUS_INSUFFICIENT, "no trajectory reached a scorable depth — nothing to measure"
    # A null-clearing cell is still an OFFLINE-ONLY UPPER BOUND when the corpus is not a deployable
    # estimate (feature mismatch and/or step cadence). The badge must carry that itself, because it
    # is read before `reason`. Both halves share the gate and the scope sentence.
    offline_only = deployability is not None and not deployability.deployable
    ok_status = _STATUS_OK_OFFLINE_ONLY if offline_only else _STATUS_OK
    scope_caveat = _offline_only_caveat(deployability)
    # THE POLICY HALF GATES TOO, because it is the shipped mechanism: a configuration that (a)
    # actually fires, (b) clears its family-wise null, and (c) has a precision interval above the
    # base rate is skill the prefix model never had a chance to express. The sweep is read as ONE
    # family (OK if any cell clears) so each cell's null is the max-over-cells maxT reference.
    # BOTH families are considered: each carries its OWN family-wise null (30 cells each on this
    # corpus), and `_best_skilled_cell` reads across them — a mild, uncorrected selection over
    # two corrected tests, which the reason string states rather than hides. The reason names
    # which family the best cell belongs to.
    all_cells = [*report_cells, *edit_gated_cells]
    best = _best_skilled_cell(all_cells)
    if best is not None:
        # Identity, never structural equality: a report_cells cell that happens to equal an
        # edit-gated one is still gated against ITS OWN family's null, and the family sentence
        # (and the multiplicity quoted below) must follow which list the winning OBJECT is in.
        in_edit_family = any(best is c for c in edit_gated_cells)
        family = "edit-gated (post-first-edit recurrence)" if in_edit_family else "as-shipped"
        # The adjusted p is family-wise WITHIN the best cell's own family (the maxT reference it
        # was gated against), NOT across both families — quoting len(all_cells)=60 would overstate
        # the multiplicity correction.
        family_size = len(edit_gated_cells) if in_edit_family else len(report_cells)
        # The scope sentence must be the WINNING family's. Quoting the shipped counter's verdict
        # beside an edit-gated cell states two mismatches for a number that has three.
        if in_edit_family and canonical_deployability is not None:
            scope_caveat = _offline_only_caveat(canonical_deployability)
        return (
            ok_status,
            f"escalation policy ({family}) at escalate_after_n={best.escalate_after_n}/"
            f"stale_window={best.stale_window} fires on {best.n_escalated} of "
            f"{best.n_trajectories} trajectories and clears its family-wise permutation null: "
            f"AUROC {best.null_auroc.observed:.3f} against the max-over-cells null 95% "
            f"[{best.gate_null.ci_low:.3f}, {best.gate_null.ci_high:.3f}] (adjusted "
            f"p={best.gate_null.p_value:.3f} within its {family_size}-cell sweep, selected across "
            f"the as-shipped and edit-gated families), with "
            f"P(fail|fired)={best.precision:.3f} [{best.precision_ci[0]:.3f}, "
            f"{best.precision_ci[1]:.3f}] above the base failure rate {best.base_failure_rate:.3f}"
            + _length_disclosure(best)
            + scope_caveat,
        )
    # This "any depth clears" rule is WHY the nulls read below are the family-wise ones: taking the
    # best of the reported depths at a nominal 2.5% each is a max over that many tests.
    # `gate_null_incremental` is that max's distribution, so its 97.5th percentile is the
    # family-wise critical value and its p-value is already adjusted. See `prefix_eval`'s module
    # header.
    skilled = [d for d in depths if d.has_skill]
    if skilled:
        best_depth = max(skilled, key=lambda d: d.incremental_auroc)
        return (
            ok_status,
            f"prefix risk model clears its family-wise permutation null at depth "
            f"{best_depth.depth}: "
            f"incremental AUROC {best_depth.incremental_auroc:+.3f} over the t=0 task prior "
            f"floored at chance "
            f"(family-wise null 97.5th pct {best_depth.gate_null_incremental.ci_high:+.3f}, "
            f"adjusted p={best_depth.gate_null_incremental.p_value:.3f} "
            f"over {len(depths)} reported depth(s))" + scope_caveat,
        )
    return (
        _STATUS_NO_SKILL,
        _no_skill_reason(report_cells, depths, edit_gated_cells)
        + _coverage_gap_caveat(depths, report_cells),
    )


def _coverage_gap_caveat(depths: Sequence[DepthReport], cells: Sequence[PolicyCell] = ()) -> str:
    """Say WHO the prefix null is about, since it is not measured on the corpus it names."""
    # A null on an instrument never shown to detect a positive is a coverage gap, not a
    # falsification; this null is also measured on a different, length-selected population
    # from the one the sentence above names.
    if not depths:
        return ""
    depth = max(depths, key=lambda d: d.n_rows)
    reached = depth.n_rows + depth.n_excluded_by_margin
    if not reached or not depth.n_rows:
        return ""
    corpus_rate = next((c.base_failure_rate for c in cells), None)
    contrast = "" if corpus_rate is None else f" against the corpus's {corpus_rate:.3f}"
    return (
        f". COVERAGE-LIMITED NULL, NOT A FALSIFICATION: the anti-leak margin admits only "
        f"{depth.n_rows} of the {reached} runs that reach depth {depth.depth} "
        f"({depth.n_rows / reached:.0%}), and that admitted population fails at "
        f"{depth.base_rate:.3f}{contrast} — it is length-selected and more failure-prone, so this "
        "is a null on a different population, on an instrument never shown to detect a positive"
    )


def _offline_only_caveat(deployability: Deployability | None) -> str:
    """The scope sentence a non-deployable OK reason must carry, or the empty string."""
    if deployability is None or deployability.deployable:
        return ""
    # The counting clause is added whenever it applies, and it is the strongest of the three: a
    # cadence gap describes the shipped rule measured at the wrong granularity, while an
    # unsupported counter describes a rule the product has no code path for at all.
    counting = ""
    if deployability.counting_unsupported:
        counting = (
            f" and it is scored with the eval-only '{deployability.counting}' counter, which the "
            f"product does not implement"
        )
    return (
        f" — {deployability.label}, so this is a per-step signal, not a shipped "
        f"escalation: production decides once per session{counting}"
    )


def _length_disclosure(cell: PolicyCell) -> str:
    """The run-length honesty sentence a skilled cell's reason must carry, or the empty string."""
    # The challenge-block gate removes the length→failure association along with everything else,
    # so a cell can clear it while most of its excess over chance is length selection. The two
    # length references are the disclosure: how much of the AUROC a pure length>=t predictor gets
    # at the same flag count, and whether the recurrence excess clears a null that KEEPS the
    # length association. Absent when the cell was hand-built without them.
    if cell.length_baseline_auroc is None or cell.null_auroc_length is None:
        return ""
    length_null = cell.null_auroc_length
    length_verdict = "clears" if length_null.beats_null else "does NOT clear"
    return (
        f" — run length alone scores {cell.length_baseline_auroc:.3f} at the same flag count, "
        f"and the recurrence excess {length_verdict} the length-stratified null "
        f"(AUROC {cell.null_auroc.observed:.3f} against {length_null.ci_low:.3f}-"
        f"{length_null.ci_high:.3f}, p={length_null.p_value:.3f})"
    )


def _no_skill_reason(
    report_cells: Sequence[PolicyCell],
    depths: Sequence[DepthReport],
    edit_gated_cells: Sequence[PolicyCell] = (),
) -> str:
    """Name the skill conditions that actually failed, with their numbers, at the closest depth."""
    if not depths and not report_cells:
        return "no trajectory reached a scorable depth — nothing to measure"
    if not depths:
        return "the prefix half has no estimable depth; the policy sweep is reported below"
    # Prefer a depth the design could actually evaluate: a rank-deficient depth's incremental is
    # arithmetic (depth 5 on the rebuilt corpus ranks 3 of 4), so naming it as the closest depth
    # would report an instrument artifact as the failure that matters. Same preference as
    # `EvalReport.best_depth`.
    estimable = [d for d in depths if d.design_full_rank]
    best = max(estimable or depths, key=lambda d: d.incremental_auroc)
    conditions = _skill_conditions(best)
    failures = [clause for met, clause in conditions if not met]
    if not failures:
        # Unreachable while this enumeration mirrors `DepthReport.has_skill`. Say so loudly rather
        # than printing a confident sentence about a condition nobody checked.
        return (
            f"no depth has skill, but depth {best.depth} meets every enumerated condition — "
            "the reported conditions have drifted from the gate; treat this run as unexplained"
        )
    policy = _policy_skill_failure(report_cells, edit_gated_cells)
    return (
        f"no depth satisfies all {len(conditions)} skill conditions; at depth {best.depth} "
        + "; ".join(failures)
        + _cleared_increment_caveat(best)
        + policy
    )


def _policy_skill_failure(
    report_cells: Sequence[PolicyCell], edit_gated_cells: Sequence[PolicyCell] = ()
) -> str:
    """The policy half's closest-miss, stated so the sweep is not reported as unexplained."""
    all_cells = [*report_cells, *edit_gated_cells]
    if not all_cells:
        return ""
    fired = [c for c in all_cells if c.precision is not None]
    if not fired:
        return f"; no swept configuration fired on any trajectory ({len(all_cells)} cells swept)"
    # The cell that came closest on the two clauses that could separate it: its own null was the
    # family-wise reference, so quote the marginal conditions that failed per cell.
    best = max(fired, key=lambda c: c.precision or 0.0)
    family = "edit-gated" if any(best is c for c in edit_gated_cells) else "as-shipped"
    clauses = [
        (best.gate_null.beats_null, "its AUROC does not clear the family-wise permutation null"),
        (
            best.precision_ci[0] > best.base_failure_rate,
            "its P(fail|fired) interval does not clear the base failure rate",
        ),
    ]
    missing = [clause for met, clause in clauses if not met]
    if not missing:
        return ""
    return (
        f"; the best swept cell ({family}, escalate_after_n={best.escalate_after_n}/"
        f"stale_window={best.stale_window}, P(fail|fired)={best.precision:.3f} "
        f"[{best.precision_ci[0]:.3f}, {best.precision_ci[1]:.3f}] against base "
        f"{best.base_failure_rate:.3f}) fails: " + " and ".join(missing)
    )


def _skill_conditions(depth: DepthReport) -> list[tuple[bool, str]]:
    """Every `DepthReport.has_skill` condition, paired with the clause reporting its failure."""
    # Built as one list so the count in the message and the clauses cannot drift apart; keep this
    # in lockstep with `has_skill` — reporting a condition nobody checks is the bug this replaced.
    # The two null clauses quote `gate_null_*` — the FAMILY-WISE band the gate actually applies —
    # so the sentence and the verdict cannot disagree. Quoting the uncorrected per-depth null here
    # while gating on the corrected one is the "reported a condition nobody checked" defect above,
    # wearing a different hat.
    return [
        (len(set(depth.scores)) > 1, "the prefix score is constant, so it ranks nothing"),
        (
            depth.gate_null_prefix.beats_null,
            f"the prefix-only score does not clear its family-wise permutation null "
            f"({depth.auroc_prefix:.3f}, null 95% [{depth.gate_null_prefix.ci_low:.3f}, "
            f"{depth.gate_null_prefix.ci_high:.3f}], adjusted p="
            f"{depth.gate_null_prefix.p_value:.3f})",
        ),
        (
            depth.gate_null_incremental.beats_null,
            f"the incremental AUROC does not clear its family-wise permutation null "
            f"({depth.incremental_auroc:+.3f}, null 95% "
            f"[{depth.gate_null_incremental.ci_low:+.3f}, "
            f"{depth.gate_null_incremental.ci_high:+.3f}], adjusted p="
            f"{depth.gate_null_incremental.p_value:.3f})",
        ),
        (
            depth.ci_incremental[0] > 0.0,
            f"the paired bootstrap over challenges puts the increment's 95% interval across zero "
            f"([{depth.ci_incremental[0]:+.3f}, {depth.ci_incremental[1]:+.3f}])",
        ),
    ]


def _cleared_increment_caveat(depth: DepthReport) -> str:
    """The case a reader will misread: the increment clears its null, the gate still refuses."""
    null = depth.gate_null_incremental
    if not null.beats_null:
        return ""
    cleared = (
        f" — the incremental AUROC DOES clear its own family-wise null "
        f"({depth.incremental_auroc:+.3f}, null 95% [{null.ci_low:+.3f}, {null.ci_high:+.3f}], "
        f"adjusted p={null.p_value:.3f}), but "
    )
    if not depth.gate_null_prefix.beats_null:
        return cleared + (
            "a prefix score that cannot clear its own null carries no discrimination to add, so "
            "that clearance reflects the t=0 prior it is measured against, not prefix evidence"
        )
    return cleared + (
        "it is not stable across resampled challenges, so the point estimate is a property of "
        "this particular set of challenges rather than of the prefix features"
    )


def evaluate(
    trajectories: Sequence[Trajectory],
    grid: Sequence[replay.GridPoint] = tuple(datasets.DEFAULT_GRID),
    *,
    depths: Sequence[int] = features.DEFAULT_DEPTHS,
    n_permutations: int = metrics.DEFAULT_PERMUTATIONS,
    exploration_rows: Sequence[Mapping[str, object]] | None = None,
) -> EvalReport:
    """Score the shipped policy and the prefix risk model, then gate on the permutation null."""
    # `stamped` is still computed here for the census (`n_stamped`), the policy half and the
    # coverage note. `prefix_eval` re-applies the same filter inside `corpus_census` — that is not
    # redundancy to be tidied away: the gate must live where a direct `evaluate_depth` caller
    # cannot skip it, and filtering an already-filtered corpus is idempotent, so this path's
    # numbers are unchanged (`n_excluded_unstamped` reads 0 here and 8 on the raw 799).
    stamped = [t for t in trajectories if features.is_stamped(t)]
    multistep = [t for t in trajectories if not datasets.is_degenerate(t)]
    policy_cells = policy_eval.evaluate(stamped, _with_shipped(grid), n_permutations=n_permutations)
    # The SAME grid replayed with the reproduction phase excluded. A family of its own, with its
    # own family-wise null — reported beside the shipped sweep, never merged into it. See the
    # field's comment for why it exists.
    edit_gated = policy_eval.evaluate(
        stamped, _with_shipped(grid), n_permutations=n_permutations, count_from_first_edit=True
    )
    depth_reports = prefix_eval.evaluate(stamped, depths, n_permutations=n_permutations)
    authenticity_errors = sum(len(errors(verify_trajectory(t))) for t in trajectories)
    # Named `deployability_verdict`, not `deployability`: the module of that name is what
    # `_deployability` calls, and the old local shadowed it for the rest of this function.
    deployability_verdict = _deployability(stamped)
    status, reason = _status(
        policy_cells,
        depth_reports,
        authenticity_errors,
        deployability_verdict,
        edit_gated,
        _deployability(stamped, CountingMode.EDIT_GATED),
    )
    recurrence = _recurrence_scores(stamped)
    recurrence_null, recurrence_null_band = (
        _recurrence_null(recurrence, n_permutations=n_permutations)
        if recurrence.labels
        else (None, None)
    )
    return EvalReport(
        status=status,
        reason=reason,
        n_trajectories=len(trajectories),
        n_stamped=len(stamped),
        n_multistep=len(multistep),
        authenticity_errors=authenticity_errors,
        policy_cells=policy_cells,
        policy_cells_edit_gated=edit_gated,
        canonical_deployability=_deployability(stamped, CountingMode.EDIT_GATED),
        depth_reports=depth_reports,
        coverage=features.model_coverage(trajectories),
        deployability=deployability_verdict,
        notes=_corpus_notes(trajectories, stamped, policy_cells),
        # The session-cadence escalation value, computed whenever the corpus carries enough
        # multi-arm instances to estimate it. Observational context, never a gate.
        session_cadence=session_cadence(trajectories),
        recurrence_scores=recurrence,
        recurrence_null=recurrence_null,
        recurrence_null_band=recurrence_null_band,
        window_sensitivity=_window_sensitivity(stamped),
        canonical_model_arms=_canonical_model_arms(stamped),
        prefix_admission=_prefix_admission(stamped, depth_reports),
        n_permutations=n_permutations,
        # Only ever computed from a REAL exploration log. Trajectories carry no propensity
        # (the committable whitelist has no `action`/`propensity`), so deriving rows from them
        # would manufacture propensity=1.0 for every decision — a fabricated input, not data.
        policy_value=(
            None
            if exploration_rows is None
            else estimate_policy_value(rows_from_records(exploration_rows))
        ),
    )


def _deployability(
    stamped: Sequence[Trajectory], counting: CountingMode = CountingMode.AS_SHIPPED
) -> Deployability:
    """Gate this run's scored feature set against what a production decision actually holds."""
    # `unfilled` is MEASURED over the corpus that was actually scored, never assumed: a field
    # production has is only a mismatch if no record fills it, and that is a property of the
    # corpus on disk, which a rebuild can change.
    unfilled = deployability.unfilled_context_fields(
        step for traj in stamped for step in features.scorable_steps(traj)
    )
    # `Cadence.STEP`: both halves score a decision point at a step boundary — the policy replay
    # walks every step, and the prefix model reads a prefix at step depth d. Production decides
    # once per session, so this is the cadence gap, declared rather than inferred.
    #
    # `counting` is the THIRD mismatch, and it is the one that separates the two reported families:
    # the as-shipped counter is the product's own, while the edit-gated one reads `StepView.action`,
    # which no production decision holds and no `EscalationPolicy` knob can ask for. Passing it
    # here is what makes the canonical cell's scope machine-readable instead of a doc sentence.
    return deployability.assess(
        features.FEATURE_NAMES, Cadence.STEP, unfilled=unfilled, counting=counting
    )


def _corpus_notes(
    trajectories: Sequence[Trajectory],
    stamped: Sequence[Trajectory],
    cells: Sequence[PolicyCell],
) -> list[str]:
    """Caveats the numbers cannot carry themselves: coverage loss and the shipped-default gap."""
    notes: list[str] = []
    dropped = len(trajectories) - len(stamped)
    if dropped:
        notes.append(
            f"{dropped}/{len(trajectories)} trajectories carry no per-step verified outcomes "
            "(the stamping stage never ran on them) and are excluded from every metric here. "
            "Their fields are parser defaults, so including them would feed the model a "
            "collection-date proxy rather than escalation evidence."
        )
    notes.extend(_default_gap_note(cells))
    return notes


def _default_gap_note(cells: Sequence[PolicyCell]) -> list[str]:
    """Flag — never silently apply — a swept configuration that beats the shipped default."""
    shipped = next((c for c in cells if _is_shipped(c)), None)
    # A cell that never fired has NO precision, so it is excluded from the ranking rather than
    # ranked at a fabricated 0.0 (see `PolicyCell.precision`).
    scored = [c for c in cells if c.precision is not None]
    if shipped is None or shipped.precision is None or not scored:
        return []
    best = max(scored, key=lambda c: c.precision or 0.0)
    if _is_shipped(best):
        return []
    return [
        f"escalate_after_n={best.escalate_after_n} (stale_window={best.stale_window}) reaches "
        f"P(fail|fired)={best.precision:.3f} against the SHIPPED default "
        f"n={shipped.escalate_after_n}/stale_window={shipped.stale_window}'s "
        f"{shipped.precision:.3f}. "
        "The shipped default is unchanged here: this is a measurement, and the intervals "
        f"([{best.precision_ci[0]:.3f}, {best.precision_ci[1]:.3f}] vs "
        f"[{shipped.precision_ci[0]:.3f}, {shipped.precision_ci[1]:.3f}]) must be read before "
        "anyone acts on it."
    ]


def _status_limitations(report: EvalReport) -> list[str]:
    """The corpus-status limitations that hold whatever population a figure scored."""
    limitations: list[str] = []
    if report.status == _STATUS_NO_SKILL:
        limitations.append(f"NO USABLE SIGNAL — {report.reason}.")
    if report.status == _STATUS_INSUFFICIENT:
        limitations.append(f"INSUFFICIENT DATA — {report.reason}.")
    if report.status == _STATUS_UNVERIFIED:
        limitations.append(f"CORPUS FAILED ITS INTEGRITY CHECK — {report.reason}.")
    return limitations


def _run_annotations(report: EvalReport) -> Annotations:
    """The run-level facts and, at most, the ONE caveat every STAMPED-corpus figure carries."""
    # This used to append the deployability line to the limitations of EVERY figure, which was a
    # correctness bug rather than a style one: it told a reader the recurrence ROC "depends on
    # features a production decision does not hold", and the AS-SHIPPED recurrence score reads
    # exactly two fields — `failing_check_id` and `success` — both of which production has. The
    # FEATURE mismatch is real but belongs to the PREFIX model, so it is stated on the one panel
    # that draws the prefix model (`plots.CORPUS_COVERAGE_SPEC`) and nowhere else.
    #
    # That narrowing was correct in direction and over-broad in extent, and this is the fix: the
    # EDIT-GATED score reads a third field, `StepView.action`, which production does NOT have. So
    # the canonical (edit-gated) verdict carries its own scope line, added below to the figures
    # that read the canonical cell — `_canonical_annotations`.
    #
    # The limitations/notes output is kept: it never reaches the canvas any more, but it is what
    # the manifest row and the docs section are written from.
    #
    # This block describes the PER-STEP-STAMPED sub-corpus, so it belongs only to figures that
    # actually score it. `session_value.png` does not — see `_session_run_annotations`.
    limitations = _status_limitations(report)
    dropped = report.n_trajectories - report.n_stamped
    if dropped:
        limitations.append(
            f"{dropped}/{report.n_trajectories} trajectories have no per-step verified outcomes "
            "and are excluded from this figure."
        )
    return Annotations(
        subtitle_facts=(f"{report.n_stamped}/{report.n_trajectories} runs scored",),
        caveat=_run_caveat(report),
        notes=(
            f"{report.n_stamped} scored trajectories, status={report.status}",
            f"{report.deployability.label} — {report.deployability.reason}",
        ),
        limitations=tuple(limitations),
    )


def _canonical_annotations(report: EvalReport) -> Annotations:
    """The scope line a figure drawn from the CANONICAL (edit-gated) cell must carry."""
    # Keyed on the verdict, not on prose: if the counter ever became something production holds,
    # `counting_unsupported` empties and this line disappears on its own. That is the whole point
    # of routing the counting mode through `deployability.assess` rather than restating it here.
    verdict = report.canonical_deployability
    if verdict is None or not verdict.counting_unsupported:
        return Annotations()
    return Annotations(
        notes=(f"canonical cell: {verdict.label} — {verdict.reason}",),
        limitations=(
            f"EVAL-ONLY COUNTER: this figure is drawn from the '{verdict.counting}' cell, which "
            f"ignores failures before the agent's first edit-like action — a rule that reads "
            f"{', '.join(verdict.counting_unsupported)}, a per-step field the live router never "
            f"sees, and that no EscalationPolicy knob can ask for. The counter the product does "
            f"run fires on almost every run and reads the base rate.",
        ),
    )


def _session_run_annotations(report: EvalReport) -> Annotations:
    """Run-level facts for the ONE figure scored on the whole corpus, not the stamped subset."""
    # `session_cadence()` is handed every trajectory and reads only `header.terminal_resolved`
    # and `header.n_steps` — both present on an unstamped run — so the per-step stamping split
    # is irrelevant to it. Inheriting `_run_annotations` told a reader this figure scored
    # 727/799 and excluded 72, and neither is true of it.
    return Annotations(
        subtitle_facts=(f"{report.n_trajectories}/{report.n_trajectories} runs read",),
        # Deliberately NOT `_run_caveat`: its lowest-severity branch is the per-step drop line,
        # which does not apply here. The status caveats still do.
        caveat=_status_caveat(report),
        notes=(
            f"{report.n_trajectories} trajectories read at session cadence "
            f"(per-step stamping not required), status={report.status}",
        ),
        limitations=(
            *_status_limitations(report),
            f"Scored on ALL {report.n_trajectories} trajectories, not the "
            f"{report.n_stamped}-run per-step-stamped subset the other escalation figures use: "
            "a session outcome is read from the run header, so an unstamped run is still "
            "scorable here.",
        ),
    )


def _status_caveat(report: EvalReport) -> str | None:
    """The corpus-status red lines, severity-ordered — true whatever population was scored."""
    # Severity order, most-severe first: a corpus that fails its own integrity recompute outranks
    # a no-skill verdict, which outranks insufficient data. The frame keeps the FIRST non-None
    # caveat, and a per-figure caveat is merged ahead of this one, so a figure with its own
    # specific warning keeps it.
    if report.status == _STATUS_UNVERIFIED:
        return (
            f"CORPUS FAILED ITS INTEGRITY CHECK: {report.authenticity_errors} Layer-1 error(s). "
            "No number on this figure is trustworthy."
        )
    if report.status == _STATUS_NO_SKILL:
        return "NO USABLE SIGNAL on this corpus — see the report's `reason` field."
    if report.status == _STATUS_INSUFFICIENT:
        return "INSUFFICIENT DATA — no trajectory reached a scorable depth."
    return None


def _run_caveat(report: EvalReport) -> str | None:
    """At most ONE red line, severity-ordered, or None. A canvas gets one or it gets none."""
    status = _status_caveat(report)
    if status is not None:
        return status
    dropped = report.n_trajectories - report.n_stamped
    if dropped:
        return (
            f"{dropped} of {report.n_trajectories} runs carry no per-step outcomes and are "
            "excluded; the drop rate is model-correlated."
        )
    return None


def _merge(*parts: Annotations) -> Annotations:
    """Concatenate annotation blocks; FigureSpec.merged dedups and drops blanks."""
    # EVERY field, including the rendered ones. Dropping `subtitle_facts`/`caveat`/`counts` here
    # is not a partial merge, it is a silent one: the figure's own subtitle numbers never reached
    # the canvas and its manifest row carried no counts, while the code that computed them looked
    # correct at both ends. Callers merge most-severe-first, and `FigureSpec.merged` keeps the
    # FIRST non-None caveat, so a figure-specific caveat still outranks the run-level one.
    return Annotations(
        subtitle_facts=tuple(f for part in parts for f in part.subtitle_facts),
        caveat=next((part.caveat for part in parts if part.caveat is not None), None),
        definitions=tuple(d for part in parts for d in part.definitions),
        notes=tuple(n for part in parts for n in part.notes),
        limitations=tuple(lim for part in parts for lim in part.limitations),
        counts=tuple(c for part in parts for c in part.counts),
    )


# The PNGs live in `docs/assets/figures/escalation/` — one subdirectory per benchmark half —
# but the manifest that describes them stays beside the code that writes it: it is source, not
# a published asset. Same for metrics.json: it is the numbers behind the images, not an image.
_PKG_DIR: Path = Path(__file__).resolve().parent
MANIFEST: Path = _PKG_DIR / "figures.json"
CANONICAL_PLOTS_DIR: Path = _PKG_DIR.parents[1] / "docs/assets/figures/escalation"
CANONICAL_METRICS_DIR: Path = _PKG_DIR / "reports"


def _committed_home(plots_dir: Path) -> bool:
    """Is this run writing the real committed figure set, or a throwaway copy?"""
    # A test or a scratch render must never touch the committed manifest or metrics.json.
    # A directory NAME proves nothing — a tmp dir can be called `escalation` too — so the
    # test is "is this THE committed directory", not "is it named like one".
    return plots_dir.resolve() == CANONICAL_PLOTS_DIR.resolve()


def _manifest_for(plots_dir: Path) -> Path:
    return MANIFEST if _committed_home(plots_dir) else plots_dir.parent / "figures.json"


def _metrics_dir_for(plots_dir: Path) -> Path:
    return CANONICAL_METRICS_DIR if _committed_home(plots_dir) else plots_dir


def _provenance(report: EvalReport, out_dir: Path) -> plot_frame.Provenance:
    """Which module drew the figure and over what corpus — the manifest's identity columns."""
    return plot_frame.Provenance(
        generator="benchmark.escalation.run_eval",
        # The real content fingerprint, not a human label: `data_digest` exists so a stale
        # figure can be DETECTED, and a counts-only string collides across two corpora with
        # the same census. `_corpus_digest` is the same value metrics.json records.
        data_digest=_corpus_digest(report),
        manifest=_manifest_for(out_dir),
    )


def _save_plots(report: EvalReport, out_dir: Path, metrics_dir: Path | None = None) -> None:
    """Render the six figures to `out_dir`. A figure whose data is absent is skipped, not faked."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run = _run_annotations(report)
    # The three figures below that READ THE CANONICAL CELL get the run block plus the counting
    # mismatch. `escalation_decision.png` is excluded on purpose: it draws both curves side by
    # side and its own spec already carries the eval-only sentence, so a second copy would be
    # duplication, not disclosure. `session_value.png` is session-cadence and reads no cell.
    canonical_scope = _merge(run, _canonical_annotations(report))
    provenance = _provenance(report, out_dir)
    cell = canonical_cell(report)
    _draw_decision(report, out_dir, run, provenance)
    _draw_corpus(report, out_dir, canonical_scope, provenance)
    if cell is not None:
        _draw_operating_point(report, cell, out_dir, canonical_scope, provenance)
        _draw_budget(cell, out_dir, canonical_scope, provenance)
    else:
        # Only reachable on a hand-built report: `evaluate` guarantees the shipped point is in
        # both grids. The per-cell figures are skipped rather than drawn off some other cell — a
        # figure titled with the canonical operating point that plots a different one IS the
        # defect this whole redesign removes.
        logger.error("no edit-gated cell matches the shipped knobs; per-cell figures skipped")
    if report.policy_cells:
        _draw_sweep(report, out_dir, run, provenance)
    if report.session_cadence is not None:
        # NOT `run`: this figure scores every trajectory, not the stamped subset.
        _draw_session(report.session_cadence, out_dir, _session_run_annotations(report), provenance)
    _write_metrics(report, metrics_dir if metrics_dir is not None else _metrics_dir_for(out_dir))
    if _committed_home(out_dir):
        # Record the pipeline's freshness manifest now that the figures ARE drawn, so
        # `benchmark.pipeline --check-figures` stays green after a canonical `make escalation-eval`
        # — otherwise this run leaves the escalation job's digest stale in
        # `benchmark/routing/figure_inputs.json` and the gate that proves the committed figures
        # current fails until someone re-runs the full pipeline. After, not before: a draw that
        # failed must leave the manifest stale (red gate), never falsely current.
        try:
            from benchmark.pipeline import write_figure_manifest  # noqa: PLC0415

            write_figure_manifest()
        except Exception:  # noqa: BLE001 — freshness bookkeeping must never fail the eval itself
            logger.warning(
                "escalation eval: could not refresh the figure-inputs manifest", exc_info=True
            )


def _draw_decision(
    report: EvalReport, out_dir: Path, run: Annotations, provenance: plot_frame.Provenance
) -> None:
    """Figure 1 — does the trigger detect, and does the counting mode decide the answer."""
    scores = report.recurrence_scores
    if scores is None or not scores.labels:
        logger.error("no recurrence scores on this report; escalation_decision.png skipped")
        return
    size = plot_frame.WIDE_TALL
    fig = plot_frame.new_figure(size)
    grid = fig.add_gridspec(2, 3, height_ratios=[16, 1])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    axes.append(fig.add_subplot(grid[1, :]))
    extra = plots.escalation_decision(
        scores.plain,
        scores.edit,
        scores.labels,
        axes,
        null=report.recurrence_null,
        band=report.recurrence_null_band,
        shipped_n=SHIPPED_ESCALATE_AFTER_N,
    )
    plot_frame.save(
        fig,
        out_dir / "escalation_decision.png",
        plots.ESCALATION_DECISION_SPEC,
        extra=_merge(extra, run),
        provenance=provenance,
        size=size,
    )


def _draw_operating_point(
    report: EvalReport,
    cell: PolicyCell,
    out_dir: Path,
    run: Annotations,
    provenance: plot_frame.Provenance,
) -> None:
    """Figure 2 — BOTH counting modes at the shipped knobs, plus the canonical cell's nulls."""
    size = plot_frame.WIDE_TALL
    fig = plot_frame.new_figure(size)
    grid = fig.add_gridspec(2, 2, height_ratios=[16, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    extra = plots.operating_point(report.headline_cell, cell, axes)
    plot_frame.save(
        fig,
        out_dir / "operating_point.png",
        plots.OPERATING_POINT_SPEC,
        extra=_merge(extra, run),
        provenance=provenance,
        size=size,
    )


def _draw_sweep(
    report: EvalReport, out_dir: Path, run: Annotations, provenance: plot_frame.Provenance
) -> None:
    """Figure 3 — every configuration in both counting modes, as one table."""
    cells = report.policy_cells
    # Wider than the default table canvas: twelve columns (two families x fired / P(fail) / 95% CI
    # / AUROC / len-only, plus the two knobs) do not stay legible at 11 inches, and the answer to
    # a column that will not fit is a wider canvas — never a dropped confound control.
    size = plot_frame.table_size(len(cells) + 1, width_in=16.0)
    fig = plot_frame.new_figure(size)
    ax = fig.subplots()
    extra = plots.policy_sweep(
        cells,
        report.policy_cells_edit_gated,
        ax,
        shipped_index=next((i for i, c in enumerate(cells) if _is_shipped(c)), None),
    )
    plot_frame.save(
        fig,
        out_dir / "policy_sweep.png",
        plots.POLICY_SWEEP_SPEC,
        extra=_merge(extra, run),
        provenance=provenance,
        size=size,
    )


def _draw_session(
    sc: SessionCadenceReport,
    out_dir: Path,
    run: Annotations,
    provenance: plot_frame.Provenance,
) -> None:
    """Figure 4 — escalate vs retry at the cadence production actually runs."""
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2)
    extra = plots.session_value(sc, axes)
    plot_frame.save(
        fig,
        out_dir / "session_value.png",
        plots.SESSION_VALUE_SPEC,
        extra=_merge(extra, run),
        provenance=provenance,
        size=size,
    )


def _draw_corpus(
    report: EvalReport, out_dir: Path, run: Annotations, provenance: plot_frame.Provenance
) -> None:
    """Figure 5 — who is in the sample, and whether the edge survives the confounds."""
    size = plot_frame.WIDE_TALL
    fig, axes = plot_frame.subplots(size, 2, 2)
    extra = plots.corpus_and_coverage(
        report.coverage,
        report.canonical_model_arms,
        _stratified(report),
        _admission_panel(report),
        list(axes.flat),
    )
    plot_frame.save(
        fig,
        out_dir / "corpus_and_coverage.png",
        plots.CORPUS_COVERAGE_SPEC,
        extra=_merge(extra, run),
        provenance=provenance,
        size=size,
    )


def _draw_budget(
    cell: PolicyCell, out_dir: Path, run: Annotations, provenance: plot_frame.Provenance
) -> None:
    """Figure 6 — what firing costs, and what it pre-empts."""
    size = plot_frame.WIDE_TALL
    fig = plot_frame.new_figure(size)
    grid = fig.add_gridspec(2, 2, height_ratios=[16, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    extra = plots.escalation_budget(cell, axes)
    plot_frame.save(
        fig,
        out_dir / "escalation_budget.png",
        plots.ESCALATION_BUDGET_SPEC,
        extra=_merge(extra, run),
        provenance=provenance,
        size=size,
    )


def _write_metrics(report: EvalReport, out_dir: Path) -> None:
    """Commit the exact scalars each figure renders, keyed by figure slug.

    Until this existed the only committed artifact was the PNG, so no number on a canvas had a
    machine-readable source and no reader could check one after the fact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        # `_finite` last, as belt-and-suspenders: a cell that never fires legitimately carries a
        # (nan, nan) interval in memory, and JSON has no NaN literal (RFC 8259) — a strict parser
        # rejects the file outright, so non-finite floats serialize as null instead.
        json.dumps(_finite_json(_metrics_payload(report)), indent=2, sort_keys=True) + "\n"
    )


def _finite_json(value: object) -> object:
    """Recursively replace non-finite floats with ``None`` so metrics.json stays RFC 8259-valid."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    return value


def _metrics_payload(report: EvalReport) -> dict[str, object]:
    """Every figure's rendered scalars, plus the run identity they were all computed under."""
    cell = canonical_cell(report)
    scores = report.recurrence_scores
    strat = _stratified(report)
    sc = report.session_cadence
    payload: dict[str, object] = {
        "run": {
            "status": report.status,
            "n_trajectories": report.n_trajectories,
            "n_stamped": report.n_stamped,
            "n_permutations": report.n_permutations,
            "canonical_counting": CANONICAL_COUNTING,
            # Both scope verdicts, beside the numbers they scope. The shipped counter's names two
            # mismatches; the canonical one names three, and the third is that the product has no
            # such counter — the fact this whole record exists to keep machine-readable.
            "deployability": report.deployability.to_dict(),
            "canonical_deployability": (
                None
                if report.canonical_deployability is None
                else report.canonical_deployability.to_dict()
            ),
            "shipped_escalation": {
                "escalate_after_n": SHIPPED_ESCALATE_AFTER_N,
                "stale_window": SHIPPED_STALE_WINDOW,
                "ladder": SHIPPED_LADDER,
            },
            "corpus_digest": _corpus_digest(report),
        },
        "escalation_decision.png": {
            "auroc_as_shipped": (
                None if scores is None else round(metrics.auroc(scores.plain, scores.labels), 4)
            ),
            "auroc_edit_gated": (
                None if scores is None else round(metrics.auroc(scores.edit, scores.labels), 4)
            ),
            "base_rate": (None if scores is None else round(metrics.prevalence(scores.labels), 4)),
            "null": (None if report.recurrence_null is None else report.recurrence_null.to_dict()),
            "window_sensitivity": {k: round(v, 4) for k, v in report.window_sensitivity.items()},
            "shipped_escalate_after_n": SHIPPED_ESCALATE_AFTER_N,
        },
        # BOTH bar pairs the canvas draws, at the same shipped knobs. The as-shipped row is the
        # negative finding — it fires on essentially everything and reads the base rate — so it
        # is committed beside the canonical one rather than only inside the sweep list.
        "operating_point.png": {
            "as_shipped": (
                None if report.headline_cell is None else report.headline_cell.to_dict()
            ),
            "edit_gated": (None if cell is None else cell.to_dict()),
        },
        "policy_sweep.png": {
            "n_configurations": len(report.policy_cells),
            "as_shipped": [_sweep_scalars(c) for c in report.policy_cells],
            "edit_gated": [_sweep_scalars(c) for c in report.policy_cells_edit_gated],
        },
        "session_value.png": (None if sc is None else sc.to_dict()),
        "corpus_and_coverage.png": {
            # Panels B and C only: the coverage census and the prefix waterfall are counter-free.
            "counting_panels_b_c": CANONICAL_COUNTING,
            "coverage": [c.to_dict() for c in report.coverage],
            "canonical_model_arms": [
                {
                    "model": a.model,
                    "n": a.n,
                    "p_fail_fired": None if a.p_fail_fired is None else round(a.p_fail_fired, 4),
                    "p_fail_quiet": None if a.p_fail_quiet is None else round(a.p_fail_quiet, 4),
                }
                for a in report.canonical_model_arms
            ],
            "stratified_auroc": {
                "pooled": round(strat.pooled, 4),
                "within_model": (
                    None if strat.within_model is None else round(strat.within_model, 4)
                ),
                "within_challenge": (
                    None if strat.within_challenge is None else round(strat.within_challenge, 4)
                ),
            },
            "prefix_admission": (
                None if report.prefix_admission is None else report.prefix_admission.to_dict()
            ),
        },
        # Wholly canonical: every number here comes from the edit-gated cell, so the counting mode
        # travels with it rather than being inferable only from the prose on the canvas.
        "escalation_budget.png": (
            None if cell is None else {**cell.budget.to_dict(), "counting": CANONICAL_COUNTING}
        ),
    }
    return payload


def _sweep_scalars(cell: PolicyCell) -> dict[str, object]:
    """The four numbers a sweep-table row actually renders."""
    return {
        "escalate_after_n": cell.escalate_after_n,
        "stale_window": cell.stale_window,
        "n_escalated": cell.n_escalated,
        "precision": None if cell.precision is None else round(cell.precision, 4),
        "precision_ci95": policy_eval.rounded_interval(cell.precision_ci),
        "auroc": round(cell.null_auroc.observed, 4),
        "length_baseline_auroc": (
            None if cell.length_baseline_auroc is None else round(cell.length_baseline_auroc, 4)
        ),
    }


def _corpus_digest(report: EvalReport) -> str:
    """A short, deterministic fingerprint of the scored corpus — no timestamp, no git sha."""
    # Either of those would dirty this file on every regeneration and turn a meaningful diff into
    # noise, which is the same reasoning `plot_frame.record` states for the manifest.
    scores = report.recurrence_scores
    material = "|".join(sorted(scores.groups)) if scores is not None else ""
    return hashlib.sha256(
        f"{report.n_trajectories}:{report.n_stamped}:{material}".encode()
    ).hexdigest()[:16]


def _stratified(report: EvalReport) -> plots.StratifiedAuroc:
    """The canonical (edit-gated) recurrence score pooled, then within model, then within task."""
    scores = report.recurrence_scores
    if scores is None or not scores.labels:
        return plots.StratifiedAuroc(pooled=0.5, within_model=None, within_challenge=None)
    return plots.StratifiedAuroc(
        pooled=metrics.auroc(scores.edit, scores.labels),
        within_model=metrics.stratified_auroc(scores.edit, scores.labels, scores.models),
        within_challenge=metrics.stratified_auroc(scores.edit, scores.labels, scores.groups),
    )


def _admission_panel(report: EvalReport) -> plots.Admission | None:
    """The prefix admission waterfall's data, or None when no depth was estimable."""
    admission = report.prefix_admission
    if admission is None:
        return None
    return plots.Admission(
        depth=admission.depth,
        n_stamped=admission.n_stamped,
        n_too_short=admission.n_excluded_too_short,
        n_by_margin=admission.n_excluded_by_margin,
        n_admitted=admission.n_admitted,
        admitted_base_rate=admission.admitted_base_rate,
        corpus_base_rate=admission.corpus_base_rate,
    )


def _shipped_config(stale_window: int | None = None) -> EscalationConfig:
    """The shipped escalation knobs, with the window optionally overridden."""
    shipped = _shipped_escalation()
    return EscalationConfig(
        enabled=True,
        escalate_after_n=shipped[0],
        stale_window=shipped[1] if stale_window is None else stale_window,
        ladder=shipped[2],
    )


def _recurrence_scores(stamped: Sequence[Trajectory]) -> RecurrenceScores:
    """Per-run continuous recurrence scores (as-shipped + edit-gated) and the terminal labels."""
    # The shipped config's stale_window, held fixed for BOTH scores. The continuous score's whole
    # purpose is to be threshold-free in `escalate_after_n`, and the window only gates retirement,
    # so the shipped value is the honest default — and pinning it is what makes the two curves
    # differ in exactly one thing. Uses the shipped config directly (never the user override) so
    # the figure means the same thing wherever it is rendered.
    cfg = _shipped_config()
    plain: list[float] = []
    edit: list[float] = []
    labels: list[bool] = []
    groups: list[str] = []
    models: list[str] = []
    for traj in stamped:
        plain.append(float(replay.max_recurrence(traj, cfg)))
        edit.append(float(replay.max_recurrence(traj, cfg, count_from_first_edit=True)))
        labels.append(not traj.header.terminal_resolved)
        # The SAME grouping key the policy half uses — the figure's detection floor must permute
        # at the same exchangeable unit the rest of the eval does.
        groups.append(features.group_of(traj))
        models.append(features.model_of(traj))
    return RecurrenceScores(tuple(plain), tuple(edit), tuple(labels), tuple(groups), tuple(models))


# An effectively unbounded window: no run in the corpus is anywhere near this long, so the
# recurrence counter never retires an event. The contrast against the shipped window is what the
# `window_sensitivity` field publishes.
_UNBOUNDED_WINDOW: Final[int] = 1000


def _window_sensitivity(stamped: Sequence[Trajectory]) -> dict[str, float]:
    """Both scores' AUROC at the shipped window and at an unbounded one — the knob's own size."""
    labels = [not t.header.terminal_resolved for t in stamped]
    if not labels or all(labels) or not any(labels):
        return {}
    shipped_window = _shipped_escalation()[1]
    out: dict[str, float] = {}
    for window in (shipped_window, _UNBOUNDED_WINDOW):
        cfg = _shipped_config(window)
        for gated, name in ((False, "as_shipped"), (True, "edit_gated")):
            scores = [
                float(replay.max_recurrence(t, cfg, count_from_first_edit=gated)) for t in stamped
            ]
            out[f"{name}_auroc_at_stale_window_{window}"] = metrics.auroc(scores, labels)
    return out


def _canonical_model_arms(stamped: Sequence[Trajectory]) -> list[plots.ModelArm]:
    """The canonical cell's two outcome arms, split per model — one replay pass, aggregated."""
    # Per-model, not per-run: the figure asks whether the separation survives inside a model, and
    # a pooled edge that only exists because hard models both fire more and fail more would be a
    # confound the pooled bars cannot show.
    cfg = _shipped_config()
    counts: dict[str, list[int]] = {}
    for traj in stamped:
        fired = replay.replay_config(traj, cfg, count_from_first_edit=True).escalated
        failed = not traj.header.terminal_resolved
        row = counts.setdefault(features.model_of(traj), [0, 0, 0, 0])
        row[0 if fired else 2] += int(failed)
        row[1 if fired else 3] += int(not failed)
    arms: list[plots.ModelArm] = []
    for model in sorted(counts):
        tp, fp, fn, tn = counts[model]
        arms.append(
            plots.ModelArm(
                model=model,
                n=tp + fp + fn + tn,
                p_fail_fired=tp / (tp + fp) if tp + fp else None,
                p_fail_quiet=fn / (fn + tn) if fn + tn else None,
            )
        )
    return arms


def _prefix_admission(
    stamped: Sequence[Trajectory], depths: Sequence[DepthReport]
) -> PrefixAdmission | None:
    """The scope the prefix half's verdict is actually about — its own admitted population."""
    if not depths:
        return None
    depth = max(depths, key=lambda d: d.n_rows)
    return PrefixAdmission(
        depth=depth.depth,
        n_stamped=len(stamped),
        n_excluded_too_short=depth.n_excluded_too_short,
        n_excluded_by_margin=depth.n_excluded_by_margin,
        n_admitted=depth.n_rows,
        admitted_base_rate=depth.base_rate,
        corpus_base_rate=metrics.prevalence([not t.header.terminal_resolved for t in stamped]),
    )


def _recurrence_null(
    scores: RecurrenceScores,
    *,
    n_permutations: int,
) -> tuple[metrics.NullResult, tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    """The recurrence score's detection floor: a scalar null and the drawn chance-curve band."""
    # Both the scalar AUROC null and the ROC band come from the SAME challenge-block shuffles
    # (`policy_eval.null_roc_band`), so the footnote's "null 95% [x, y]" is the band's own
    # construction, not a second estimator. The family-wise max over the two score vectors keeps
    # the floor honest to the figure showing both curves: either curve clearing it is the claim.
    score_vectors = (scores.plain, scores.edit)
    labels = scores.labels
    band = policy_eval.null_roc_band(
        score_vectors, labels, scores.groups, n_permutations=n_permutations, seed=0
    )
    observed = max(metrics.auroc(vector, labels) for vector in score_vectors)
    draws = policy_eval.family_null_draws(
        score_vectors, labels, scores.groups, n_permutations=n_permutations, seed=0
    )
    return metrics.permutation_null(observed, draws), band


def _best_skilled_cell(cells: Sequence[PolicyCell]) -> PolicyCell | None:
    """The strongest cell that clears its OWN family-wise null, ranked by AUROC (detection)."""
    # AUROC first, precision second — the verdict is a detection claim; ranking by precision
    # alone would headline a 69-run tail cell over the 435-run cell that actually separates. The
    # single definition shared by the report property and the status gate so they cannot disagree.
    skilled = [c for c in cells if c.has_skill]
    return max(skilled, key=lambda c: (c.null_auroc.observed, c.precision or 0.0), default=None)


def canonical_cell(report: EvalReport) -> PolicyCell | None:
    """The one operating point every per-cell figure reads: edit-gated at the SHIPPED knobs.

    None only on a hand-built report — `evaluate` guarantees the shipped point is in both grids.
    """
    return next((c for c in report.policy_cells_edit_gated if _is_shipped(c)), None)


def _print_summary(report: EvalReport) -> None:
    """Print the policy sweep and the prefix-model table to stdout."""
    print(f"\nstatus: {report.status} — {report.reason}")  # noqa: T201
    # Immediately under the status, because the two are read together: `status` is about signal,
    # this is about scope, and a skill verdict quoted without it is quoted as if deployable.
    print(  # noqa: T201
        f"deployability: {report.deployability.label} — {report.deployability.reason}"
    )
    print(  # noqa: T201
        f"\n{'n':>3} {'sw':>3} {'esc':>5} {'P(fail|fired)':>14} {'95% CI':>18} "
        f"{'base':>6} {'lift':>6}"
    )
    for c in report.policy_cells:
        marker = "  <- shipped default" if _is_shipped(c) else ""
        precision = "n/a" if c.precision is None else f"{c.precision:.3f}"
        lift = "n/a" if c.lift is None else f"{c.lift:.2f}x"
        print(  # noqa: T201
            f"{c.escalate_after_n:>3} {c.stale_window:>3} {c.n_escalated:>5} {precision:>14} "
            f"{f'[{c.precision_ci[0]:.3f}, {c.precision_ci[1]:.3f}]':>18} "
            f"{c.base_failure_rate:>6.3f} {lift:>6}{marker}"
        )
    if report.policy_cells_edit_gated:
        # The reproduction-phase-gated family, printed under the shipped one. Read the precision
        # against the same base rate: the gated cells separate where the ungated ones mask the
        # signal (n=2 as-shipped fires on everything; n=2 edit-gated carries a real edge).
        print("\nEDIT-GATED sweep (failures before the first edit are not escalation evidence):")  # noqa: T201
        for c in report.policy_cells_edit_gated:
            precision = "n/a" if c.precision is None else f"{c.precision:.3f}"
            lift = "n/a" if c.lift is None else f"{c.lift:.2f}x"
            length = "n/a" if c.length_baseline_auroc is None else f"{c.length_baseline_auroc:.3f}"
            print(  # noqa: T201
                f"{c.escalate_after_n:>3} {c.stale_window:>3} {c.n_escalated:>5} {precision:>14} "
                f"{f'[{c.precision_ci[0]:.3f}, {c.precision_ci[1]:.3f}]':>18} "
                f"{c.base_failure_rate:>6.3f} {lift:>6}  len-only {length}"
            )
    # Both p-values, side by side: `raw p` is this depth's own null and gates nothing, `fw p` is
    # the family-wise adjusted p the verdict is read off. Printing only the first is how a max over
    # depths came to be quoted as if it were one test.
    print(  # noqa: T201
        f"\n{'depth':>5} {'n':>5} {'prior':>7} {'leaked':>7} {'prefix':>7} {'combined':>9} "
        f"{'incr':>7} {'raw p':>7} {'fw p':>7}"
    )
    for d in report.depth_reports:
        print(  # noqa: T201
            f"{d.depth:>5} {d.n_rows:>5} {d.auroc_prior:>7.3f} "
            f"{d.auroc_prior_leaked:>7.3f} {d.auroc_prefix:>7.3f} "
            f"{d.auroc_combined:>9.3f} {d.incremental_auroc:>+7.3f} "
            f"{d.null_incremental.p_value:>7.3f} {d.gate_null_incremental.p_value:>7.3f}"
        )
    if report.policy_value is not None:
        # Printed even (especially) when it refuses: a visible `not_identified` is the point —
        # it says the escalation policy's value is UNMEASURED, not that it is zero.
        pv = report.policy_value
        print(f"\nescalation OPE: {pv.status} — {pv.reason}")  # noqa: T201
        if pv.contrast_estimate is not None:
            # The CONTRAST, not the level, is the answer to "does escalating help" — printed on
            # its own line so a reader cannot mistake V(always_escalate) for a comparison.
            verdict = "" if pv.contrast_excludes_zero else " — interval spans 0, no measured effect"
            print(  # noqa: T201
                f"escalation OPE contrast V(escalate) - V(hold): {pv.contrast_estimate:+.4f} "
                f"[{pv.contrast_ci_low:+.4f}, {pv.contrast_ci_high:+.4f}]{verdict}"
            )
    if report.session_cadence is not None:
        sc = report.session_cadence
        lift = "n/a" if sc.lift is None else f"{sc.lift:.2f}x"
        print("\nsession-cadence escalation value (observational, same-instance subset):")  # noqa: T201
        print(  # noqa: T201
            f"  P(frontier resolves | cheap failed) = {sc.escalate_rate:.3f} "
            f"[{sc.escalate_ci[0]:.3f}, {sc.escalate_ci[1]:.3f}] "
            f"({sc.n_escalated_resolved}/{sc.n_escalated})"
        )
        print(  # noqa: T201
            f"  P(cheap retry resolves | cheap failed) = {sc.retry_rate:.3f} "
            f"[{sc.retry_ci[0]:.3f}, {sc.retry_ci[1]:.3f}] "
            f"({sc.n_retried_resolved}/{sc.n_retried})"
        )
        print(f"  lift {lift} over {sc.n_overlap_instances} overlap instances")  # noqa: T201
    for note in report.notes:
        print(f"\nNOTE: {note}", file=sys.stderr)  # noqa: T201


def _permutations(raw: str) -> int:
    """argparse type: a draw count at or above the floor the null estimator requires."""
    # Below the floor `metrics.permutation_null` raises deep inside the pipeline, so
    # `--permutations 100` printed a ValueError traceback instead of a usage error — a CLI
    # contract enforced by a crash. The floor is checked where the argument is parsed.
    value = int(raw)
    if value < metrics.MIN_PERMUTATIONS:
        raise argparse.ArgumentTypeError(
            f"must be >= {metrics.MIN_PERMUTATIONS} (metrics.MIN_PERMUTATIONS): a null estimated "
            f"off fewer draws has a 97.5th percentile that is itself noise; got {value}"
        )
    return value


def _depth(raw: str) -> int:
    """argparse type: a decision depth is an absolute step count, so it must be positive."""
    # A non-positive depth used to fail deep inside the prefix admission/refit path — the same
    # CLI-contract-enforced-by-a-crash defect `_permutations` closes. Checked at parse time.
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"must be > 0 (a decision depth is an absolute step count); got {value}"
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="escalation-detector offline eval")
    here = Path(__file__).resolve()
    parser.add_argument(
        "--live-dir",
        type=Path,
        default=here.parent / "data/live",
        help="Real captured live trajectories (schema JSONL).",
    )
    # Both default to their committed homes: `make escalation-eval` must actually refresh the
    # figures, or they silently rot while the JSON report says everything is current. The PNGs
    # live inside the published docs tree so the docs can link them relatively; metrics.json
    # stays beside the code, because it is the numbers behind the images, not an image.
    # CANONICAL_PLOTS_DIR itself, so the default and `_committed_home` cannot drift apart.
    parser.add_argument("--plots-dir", type=Path, default=CANONICAL_PLOTS_DIR)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Where metrics.json lands. Defaults to benchmark/escalation/reports/ for the "
        "committed figure set, and beside the plots for any other --plots-dir.",
    )
    parser.add_argument(
        "--permutations",
        type=_permutations,
        default=metrics.DEFAULT_PERMUTATIONS,
        help=(
            f"Label shuffles behind every null band (the admissibility gate reads this). "
            f"Default {metrics.DEFAULT_PERMUTATIONS}, minimum {metrics.MIN_PERMUTATIONS}. At the "
            f"minimum every reported p-value is the floor artifact 1/(n+1)."
        ),
    )
    parser.add_argument(
        "--depths",
        type=_depth,
        nargs="+",
        default=list(features.DEFAULT_DEPTHS),
        help="Decision depths (absolute, positive step counts) to score prefixes at.",
    )
    parser.add_argument(
        "--exploration-db",
        type=Path,
        default=None,
        help=(
            "OutcomeStore sqlite path. When given, ALSO report the off-policy escalation "
            "value — or, on a deterministic log, its honest refusal."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the eval on the live corpus and print the JSON report."""
    args = _build_parser().parse_args(argv)
    if not args.live_dir.exists():
        logger.error("live trajectory directory not found: %s", args.live_dir)
        return 1
    trajectories = [load_jsonl(p) for p in sorted(args.live_dir.glob("*.jsonl"))]
    if not trajectories:
        logger.error("no trajectories found under %s", args.live_dir)
        return 1
    exploration_rows = None
    if args.exploration_db is not None:
        from shunt.db.store import OutcomeStore

        store = OutcomeStore(db_path=str(args.exploration_db))
        try:
            exploration_rows = store.escalation_exploration_rows()
        finally:
            store.close()
    report = evaluate(
        trajectories,
        depths=args.depths,
        n_permutations=args.permutations,
        exploration_rows=exploration_rows,
    )
    if args.plots_dir is not None:
        _save_plots(report, args.plots_dir, args.metrics_dir)
    print(json.dumps(report.to_dict(), indent=2))  # noqa: T201 (CLI report to stdout)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
