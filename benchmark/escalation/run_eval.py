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
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
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
from benchmark.escalation.deployability import Cadence, Deployability
from benchmark.escalation.ope import (
    PolicyValueEstimate,
    estimate_policy_value,
    rows_from_records,
)
from benchmark.escalation.schema import Trajectory, load_jsonl
from benchmark.plot_frame import Annotations, FigureSpec
from shunt.router.policy import load_router_policy, packaged_policy_path

matplotlib.use("Agg")

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from benchmark.escalation.features import ModelCoverage
    from benchmark.escalation.policy_eval import PolicyCell
    from benchmark.escalation.prefix_eval import DepthReport

logger = logging.getLogger(__name__)

_FIGSIZE: Final[tuple[float, float]] = (9.0, 5.5)
_DPI: Final[int] = 150
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
    depth_reports: list[DepthReport]
    coverage: list[ModelCoverage]
    # REQUIRED, deliberately not defaulted: a skill verdict says whether a signal is there, and
    # this says whether that signal is one production could act on. A default would let a report
    # be built — and its numbers quoted — without anyone stating which of the two it is.
    deployability: Deployability
    notes: list[str] = field(default_factory=list)
    # The off-policy escalation-value estimate, present only when a real exploration log was
    # supplied. None means "not asked", NOT "no effect" — the two must stay distinguishable.
    policy_value: PolicyValueEstimate | None = None

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
        return {
            "status": self.status,
            "reason": self.reason,
            # Beside `status`, not buried: `status` says whether a signal was found, this says
            # whether the thing that found it is a policy production could run.
            "deployability": self.deployability.to_dict(),
            "n_trajectories": self.n_trajectories,
            "n_stamped": self.n_stamped,
            "n_multistep": self.n_multistep,
            "authenticity_errors": self.authenticity_errors,
            "headline_policy_cell": None if cell is None else cell.to_dict(),
            "policy_cells": [c.to_dict() for c in self.policy_cells],
            "prefix_model": [d.to_dict() for d in self.depth_reports],
            "capture_coverage": [c.to_dict() for c in self.coverage],
            "notes": self.notes,
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
    skilled_cells = [c for c in report_cells if c.has_skill]
    if skilled_cells:
        best = max(skilled_cells, key=lambda c: c.precision or 0.0)
        return (
            ok_status,
            f"escalation policy at escalate_after_n={best.escalate_after_n}/"
            f"stale_window={best.stale_window} fires on {best.n_escalated} of "
            f"{best.n_trajectories} trajectories and clears its family-wise permutation null: "
            f"AUROC {best.null_auroc.observed:.3f} against the max-over-cells null 95% "
            f"[{best.gate_null.ci_low:.3f}, {best.gate_null.ci_high:.3f}] (adjusted "
            f"p={best.gate_null.p_value:.3f} over {len(report_cells)} swept cell(s)), with "
            f"P(fail|fired)={best.precision:.3f} [{best.precision_ci[0]:.3f}, "
            f"{best.precision_ci[1]:.3f}] above the base failure rate {best.base_failure_rate:.3f}"
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
    return _STATUS_NO_SKILL, _no_skill_reason(report_cells, depths)


def _offline_only_caveat(deployability: Deployability | None) -> str:
    """The scope sentence a non-deployable OK reason must carry, or the empty string."""
    if deployability is None or deployability.deployable:
        return ""
    return (
        f" — {deployability.label}, so this is a per-step signal, not a shipped "
        "escalation: production decides once per session"
    )


def _no_skill_reason(report_cells: Sequence[PolicyCell], depths: Sequence[DepthReport]) -> str:
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
    policy = _policy_skill_failure(report_cells)
    return (
        f"no depth satisfies all {len(conditions)} skill conditions; at depth {best.depth} "
        + "; ".join(failures)
        + _cleared_increment_caveat(best)
        + policy
    )


def _policy_skill_failure(report_cells: Sequence[PolicyCell]) -> str:
    """The policy half's closest-miss, stated so the sweep is not reported as unexplained."""
    if not report_cells:
        return ""
    fired = [c for c in report_cells if c.precision is not None]
    if not fired:
        return f"; no swept configuration fired on any trajectory ({len(report_cells)} cells swept)"
    # The cell that came closest on the two clauses that could separate it: its own null was the
    # family-wise reference, so quote the marginal conditions that failed per cell.
    best = max(fired, key=lambda c: c.precision or 0.0)
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
        f"; the best swept cell (escalate_after_n={best.escalate_after_n}/"
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
    n_permutations: int = metrics.MIN_PERMUTATIONS,
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
    depth_reports = prefix_eval.evaluate(stamped, depths, n_permutations=n_permutations)
    authenticity_errors = sum(len(errors(verify_trajectory(t))) for t in trajectories)
    deployability = _deployability(stamped)
    status, reason = _status(policy_cells, depth_reports, authenticity_errors, deployability)
    return EvalReport(
        status=status,
        reason=reason,
        n_trajectories=len(trajectories),
        n_stamped=len(stamped),
        n_multistep=len(multistep),
        authenticity_errors=authenticity_errors,
        policy_cells=policy_cells,
        depth_reports=depth_reports,
        coverage=features.model_coverage(trajectories),
        deployability=_deployability(stamped),
        notes=_corpus_notes(trajectories, stamped, policy_cells),
        # Only ever computed from a REAL exploration log. Trajectories carry no propensity
        # (the committable whitelist has no `action`/`propensity`), so deriving rows from them
        # would manufacture propensity=1.0 for every decision — a fabricated input, not data.
        policy_value=(
            None
            if exploration_rows is None
            else estimate_policy_value(rows_from_records(exploration_rows))
        ),
    )


def _deployability(stamped: Sequence[Trajectory]) -> Deployability:
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
    return deployability.assess(features.FEATURE_NAMES, Cadence.STEP, unfilled=unfilled)


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


def _run_annotations(report: EvalReport) -> Annotations:
    """Status caveats every figure must carry, whatever it plots."""
    limitations: list[str] = []
    if report.status == _STATUS_NO_SKILL:
        limitations.append(f"NO USABLE SIGNAL — {report.reason}.")
    if report.status == _STATUS_INSUFFICIENT:
        limitations.append(f"INSUFFICIENT DATA — {report.reason}.")
    if report.status == _STATUS_UNVERIFIED:
        limitations.append(f"CORPUS FAILED ITS INTEGRITY CHECK — {report.reason}.")
    dropped = report.n_trajectories - report.n_stamped
    if dropped:
        limitations.append(
            f"{dropped}/{report.n_trajectories} trajectories have no per-step verified outcomes "
            "and are excluded from this figure."
        )
    return Annotations(
        notes=(
            f"{report.n_stamped} scored trajectories, status={report.status}",
            # On EVERY figure, whatever it plots: a reader who takes a number off a plot must see
            # which question it answers without going back to the JSON.
            f"{report.deployability.label} — {report.deployability.reason}",
        ),
        limitations=tuple(limitations),
    )


def _merge(*parts: Annotations) -> Annotations:
    """Concatenate annotation blocks; FigureSpec.merged dedups and drops blanks."""
    return Annotations(
        definitions=tuple(d for part in parts for d in part.definitions),
        notes=tuple(n for part in parts for n in part.notes),
        limitations=tuple(lim for part in parts for lim in part.limitations),
    )


def _save_plots(report: EvalReport, out_dir: Path) -> None:
    """Render every figure to `out_dir`. Figures whose data is absent are skipped, not faked."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run = _run_annotations(report)
    _render(
        out_dir / "failure_capture_coverage.png",
        plots.CAPTURE_COVERAGE_SPEC,
        lambda ax: _merge(plots.capture_coverage(report.coverage, ax), run),
    )
    if report.policy_cells:
        _save_policy_plots(report, out_dir, run)
        # The prefix risk model's own figures are NOT drawn: its score is constant at the
        # evaluated depths (early steps are all bug-reproduction, so the model ranks nothing), and
        # a degenerate curve invites the reader to see "escalation does not work" when the
        # escalation METHOD (the policy) carries the measurable edge. The prefix model's full
        # numbers stay in the JSON report, honestly NO_SKILL.


def _save_policy_plots(report: EvalReport, out_dir: Path, run: Annotations) -> None:
    """The escalation method's figures: sweep table, outcome bars, and the operating
    characteristic (PR/ROC), the separating cell's confusion, and its family-wise null."""
    cells = report.policy_cells
    cell = report.headline_cell
    best = _best_separating_cell(cells)
    _render(
        out_dir / "sweep_table.png",
        plots.SWEEP_TABLE_SPEC,
        lambda ax: _merge(
            plots.sweep_table(
                cells,
                ax,
                shipped_index=next((i for i, c in enumerate(cells) if _is_shipped(c)), None),
            ),
            run,
        ),
    )
    if best is None:
        # Only reachable on a hand-built report with no fired cell. The curve/confusion/null
        # figures are SKIPPED rather than drawn off nothing — see `policy_pr_curve`'s own refusal.
        logger.error("no swept cell fired; operating-characteristic figures skipped")
        return
    _render(
        out_dir / "pr_curve.png",
        plots.PR_CURVE_SPEC,
        lambda ax: _merge(plots.policy_pr_curve(cells, ax), run),
    )
    _render(
        out_dir / "roc_curve.png",
        plots.ROC_CURVE_SPEC,
        lambda ax: _merge(plots.policy_roc_curve(cells, ax), run),
    )
    _render(
        out_dir / "confusion_matrix.png",
        plots.CONFUSION_MATRIX_SPEC,
        lambda ax: _merge(
            plots.policy_confusion(best, ax, flag_budget=best.n_escalated),
            Annotations(notes=(_policy_cell_note(best),)),
            run,
        ),
    )
    _render(
        out_dir / "permutation_null.png",
        plots.PERMUTATION_NULL_SPEC,
        lambda ax: _merge(
            plots.permutation_null_plot(
                best.gate_null,
                ax,
                label=(
                    f"policy AUROC at escalate_after_n={best.escalate_after_n} "
                    "(family-wise null across the sweep)"
                ),
            ),
            Annotations(notes=(_policy_cell_note(best),)),
            run,
        ),
    )
    if cell is None:
        # Only reachable on a hand-built report: `evaluate` guarantees the shipped cell exists.
        # The per-cell figure is SKIPPED rather than drawn off an arbitrary cell — a figure
        # titled with the shipped default that plots a different configuration is the defect.
        logger.error("no swept cell matches the shipped configuration; per-cell figure skipped")
        return
    _render(
        out_dir / "trajectory_outcomes.png",
        plots.OUTCOME_BARS_SPEC,
        lambda ax: _merge(plots.outcome_bars(cell, ax), run),
    )


def _best_separating_cell(cells: Sequence[PolicyCell]) -> PolicyCell | None:
    """The most readable operating point: skilled, else the highest-precision fired cell."""
    skilled = [c for c in cells if c.has_skill]
    pool = skilled or [c for c in cells if c.precision is not None]
    return max(pool, key=lambda c: c.precision or 0.0, default=None)


def _policy_cell_note(cell: PolicyCell) -> str:
    """The one line that anchors a policy figure to its operating point."""
    return (
        f"cell: escalate_after_n={cell.escalate_after_n}, stale_window={cell.stale_window}; "
        f"fired on {cell.n_escalated}/{cell.n_trajectories}; P(fail|fired)={cell.precision:.3f} "
        f"[{cell.precision_ci[0]:.3f}, {cell.precision_ci[1]:.3f}] vs base "
        f"{cell.base_failure_rate:.3f}"
    )


def _render(path: Path, spec: FigureSpec, draw: Callable[[Axes], Annotations]) -> None:
    """Every figure goes through the annotated frame — the one legal savefig site (SH007)."""
    plot_frame.render(path, spec, draw, figsize=_FIGSIZE, dpi=_DPI)


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
    # Defaults to the committed reports dir: `make escalation-eval` must actually refresh the
    # figures, or they silently rot while the JSON report says everything is current.
    parser.add_argument("--plots-dir", type=Path, default=here.parent / "reports")
    parser.add_argument(
        "--permutations",
        type=_permutations,
        default=metrics.MIN_PERMUTATIONS,
        help=(
            f"Label shuffles behind every null band (the admissibility gate reads this). "
            f"Minimum {metrics.MIN_PERMUTATIONS}."
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
        _save_plots(report, args.plots_dir)
    print(json.dumps(report.to_dict(), indent=2))  # noqa: T201 (CLI report to stdout)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
