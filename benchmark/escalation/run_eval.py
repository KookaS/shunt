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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib

from benchmark import plot_frame
from benchmark.escalation import (
    datasets,
    features,
    metrics,
    plots,
    policy_eval,
    prefix_eval,
    replay,
)
from benchmark.escalation.authenticity import errors, verify_trajectory
from benchmark.escalation.schema import Trajectory, load_jsonl
from benchmark.plot_frame import Annotations, FigureSpec

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
_STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
_STATUS_NO_SKILL = "NO_SKILL"
# The configuration the product ships; the sweep must report it even when another cell scores
# better, so a quiet default change can never hide inside an "argmax" line.
SHIPPED_ESCALATE_AFTER_N: Final[int] = 2


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
    notes: list[str] = field(default_factory=list)

    @property
    def headline_cell(self) -> PolicyCell | None:
        """The shipped configuration — the one the product actually runs, never an argmax."""
        shipped = [c for c in self.policy_cells if c.escalate_after_n == SHIPPED_ESCALATE_AFTER_N]
        return shipped[0] if shipped else (self.policy_cells[0] if self.policy_cells else None)

    @property
    def best_depth(self) -> DepthReport | None:
        """The depth with the largest incremental AUROC — reported, never presented as chosen."""
        return max(self.depth_reports, key=lambda d: d.incremental_auroc, default=None)

    def to_dict(self) -> dict[str, object]:
        cell = self.headline_cell
        return {
            "status": self.status,
            "reason": self.reason,
            "n_trajectories": self.n_trajectories,
            "n_stamped": self.n_stamped,
            "n_multistep": self.n_multistep,
            "authenticity_errors": self.authenticity_errors,
            "headline_policy_cell": None if cell is None else cell.to_dict(),
            "policy_cells": [c.to_dict() for c in self.policy_cells],
            "prefix_model": [d.to_dict() for d in self.depth_reports],
            "capture_coverage": [c.to_dict() for c in self.coverage],
            "notes": self.notes,
        }


def _status(report_cells: Sequence[PolicyCell], depths: Sequence[DepthReport]) -> tuple[str, str]:
    """Gate on the permutation null: only a result outside its own null band may report OK."""
    if not report_cells and not depths:
        return _STATUS_INSUFFICIENT, "no trajectory reached a scorable depth — nothing to measure"
    skilled = [d for d in depths if d.has_skill]
    if skilled:
        best = max(skilled, key=lambda d: d.incremental_auroc)
        return (
            _STATUS_OK,
            f"prefix risk model clears its permutation null at depth {best.depth}: "
            f"incremental AUROC {best.incremental_auroc:+.3f} over the t=0 task prior "
            f"(null 97.5th pct {best.null_incremental.ci_high:+.3f}, "
            f"p={best.null_incremental.p_value:.3f})",
        )
    return _STATUS_NO_SKILL, _no_skill_reason(depths)


def _no_skill_reason(depths: Sequence[DepthReport]) -> str:
    """State which half of the gate failed, with the number, at the depth that came closest."""
    if not depths:
        return "no depth was estimable — the corpus cannot support a grouped-CV fit"
    best = max(depths, key=lambda d: d.incremental_auroc)
    return (
        f"no depth clears its permutation null; best is depth {best.depth} with incremental "
        f"AUROC {best.incremental_auroc:+.3f} (null 95% "
        f"[{best.null_incremental.ci_low:+.3f}, {best.null_incremental.ci_high:+.3f}], "
        f"p={best.null_incremental.p_value:.3f}) — no usable signal"
    )


def evaluate(
    trajectories: Sequence[Trajectory],
    grid: Sequence[replay.GridPoint] = tuple(datasets.DEFAULT_GRID),
    *,
    depths: Sequence[int] = features.DEFAULT_DEPTHS,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
) -> EvalReport:
    """Score the shipped policy and the prefix risk model, then gate on the permutation null."""
    stamped = [t for t in trajectories if features.is_stamped(t)]
    multistep = [t for t in trajectories if not datasets.is_degenerate(t)]
    policy_cells = policy_eval.evaluate(stamped, list(grid), n_permutations=n_permutations)
    depth_reports = prefix_eval.evaluate(stamped, depths, n_permutations=n_permutations)
    status, reason = _status(policy_cells, depth_reports)
    return EvalReport(
        status=status,
        reason=reason,
        n_trajectories=len(trajectories),
        n_stamped=len(stamped),
        n_multistep=len(multistep),
        authenticity_errors=sum(len(errors(verify_trajectory(t))) for t in trajectories),
        policy_cells=policy_cells,
        depth_reports=depth_reports,
        coverage=features.model_coverage(trajectories),
        notes=_corpus_notes(trajectories, stamped, policy_cells),
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
    shipped = next((c for c in cells if c.escalate_after_n == SHIPPED_ESCALATE_AFTER_N), None)
    if shipped is None or not cells:
        return []
    best = max(cells, key=lambda c: c.precision)
    if best.escalate_after_n == shipped.escalate_after_n:
        return []
    return [
        f"escalate_after_n={best.escalate_after_n} reaches P(fail|fired)={best.precision:.3f} "
        f"against the SHIPPED default n={shipped.escalate_after_n}'s {shipped.precision:.3f}. "
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
    dropped = report.n_trajectories - report.n_stamped
    if dropped:
        limitations.append(
            f"{dropped}/{report.n_trajectories} trajectories have no per-step verified outcomes "
            "and are excluded from this figure."
        )
    return Annotations(
        notes=(f"{report.n_stamped} scored trajectories, status={report.status}",),
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
    if report.best_depth is not None:
        _save_prefix_plots(report.best_depth, out_dir, run)


def _save_policy_plots(report: EvalReport, out_dir: Path, run: Annotations) -> None:
    """The shipped policy's trajectory-level figures (sweep table, outcome bars, lead time)."""
    cells = report.policy_cells
    cell = report.headline_cell
    assert cell is not None  # noqa: S101 (guarded by the caller's `if report.policy_cells`)
    _render(
        out_dir / "sweep_table.png",
        plots.SWEEP_TABLE_SPEC,
        lambda ax: _merge(plots.sweep_table(cells, ax), run),
    )
    _render(
        out_dir / "trajectory_outcomes.png",
        plots.OUTCOME_BARS_SPEC,
        lambda ax: _merge(plots.outcome_bars(cell, ax), run),
    )
    _render(
        out_dir / "lead_time_by_outcome.png",
        plots.LEAD_TIME_SPEC,
        lambda ax: _merge(plots.lead_time_by_outcome(cell, ax), run),
    )


def _save_prefix_plots(depth: DepthReport, out_dir: Path, run: Annotations) -> None:
    """The risk model's figures — every one drawn against its own permutation null."""
    scores = list(depth.scores)
    labels = list(depth.labels)
    detail = Annotations(notes=(_depth_note(depth),))
    _render(
        out_dir / "pr_curve.png",
        plots.PR_CURVE_SPEC,
        lambda ax: _merge(plots.pr_curve(scores, labels, ax), detail, run),
    )
    _render(
        out_dir / "roc_curve.png",
        plots.ROC_CURVE_SPEC,
        lambda ax: _merge(plots.roc_curve(scores, labels, depth.null_prefix, ax), detail, run),
    )
    _render(
        out_dir / "confusion_matrix.png",
        plots.CONFUSION_MATRIX_SPEC,
        lambda ax: _merge(
            plots.confusion_matrix_plot(metrics.detection_metrics(scores, labels).confusion, ax),
            detail,
            run,
        ),
    )
    _render(
        out_dir / "permutation_null.png",
        plots.PERMUTATION_NULL_SPEC,
        lambda ax: _merge(
            plots.permutation_null_plot(
                depth.null_incremental, ax, label="incremental AUROC over the t=0 task prior"
            ),
            detail,
            run,
        ),
    )


def _depth_note(depth: DepthReport) -> str:
    """The one line that stops a reader mistaking raw discrimination for incremental value."""
    return (
        f"prefix depth {depth.depth} decisions, {depth.n_rows} trajectories over "
        f"{depth.n_groups} challenges; AUROC prior-only={depth.auroc_prior:.3f}, "
        f"prefix-only={depth.auroc_prefix:.3f}, combined={depth.auroc_combined:.3f}, "
        f"incremental={depth.incremental_auroc:+.3f}"
    )


def _render(path: Path, spec: FigureSpec, draw: Callable[[Axes], Annotations]) -> None:
    """Every figure goes through the annotated frame — the one legal savefig site (SH007)."""
    plot_frame.render(path, spec, draw, figsize=_FIGSIZE, dpi=_DPI)


def _print_summary(report: EvalReport) -> None:
    """Print the policy sweep and the prefix-model table to stdout."""
    print(f"\nstatus: {report.status} — {report.reason}")  # noqa: T201
    print(  # noqa: T201
        f"\n{'n':>3} {'esc':>5} {'P(fail|fired)':>14} {'95% CI':>18} {'base':>6} {'lift':>6}"
    )
    for c in report.policy_cells:
        marker = "  <- shipped default" if c.escalate_after_n == SHIPPED_ESCALATE_AFTER_N else ""
        print(  # noqa: T201
            f"{c.escalate_after_n:>3} {c.n_escalated:>5} {c.precision:>14.3f} "
            f"{f'[{c.precision_ci[0]:.3f}, {c.precision_ci[1]:.3f}]':>18} "
            f"{c.base_failure_rate:>6.3f} {c.lift:>5.2f}x{marker}"
        )
    print(  # noqa: T201
        f"\n{'depth':>5} {'n':>5} {'prior':>7} {'prefix':>7} {'combined':>9} "
        f"{'incr':>7} {'null p':>7}"
    )
    for d in report.depth_reports:
        print(  # noqa: T201
            f"{d.depth:>5} {d.n_rows:>5} {d.auroc_prior:>7.3f} {d.auroc_prefix:>7.3f} "
            f"{d.auroc_combined:>9.3f} {d.incremental_auroc:>+7.3f} "
            f"{d.null_incremental.p_value:>7.3f}"
        )
    for note in report.notes:
        print(f"\nNOTE: {note}", file=sys.stderr)  # noqa: T201


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
        type=int,
        default=metrics.MIN_PERMUTATIONS,
        help="Label shuffles behind every null band (the skill gate reads this).",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=list(features.DEFAULT_DEPTHS),
        help="Decision depths (absolute, never fractional) to score prefixes at.",
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
    report = evaluate(trajectories, depths=args.depths, n_permutations=args.permutations)
    if args.plots_dir is not None:
        _save_plots(report, args.plots_dir)
    print(json.dumps(report.to_dict(), indent=2))  # noqa: T201 (CLI report to stdout)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
