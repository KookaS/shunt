"""End-to-end escalation-detector eval CLI: normalize -> replay-sweep -> full metric suite ->"""

# per-cell best-config selection -> plots -> authenticity. Runs on the checked-in fixtures, the
# terminal results.csv bootstrap, and any real captured live trajectories. On data that cannot
# exercise the recurrence trigger it reports status=INSUFFICIENT_DATA with the reason (never a
# misleading auprc==prevalence), and every plot carries a visible insufficient-data annotation.

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import matplotlib

from benchmark.escalation import datasets, metrics, plots, replay
from benchmark.escalation.authenticity import errors, verify_trajectory
from benchmark.escalation.metrics import CellReport
from benchmark.escalation.normalize.base import TrajectoryParser
from benchmark.escalation.normalize.openhands import OpenHandsParser
from benchmark.escalation.normalize.swe_agent import SweAgentParser
from benchmark.escalation.normalize.swe_smith import SweSmithParser
from benchmark.escalation.schema import Trajectory, load_jsonl
from shunt.router.escalation import EscalationAction

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

_HORIZON = 3
_STATUS_OK = "OK"
_STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
_STATUS_NO_SKILL = "NO_SKILL"
_PARSERS: Final[dict[str, TrajectoryParser]] = {
    "swe_agent.traj": SweAgentParser(),
    "swe_smith.traj": SweSmithParser(),
    "openhands.json": OpenHandsParser(),
}


@dataclass(frozen=True)
class EvalReport:
    """The eval outcome — status-gated metric suite, per-cell sweep, best config, authenticity."""

    status: str
    reason: str
    n_trajectories: int
    n_degenerate: int
    n_multistep: int
    n_escalated: int
    authenticity_errors: int
    headline: CellReport
    best_config: CellReport | None
    cells: list[CellReport]
    degeneracy_note: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        cm = self.headline.confusion
        return {
            "status": self.status,
            "reason": self.reason,
            "n_trajectories": self.n_trajectories,
            "n_degenerate": self.n_degenerate,
            "n_multistep": self.n_multistep,
            "n_escalated": self.n_escalated,
            "authenticity_errors": self.authenticity_errors,
            "prevalence": round(self.headline.prevalence, 4),
            "auprc": round(self.headline.auprc, 4),
            "auroc": round(self.headline.auroc, 4),
            "confusion": {"tp": cm.tp, "fp": cm.fp, "fn": cm.fn, "tn": cm.tn},
            "precision": round(self.headline.precision, 4),
            "recall": round(self.headline.recall, 4),
            "f1": round(self.headline.f1, 4),
            "fpr": round(self.headline.fpr, 4),
            "cohen_kappa": round(self.headline.cohen_kappa, 4),
            "best_config": None if self.best_config is None else self.best_config.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "degeneracy_note": self.degeneracy_note,
            "notes": self.notes,
        }

    @property
    def insufficient_note(self) -> str | None:
        """Red 'insufficient data' note when data could not exercise the detector, else None."""
        return self.reason if self.status == _STATUS_INSUFFICIENT else None

    @property
    def no_skill_note(self) -> str | None:
        """Orange 'no usable signal' note when data was sufficient but the detector ≤ no-skill."""
        return self.reason if self.status == _STATUS_NO_SKILL else None


def _cumulative_detection(directives: list[EscalationAction]) -> list[float]:
    """Monotone per-prefix detection score: 0.0 until the policy first fires, 1.0 at and after."""
    # The metric asks "has this failing task been flagged by prefix t" — once flagged it stays
    # flagged. This is invariant to the post-first-flag ladder rungs (effort, tier, ceiling), an
    # isolation-model abstraction that need not match the concrete routing ladder; scoring the raw
    # per-step directive there would grade the metric on a directive stream production never runs.
    scores: list[float] = []
    flagged = False
    for action in directives:
        flagged = flagged or action is not EscalationAction.HOLD
        scores.append(1.0 if flagged else 0.0)
    return scores


def _prefix_scores(traj: Trajectory, cfg_point: replay.GridPoint) -> list[float]:
    """The cumulative detection score per prefix from the imported policy replay (see helper)."""
    return _cumulative_detection(replay.replay_config(traj, cfg_point.to_config()).directives)


def _all_labels(trajectories: list[Trajectory]) -> list[bool]:
    """Prefix-risk labels flattened across trajectories (grid-independent — computed once)."""
    labels: list[bool] = []
    for traj in trajectories:
        labels.extend(metrics.label_prefixes(traj, _HORIZON))
    return labels


def _mean_steps_to_detection(point: replay.SweepPoint) -> float | None:
    """Mean first-escalation step over the escalated trajectories of one cell, or None."""
    firsts = [
        d.first_escalation_index for d in point.decisions if d.first_escalation_index is not None
    ]
    return sum(firsts) / len(firsts) if firsts else None


def _cell_reports(sweep: replay.SweepResult, labels: list[bool]) -> list[CellReport]:
    """The full metric suite for EVERY grid cell (not just grid[0]) over the whole corpus."""
    reports: list[CellReport] = []
    for point in sweep.points:
        scores: list[float] = []
        for decision in point.decisions:
            scores.extend(_cumulative_detection(decision.directives))
        gp = point.grid_point
        reports.append(
            metrics.cell_report(
                escalate_after_n=gp.escalate_after_n,
                stale_window=gp.stale_window,
                ladder=gp.ladder,
                scores=scores,
                labels=labels,
                n_escalated=sum(d.escalated for d in point.decisions),
                mean_steps_to_detection=_mean_steps_to_detection(point),
            )
        )
    return reports


def _status(
    labels: list[bool], n_escalated: int, n_multistep: int, headline_scores: list[float]
) -> tuple[str, str]:
    """Gate the report: refuse to present a null result as signal (audit P0)."""
    if not any(labels):
        return _STATUS_INSUFFICIENT, "no positive prefix labels — nothing to detect"
    if n_escalated == 0:
        return (
            _STATUS_INSUFFICIENT,
            f"recurrence trigger never fired — N={n_multistep} multi-step, "
            "0 with recurring failing_check_id",
        )
    if len(set(headline_scores)) <= 1:
        return _STATUS_INSUFFICIENT, "constant detector score — detector not exercised"
    return _STATUS_OK, "sufficient: recurrence trigger fired and positive labels present"


def _apply_skill_gate(status: str, reason: str, headline: CellReport) -> tuple[str, str]:
    """Demote OK→NO_SKILL when data was sufficient but the detector fails to beat no-skill.

    AUPRC-vs-prevalence is primary on this imbalanced problem; a headline at/below prevalence is a
    legible *null* result, not signal — it must never present as OK (audit F2).
    """
    if status != _STATUS_OK or headline.auprc > headline.prevalence:
        return status, reason
    return (
        _STATUS_NO_SKILL,
        f"detector at/below no-skill baseline: AUPRC={headline.auprc:.3f} "
        f"≤ prevalence={headline.prevalence:.3f} — no usable signal",
    )


def evaluate(trajectories: list[Trajectory], grid: list[replay.GridPoint]) -> EvalReport:
    """Score the labeler + full metric suite per cell, pick the best config, gate on sufficiency."""
    labels = _all_labels(trajectories)
    sweep = replay.sweep(trajectories, grid)
    cells = _cell_reports(sweep, labels)
    best = metrics.select_best_config(cells)
    headline = best if best is not None else cells[0]
    n_escalated = sum(c.n_escalated for c in cells) if cells else 0
    n_degenerate = sum(datasets.is_degenerate(t) for t in trajectories)
    n_multistep = len(trajectories) - n_degenerate
    headline_scores = _prefix_scores_flat(trajectories, headline)
    status, reason = _status(labels, n_escalated, n_multistep, headline_scores)
    status, reason = _apply_skill_gate(status, reason, headline)
    return EvalReport(
        status=status,
        reason=reason,
        n_trajectories=len(trajectories),
        n_degenerate=n_degenerate,
        n_multistep=n_multistep,
        n_escalated=n_escalated,
        authenticity_errors=sum(len(errors(verify_trajectory(t))) for t in trajectories),
        headline=headline,
        best_config=best,
        cells=cells,
        degeneracy_note=_degeneracy_note(n_degenerate, len(trajectories)),
    )


def _prefix_scores_flat(trajectories: list[Trajectory], cell: CellReport) -> list[float]:
    """Flatten a cell's cumulative-detection scores across the corpus (for the status gate)."""
    point = replay.GridPoint(cell.escalate_after_n, cell.stale_window, cell.ladder)
    scores: list[float] = []
    for traj in trajectories:
        scores.extend(_prefix_scores(traj, point))
    return scores


def _degeneracy_note(n_degenerate: int, total: int) -> str:
    """State the length-1 degeneracy plainly — never imply the recurrence trigger was exercised."""
    if n_degenerate == 0:
        return "no degenerate trajectories: the recurrence trigger is exercisable."
    return (
        f"{n_degenerate}/{total} trajectories are length-1 (degenerate): the recurrence trigger "
        "CANNOT fire on a length-1 stream. This run validates the prefix-labeler, metrics, and "
        "plots on real terminal data — NOT the escalation trigger, which needs multi-step data."
    )


def _load_fixtures(fixtures_dir: Path) -> list[Trajectory]:
    out: list[Trajectory] = []
    for name, parser in _PARSERS.items():
        path = fixtures_dir / name
        if path.exists():
            out.append(
                parser.parse(json.loads(path.read_text(encoding="utf-8")), {"trajectory_id": name})
            )
    return out


def _save_plots(
    trajectories: list[Trajectory], grid: list[replay.GridPoint], report: EvalReport, out_dir: Path
) -> None:
    """Render every detector figure on the real data to `out_dir`, each degrading gracefully."""
    out_dir.mkdir(parents=True, exist_ok=True)
    note = report.insufficient_note
    no_skill = report.no_skill_note
    labels = _all_labels(trajectories)
    scores = _prefix_scores_flat(trajectories, report.headline)
    sweep = replay.sweep(trajectories, grid)

    _render(
        out_dir / "pr_curve.png",
        lambda ax: plots.pr_curve(scores, labels, ax, note=note, no_skill_note=no_skill),
    )
    _render(
        out_dir / "roc_curve.png",
        lambda ax: plots.roc_curve(scores, labels, ax, note=note, no_skill_note=no_skill),
    )
    _render(
        out_dir / "steps_to_detection.png",
        lambda ax: plots.steps_to_detection_hist(sweep, ax, note=note),
    )
    _render(
        out_dir / "confusion_matrix.png",
        lambda ax: plots.confusion_matrix_plot(report.headline.confusion, ax, note=note),
    )
    ladder = report.headline.ladder
    _render(
        out_dir / "sweep_heatmap.png",
        lambda ax: plots.sweep_heatmap(report.cells, ax, ladder=ladder, note=note),
    )
    _render(out_dir / "cost_quality.png", lambda ax: _cost_quality(report.cells, ax, note))


def _cost_quality(cells: list[CellReport], ax: object, note: str | None) -> None:
    """Cost-quality frontier: escalation count (cost proxy) vs F1 quality, per grid cell."""
    from matplotlib.axes import Axes

    assert isinstance(ax, Axes)
    perf = {f"n{c.escalate_after_n}_w{c.stale_window}_{c.ladder[:6]}": c.f1 * 100 for c in cells}
    cost = {
        f"n{c.escalate_after_n}_w{c.stale_window}_{c.ladder[:6]}": float(c.n_escalated)
        for c in cells
    }
    plots.cost_quality_frontier(perf, cost, ax)
    if note:
        plots.annotate_insufficient(ax, note)


def _render(path: Path, draw: object) -> None:
    """Make a fresh figure, run the draw callback, save, and close (no leaked figures)."""
    fig, ax = plt.subplots()
    draw(ax)  # type: ignore[operator]
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _print_summary(report: EvalReport) -> None:
    """Print the best hyperparameter + the per-cell metric table to stdout (audit P4)."""
    print(f"\nstatus: {report.status} — {report.reason}")  # noqa: T201
    if report.best_config is None:
        print("best_config: null (no cell discriminates — insufficient data)")  # noqa: T201
    else:
        b = report.best_config
        print(  # noqa: T201
            f"best_config: escalate_after_n={b.escalate_after_n} stale_window={b.stale_window} "
            f"ladder={b.ladder} | F1={b.f1:.3f} steps={b.mean_steps_to_detection}"
        )
    header = f"{'n':>3} {'window':>6} {'ladder':>16} {'F1':>6} {'prec':>6} {'rec':>6} {'esc':>5}"
    print(header)  # noqa: T201
    for c in report.cells:
        print(  # noqa: T201
            f"{c.escalate_after_n:>3} {c.stale_window:>6} {c.ladder:>16} "
            f"{c.f1:>6.3f} {c.precision:>6.3f} {c.recall:>6.3f} {c.n_escalated:>5}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the eval on fixtures + results.csv bootstrap + live data; print the JSON report."""
    parser = argparse.ArgumentParser(description="escalation-detector offline eval")
    here = Path(__file__).resolve()
    parser.add_argument(
        "--fixtures", type=Path, default=here.parents[2] / "tests/escalation/fixtures"
    )
    parser.add_argument(
        "--results-csv", type=Path, default=here.parent.parent / "routing/results.csv"
    )
    parser.add_argument(
        "--live-dir",
        type=Path,
        default=here.parent / "data/live",
        help="Real captured live trajectories (schema JSONL) to include, if present.",
    )
    parser.add_argument("--plots-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    trajectories = _load_fixtures(args.fixtures)
    if args.results_csv.exists():
        trajectories.extend(datasets.results_csv_bootstrap(args.results_csv))
    if args.live_dir.exists():
        trajectories.extend(load_jsonl(p) for p in sorted(args.live_dir.glob("*.jsonl")))
    if not trajectories:
        logger.error("no trajectories found (fixtures + results.csv both empty)")
        return 1

    report = evaluate(trajectories, datasets.DEFAULT_GRID)
    if args.plots_dir is not None:
        _save_plots(trajectories, datasets.DEFAULT_GRID, report, args.plots_dir)
    print(json.dumps(report.to_dict(), indent=2))  # noqa: T201 (CLI report to stdout)
    _print_summary(report)
    print(report.degeneracy_note, file=sys.stderr)  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
