"""The SHIPPED policy, graded at the unit it actually acts on: one trajectory, one decision."""

# The old harness pooled 29 422 prefixes from 799 runs, which inflates n by ~37x (prefixes inside a
# run are near-perfectly correlated) and summed `n_escalated` across all 12 sweep cells, printing
# 4980 escalations for 799 trajectories. Here one trajectory is one row, `n_escalated` is per cell,
# and the headline is the question the product asks: given the policy fired, is this run more
# likely to fail than a run picked at random?

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.escalation import metrics, replay

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.escalation.schema import Trajectory


@dataclass(frozen=True)
class PolicyCell:
    """One swept configuration, scored per trajectory against the terminal outcome."""

    escalate_after_n: int
    stale_window: int
    ladder: str
    n_trajectories: int
    n_escalated: int
    tp: int
    fp: int
    fn: int
    tn: int
    base_failure_rate: float
    precision_ci: tuple[float, float]
    null_auroc: metrics.NullResult
    lead_times_failed: tuple[int, ...]
    lead_times_resolved: tuple[int, ...]

    @property
    def precision(self) -> float:
        """P(run failed | policy fired) — compare against `base_failure_rate`, not against 0."""
        fired = self.tp + self.fp
        return self.tp / fired if fired else 0.0

    @property
    def recall(self) -> float:
        failures = self.tp + self.fn
        return self.tp / failures if failures else 0.0

    @property
    def p_fail_given_quiet(self) -> float:
        """P(run failed | policy did NOT fire). Above `precision` means the policy is inverted."""
        quiet = self.fn + self.tn
        return self.fn / quiet if quiet else 0.0

    @property
    def lift(self) -> float:
        """precision / base rate. Below 1.0 means firing predicts SUCCESS, not failure."""
        return self.precision / self.base_failure_rate if self.base_failure_rate else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "escalate_after_n": self.escalate_after_n,
            "stale_window": self.stale_window,
            "ladder": self.ladder,
            "n_trajectories": self.n_trajectories,
            "n_escalated": self.n_escalated,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "precision": round(self.precision, 4),
            "precision_ci95": [round(v, 4) for v in self.precision_ci],
            "recall": round(self.recall, 4),
            "p_fail_given_fired": round(self.precision, 4),
            "p_fail_given_not_fired": round(self.p_fail_given_quiet, 4),
            "base_failure_rate": round(self.base_failure_rate, 4),
            "lift": round(self.lift, 4),
            "null_auroc": self.null_auroc.to_dict(),
        }


def evaluate_cell(
    trajectories: Sequence[Trajectory],
    point: replay.GridPoint,
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
) -> PolicyCell:
    """Replay one configuration over every trajectory and score it at the trajectory level."""
    fired: list[bool] = []
    failed: list[bool] = []
    lead: list[int | None] = []
    for traj in trajectories:
        decision = replay.replay_config(traj, point.to_config())
        fired.append(decision.escalated)
        failed.append(not traj.header.terminal_resolved)
        lead.append(_lead_time(traj, decision.first_escalation_index))
    return _cell(point, fired, failed, lead, n_permutations=n_permutations, seed=seed)


def _lead_time(traj: Trajectory, first: int | None) -> int | None:
    """Decisions between the first escalation and the end of the run, or None if it never fired."""
    return None if first is None else len(traj.steps) - 1 - first


def _cell(
    point: replay.GridPoint,
    fired: Sequence[bool],
    failed: Sequence[bool],
    lead: Sequence[int | None],
    *,
    n_permutations: int,
    seed: int,
) -> PolicyCell:
    """Assemble one cell's 2x2, interval, null, and per-outcome lead-time distributions."""
    tp = sum(f and y for f, y in zip(fired, failed, strict=True))
    fp = sum(f and not y for f, y in zip(fired, failed, strict=True))
    fn = sum(not f and y for f, y in zip(fired, failed, strict=True))
    tn = sum(not f and not y for f, y in zip(fired, failed, strict=True))
    scores = [1.0 if f else 0.0 for f in fired]
    return PolicyCell(
        escalate_after_n=point.escalate_after_n,
        stale_window=point.stale_window,
        ladder=point.ladder,
        n_trajectories=len(fired),
        n_escalated=sum(fired),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        base_failure_rate=metrics.prevalence(failed),
        precision_ci=metrics.wilson_interval(tp, tp + fp),
        null_auroc=metrics.permute_statistic(
            scores, failed, metrics.auroc, n_permutations=n_permutations, seed=seed
        ),
        lead_times_failed=tuple(
            t for t, y in zip(lead, failed, strict=True) if t is not None and y
        ),
        lead_times_resolved=tuple(
            t for t, y in zip(lead, failed, strict=True) if t is not None and not y
        ),
    )


def evaluate(
    trajectories: Sequence[Trajectory],
    grid: Sequence[replay.GridPoint],
    *,
    n_permutations: int = metrics.MIN_PERMUTATIONS,
    seed: int = 0,
) -> list[PolicyCell]:
    """Every swept configuration, scored per trajectory."""
    return [
        evaluate_cell(trajectories, point, n_permutations=n_permutations, seed=seed)
        for point in grid
    ]


__all__ = ["PolicyCell", "evaluate", "evaluate_cell"]
