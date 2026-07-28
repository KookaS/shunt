"""Causal prefix features: what an online detector could actually know after k decisions."""

# TWO STRUCTURAL RULES, both of which the previous harness broke and both of which are pinned by
# tests (tests/escalation/test_features.py):
#
# 1. NO FUTURE LENGTH. Features are read from `steps[:depth]` at a FIXED ABSOLUTE depth, never
#    from a fraction of the run. A fraction needs `n_steps`, which is future information; the old
#    positional label leaked exactly that, and a content-free clock (score = t/n) scored AUROC
#    0.970 on it while a perfect task-oracle capped at 0.757. At fixed depth every row's prefix is
#    the same length, so "how far into the run am I" carries no variance and cannot be learned.
#
# 2. NO TERMINAL STEP. `_stamp_terminal` writes the harness verdict onto the last step, so any
#    feature that can see step n-1 reads the label verbatim (AUROC 1.00 including it, 0.56
#    excluding). The prefix is clipped to `n_steps - 1` before `depth` is applied.
#
# FIELD CENSUS (measured over all 799 live trajectories / 29 422 steps — do not add features on
# fields outside this list). POPULATED: success, blocking, is_infra_failure, confirmed, status,
# action, tool, exit_code (17 559 non-null), failing_check_id / dedup_key (11 537 non-null).
# CONSTANT OR 100% NULL, therefore unusable: is_revert (always False), retry_count (always 0),
# loop_signal (always False), subgoal_progress, test_passed, test_total, model, reasoning_effort,
# rank_index, effort_index, real_cost.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.escalation.schema import StepView, Trajectory

# The prefix depths the eval reports at, in decisions. Absolute, not fractional (rule 1).
DEFAULT_DEPTHS: Final[tuple[int, ...]] = (5, 10, 20)

# The tail window for the recency features — short enough to differ from the whole-prefix rate.
_RECENT: Final[int] = 3

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "fail_rate",
    "blocking_rate",
    "infra_rate",
    "error_status_rate",
    "check_id_rate",
    "distinct_check_id_rate",
    "max_key_repeat_rate",
    "nonzero_exit_rate",
    "missing_exit_rate",
    "recent_fail_rate",
    "max_action_repeat_rate",
    "distinct_action_rate",
)


@dataclass(frozen=True)
class EvalRow:
    """One trajectory as one unit of evaluation: its group, its outcome, its prefix features."""

    trajectory_id: str
    group: str
    model: str
    failed: bool
    features: tuple[float, ...]


def scorable_steps(traj: Trajectory) -> list[StepView]:
    """Every step a feature may read: the trajectory minus its label-stamped terminal step."""
    return traj.steps[:-1]


def is_stamped(traj: Trajectory) -> bool:
    """True iff the per-step verified-outcome stage ran on this trajectory."""
    # `stamp_step` sets `confirmed=True` on every step it touches, so a trajectory with no confirmed
    # non-terminal step never went through the offline container replay: its `success` is the parser
    # default and its `failing_check_id`/`exit_code` are null throughout. Including such a run in
    # the eval hands the model a *collection-date* proxy (unstamped runs cluster by capture window
    # and therefore by model), which is a data artifact, not escalation signal.
    return any(step.confirmed for step in scorable_steps(traj))


def extract_features(traj: Trajectory, depth: int) -> tuple[float, ...] | None:
    """Features over the first `depth` non-terminal steps, or None when the run is too short.

    Returning None (rather than padding) keeps the unit honest: a trajectory that never reached
    the decision depth was not observed there, and imputing it would invent data.
    """
    steps = scorable_steps(traj)
    if len(steps) < depth:
        return None
    return _features(steps[:depth])


def _features(steps: Sequence[StepView]) -> tuple[float, ...]:
    """The FEATURE_NAMES vector over exactly these steps. Every value is a rate in [0, 1]."""
    n = len(steps)
    keys = [s.failing_check_id for s in steps if s.failing_check_id is not None]
    actions = Counter(s.action for s in steps)
    recent = steps[-_RECENT:]
    exits = [s.exit_code for s in steps]
    return (
        _rate(sum(not s.success for s in steps), n),
        _rate(sum(s.blocking for s in steps), n),
        _rate(sum(s.is_infra_failure for s in steps), n),
        _rate(sum(s.status != "ok" for s in steps), n),
        _rate(len(keys), n),
        _rate(len(set(keys)), n),
        _rate(max(Counter(keys).values()) if keys else 0, n),
        _rate(sum(code not in (None, 0) for code in exits), n),
        _rate(sum(code is None for code in exits), n),
        _rate(sum(not s.success for s in recent), len(recent)),
        _rate(max(actions.values()), n),
        _rate(len(actions), n),
    )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def model_of(traj: Trajectory) -> str:
    """The model that produced a trajectory, read off the `<instance>__<model>__<effort>` id."""
    parts = traj.header.trajectory_id.split("__")
    return parts[-2] if len(parts) >= 3 else "unknown"  # noqa: PLR2004 (id has 3 segments)


def group_of(traj: Trajectory) -> str:
    """The challenge a trajectory belongs to — the CV grouping key (never split across folds)."""
    return traj.header.instance_id or traj.header.trajectory_id


def build_rows(trajectories: Sequence[Trajectory], depth: int) -> list[EvalRow]:
    """One EvalRow per trajectory that reached `depth` non-terminal decisions."""
    rows: list[EvalRow] = []
    for traj in trajectories:
        features = extract_features(traj, depth)
        if features is None:
            continue
        rows.append(
            EvalRow(
                trajectory_id=traj.header.trajectory_id,
                group=group_of(traj),
                model=model_of(traj),
                failed=not traj.header.terminal_resolved,
                features=features,
            )
        )
    return rows


@dataclass(frozen=True)
class ModelCoverage:
    """Per-model failure-capture coverage — the diagnostic behind the zero-escalation models."""

    model: str
    n_trajectories: int
    n_steps: int
    n_failed_steps: int
    n_steps_with_check_id: int
    terminal_failure_rate: float
    n_stamped: int

    @property
    def capture_rate(self) -> float:
        """Share of failed steps that carry a failing_check_id — 0.0 means the trigger is dead."""
        return self.n_steps_with_check_id / self.n_failed_steps if self.n_failed_steps else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "n_trajectories": self.n_trajectories,
            "n_steps": self.n_steps,
            "n_failed_steps": self.n_failed_steps,
            "n_steps_with_check_id": self.n_steps_with_check_id,
            "n_stamped": self.n_stamped,
            "capture_rate": round(self.capture_rate, 4),
            "terminal_failure_rate": round(self.terminal_failure_rate, 4),
        }


def model_coverage(trajectories: Sequence[Trajectory]) -> list[ModelCoverage]:
    """Failure-capture coverage per model, worst first — exposes undetectable models."""
    buckets: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        buckets.setdefault(model_of(traj), []).append(traj)
    out = [_coverage(model, group) for model, group in buckets.items()]
    return sorted(out, key=lambda c: (c.capture_rate, c.model))


def _coverage(model: str, trajs: Sequence[Trajectory]) -> ModelCoverage:
    steps = [s for t in trajs for s in scorable_steps(t)]
    failed = [s for s in steps if not s.success]
    return ModelCoverage(
        model=model,
        n_trajectories=len(trajs),
        n_steps=len(steps),
        n_failed_steps=len(failed),
        n_steps_with_check_id=sum(s.failing_check_id is not None for s in failed),
        terminal_failure_rate=_rate(sum(not t.header.terminal_resolved for t in trajs), len(trajs)),
        n_stamped=sum(is_stamped(t) for t in trajs),
    )


__all__ = [
    "DEFAULT_DEPTHS",
    "FEATURE_NAMES",
    "EvalRow",
    "ModelCoverage",
    "build_rows",
    "extract_features",
    "group_of",
    "is_stamped",
    "model_coverage",
    "model_of",
    "scorable_steps",
]
