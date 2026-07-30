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
# FIELD CENSUS (do not add features on fields outside this list).
# POPULATED: success, blocking, is_infra_failure, confirmed, status, action, tool, exit_code.
# CONSTANT OR 100% NULL, therefore unusable: is_revert (always False), retry_count (always 0),
# loop_signal (always False), subgoal_progress, test_passed, test_total, model, reasoning_effort,
# rank_index, effort_index, real_cost.
#
# NOT INDEPENDENT — one field wearing three names. `normalize/mini_swe_agent.stamp_step` writes
# `success = not failure`, `failing_check_id = <id> if failure else None` and
# `blocking = recompute_blocking(...)` in the same assignment, so
# `success == (failing_check_id is None) == (not blocking)` holds BY CONSTRUCTION. It was measured
# at 28 623/28 623 scorable steps: `failing_check_id`/`dedup_key`'s non-null count is exactly the
# complement of `success`, not an independently populated field. Rates over the three therefore
# produce ONE column under three names; only `fail_rate` is kept (dropping `blocking_rate` and
# `check_id_rate`, which were bit-identical to it at every depth).
#
# ALSO DROPPED: `missing_exit_rate` (share of prefix steps with a null exit_code). On the stamped
# corpus it is 0 at depths 5 and 10 and near-constant deeper (std 0.026 at depth 20); a null
# exit_code means the stamping stage recorded none, so it tracks capture coverage — a collection
# artifact — rather than agent behaviour, which is the same reason unstamped runs are excluded.

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

# ALSO DROPPED, for the same reason: `error_status_rate` (share of prefix steps with
# `status != "ok"`). The normalizer assigns `status` and `success` together, so it is a FOURTH
# alias of `fail_rate` — bit-identical on the corpus, caught by the duplicate-column test rather
# than by inspection, which is exactly why that test exists.
#
# TWO MORE DROPPED, for AFFINE dependence rather than bit-identity. Exact duplication is only the
# easiest case; a column that is an exact linear combination of others carries no extra rank while
# still splitting the L2 logistic's weight. Both were measured on all 791 stamped trajectories:
#
#   `nonzero_exit_rate` == `fail_rate` + `infra_rate`, EXACTLY: over every depth-5 and depth-10
#     prefix step (3 955 and 7 810 steps) a non-zero exit_code occurs iff the step failed or was an
#     infra failure, and the two are disjoint (0 steps are both). It holds at depth 20 too bar
#     27/11 680 steps. It is arithmetic, not correlation: the exit code is what the verifier reads
#     to decide `success`, so its rate cannot be independent of the failure rates it produces.
#   `distinct_action_rate` == 1 + 1/depth - `max_action_repeat_rate` at depth 5 (their sum is 1.2
#     for EVERY row) and correlates at r = -0.973 at depth 10. At a fixed prefix length the two are
#     the same fact about the action multiset — how concentrated it is — read from opposite ends.
#     `max_action_repeat_rate` is kept because repetition is the construct escalation is about
#     (an agent thrashing on one command); the variety reading adds no rank at the shallow depths.
#
# NOT dropped, but disclosed: `fail_rate` and `recent_fail_rate` correlate at r = +0.9988 at depth
# 5 (the recency window is 3 of the 5 prefix steps, so it is mostly the same steps). The dependence
# is near, not exact — the design keeps full rank with both — so removing one would be a judgement
# call on a real column rather than the arithmetic removal the two above are. It shrinks with depth
# as the window becomes a smaller share of the prefix.
#
# With both in, the design matrix ranked 7 of 8 at depths 5 and 10, so the "each column carries
# signal no other carries" claim below was false and the eval was fitting a rank-deficient design.
# `tests/escalation/test_features.py` now gates on the RANK of [features | intercept] over the real
# corpus, not on pairwise inequality, which affine dependence passes by construction.

# Six columns, each carrying rank no other column already carries — see the census above for the
# six removed as aliases of `fail_rate`, as affine combinations, or as a coverage artifact.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "fail_rate",
    "infra_rate",
    "distinct_check_id_rate",
    "max_key_repeat_rate",
    "recent_fail_rate",
    "max_action_repeat_rate",
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
    return (
        _rate(sum(not s.success for s in steps), n),
        _rate(sum(s.is_infra_failure for s in steps), n),
        _rate(len(set(keys)), n),
        _rate(max(Counter(keys).values()) if keys else 0, n),
        _rate(sum(not s.success for s in recent), len(recent)),
        _rate(max(actions.values()), n),
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
