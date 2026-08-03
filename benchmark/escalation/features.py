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
# 2. NO TERMINAL STEP, AND NO APPROACH TO IT. `_stamp_terminal` writes the harness verdict onto the
#    last step, so any feature that can see step n-1 reads the label verbatim (AUROC 1.00 including
#    it, 0.56 excluding). `scorable_steps` clips that one step, and this rule used to stop there and
#    read as complete. It was not. Clipping ONE step bounds the leak only while the prefix stays
#    SHORT RELATIVE TO THE RUN, and the admission test was merely `len(steps) >= depth` — so at
#    depth 40 a 41-step run was admitted with its prefix ending one step short of the verdict, and
#    grouped-OOF AUROC there read 0.770 against 0.608 once 20 steps were withheld.
#
#    The channel is NOT the stamp: only `steps[-1]` carries it. It is that a per-step verified
#    outcome is the SAME MEASUREMENT as the terminal grade, taken on a workspace the agent has
#    barely changed since — so proximity to termination makes a step's `success` a near-copy of the
#    label. Measured on the 216 runs with >= 45 scorable steps (a FIXED population, so the decay is
#    not a selection artifact): AUROC of one step's `success` against the final grade holds a
#    plateau of 0.77-0.79 for every step within 10 of the end, decays through 0.652 at distance 18
#    and 0.607 at 21, and first becomes statistically indistinguishable from that population's own
#    far-field floor (0.554, the mean over distances 30-42) at distance 23 — challenge-level
#    bootstrap, 400 resamples, the excess CI first including 0 at 23 and staying flat out to 42.
#    `MIN_WITHHELD` is that distance minus one: a trajectory qualifies at `depth` only if at least
#    22 non-terminal steps remain UNREAD after the prefix, which puts the last step a feature may
#    read at distance 23 and out of the bleed.
#
#    Said WITHOUT the overclaim the old wording made: this bounds the proximity channel to the
#    corpus's own noise floor; it does not drive it to zero, and no margin could. The residual at
#    distance 23+ is ~0.554 on long runs, and that is run-level difficulty rather than the verdict —
#    a hard task fails from its first step. The margin is ABSOLUTE, never a fraction of the run, for
#    rule 1's reason: a proportional margin is a function of `n_steps` and would smuggle future
#    length back in through the admission test itself.
#
#    THE PRICE, stated rather than hidden. The rule is a filter on real rows, and the reported
#    depths pay for it — MORE the deeper they go. Admission needs `depth + MIN_WITHHELD` scorable
#    steps, so the cut at depth d is the share of runs that reach d and then fail to survive
#    another 22, and the run-length distribution thins as d rises: the margin discards a larger
#    fraction of an already smaller pool. Measured on the REBUILT corpus (727 stamped runs,
#    2026-08-02): depth 5 falls 727 -> 414 rows (152 -> 116 challenges), depth 10 717 -> 344
#    (151 -> 102), and depth 20 532 -> 228 (136 -> 78) — cuts of 43%, 52% and 57%. That is the
#    cost of not scoring a leak. Every depth still clears `prefix_eval.MIN_ROWS` (40). The six
#    counts are properties of this corpus and will move on the next rebuild; what does not move
#    is that the cut rises with depth, which is the claim.
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
#
# THIS CONSTANT DECIDES THE SIGN OF THE PUBLISHED VERDICT, so five words were not enough. It used to
# carry only the line above, and the ladder itself went unjustified while `run_eval._status` read
# these three depths as the whole family it declares skill from.
#
# WHY A x2 LADDER FROM 5. Five decisions is roughly the earliest point an escalation router could
# act on and still save anything — below it the prefix is one or two commands and there is nothing
# to summarise; a router that only decides after 40 turns has already paid for the run it was
# supposed to divert. Doubling (5, 10) spans that usable window on a log scale rather than
# oversampling one neighbourhood, which is what a linear 5/10/15 would do. The ladder is a
# reporting choice about where a router could act, made independently of what the numbers say.
#
# THEY WERE NOT CHOSEN BY OUTCOME. The original three-depth tuple (5, 10, 20) was introduced whole
# and stood for its first edit (see "DEPTH 20 LEFT" below); none of its members was a local maximum
# of any reported statistic — a result-maximiser on this corpus would have picked 24/33/42, not
# 5/10/20. The defect this comment repairs is that a load-bearing choice was undocumented, not that
# it was gamed.
#
# THE CEILING, AND WHY THERE IS ONE. Do not raise these depths to chase a bigger incremental AUROC.
# Admission at depth d requires `len(scorable) >= d + MIN_WITHHELD`, so a deeper depth does not
# score the same runs more deeply — it SELECTS a different, shorter-lived population. Run length is
# outcome-correlated (a task the agent solves quickly is a task it solves), so raising the depth
# raises the admitted failure rate, and a rising incremental at depth measures that selection
# rather than prefix evidence. `MIN_WITHHELD` bounds the PROXIMITY channel (rule 2 above); it does
# nothing about this SELECTION channel, which is a property of the admission test itself.
#
# MECHANISM FIRST, NUMBERS SECOND. The corpus HAS SINCE BEEN REBUILT — the pre-rebuild one carried
# fabricated per-step outcomes on roughly 30% of it — so the two halves below have different
# vintages and are labelled rather than merged:
#   RE-MEASURED on the rebuilt corpus (2026-08-02, 727 stamped runs): over a depth-2..45 sweep the
#   admitted base rate rises from 0.475 to 0.700 — with a one-point dip here and there, so rising
#   but not strictly monotone — while n collapses from 461 rows to 70. The selection channel
#   survived the rebuild essentially unchanged (it read 0.507 -> 0.694 and 503 -> 72 before).
#   NOT re-measured, still the 2026-07-31 PRE-REBUILD figure: the incremental AUROC — negative at
#   every reported depth — first turned positive at depth 31 and reached +0.18 at depth 42. Had
#   this tuple been (30, 40, 45) the harness would have published a positive incremental at every
#   depth it reports. Read it as the shape of the risk, not as a current measurement.
# `tests/escalation/test_features.py` turns the ceiling into a checkable bound rather than a
# warning: every reported depth's admitted base rate must stay within a stated tolerance of the
# corpus base rate, which is the selection channel measured directly.
#
# DEPTH 20 LEFT, MEASURED 2026-08-02, AND THAT IS NOT A NUMBER EDIT — the tolerance section in
# `test_features.py` is the story. The bound encodes a 26% RELATIVE shift in the admitted
# population's outcome mix, and the rebuilt corpus (727 stamped runs, base rate 0.4209) put
# depth 20 at +0.1186 absolute = 28.2% relative — PAST the criterion its own comment states,
# while still inside the old ABSOLUTE 0.12 (which is 26% of the PRE-rebuild 0.460 base rate and
# therefore looser than its justification on 0.4209). The tolerance was NOT widened to keep it:
# the earlier note in this comment ("if the next corpus movement turns this red, the free
# variable is the ladder's ceiling, not the tolerance") is the recorded decision, and the ladder
# now ends where the selection channel stays inside the bound it declares. The cost is real and
# stated: depth 20 was the one depth whose prefix-only score came closest to looking nonzero
# (0.519 vs a family-wise null [0.494, 0.555] in the 2026-08-01 report), and it is now
# unreported. It is removed because at 28.2% relative drift its incremental measured the
# admission test selecting failures, not prefix evidence — the exact failure the bound exists to
# refuse. The depth-20 row does not go away quietly: `frozen_corpus.EXPECTED[20]` still documents
# what it would admit, and this paragraph is the record of why it is not reported.
#
# DEPTH 5 LEFT, MEASURED 2026-08-02, AND FOR THE OPPOSITE REASON — ITS DESIGN IS RANK-DEFICIENT,
# not selection-shifted. 412 of 414 admitted depth-5 rows carry the IDENTICAL feature vector
# (fail_rate=1.0, infra_rate=0.0, max_action_repeat_rate=0.2): in the first five replayed steps
# the agent is still reproducing the bug (running the failing test before editing anything), so
# every run "fails" identically and the design `[features | intercept]` ranks 3 of 4. A depth
# whose score is constant by construction reports an arithmetic AUROC, not evidence, and the
# eval must not headline it — `prefix_eval`'s `design_full_rank` gate refuses it and
# `run_eval.best_depth` excludes it. It is a corpus property, not a redundant column: the two
# columns that once shared the blame (max_key_repeat_rate, distinct_check_id_rate) are already
# removed, and the degeneracy survives. It is dropped from the reported ladder for the same
# reason depth 20 was: a reported depth must be measurable, and depth 5 is not. `frozen_corpus`
# records what it WOULD admit so the finding stays visible.
#
# WHAT REMAINS. The reported ladder is the shallowest depth the corpus can actually score: depth
# 10, full-rank (7 distinct vectors) and inside the selection bound (+19.5% relative). It is the
# honest floor for a prefix risk score on this corpus — and the audit that removed depth 5 also
# established WHERE the escalation edge actually lives: NOT in a shallow prefix (early steps are
# all bug-reproduction, so no prefix feature separates) but in the shipped recurrence POLICY at
# HIGH thresholds (escalate_after_n >= 10 clears the family-wise null; see `datasets.DEFAULT_GRID`
# and `policy_eval`). The prefix model is reported as the secondary instrument it is.
DEFAULT_DEPTHS: Final[tuple[int, ...]] = (10,)

# Non-terminal steps that must remain UNREAD after the prefix for a trajectory to qualify (rule 2).
# A module constant, deliberately not a parameter with a default: a caller that could pass its own
# margin could silently re-admit the leaked rows, and the whole point of the rule is that it is not
# bypassable from the call site. `extract_features` is the only admission gate, and `build_rows`
# reaches every row through it.
MIN_WITHHELD: Final[int] = 22

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
# AND ONE MORE, dropped by rule 2 rather than by arithmetic: `recent_fail_rate` (the failure rate
# over the last 3 prefix steps). It was previously kept and merely disclosed as correlating with
# `fail_rate` at r = +0.9988 at depth 5, on the stated grounds that "the design keeps full rank with
# both". That grounds was itself an artifact of the leak. Measured: at depth 5 the two columns are
# not merely correlated but EXACTLY EQUAL on 787 of 791 trajectories, and the four rows that made
# the design full-rank are the four SHORTEST runs in the corpus — 7, 9, 10 and 11 scorable steps,
# i.e. prefixes ending 2 to 6 steps from the verdict. The depth-5 rank guard was being satisfied
# only by the leakiest rows there are; once `MIN_WITHHELD` excludes them the equality is 452 of 452
# and the design is rank-deficient. No window width repairs it: on runs long enough to qualify, the
# opening 5 steps are homogeneous, so every sub-window of them equals the whole.
#
# It is also the channel the leak travelled down. Permutation importance at depth 40 put
# `recent_fail_rate` at +0.1368 dAUROC against +0.0122 for `fail_rate` — an order of magnitude
# above every other column, because the recency window is the part of the prefix nearest the end.
# Removing it costs almost nothing once the margin is in: at depth 10 the grouped-OOF AUROC moves
# 0.4082 -> 0.4054 and at depth 20 0.3601 -> 0.3533. It carries genuine rank at depths 10 and 20
# (it differs from `fail_rate` on 96/781 and 130/584 rows), so this is a real column being given
# up — not a free arithmetic removal like the two above — and it is given up because a design that
# is rank-deficient at a reported depth cannot be fitted honestly at that depth.
#
# With both in, the design matrix ranked 7 of 8 at depths 5 and 10, so the "each column carries
# signal no other carries" claim below was false and the eval was fitting a rank-deficient design.
# `tests/escalation/test_features.py` now gates on the RANK of [features | intercept] over the real
# corpus, not on pairwise inequality, which affine dependence passes by construction.

# THREE columns, each carrying rank no other column already carries at EVERY reported depth — the
# five-column claim that stood here was false and is why this comment exists.
#
# TWO MORE REMOVED, 2026-08-02, FOR NON-INDEPENDENCE ON THE REBUILT CORPUS — the measure that
# killed `recent_fail_rate` runs again, and this time it is measured corpus-wide rather than on a
# depth-5 slice:
#
#   `max_key_repeat_rate` == `fail_rate` on 414/414 depth-5 rows, 342/344 at depth 10 and
#     226/228 at depth 20. The mechanism is the same one that degrades the whole shallow design:
#     until the agent edits the workspace, the selector fails on the SAME test id every step, so
#     one distinct key repeated n times makes the max-repeat rate and the failure rate the same
#     number. The 2/344 and 2/228 exceptions are the "two outlier rows" that used to let the
#     design-matrix rank guard read as green on a margin of two observations.
#   `distinct_check_id_rate` takes exactly ONE value at depth 5 (1/5) and TWO at depths 10 and
#     20 (1/depth, 2/depth). At a fixed absolute depth it is a binary "did more than one test id
#     fail" flag, and its one informative state is a near-copy of the same constant-key mechanism
#     `max_key_repeat_rate` was a copy of.
#   Both were kept on the old argument that they were not EXACT aliases; the design-matrix rank
#   guard, which catches affine dependence by construction, was staying green on the handful of
#   rows where they differ. That is the same failure `recent_fail_rate` shipped — the guard
#   satisfied only by the leakiest rows there are — and it is now closed by removing the columns
#   rather than re-scoring them. What remains — `fail_rate`, `infra_rate`,
#   `max_action_repeat_rate` — is pairwise affinely independent at every reported depth on the
#   rebuilt corpus, and the depth-5 rank guard now fails for the honest reason (the DATA has three
#   distinct rows, not because a redundant column split the rank).
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "fail_rate",
    "infra_rate",
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
    # `depth + MIN_WITHHELD`, not `depth`: reaching the depth is necessary but not sufficient. The
    # run must also still have somewhere to go, or the prefix is reading the outcome (rule 2).
    steps = scorable_steps(traj)
    if len(steps) < depth + MIN_WITHHELD:
        return None
    return _features(steps[:depth])


def _features(steps: Sequence[StepView]) -> tuple[float, ...]:
    """The FEATURE_NAMES vector over exactly these steps. Every value is a rate in [0, 1]."""
    n = len(steps)
    actions = Counter(s.action for s in steps)
    return (
        _rate(sum(not s.success for s in steps), n),
        _rate(sum(s.is_infra_failure for s in steps), n),
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
    """One EvalRow per trajectory that reached `depth` decisions with room still left to run."""
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
    "MIN_WITHHELD",
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
