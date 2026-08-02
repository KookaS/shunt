"""Post-hoc audit of the per-step state capture: mark steps whose state was never observed."""

# WHY THIS EXISTS. `step_snapshots.DIFF_COMMAND` WAS `git -C /testbed diff` — UNSTAGED changes to
# TRACKED files only. The moment an agent ran `git add` (or committed, or stashed) the capture
# collapsed to 0 bytes while the work itself was still there, invisible to the instrument. The
# offline replay then rebuilds that step as "base commit + empty diff + gold test_patch", the
# FAIL_TO_PASS tests necessarily fail, and `stamp_step` writes `success=False`, `blocking=True`
# and a real `failing_check_id`. The result is indistinguishable from a measured capability
# failure and is not one: it is the instrument's blindness recorded as the agent's failure.
#
# Measured on the collection host: 310 of 799 trajectories and 1 897 steps carry an empty capture
# that follows a non-empty one. This module detects them and marks them UNMEASURED. It is a
# post-processing pass over data ALREADY collected, and it never re-derives an outcome.
#
# THE FORWARD FIX IS SEPARATE AND HAS LANDED. `DIFF_COMMAND` is now `git diff HEAD`, which is
# worktree-vs-HEAD and so captures staged edits too; runs collected after that change cannot
# reproduce the staging half of this defect. What `git diff HEAD` still cannot see is a COMMIT —
# it moves HEAD, taking the delta behind it — because the recorder is never told the instance's
# base_commit. Closing that needs the recorder to resolve and pin the base SHA at attach time;
# until it does, this module's rules stay live for committing scaffolds.
#
# THE SECOND TRANCHE: A CAPTURE THAT IS INCOMPLETE RATHER THAN EMPTY. `git add` removes a path
# from `git diff` WITHOUT changing the working tree. When every edited path is staged the capture
# goes to 0 bytes and the rule above sees it. When the agent then edits a NEW file, the next
# capture is non-empty again — and it is missing the staged portion for the rest of the run. The
# replay applies that partial diff to the base commit (`offline_replay.replay_step`), so it
# rebuilds a tree the agent never had, and the FAIL_TO_PASS tests fail for a reason the agent did
# not cause. Measured on the collection host: 62 such steps across 16 trajectories, 22 of them
# currently stamped as capability failures (23 before the C2 build-state rebuild re-derived the
# stamps; the 62/16 shape is unchanged). All 16 are `gpt-5-mini` arms — this scaffold is the
# only one in the corpus that commits as it goes, so leaving the class unmarked biases the corpus
# against ONE model rather than adding symmetric noise.
#
# THE PARTIAL RULE, AND THE THREE THINGS IT REFUSES TO GUESS. A file dropping out of a cumulative
# diff has exactly two causes — it was staged (capture now wrong) or it was reverted (capture
# still right) — and the reverted case is the MAJORITY: of 41 measured drop events in this corpus,
# 37 are an explicit `git checkout -- <path>`. Marking those would delete real measurements, so
# the rule fires only on positive evidence of staging and abstains everywhere else:
#   * the path must have been VISIBLE in an earlier capture (the instrument itself recorded the
#     edit — not inferred), and must not be part of the image's own build-time modification, which
#     `offline_replay._restore_build_state` re-applies from `_BUILD_STATE` on every step whose diff
#     did not already carry it, and which therefore costs the reconstruction nothing when it goes
#     missing from a capture. That exclusion alone halves
#     the candidate set (121 -> 62 steps): `pyproject.toml` / `tox.ini` dominate the drops.
#   * the step it vanished at must run a real `git add` whose PATHSPEC covers it (`-N` /
#     `--intent-to-add` excluded: it adds to the diff, never removes), and must contain no command
#     that could have rewritten the file in the same turn. `git add` is the one command whose
#     effect here is deterministic and known: `git diff` is worktree-vs-index, so staging moves
#     the delta out of the capture while leaving the bytes on disk.
#   * ANY undo command (`checkout` / `restore` / `reset` / `stash` / `revert` / `clean`) anywhere
#     from that step onward retires the claim permanently — measured counter-example
#     `matplotlib-25775__gpt-5-mini__medium` step 39, where a `git commit -- <path> || true; git
#     reset --soft HEAD~1` follows a stage: the file is still staged in fact, but the tree can no
#     longer be reasoned about from the command record, so the audit abstains and keeps the stamp.
# The cost is recall, and it is deliberate: this rule is a floor on the class, not a census of it.
# It cannot see a file staged in the SAME step that created it (never visible, so never missed),
# and it stops at the first undo. Corroboration where the agent happened to print its own index
# (`git status --porcelain` / `git show --name-only`) agrees on 11 of the 22 (file, trajectory)
# pairs it marks and contradicts none; the other 11 simply never ran such a command.
#
# THE DETECTION RULE, AND WHY IT SEPARATES A NO-OP FROM A LOSS. A `git diff` capture is
# CUMULATIVE: at step k it is the whole working-tree delta from the base commit, not step k's
# increment. So the empty/non-empty sequence within one trajectory is the evidence:
#
#   * empty BEFORE any non-empty capture  -> the agent had not yet edited a tracked file. The
#     empty capture is a TRUE observation and the step keeps its stamp. 11 908 of the corpus's
#     13 829 empty captures (86.1%) are this case, which is exactly why the rule is not "empty".
#   * empty AFTER a non-empty capture     -> the trajectory had accumulated tracked edits and the
#     instrument now reports none. Whatever moved them (index, stash, HEAD, or a revert), what
#     the replay reconstructs from this capture is the BASE COMMIT, whose FAIL_TO_PASS outcome is
#     fixed by SWE-bench's own construction. The derived "failure" therefore carries zero
#     information about the agent, so the step is not a measurement of one.
#
# That is the primary signal: state cannot silently un-accumulate inside a cumulative diff.
#
# THE CORROBORATING EVIDENCE IS RECORDED, NOT USED TO NARROW THE RULE. Each run of empties is
# classified from the onset step's own `action` (staged / committed / stashed / reverted /
# unexplained) and its `<returncode>`, and that classification is written to the committed report
# so a reader can see WHY the capture went blind. It deliberately does not gate the marking:
#   * an action string is a command, not its effect. Measured counter-example in the corpus —
#     `astropy__astropy-13236__kimi-k3__max` step 30 is `git stash list && git status --short`,
#     a READ-ONLY action, yet its capture is empty because the previous step's `docker exec`
#     timed out (`<exception>`, no `<returncode>`) and its `git stash -q` completed inside the
#     container after that step's snapshot was taken. Keying the decision on the onset action
#     would have called that step a no-op.
#   * `reverted` (129 of the 1 897 steps, 6.8%) is the one class where the working tree may
#     genuinely equal base, i.e. where the empty capture is faithful. Marking it is a deliberate,
#     bounded UNDER-claim: what is cleared is only the instrument's verdict — which for a base
#     tree restates the instance definition and measures nothing about the agent — while the
#     revert itself stays fully readable in the preserved `action`/`result`.
# The reverse error is not symmetric: a kept stamp is a fabricated failure in git, which is the
# thing the owner's bar forbids.
#
# WHAT MARKING DOES. `mark_step` = `unstamp_step` (the existing clearing convention) plus
# `is_infra_failure=True` — the outcome could not be obtained for an instrument reason. That is
# the pre-decided shape while the schema split is not in. It routes the step to
# `replay.VerifiedOutcome.NONE` and out of `features.is_stamped`, and the triple
# `(success=True, confirmed=False, is_infra_failure=True)` is written by no other path in the
# repo, so a marked step is greppable and distinct from a never-stamped one (infra False) and
# from a replay-stamped infra step (confirmed True). No new sentinel field is invented.
# Everything the agent did — `action`, `args`, `observation`, `result`, `tool`, `metadata`,
# `status` — is preserved verbatim: this module edits the instrument's verdict, never behaviour.
#
# WHY A COMMITTED SIDECAR. The snapshots are gitignored scratch, so on any other checkout the
# evidence for the marking is gone — the same asymmetry that makes `header.snapshot_steps` and
# `admissibility.json` committed rather than recomputed. `state_capture.json` is the per-step
# analogue of `admissibility.json`: same role (a diffable record of WHY something was excluded),
# same shape (a flat map keyed by id, each record carrying a `reason`), same writer discipline
# (corpus lock + atomic write). It is DERIVED — the trajectories stay the source of truth — and
# `verify_state_capture` cross-checks the two whenever the scratch is present.
#
# LAYER-1 CEILING. Like `escalation.authenticity`, everything here is recomputed from committed
# data, so it proves a corpus is internally consistent with its own audit record. It cannot
# testify that the capture describes a run that happened.

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from benchmark import corpus_lock
from benchmark.escalation import schema
from benchmark.escalation.normalize.mini_swe_agent import unstamp_step
from benchmark.routing.authenticity import ERROR, WARN, Finding
from benchmark.runner import step_snapshots

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from benchmark.escalation.schema import StepView, Trajectory

_LOG = logging.getLogger(__name__)

# The committed audit record, next to the corpus it describes. Plain JSON, deliberately outside
# the `*.jsonl` LFS pattern so it diffs in review — the same call `manifest.json` makes.
REPORT_FILENAME: Final[str] = "state_capture.json"
DETECTION_RULE: Final[str] = "empty-after-non-empty"
PARTIAL_DETECTION_RULE: Final[str] = "staged-path-missing-from-non-empty-capture"

# The two loss classes, kept separable in the committed record so the corpus can report them apart.
CLASS_EMPTY: Final[str] = "empty_after_nonempty"
CLASS_PARTIAL: Final[str] = "partial_stage_loss"

# Evidence classes for the onset of a run of empty captures, read off the onset step's action.
STAGED: Final[str] = "staged"
COMMITTED: Final[str] = "committed"
STASHED: Final[str] = "stashed"
REVERTED: Final[str] = "reverted"
UNEXPLAINED: Final[str] = "unexplained"
CONTINUATION: Final[str] = "continuation"

_GIT: Final[str] = r"\bgit\s+(?:-C\s+\S+\s+)?"
_STAGE_RE: Final[re.Pattern[str]] = re.compile(_GIT + r"add\b")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(_GIT + r"commit\b")
_STASH_RE: Final[re.Pattern[str]] = re.compile(_GIT + r"stash\b")
_REVERT_RE: Final[re.Pattern[str]] = re.compile(_GIT + r"(?:checkout|reset|restore)\b")
_RETURNCODE_RE: Final[re.Pattern[str]] = re.compile(r"<returncode>(-?\d+)</returncode>")

# Which paths a captured diff touches. `git diff` always emits the b-side path for a modification.
_DIFF_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^diff --git a/(?:\S+) b/(\S+)$", re.M)
# Every command that can put a tracked path back — each one makes a drop ambiguous, for good.
_UNDO_RE: Final[re.Pattern[str]] = re.compile(
    _GIT + r"(?:checkout|restore|reset|stash|revert|clean)\b"
)
# `git add` with its pathspec, up to the next shell separator.
_ADD_ARGS_RE: Final[re.Pattern[str]] = re.compile(_GIT + r"add\b([^\n;&|]*)")
# Commands that could have rewritten a file in the SAME turn as the stage — the one competing
# explanation for a drop that no `git` verb in the action would reveal.
_WRITES_RE: Final[re.Pattern[str]] = re.compile(
    r"\bsed\s+-i\b|\bcat\s*>|\btee\b|\bpython[0-9.]*\s+-(?:c\b|\s*<<)|\bpatch\b"
    r"|\b(?:rm|mv|cp)\s|>\s*\S+\.(?:py|pyx|toml|ini|cfg|txt|rst)\b|" + _GIT + r"apply\b"
)
# A pathspec that stages everything, so it covers any path by construction.
_BULK_PATHSPECS: Final[frozenset[str]] = frozenset(
    {"-A", "--all", "-u", "--update", ".", "*", ":/"}
)
_INTENT_FLAGS: Final[frozenset[str]] = frozenset({"-N", "--intent-to-add"})


@dataclass(frozen=True)
class StepLoss:
    """One step whose captured state is empty after an earlier step's was not."""

    step_index: int
    onset: bool
    evidence: str
    returncode: int | None


@dataclass(frozen=True)
class Capture:
    """What the partial rule reads for one step: the captured diff, plus the action that made it."""

    size: int
    files: frozenset[str]
    action: str


@dataclass(frozen=True)
class PartialLoss:
    """One step whose NON-EMPTY capture is missing a path an earlier capture showed staged away."""

    step_index: int
    onset: bool
    missing: tuple[str, ...]
    staged_at: tuple[int, ...]


@dataclass(frozen=True)
class TrajectoryAudit:
    """One trajectory's state-capture verdict: which steps lost their state, and the evidence."""

    trajectory_id: str
    losses: tuple[StepLoss, ...]
    snapshots_seen: int
    orphan_snapshots: tuple[int, ...]
    terminal_skipped: tuple[int, ...]
    partial_losses: tuple[PartialLoss, ...] = ()

    @property
    def marked_steps(self) -> tuple[int, ...]:
        """The step indices this audit marks unmeasured (terminal and orphan steps excluded)."""
        # The two classes are disjoint by construction — one needs an empty capture, the other a
        # non-empty one — so the union never double-counts a step.
        return tuple(sorted({loss.step_index for loss in self.losses} | set(self.partial_steps)))

    @property
    def partial_steps(self) -> tuple[int, ...]:
        """The step indices marked by the partial-stage rule alone."""
        return tuple(loss.step_index for loss in self.partial_losses)

    def step_classes(self) -> dict[int, str]:
        """Which loss class marked each step, for the record and for the guard's message."""
        classes = {loss.step_index: CLASS_EMPTY for loss in self.losses}
        classes.update({index: CLASS_PARTIAL for index in self.partial_steps})
        return classes


@dataclass(frozen=True)
class MarkSummary:
    """What one `apply_marks` pass did, for the CLI and for the caller's own reporting."""

    trajectories_examined: int
    trajectories_with_loss: int
    steps_marked: int
    trajectories_written: int
    steps_already_marked: int
    unaudited: tuple[str, ...]
    steps_marked_partial: int = 0


# ---------------------------------------------------------------------------
# Detection — pure over the snapshot size sequence, then joined to the steps.
# ---------------------------------------------------------------------------


def snapshot_sizes(trajectory_id: str, root: Path = step_snapshots.SNAPSHOT_ROOT) -> dict[int, int]:
    """Byte size of every captured per-step diff, keyed by step index (stat only, no read)."""
    out = step_snapshots.snapshot_dir(trajectory_id, root)
    if not out.is_dir():
        return {}
    return {
        int(path.stem.removeprefix("step_")): path.stat().st_size
        for path in out.glob("step_*.diff")
    }


def lost_snapshot_indices(sizes: Mapping[int, int]) -> dict[int, bool]:
    """Indices whose capture is empty AFTER a non-empty one; the value is True at a run's onset."""
    # THE PRIMARY SIGNAL. The captured diff is cumulative, so `seen_nonempty` is a latch: once a
    # trajectory has shown tracked edits, a later empty capture is the instrument reporting none
    # of them. Empties before the latch are left alone — those are the agent genuinely not having
    # edited a tracked file yet, and they are 86% of all empty captures in the corpus.
    # A missing index (the recorder skipped a step) is a gap, not a reset: it neither clears the
    # latch nor is itself reported, because there is no capture to judge.
    out: dict[int, bool] = {}
    seen_nonempty = False
    prev_lost = False
    for index in sorted(sizes):
        if sizes[index] > 0:
            seen_nonempty = True
            prev_lost = False
            continue
        if not seen_nonempty:
            continue
        out[index] = not prev_lost
        prev_lost = True
    return out


def snapshot_files(
    trajectory_id: str, root: Path = step_snapshots.SNAPSHOT_ROOT
) -> dict[int, frozenset[str]]:
    """The paths each captured diff touches, keyed by step index (reads every diff, ~74 MB)."""
    out = step_snapshots.snapshot_dir(trajectory_id, root)
    if not out.is_dir():
        return {}
    return {
        int(path.stem.removeprefix("step_")): frozenset(
            _DIFF_HEADER_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
        )
        for path in out.glob("step_*.diff")
    }


def add_covers(action: str, path: str) -> bool:
    """True iff a real `git add` in *action* has a pathspec that provably covers *path*."""
    # `-N` / `--intent-to-add` is the inverse of the bug: it ADDS an untracked path to `git diff`,
    # so it can never be the reason one disappeared. A bare `git add` with no positional pathspec
    # is a no-op in git, but treating it as bulk only ever widens abstention, never marking.
    for match in _ADD_ARGS_RE.finditer(action):
        tokens = match.group(1).split()
        if _INTENT_FLAGS & set(tokens):
            continue
        pathspecs = [token for token in tokens if not token.startswith("-")]
        if not pathspecs or _BULK_PATHSPECS & set(pathspecs):
            return True
        if any(path == spec or path.startswith(spec.rstrip("/") + "/") for spec in pathspecs):
            return True
    return False


def _staging_explains(action: str, path: str) -> bool:
    """True iff staging is the ONLY reading of *action* under which *path* leaves the capture."""
    return add_covers(action, path) and not _WRITES_RE.search(action)


def _absorb(
    capture: Capture, index: int, vanished: Iterable[str], staged: dict[str, int], hazy: set[str]
) -> None:
    """Fold one capture into the staged/ambiguous ledger, in step order (mutates both ledgers)."""
    undo = bool(_UNDO_RE.search(capture.action))
    for path in sorted(vanished):
        if path in staged or path in hazy:
            continue
        if not undo and _staging_explains(capture.action, path):
            staged[path] = index
        else:
            hazy.add(path)
    if undo:
        # An undo can put ANY tracked path back, so every open claim retires here — permanently.
        hazy.update(staged)
        staged.clear()
    for path in capture.files:
        # The path is visible again: whatever hid it is over, and the capture is whole for it.
        staged.pop(path, None)
        hazy.discard(path)


def partial_loss_indices(captures: Mapping[int, Capture]) -> dict[int, PartialLoss]:
    """Steps whose non-empty capture is missing a path a `git add` provably removed from it."""
    order = sorted(captures)
    if not order:
        return {}
    # The image's OWN uncommitted edits: `offline_replay._restore_build_state` re-applies them from
    # `_BUILD_STATE` on any step whose diff does not already carry them, so losing them from a
    # capture costs the rebuild nothing.
    build_state = captures[order[0]].files
    seen: set[str] = set()
    staged: dict[str, int] = {}
    hazy: set[str] = set()
    out: dict[int, PartialLoss] = {}
    prev_marked = False
    for index in order:
        capture = captures[index]
        _absorb(capture, index, (seen - capture.files) - build_state, staged, hazy)
        seen |= capture.files
        marked = bool(capture.size and staged and not hazy)
        if marked:
            out[index] = PartialLoss(
                step_index=index,
                onset=not prev_marked,
                missing=tuple(sorted(staged)),
                staged_at=tuple(sorted(set(staged.values()))),
            )
        prev_marked = marked
    return out


def onset_evidence(step: StepView) -> str:
    """Classify WHY the capture went blind, from the onset step's own recorded action."""
    # Recorded for the audit trail only — see the module note on why it does not gate marking.
    action = step.action or ""
    if _STAGE_RE.search(action):
        return STAGED
    if _COMMIT_RE.search(action):
        return COMMITTED
    if _STASH_RE.search(action):
        return STASHED
    if _REVERT_RE.search(action):
        return REVERTED
    return UNEXPLAINED


def step_returncode(step: StepView) -> int | None:
    """The `<returncode>` the scaffold recorded in `result`, or None (an exception / timeout)."""
    # None is the interesting value: it means the step's command did not complete, so the capture
    # that follows it may have raced a container-side process that was still running.
    match = _RETURNCODE_RE.search(step.result or "")
    return int(match.group(1)) if match else None


def audit_trajectory(
    traj: Trajectory,
    sizes: Mapping[int, int],
    files: Mapping[int, frozenset[str]] | None = None,
) -> TrajectoryAudit:
    """Join the detected snapshot losses to this trajectory's steps, with their evidence."""
    # `files` is optional and its absence is a COVERAGE GAP, not a pass: without the paths a
    # capture touches there is no evidence for the partial rule, so it makes no claim at all
    # rather than a weaker one. Callers that hold the scratch always pass it.
    lost = lost_snapshot_indices(sizes)
    by_index = {step.step_index: step for step in traj.steps}
    terminal_index = traj.steps[-1].step_index if traj.steps else None
    losses: list[StepLoss] = []
    orphans: list[int] = []
    terminal_skipped: list[int] = []
    for index in sorted(lost):
        step = by_index.get(index)
        if step is None:
            # A captured diff with no StepView (40 in the corpus, 24 of them in the loss set):
            # there is no stamp to clear, so it is reported and not marked.
            orphans.append(index)
            continue
        if index == terminal_index:
            # The terminal step's stamp comes from the SWE-bench grading harness
            # (`header.terminal_resolved`), a different instrument that a lost per-step capture
            # says nothing about. Clearing it here would destroy the eval's own label.
            terminal_skipped.append(index)
            continue
        onset = lost[index]
        losses.append(
            StepLoss(
                step_index=index,
                onset=onset,
                evidence=onset_evidence(step) if onset else CONTINUATION,
                returncode=step_returncode(step),
            )
        )
    partial = _partial_losses(sizes, files, by_index, terminal_index)
    return TrajectoryAudit(
        trajectory_id=traj.header.trajectory_id,
        losses=tuple(losses),
        snapshots_seen=len(sizes),
        orphan_snapshots=tuple(orphans),
        terminal_skipped=tuple(terminal_skipped),
        partial_losses=partial,
    )


def _partial_losses(
    sizes: Mapping[int, int],
    files: Mapping[int, frozenset[str]] | None,
    by_index: Mapping[int, StepView],
    terminal_index: int | None,
) -> tuple[PartialLoss, ...]:
    """Run the partial-stage rule over the captures, skipping terminal and orphan steps."""
    if files is None:
        return ()
    captures = {
        index: Capture(
            size=size, files=files.get(index, frozenset()), action=_action(by_index, index)
        )
        for index, size in sizes.items()
    }
    return tuple(
        loss
        for index, loss in sorted(partial_loss_indices(captures).items())
        # Same two exclusions the empty rule makes: the terminal stamp is the harness grade, and
        # a capture with no StepView has no stamp to clear.
        if index != terminal_index and index in by_index
    )


def _action(by_index: Mapping[int, StepView], index: int) -> str:
    """The action recorded for a captured step, or '' when the recorder outran the parser."""
    step = by_index.get(index)
    return (step.action or "") if step is not None else ""


# ---------------------------------------------------------------------------
# Marking — clear the instrument's verdict, preserve the agent's behaviour.
# ---------------------------------------------------------------------------


def mark_step(step: StepView) -> StepView:
    """Render one step unmeasured: the existing unstamp, plus the instrument-reason flag."""
    # `unstamp_step` restores the parser defaults for every field the replay writes
    # (success/failing_check_id/exit_code/is_infra_failure/confirmed/dedup_key/blocking) and
    # touches nothing else. `is_infra_failure=True` on top says the outcome is absent for an
    # INSTRUMENT reason rather than never having been attempted, which is what makes a marked
    # step distinguishable from a never-stamped one. `blocking` is re-derived through the shared
    # predicate so Layer-1 stays clean.
    marked = replace(unstamp_step(step), is_infra_failure=True)
    return replace(marked, blocking=schema.recompute_blocking(marked))


def is_marked(step: StepView) -> bool:
    """True iff this step already carries the unmeasured-because-state-lost shape."""
    return step.success and not step.confirmed and step.is_infra_failure


def mark_trajectory(traj: Trajectory, step_indices: Iterable[int]) -> Trajectory:
    """A copy of *traj* with the named steps marked unmeasured and the header hash rebound."""
    wanted = set(step_indices)
    steps = [mark_step(s) if s.step_index in wanted else s for s in traj.steps]
    header = replace(traj.header, content_sha256=schema.content_sha256(steps), n_steps=len(steps))
    return schema.Trajectory(header=header, steps=steps)


# ---------------------------------------------------------------------------
# The committed audit record.
# ---------------------------------------------------------------------------


def _reason(audit: TrajectoryAudit) -> str:
    """The human sentence stored beside the marked indices, mirroring an admissibility reason."""
    if not audit.losses and not audit.partial_losses:
        return (
            f"CLEAN: no captured state is empty after an earlier one was non-empty, and no "
            f"non-empty capture is missing a staged path ({audit.snapshots_seen} captures "
            f"examined)."
        )
    return " ".join(part for part in (_empty_reason(audit), _partial_reason(audit)) if part)


def _empty_reason(audit: TrajectoryAudit) -> str:
    """The class-A sentence, or '' when this trajectory has no empty-after-non-empty loss."""
    if not audit.losses:
        return ""
    onsets = [loss for loss in audit.losses if loss.onset]
    classes = ", ".join(sorted({loss.evidence for loss in onsets})) or UNEXPLAINED
    return (
        f"STATE NOT OBSERVED: {len(audit.losses)} of {audit.snapshots_seen} captures are empty "
        f"after an earlier one was non-empty, across {len(onsets)} run(s) whose onset evidence is "
        f"[{classes}]. Those steps are marked unmeasured; their behaviour fields are unchanged."
    )


def _partial_reason(audit: TrajectoryAudit) -> str:
    """The class-B sentence, naming the paths a `git add` provably removed from the capture."""
    if not audit.partial_losses:
        return ""
    missing = sorted({path for loss in audit.partial_losses for path in loss.missing})
    return (
        f"STATE OBSERVED IN PART: {len(audit.partial_losses)} non-empty captures are missing "
        f"{len(missing)} path(s) a `git add` removed from the diff while leaving them on disk "
        f"[{', '.join(missing)}]. The replay would rebuild a tree the agent never had, so those "
        f"steps are marked unmeasured; their behaviour fields are unchanged."
    )


def report_record(audit: TrajectoryAudit) -> dict[str, object]:
    """One trajectory's committed audit record."""
    return {
        "detection_rule": DETECTION_RULE,
        "marked_steps": list(audit.marked_steps),
        "classes": {
            CLASS_EMPTY: {
                "rule": DETECTION_RULE,
                "marked_steps": [loss.step_index for loss in audit.losses],
            },
            CLASS_PARTIAL: {
                "rule": PARTIAL_DETECTION_RULE,
                "marked_steps": list(audit.partial_steps),
            },
        },
        "onsets": [
            {
                "step_index": loss.step_index,
                "evidence": loss.evidence,
                "returncode": loss.returncode,
            }
            for loss in audit.losses
            if loss.onset
        ],
        "partial_onsets": [
            {
                "step_index": loss.step_index,
                "missing": list(loss.missing),
                "staged_at": list(loss.staged_at),
            }
            for loss in audit.partial_losses
            if loss.onset
        ],
        "orphan_snapshots": list(audit.orphan_snapshots),
        "terminal_skipped": list(audit.terminal_skipped),
        "snapshots_seen": audit.snapshots_seen,
        "reason": _reason(audit),
    }


def read_report(data_dir: Path) -> dict[str, dict[str, object]]:
    """The committed audit record, or {} when the corpus has never been audited."""
    path = data_dir / REPORT_FILENAME
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _serialize(records: Mapping[str, dict[str, object]]) -> str:
    """One line per trajectory — still a reviewable text diff, without the indent tax."""
    # WHY NOT `indent=2`, WHICH `manifest.json` AND `admissibility.json` BOTH USE. This record is
    # per-TRAJECTORY and carries nine fields including a prose `reason`, so at 792 trajectories the
    # pretty form is 643 KB — over the repo's 500 KB `check-added-large-files` ceiling, where the
    # siblings (189 KB / 135 KB) are nowhere near it. 171 KB of that (26%) is indentation, which
    # carries no information. Compacting it is the only fix that neither weakens the ceiling for
    # every future file nor moves a DERIVED, re-marked-on-every-pass artifact into LFS, where each
    # regeneration would push a whole new object with no delta (the objection that already ruled
    # LFS out for the snapshot archive). No field is dropped: `json.loads` of this and of the
    # pretty form are the same object.
    # The line break stays at the RECORD boundary rather than disappearing entirely: a single
    # 476 KB line would take the file back to the thing LFS was rejected for — a change you cannot
    # read in review. Per-record lines keep `git diff` naming exactly which trajectories moved.
    if not records:
        return "{}\n"
    body = ",\n".join(
        f"{json.dumps(tid)}: {json.dumps(record, separators=(',', ':'), sort_keys=True)}"
        for tid, record in sorted(records.items())
    )
    return "{\n" + body + "\n}\n"


def write_report(records: Mapping[str, dict[str, object]], data_dir: Path) -> Path:
    """Persist the audit record so the exclusion is auditable off this host, not silent."""
    # Same discipline as `replay_admissibility.record_verdict`: the read-modify-write is one
    # locked transaction and the file is whole at every instant, because the stamping stage's
    # workers share this directory.
    path = data_dir / REPORT_FILENAME
    with corpus_lock.corpus_lock(data_dir):
        merged = dict(read_report(data_dir))
        merged.update(records)
        corpus_lock.atomic_write_text(path, _serialize(merged))
    return path


# ---------------------------------------------------------------------------
# The passes: audit, apply, verify.
# ---------------------------------------------------------------------------


def audit_corpus(
    data_dir: Path, snapshot_root: Path = step_snapshots.SNAPSHOT_ROOT
) -> dict[str, TrajectoryAudit]:
    """Audit every committed trajectory against the per-step scratch that produced it."""
    out: dict[str, TrajectoryAudit] = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        traj = schema.load_jsonl(path)
        tid = traj.header.trajectory_id
        if not step_snapshots.snapshot_dir(tid, snapshot_root).is_dir():
            # No scratch on THIS host says nothing about the capture — see
            # `offline_replay._handle_absent_snapshots`. Absence is a coverage gap, never a pass.
            continue
        out[tid] = audit_trajectory(
            traj, snapshot_sizes(tid, snapshot_root), snapshot_files(tid, snapshot_root)
        )
    return out


def apply_marks(
    data_dir: Path,
    snapshot_root: Path = step_snapshots.SNAPSHOT_ROOT,
    *,
    dry_run: bool = False,
) -> MarkSummary:
    """Mark every state-lost step in the corpus and write the committed audit record."""
    # Each rewrite goes through `offline_replay.commit_trajectory`, which re-binds the manifest in
    # the SAME locked transaction as the file — the pairing that a hand-rolled write would break.
    # It re-hashes every trajectory each time (~0.7s over 799), so a full pass costs minutes. That
    # is the right trade for a one-shot honesty pass; do not open-code the write to save it.
    from benchmark.runner.offline_replay import commit_trajectory  # noqa: PLC0415

    records: dict[str, dict[str, object]] = {}
    unaudited: list[str] = []
    written = already = marked = with_loss = partial = 0
    for path in sorted(data_dir.glob("*.jsonl")):
        traj = schema.load_jsonl(path)
        tid = traj.header.trajectory_id
        if not step_snapshots.snapshot_dir(tid, snapshot_root).is_dir():
            # No scratch on THIS host says nothing about the capture — see
            # `offline_replay._handle_absent_snapshots`. Absence is a coverage gap, never a pass,
            # so the trajectory is reported unaudited and deliberately gets no record.
            unaudited.append(tid)
            continue
        audit = audit_trajectory(
            traj, snapshot_sizes(tid, snapshot_root), snapshot_files(tid, snapshot_root)
        )
        records[tid] = report_record(audit)
        if not audit.marked_steps:
            continue
        with_loss += 1
        partial += len(audit.partial_steps)
        wanted = set(audit.marked_steps)
        already += sum(1 for s in traj.steps if s.step_index in wanted and is_marked(s))
        marked += len(wanted)
        updated = mark_trajectory(traj, wanted)
        if updated == traj:
            # IDEMPOTENCE, structurally: re-running produces byte-identical steps, so the second
            # pass writes nothing at all rather than merely writing the same bytes again.
            continue
        if not dry_run:
            commit_trajectory(updated, path)
        written += 1
    if not dry_run:
        write_report(records, data_dir)
    return MarkSummary(
        trajectories_examined=len(records),
        trajectories_with_loss=with_loss,
        steps_marked=marked,
        trajectories_written=written,
        steps_already_marked=already,
        unaudited=tuple(sorted(unaudited)),
        steps_marked_partial=partial,
    )


def verify_state_capture(
    data_dir: Path, snapshot_root: Path = step_snapshots.SNAPSHOT_ROOT
) -> list[Finding]:
    """The re-runnable guard: no step whose state was never captured may carry a stamp."""
    # Runs with no Docker, no network and no model. The scratch is authoritative when this
    # checkout has it; elsewhere the committed record is the only surviving evidence, and a
    # trajectory covered by NEITHER is reported as unaudited rather than passed silently.
    out: list[Finding] = []
    report = read_report(data_dir)
    for path in sorted(data_dir.glob("*.jsonl")):
        traj = schema.load_jsonl(path)
        tid = traj.header.trajectory_id
        record = report.get(tid)
        has_scratch = step_snapshots.snapshot_dir(tid, snapshot_root).is_dir()
        if not has_scratch and record is None:
            out.append(
                Finding(
                    ERROR,
                    "state_capture.unaudited",
                    tid,
                    f"neither a per-step scratch nor a {REPORT_FILENAME} entry — whether this "
                    "trajectory's captured state was lost is UNKNOWN, so its stamps are unverified",
                )
            )
            continue
        if has_scratch:
            audit = audit_trajectory(
                traj, snapshot_sizes(tid, snapshot_root), snapshot_files(tid, snapshot_root)
            )
            classes = audit.step_classes()
            out.extend(_record_findings(record, set(classes), tid))
        else:
            classes = _recorded_classes(record)
        out.extend(_stamp_findings(traj, classes, tid))
    return out


def _recorded_steps(record: dict[str, object] | None) -> list[int]:
    """The step indices a committed audit record marks, defensively parsed."""
    steps = (record or {}).get("marked_steps", [])
    return [int(i) for i in steps] if isinstance(steps, list) else []


def _recorded_classes(record: dict[str, object] | None) -> dict[int, str]:
    """Which class the committed record assigns each marked step (class A when unsplit)."""
    # A record written before the partial rule existed carries `marked_steps` but no `classes`;
    # its steps are class A by construction, which is exactly the fallback here.
    classes = dict.fromkeys(_recorded_steps(record), CLASS_EMPTY)
    split = (record or {}).get("classes")
    if isinstance(split, dict):
        for name, body in split.items():
            steps = body.get("marked_steps", []) if isinstance(body, dict) else []
            classes.update(dict.fromkeys((int(i) for i in steps), str(name)))
    return classes


def _record_findings(
    record: dict[str, object] | None, expected: set[int], tid: str
) -> list[Finding]:
    """Cross-check the committed record against what the scratch on this host actually shows."""
    if record is None:
        if not expected:
            return [
                Finding(
                    WARN,
                    "state_capture.unrecorded",
                    tid,
                    f"no {REPORT_FILENAME} entry; the scratch shows no loss, so nothing is "
                    "wrong here, but the corpus carries no committed proof of that",
                )
            ]
        return [
            Finding(
                ERROR,
                "state_capture.unrecorded",
                tid,
                f"{len(expected)} steps lost their captured state and no {REPORT_FILENAME} "
                "entry records it — off this host the exclusion would be invisible",
            )
        ]
    recorded = set(_recorded_steps(record))
    if recorded != expected:
        return [
            Finding(
                ERROR,
                "state_capture.record_mismatch",
                tid,
                f"{REPORT_FILENAME} records {sorted(recorded)} but the scratch shows "
                f"{sorted(expected)} — the committed audit record is stale",
            )
        ]
    return []


_LOSS_DETAIL: Final[dict[str, str]] = {
    CLASS_EMPTY: "captured state is empty after an earlier step's was not",
    CLASS_PARTIAL: "captured state is missing a path a `git add` removed from the diff",
}


def _stamp_findings(traj: Trajectory, expected: Mapping[int, str], tid: str) -> list[Finding]:
    """The gate itself: every state-lost step must carry the unmeasured shape, not a verdict."""
    out: list[Finding] = []
    for step in traj.steps:
        if step.step_index not in expected or is_marked(step):
            continue
        what = _LOSS_DETAIL.get(expected[step.step_index], _LOSS_DETAIL[CLASS_EMPTY])
        out.append(
            Finding(
                ERROR,
                "state_capture.stamped_after_loss",
                f"{tid}#{step.step_index}",
                f"{what}, yet the step carries "
                f"success={step.success} confirmed={step.confirmed} "
                f"failing_check_id={step.failing_check_id!r} — an outcome derived from a state "
                "that was never observed",
            )
        )
    return out


def _main() -> int:
    import argparse  # noqa: PLC0415

    from benchmark.escalation.live_capture import LIVE_DIR  # noqa: PLC0415
    from benchmark.routing.authenticity import errors, warnings  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=LIVE_DIR, help="committed corpus directory")
    ap.add_argument(
        "--snapshot-root",
        type=Path,
        default=step_snapshots.SNAPSHOT_ROOT,
        help="per-step diff scratch (gitignored; absent on any other checkout)",
    )
    ap.add_argument("--apply", action="store_true", help="mark the steps and write the record")
    ap.add_argument("--dry-run", action="store_true", help="report what --apply would change")
    args = ap.parse_args()

    if args.apply or args.dry_run:
        summary = apply_marks(args.data_dir, args.snapshot_root, dry_run=args.dry_run)
        print(f"{'DRY-RUN ' if args.dry_run else ''}{summary}")
        if summary.unaudited:
            print(f"UNAUDITED (no scratch on this host): {len(summary.unaudited)} trajectories")
            for tid in summary.unaudited:
                print(f"  {tid}")
        return 0

    findings = verify_state_capture(args.data_dir, args.snapshot_root)
    for finding in warnings(findings):
        print(f"WARN  {finding.rule} {finding.key}: {finding.detail}")
    failures = errors(findings)
    for finding in failures:
        print(f"ERROR {finding.rule} {finding.key}: {finding.detail}")
    print(f"state-capture check: {len(failures)} errors, {len(warnings(findings))} warnings")
    return 1 if failures else 0


__all__ = [
    "CLASS_EMPTY",
    "CLASS_PARTIAL",
    "COMMITTED",
    "CONTINUATION",
    "DETECTION_RULE",
    "PARTIAL_DETECTION_RULE",
    "REPORT_FILENAME",
    "REVERTED",
    "STAGED",
    "STASHED",
    "UNEXPLAINED",
    "Capture",
    "MarkSummary",
    "PartialLoss",
    "StepLoss",
    "TrajectoryAudit",
    "add_covers",
    "apply_marks",
    "audit_corpus",
    "audit_trajectory",
    "is_marked",
    "lost_snapshot_indices",
    "mark_step",
    "mark_trajectory",
    "onset_evidence",
    "partial_loss_indices",
    "read_report",
    "report_record",
    "snapshot_files",
    "snapshot_sizes",
    "step_returncode",
    "verify_state_capture",
    "write_report",
]


if __name__ == "__main__":
    raise SystemExit(_main())
