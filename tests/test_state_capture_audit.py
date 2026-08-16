"""State-capture audit: separate a genuine no-op from a lost capture, then mark the loss
unmeasured without touching a single byte of the agent's recorded behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from benchmark.escalation import authenticity, features, replay, schema
from benchmark.escalation.normalize.mini_swe_agent import stamp_step, unstamp_step
from benchmark.routing.authenticity import errors, warnings
from benchmark.runner import state_capture_audit as sca
from shunt.verifiers.parse import parse_test_outcome
from tests.escalation.factories import make_step, make_trajectory

_KEY = "tests/t.py::test_widget"
_ROOT = Path(__file__).resolve().parents[1]
_LIVE_DIR = _ROOT / "benchmark" / "escalation" / "data" / "live"
_SCRATCH = _ROOT / "benchmark" / "runner" / "artifacts" / "step_snapshots"

# The corpus is LFS-tracked; an un-hydrated checkout holds pointer files, not JSON.
_CORPUS_PRESENT = _LIVE_DIR.is_dir() and any(
    p.read_bytes()[:1] == b"{" for p in list(_LIVE_DIR.glob("*.jsonl"))[:1]
)
_needs_corpus = pytest.mark.skipif(
    not _CORPUS_PRESENT, reason="live corpus absent or LFS pointers not hydrated"
)


# ---------------------------------------------------------------------------
# The rule: which empty captures are a no-op and which are a loss.
# ---------------------------------------------------------------------------


def test_empty_prefix_is_a_genuine_no_op_and_is_never_flagged():
    # The agent had not yet touched a tracked file. `git diff` truthfully reports nothing, so
    # these steps are real measurements and must keep their stamps. 86% of the corpus's empty
    # captures are this case — flagging them would discard real data.
    assert sca.lost_snapshot_indices({0: 0, 1: 0, 2: 0, 3: 400}) == {}


def test_empty_after_non_empty_is_a_loss_with_the_first_marked_as_the_run_onset():
    assert sca.lost_snapshot_indices({0: 0, 1: 500, 2: 0, 3: 0}) == {2: True, 3: False}


def test_a_recovered_capture_that_goes_empty_again_opens_a_second_run():
    lost = sca.lost_snapshot_indices({0: 300, 1: 0, 2: 400, 3: 0, 4: 0})
    assert lost == {1: True, 3: True, 4: False}


def test_a_missing_capture_is_a_gap_that_neither_resets_the_latch_nor_is_reported():
    # Step 2 has no snapshot file at all: there is nothing to judge, and the accumulated state
    # from step 1 has not gone away, so step 3 is still a loss.
    assert sca.lost_snapshot_indices({1: 500, 3: 0}) == {3: True}


def test_an_all_empty_trajectory_flags_nothing():
    assert sca.lost_snapshot_indices({0: 0, 1: 0, 2: 0}) == {}


def test_indices_are_ordered_numerically_not_lexically():
    # step_9 < step_10 numerically; a string sort would put "10" first and invert the rule.
    assert sca.lost_snapshot_indices({9: 500, 10: 0}) == {10: True}


# ---------------------------------------------------------------------------
# The second class: a capture that is INCOMPLETE rather than empty.
#
# Every test below is written against the one thing the instrument can prove — a path was VISIBLE
# in an earlier capture and is not in this one — plus the only command whose effect on `git diff`
# is deterministic (`git add` moves a delta into the index without touching the worktree). The
# negative tests are the load-bearing ones: the majority cause of a path leaving a cumulative diff
# in the real corpus is a genuine `git checkout`, and marking those would delete real measurements.
# ---------------------------------------------------------------------------


def _cap(size: int, files: tuple[str, ...] = (), action: str = "echo hi") -> sca.Capture:
    """One captured step: how many bytes the diff was, which paths it named, what ran."""
    return sca.Capture(size=size, files=frozenset(files), action=action)


# Step 0 is the agent's first turn, before it has edited anything: whatever `git diff` shows there
# is the IMAGE's own build-time modification, which the rule uses as its build-state reference.
_EXPLORE = _cap(0, (), "ls -la")
_EDIT_A = _cap(400, ("pkg/a.py",), "python - <<'PY'\nopen('pkg/a.py','w')\nPY")
_STAGE_A = _cap(0, (), "git add pkg/a.py && git commit -m wip")
_EDIT_B = _cap(500, ("pkg/b.py",), "cat > pkg/b.py <<'PY'\nx=1\nPY")


def test_a_non_empty_capture_missing_a_staged_path_is_a_partial_loss():
    # The archetype: edit a.py (visible), stage it (capture goes empty — class A owns step 2),
    # then edit b.py. Step 3's capture is NON-EMPTY and describes only half the agent's work, so
    # replaying it rebuilds a tree the agent never had.
    lost = sca.partial_loss_indices({0: _EXPLORE, 1: _EDIT_A, 2: _STAGE_A, 3: _EDIT_B})
    assert set(lost) == {3}
    assert lost[3].missing == ("pkg/a.py",)
    assert lost[3].staged_at == (2,)
    assert lost[3].onset is True


def test_an_explicit_checkout_is_a_genuine_revert_and_is_never_flagged():
    # THE PRECISION TEST. 37 of the 41 measured drop events in the corpus are exactly this shape.
    # The capture is CORRECT here — the file really is back at base — so marking it would destroy
    # a real measurement.
    revert = _cap(0, (), "cd /testbed && git checkout -- pkg/a.py")
    assert sca.partial_loss_indices({0: _EXPLORE, 1: _EDIT_A, 2: revert, 3: _EDIT_B}) == {}


@pytest.mark.parametrize(
    "undo",
    [
        "git checkout -- pkg/a.py",
        "git restore pkg/a.py",
        "git reset --hard",
        "git stash",
        "git revert HEAD",
        "git clean -fd",
    ],
)
def test_any_undo_command_after_a_stage_retires_the_claim_permanently(undo):
    # Measured counter-example (matplotlib-25775 step 39): `git commit -- <path> || true; git
    # reset --soft HEAD~1`. The path is in fact still staged, but the command record no longer
    # pins the tree, so the audit abstains and the step KEEPS its stamp.
    caps = {0: _EXPLORE, 1: _EDIT_A, 2: _STAGE_A, 3: _cap(0, (), undo), 4: _EDIT_B}
    assert sca.partial_loss_indices(caps) == {}


def test_the_images_own_build_time_edit_is_excluded_because_the_replay_re_applies_it():
    # `offline_replay._restore_build_state` restores `_BUILD_STATE` on any step whose diff does not
    # already carry it, so a build-time path going missing from a capture costs the rebuild
    # nothing. Excluding it halves the real candidate set.
    caps = {
        0: _cap(120, ("pyproject.toml",), "ls -la"),
        1: _cap(400, ("pyproject.toml", "pkg/a.py"), "sed -i s/x/y/ pkg/a.py"),
        2: _cap(0, (), "git add -A && git commit -m wip"),
        3: _cap(500, ("pkg/b.py",), "echo done"),
    }
    assert sca.partial_loss_indices(caps)[3].missing == ("pkg/a.py",)


def test_intent_to_add_can_never_explain_a_drop_because_it_only_widens_the_diff():
    caps = {0: _EXPLORE, 1: _EDIT_A, 2: _cap(0, (), "git add -N pkg/a.py"), 3: _EDIT_B}
    assert sca.partial_loss_indices(caps) == {}


def test_an_add_whose_pathspec_misses_the_dropped_path_explains_nothing():
    caps = {0: _EXPLORE, 1: _EDIT_A, 2: _cap(0, (), "git add pkg/other.py"), 3: _EDIT_B}
    assert sca.partial_loss_indices(caps) == {}


def test_a_stage_step_that_could_also_have_rewritten_the_file_is_not_attributed():
    # The one competing explanation no `git` verb in the action would reveal: the agent edits the
    # file back to base AND stages it in the same turn, leaving a capture that is CORRECT.
    rewrite = _cap(0, (), "sed -i s/y/x/ pkg/a.py && git add -A")
    assert sca.partial_loss_indices({0: _EXPLORE, 1: _EDIT_A, 2: rewrite, 3: _EDIT_B}) == {}


def test_a_path_that_comes_back_into_the_capture_clears_its_claim():
    back = _cap(900, ("pkg/a.py", "pkg/b.py"), "echo")
    caps = {0: _EXPLORE, 1: _EDIT_A, 2: _STAGE_A, 3: _EDIT_B, 4: back}
    assert set(sca.partial_loss_indices(caps)) == {3}


def test_an_empty_capture_is_left_to_the_other_class_and_never_double_marked():
    # Disjointness by construction: the partial rule needs a NON-EMPTY capture, the empty rule
    # needs an empty one. Steps 2 and 4 are empty; only the non-empty step 3 is a partial loss.
    caps = {0: _EXPLORE, 1: _EDIT_A, 2: _STAGE_A, 3: _EDIT_B, 4: _cap(0, (), "git add -A")}
    sizes = {index: cap.size for index, cap in caps.items()}
    assert set(sca.partial_loss_indices(caps)) == {3}
    assert set(sca.lost_snapshot_indices(sizes)) == {2, 4}
    assert not set(sca.partial_loss_indices(caps)) & set(sca.lost_snapshot_indices(sizes))


def test_a_legitimate_partial_change_where_nothing_was_ever_staged_flags_nothing():
    # The agent simply edits more files as it goes. Every capture is a true observation.
    caps = {
        0: _EXPLORE,
        1: _cap(100, ("pkg/a.py",), "sed -i s/x/y/ pkg/a.py"),
        2: _cap(300, ("pkg/a.py", "pkg/b.py"), "sed -i s/x/y/ pkg/b.py"),
        3: _cap(600, ("pkg/a.py", "pkg/b.py", "pkg/c.py"), "sed -i s/x/y/ pkg/c.py"),
    }
    assert sca.partial_loss_indices(caps) == {}


def test_a_run_of_partial_steps_marks_only_its_first_as_the_onset():
    later = _cap(510, ("pkg/b.py",), "pytest -q")
    caps = {0: _EXPLORE, 1: _EDIT_A, 2: _STAGE_A, 3: _EDIT_B, 4: later}
    lost = sca.partial_loss_indices(caps)
    assert [(index, loss.onset) for index, loss in sorted(lost.items())] == [(3, True), (4, False)]


def test_no_captures_at_all_makes_no_claim():
    assert sca.partial_loss_indices({}) == {}


@pytest.mark.parametrize(
    ("action", "path", "covered"),
    [
        ("git add -A", "pkg/a.py", True),
        ("git add --all", "pkg/a.py", True),
        ("git add .", "pkg/a.py", True),
        ("git -C /testbed add -u", "pkg/a.py", True),
        ("git add pkg/a.py", "pkg/a.py", True),
        ("git add pkg/", "pkg/a.py", True),
        ("git add pkg/b.py", "pkg/a.py", False),
        ("git add -N pkg/a.py", "pkg/a.py", False),
        ("git add --intent-to-add pkg/a.py", "pkg/a.py", False),
        ("git status --short", "pkg/a.py", False),
        ("git add -N pkg/a.py; git add -A", "pkg/a.py", True),
    ],
)
def test_add_covers_reads_the_pathspec_not_just_the_verb(action, path, covered):
    assert sca.add_covers(action, path) is covered


# ---------------------------------------------------------------------------
# Corroborating evidence: recorded, and deliberately not used to narrow the rule.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("git add -A", sca.STAGED),
        ("cd /testbed && git add astropy/table.py", sca.STAGED),
        ("git -C /testbed commit -m wip", sca.COMMITTED),
        ("cd /testbed && git stash && pytest -q", sca.STASHED),
        ("git checkout -- astropy/table.py", sca.REVERTED),
        ("git restore --staged --worktree x.py", sca.REVERTED),
        ("git reset --hard", sca.REVERTED),
        ("sed -n '1,20p' README.rst", sca.UNEXPLAINED),
    ],
)
def test_onset_evidence_reads_the_recorded_action(action, expected):
    assert sca.onset_evidence(make_step(action=action)) == expected


def test_returncode_is_read_from_the_result_and_is_none_when_the_step_did_not_complete():
    assert sca.step_returncode(make_step(result="<returncode>0</returncode>\n<output>ok")) == 0
    assert sca.step_returncode(make_step(result="<returncode>-1</returncode>")) == -1
    assert sca.step_returncode(make_step(result="<exception>docker exec timed out")) is None


def test_a_read_only_onset_action_is_still_marked_and_its_evidence_is_only_a_hint():
    # The measured counter-example that decides the design (astropy-13236__kimi-k3__max step 30):
    # the onset action is READ-ONLY (`git stash list`) and the capture is empty only because the
    # previous, timed-out step's `git stash -q` completed inside the container after that step's
    # snapshot was taken. The evidence class even reads "stashed" here, from a command that
    # stashes nothing — which is exactly why marking is keyed on the capture sequence, not on it.
    steps = [
        make_step(step_index=0, action="sed -i s/a/b/ x.py"),
        make_step(step_index=1, action="cd /testbed && git stash list", result="<exception>"),
        make_step(step_index=2, action="echo done"),
    ]
    traj = make_trajectory(steps, trajectory_id="t-readonly")
    audit = sca.audit_trajectory(traj, {0: 500, 1: 0})
    assert audit.marked_steps == (1,)
    assert audit.losses[0].evidence == sca.STASHED
    assert audit.losses[0].returncode is None


def test_continuation_steps_carry_the_continuation_evidence_class():
    steps = [make_step(step_index=i, action=f"a{i}") for i in range(4)]
    steps[1] = make_step(step_index=1, action="git add -A")
    traj = make_trajectory(steps, trajectory_id="t-run")
    audit = sca.audit_trajectory(traj, {0: 500, 1: 0, 2: 0})
    assert [(loss.step_index, loss.evidence) for loss in audit.losses] == [
        (1, sca.STAGED),
        (2, sca.CONTINUATION),
    ]


# ---------------------------------------------------------------------------
# The join: what the audit refuses to mark.
# ---------------------------------------------------------------------------


def test_the_terminal_step_is_never_marked_because_its_stamp_is_the_harness_grade():
    steps = [make_step(step_index=i) for i in range(3)]
    traj = make_trajectory(steps, trajectory_id="t-term", terminal_resolved=True)
    audit = sca.audit_trajectory(traj, {0: 500, 1: 0, 2: 0})
    assert audit.marked_steps == (1,)
    assert audit.terminal_skipped == (2,)


def test_a_snapshot_with_no_stepview_is_reported_as_an_orphan_and_not_marked():
    traj = make_trajectory([make_step(step_index=i) for i in range(2)], trajectory_id="t-orphan")
    audit = sca.audit_trajectory(traj, {0: 500, 1: 0, 7: 0})
    assert audit.marked_steps == ()  # step 1 is terminal here; step 7 has no StepView
    assert audit.orphan_snapshots == (7,)


def test_the_partial_rule_makes_no_claim_when_the_capture_paths_are_unavailable():
    # Absence of the diff TEXT is a coverage gap, not a pass: with no paths there is no evidence,
    # so the rule reports nothing rather than something weaker.
    steps = [make_step(step_index=i, action=a) for i, a in enumerate(["edit", "git add -A", "e2"])]
    traj = make_trajectory([*steps, make_step(step_index=3)], trajectory_id="t-nofiles")
    audit = sca.audit_trajectory(traj, {0: 400, 1: 0, 2: 500})
    assert audit.partial_losses == ()


_PARTIAL_ACTIONS = (
    "ls -la",
    "sed -i s/x/y/ pkg/a.py",
    "git add pkg/a.py && git commit -m wip",
    "pytest -q",
    "echo done",
)
_PARTIAL_SIZES: Final[dict[int, int]] = {0: 0, 1: 400, 2: 0, 3: 500, 4: 500}
_PARTIAL_FILES: Final[dict[int, frozenset[str]]] = {
    0: frozenset(),
    1: frozenset({"pkg/a.py"}),
    2: frozenset(),
    3: frozenset({"pkg/b.py"}),
    4: frozenset({"pkg/b.py"}),
}


def _partial_traj(tid: str = "t-partial", n_steps: int = 5) -> schema.Trajectory:
    """explore -> edit a.py -> stage it -> edit b.py -> terminal; step 3's capture is partial."""
    steps = [make_step(step_index=i, action=_PARTIAL_ACTIONS[i]) for i in range(n_steps)]
    return make_trajectory(steps, trajectory_id=tid, terminal_resolved=True)


def test_the_two_classes_are_joined_into_one_marked_set_and_stay_separable():
    audit = sca.audit_trajectory(_partial_traj(), _PARTIAL_SIZES, _PARTIAL_FILES)
    assert audit.marked_steps == (2, 3)  # step 2 empty (class A), step 3 partial (class B)
    assert audit.partial_steps == (3,)
    assert audit.step_classes() == {2: sca.CLASS_EMPTY, 3: sca.CLASS_PARTIAL}


def test_the_terminal_step_is_never_marked_by_the_partial_rule_either():
    # Step 4 is terminal and its capture is also missing pkg/a.py, but its stamp is the harness
    # grade — a different instrument that a partial per-step capture says nothing about.
    audit = sca.audit_trajectory(_partial_traj(), _PARTIAL_SIZES, _PARTIAL_FILES)
    assert 4 not in audit.marked_steps


def test_a_partial_capture_with_no_stepview_is_not_marked():
    traj = make_trajectory(
        [make_step(step_index=i, action=_PARTIAL_ACTIONS[i]) for i in range(3)],
        trajectory_id="t-partial-orphan",
    )
    audit = sca.audit_trajectory(traj, _PARTIAL_SIZES, _PARTIAL_FILES)
    assert audit.partial_steps == ()  # steps 3-4 have no StepView to unstamp


# ---------------------------------------------------------------------------
# The marker: unmeasured, not failed; behaviour preserved byte-for-byte.
# ---------------------------------------------------------------------------


def _failed_step(index: int) -> schema.StepView:
    """A step stamped by the replay as a real capability failure."""
    return stamp_step(
        make_step(
            step_index=index,
            action=f"act{index}",
            result=f"<returncode>1</returncode>\n<output>{_KEY} FAILED</output>",
        ),
        parse_test_outcome(f"{_KEY} FAILED\nAssertionError", 1),
    )


def test_marking_routes_the_step_to_none_rather_than_to_a_verified_failure():
    marked = sca.mark_step(_failed_step(1))
    assert replay.verified_outcome(marked) is replay.VerifiedOutcome.NONE
    assert marked.success is True
    assert marked.confirmed is False
    assert marked.blocking is False
    assert marked.failing_check_id is None
    assert marked.dedup_key is None
    assert marked.exit_code is None


def test_a_marked_step_is_distinguishable_from_every_other_writer_of_the_same_fields():
    # (success, confirmed, is_infra_failure) — the marked triple must be unique, or the audit
    # trail would be indistinguishable from a step the replay simply never reached.
    plain = make_step(step_index=0)
    shapes = {
        "parser default": plain,
        "unstamped": unstamp_step(_failed_step(0)),
        "replayed failure": _failed_step(0),
        "replayed pass": stamp_step(plain, parse_test_outcome("2 passed", 0)),
        "replayed infra": stamp_step(plain, parse_test_outcome("ERROR collecting\nImportError", 2)),
    }
    marked = sca.mark_step(_failed_step(0))
    triple = (marked.success, marked.confirmed, marked.is_infra_failure)
    assert triple == (True, False, True)
    assert sca.is_marked(marked)
    for name, step in shapes.items():
        assert (step.success, step.confirmed, step.is_infra_failure) != triple, name
        assert not sca.is_marked(step), name


def test_marking_preserves_every_collected_behaviour_field():
    original = _failed_step(3)
    marked = sca.mark_step(original)
    for field in ("action", "args", "observation", "result", "tool", "metadata", "status"):
        assert getattr(marked, field) == getattr(original, field), field
    # ...and every identity / drift field the replay does not own.
    for field in ("step_index", "decision_index", "parent_step_index", "is_revert", "retry_count"):
        assert getattr(marked, field) == getattr(original, field), field


def test_marking_a_trajectory_drops_it_out_of_is_stamped_when_every_step_is_lost():
    steps = [_failed_step(0), _failed_step(1), _failed_step(2)]
    traj = make_trajectory(steps, trajectory_id="t-all")
    assert features.is_stamped(traj)
    marked = sca.mark_trajectory(traj, {0, 1})
    assert not features.is_stamped(marked)
    assert marked.header.content_sha256 == schema.content_sha256(marked.steps)


def test_marking_is_idempotent_at_the_step_and_at_the_trajectory():
    step = _failed_step(1)
    assert sca.mark_step(sca.mark_step(step)) == sca.mark_step(step)
    traj = make_trajectory([_failed_step(i) for i in range(3)], trajectory_id="t-idem")
    once = sca.mark_trajectory(traj, {1})
    assert sca.mark_trajectory(once, {1}) == once


# ---------------------------------------------------------------------------
# The corpus pass, on a real on-disk corpus + scratch (no Docker, no network).
# ---------------------------------------------------------------------------


def _corpus(tmp_path: Path, *, sizes: dict[int, int], n_steps: int = 4) -> tuple[Path, Path]:
    """A one-trajectory corpus with a per-step scratch, written the way the real pipeline does."""
    data_dir = tmp_path / "live"
    data_dir.mkdir()
    scratch = tmp_path / "snapshots"
    traj = make_trajectory(
        [_failed_step(i) for i in range(n_steps)], trajectory_id="tid", terminal_resolved=True
    )
    schema.dump_jsonl(traj, data_dir / "tid.jsonl")
    (data_dir / authenticity.MANIFEST_NAME).write_text(
        json.dumps(authenticity.manifest(data_dir), indent=2, sort_keys=True) + "\n"
    )
    (scratch / "tid").mkdir(parents=True)
    for index, size in sizes.items():
        (scratch / "tid" / f"step_{index:04d}.diff").write_text("x" * size)
    return data_dir, scratch


def _partial_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A corpus whose scratch holds REAL diff headers, so the partial rule has paths to read."""
    data_dir = tmp_path / "live"
    data_dir.mkdir()
    scratch = tmp_path / "snapshots"
    steps = [
        stamp_step(
            make_step(step_index=i, action=_PARTIAL_ACTIONS[i]),
            parse_test_outcome(f"{_KEY} FAILED\nAssertionError", 1),
        )
        for i in range(5)
    ]
    traj = make_trajectory(steps, trajectory_id="tid", terminal_resolved=True)
    schema.dump_jsonl(traj, data_dir / "tid.jsonl")
    (data_dir / authenticity.MANIFEST_NAME).write_text(
        json.dumps(authenticity.manifest(data_dir), indent=2, sort_keys=True) + "\n"
    )
    (scratch / "tid").mkdir(parents=True)
    for index, paths in _PARTIAL_FILES.items():
        body = "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@\n+x\n" for p in paths)
        (scratch / "tid" / f"step_{index:04d}.diff").write_text(body)
    return data_dir, scratch


def test_apply_marks_marks_both_classes_and_records_them_separately(tmp_path):
    data_dir, scratch = _partial_corpus(tmp_path)
    summary = sca.apply_marks(data_dir, scratch)
    assert (summary.steps_marked, summary.steps_marked_partial) == (2, 1)
    traj = schema.load_jsonl(data_dir / "tid.jsonl")
    assert [sca.is_marked(s) for s in traj.steps] == [False, False, True, True, False]
    assert traj.steps[1].confirmed  # the one fully-observed measurement is untouched
    assert traj.steps[4].confirmed  # the terminal harness grade survives
    record = sca.read_report(data_dir)["tid"]
    assert record["marked_steps"] == [2, 3]
    assert record["classes"][sca.CLASS_EMPTY]["marked_steps"] == [2]
    assert record["classes"][sca.CLASS_PARTIAL]["marked_steps"] == [3]
    assert record["classes"][sca.CLASS_PARTIAL]["rule"] == sca.PARTIAL_DETECTION_RULE
    assert record["partial_onsets"] == [
        {"step_index": 3, "missing": ["pkg/a.py"], "staged_at": [2]}
    ]
    assert "missing" in record["reason"] and "pkg/a.py" in record["reason"]
    assert errors(authenticity.verify_manifest(data_dir)) == []


def test_the_report_is_one_line_per_trajectory_and_an_empty_write_is_still_json(tmp_path):
    # The record is per-TRAJECTORY and carries a prose `reason`, so at corpus scale the pretty form
    # its two siblings use crosses the repo's 500 KB added-file ceiling on indentation alone. The
    # line break stays at the record boundary so `git diff` still names WHICH trajectory moved —
    # and the no-records case is the edge where a naive join emits `{\n\n}`, which is not JSON.
    data_dir, scratch = _partial_corpus(tmp_path)
    sca.apply_marks(data_dir, scratch)
    lines = (data_dir / sca.REPORT_FILENAME).read_text().splitlines()
    assert lines[0] == "{" and lines[-1] == "}"
    assert [line.split(":")[0] for line in lines[1:-1]] == ['"tid"']
    empty = tmp_path / "empty"
    empty.mkdir()
    sca.write_report({}, empty)
    assert sca.read_report(empty) == {}


def test_the_guard_names_the_partial_class_when_a_partial_step_still_carries_a_stamp(tmp_path):
    data_dir, scratch = _partial_corpus(tmp_path)
    failures = errors(sca.verify_state_capture(data_dir, scratch))
    detail = {f.key: f.detail for f in failures if f.rule == "state_capture.stamped_after_loss"}
    assert "missing a path a `git add` removed" in detail["tid#3"]
    assert "empty after an earlier step's was not" in detail["tid#2"]
    sca.apply_marks(data_dir, scratch)
    assert errors(sca.verify_state_capture(data_dir, scratch)) == []


def test_the_guard_reads_the_partial_class_back_off_the_record_with_no_scratch(tmp_path):
    data_dir, scratch = _partial_corpus(tmp_path)
    sca.apply_marks(data_dir, scratch)
    missing = tmp_path / "no-scratch"
    assert errors(sca.verify_state_capture(data_dir, missing)) == []
    traj = schema.load_jsonl(data_dir / "tid.jsonl")
    steps = [_failed_step(3) if s.step_index == 3 else s for s in traj.steps]
    schema.dump_jsonl(make_trajectory(steps, trajectory_id="tid"), data_dir / "tid.jsonl")
    failures = errors(sca.verify_state_capture(data_dir, missing))
    assert [f.key for f in failures] == ["tid#3"]
    assert "missing a path a `git add` removed" in failures[0].detail


def test_a_record_written_before_the_partial_rule_existed_still_reads_as_class_a(tmp_path):
    data_dir, scratch = _partial_corpus(tmp_path)
    sca.apply_marks(data_dir, scratch)
    path = data_dir / sca.REPORT_FILENAME
    record = json.loads(path.read_text())
    del record["tid"]["classes"]  # the shape this file had before the second class landed
    path.write_text(json.dumps(record))
    assert errors(sca.verify_state_capture(data_dir, tmp_path / "no-scratch")) == []


def test_audit_corpus_reports_read_only_and_skips_trajectories_with_no_scratch(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    before = (data_dir / "tid.jsonl").read_bytes()
    assert sca.audit_corpus(data_dir, scratch)["tid"].marked_steps == (2,)
    assert sca.audit_corpus(data_dir, tmp_path / "no-scratch") == {}
    assert (data_dir / "tid.jsonl").read_bytes() == before


def test_apply_marks_marks_the_lost_steps_and_leaves_the_corpus_self_consistent(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    summary = sca.apply_marks(data_dir, scratch)
    assert (summary.trajectories_with_loss, summary.steps_marked) == (1, 1)  # step 3 is terminal
    traj = schema.load_jsonl(data_dir / "tid.jsonl")
    assert sca.is_marked(traj.steps[2])
    assert not sca.is_marked(traj.steps[1])  # a real, observed measurement is untouched
    assert traj.steps[3].confirmed  # the terminal harness grade survives
    assert errors(authenticity.verify_manifest(data_dir)) == []


def test_apply_marks_writes_a_committed_record_naming_the_reason(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    sca.apply_marks(data_dir, scratch)
    record = sca.read_report(data_dir)["tid"]
    assert record["marked_steps"] == [2]
    assert record["detection_rule"] == sca.DETECTION_RULE
    assert "empty after an earlier one was non-empty" in record["reason"]
    assert record["onsets"] == [{"step_index": 2, "evidence": sca.UNEXPLAINED, "returncode": 1}]


def test_a_clean_trajectory_is_recorded_too_so_audited_is_distinguishable_from_unaudited(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 0, 2: 400, 3: 400})
    sca.apply_marks(data_dir, scratch)
    record = sca.read_report(data_dir)["tid"]
    assert record["marked_steps"] == []
    assert record["reason"].startswith("CLEAN")


def test_running_the_marker_twice_writes_nothing_the_second_time(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    sca.apply_marks(data_dir, scratch)
    first = (data_dir / "tid.jsonl").read_bytes()
    second_summary = sca.apply_marks(data_dir, scratch)
    assert second_summary.trajectories_written == 0
    assert second_summary.steps_already_marked == second_summary.steps_marked == 1
    assert (data_dir / "tid.jsonl").read_bytes() == first


def test_the_marker_is_safe_on_a_corpus_where_some_steps_are_already_marked(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 500, 1: 0, 2: 0, 3: 0})
    traj = schema.load_jsonl(data_dir / "tid.jsonl")
    schema.dump_jsonl(sca.mark_trajectory(traj, {1}), data_dir / "tid.jsonl")
    summary = sca.apply_marks(data_dir, scratch)
    assert summary.steps_already_marked == 1
    assert summary.steps_marked == 2
    after = schema.load_jsonl(data_dir / "tid.jsonl")
    assert [sca.is_marked(s) for s in after.steps] == [False, True, True, False]


def test_dry_run_changes_nothing_on_disk(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    before = (data_dir / "tid.jsonl").read_bytes()
    summary = sca.apply_marks(data_dir, scratch, dry_run=True)
    assert summary.steps_marked == 1
    assert (data_dir / "tid.jsonl").read_bytes() == before
    assert not (data_dir / sca.REPORT_FILENAME).exists()


# ---------------------------------------------------------------------------
# The re-runnable guard.
# ---------------------------------------------------------------------------


def test_the_guard_fails_on_a_step_that_lost_its_state_and_still_carries_a_stamp(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    failures = errors(sca.verify_state_capture(data_dir, scratch))
    assert [f.rule for f in failures] == [
        "state_capture.unrecorded",
        "state_capture.stamped_after_loss",
    ]
    assert "tid#2" in {f.key for f in failures}


def test_the_guard_passes_once_the_marker_has_run(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    sca.apply_marks(data_dir, scratch)
    assert errors(sca.verify_state_capture(data_dir, scratch)) == []


def test_the_guard_runs_off_the_committed_record_when_the_scratch_is_absent(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    sca.apply_marks(data_dir, scratch)
    missing = tmp_path / "no-scratch"
    assert errors(sca.verify_state_capture(data_dir, missing)) == []
    # ...and it still catches a re-stamped step with no scratch to consult.
    traj = schema.load_jsonl(data_dir / "tid.jsonl")
    steps = [_failed_step(2) if s.step_index == 2 else s for s in traj.steps]
    schema.dump_jsonl(make_trajectory(steps, trajectory_id="tid"), data_dir / "tid.jsonl")
    failures = errors(sca.verify_state_capture(data_dir, missing))
    assert [f.rule for f in failures] == ["state_capture.stamped_after_loss"]


def test_a_corpus_with_neither_scratch_nor_record_is_reported_unaudited_not_passed(tmp_path):
    data_dir, _scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    failures = errors(sca.verify_state_capture(data_dir, tmp_path / "no-scratch"))
    assert [f.rule for f in failures] == ["state_capture.unaudited"]


def test_a_stale_committed_record_is_an_error_not_a_pass(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 500, 2: 0, 3: 0})
    sca.apply_marks(data_dir, scratch)
    (scratch / "tid" / "step_0001.diff").write_text("")  # the loss now starts earlier
    (scratch / "tid" / "step_0000.diff").write_text("x" * 90)
    failures = errors(sca.verify_state_capture(data_dir, scratch))
    assert "state_capture.record_mismatch" in {f.rule for f in failures}


def test_a_clean_trajectory_with_no_record_warns_rather_than_failing(tmp_path):
    data_dir, scratch = _corpus(tmp_path, sizes={0: 0, 1: 0, 2: 400, 3: 400})
    findings = sca.verify_state_capture(data_dir, scratch)
    assert errors(findings) == []
    assert [f.rule for f in warnings(findings)] == ["state_capture.unrecorded"]


# ---------------------------------------------------------------------------
# The ratchet on the real committed corpus.
# ---------------------------------------------------------------------------


# THE MARKING HAS LANDED, AND THIS RATCHET STILL HOLDS FOR A DIFFERENT REASON. `--apply` ran over
# the rebuilt corpus and marked 1 959 steps across 310 trajectories (1 897 `empty_after_nonempty` /
# 310 trajectories, 62 `partial_stage_loss` / 16), clearing 1 825 steps that had been asserting a
# capability failure. Every `state_capture.stamped_after_loss` error is gone. What remains is a
# NARROWER and different defect, so the marker's original reason no longer describes it:
#
#   7 trajectories have neither a per-step scratch nor a `state_capture.json` entry, so
#   `verify_state_capture` reports `state_capture.unaudited` — it cannot PROVE their stamps were
#   not derived from a lost state. They are exactly the rebuild's `SnapshotsMissingError` failures
#   (`snapshot_steps=None` in the header, scratch gone, so the replay refused to clear real
#   stamps). Measured: each carries 0 confirmed scorable steps (`features.is_stamped` is False), so
#   no per-step verdict of theirs reaches the eval; their only confirmed step is the terminal one,
#   whose stamp is the SWE-bench harness grade — a different instrument this bug never touched.
#
# So this is a coverage gap the gate correctly refuses to pass silently, not fabricated data. It
# cannot be closed from committed data alone: it needs the scratch restored or those trajectories
# re-collected. XPASS here means the 7 were resolved — delete this marker.
@_needs_corpus
@pytest.mark.xfail(
    strict=True,
    reason="7 trajectories have neither a per-step scratch nor a state_capture.json entry, so "
    "their stamps are unverifiable (state_capture.unaudited). The marking itself has landed.",
)
def test_the_committed_corpus_carries_no_stamp_on_a_step_whose_state_was_never_captured():
    failures = errors(sca.verify_state_capture(_LIVE_DIR, _SCRATCH))
    assert failures == [], f"{len(failures)} steps assert an outcome derived from a lost state"
