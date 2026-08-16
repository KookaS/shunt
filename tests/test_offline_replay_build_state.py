"""Build-state ordering in the offline replay, driven through REAL git in a REAL temp repo."""

# NO DOCKER AND NO STUBBED `git`. The thing under test is what git itself does to a working tree
# when `git apply` meets a hunk that is already applied, so a fake `git` could only assert the
# commands we chose to write. `exec_fn` runs the container command string through `bash` against a
# throwaway repository — the same shell, the same git, the same atomicity — and every assertion is
# made on the resulting FILE CONTENTS, which is the quantity the replay's verdict depends on. The
# one stand-in is the adjudicator (`classify`), exactly as in `test_offline_replay.py`.
#
# FIXTURE NAMING RULE: no filename here is a substring of any other. A sibling fix in this repo was
# briefly "proven" by a mock-based test whose bug a substring collision between two fixture
# filenames hid, so `alpha_helper.py` / `bravo_widget.py` / `source_module.py` / `tox_config.ini`
# are deliberately unrelated as strings.

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from benchmark.runner import offline_replay
from shunt.verifiers.base import VerifierResult

# The committed base state of the fake instance repo.
_BASE: Final[dict[str, str]] = {
    "tox_config.ini": "[testenv]\ncommands = pytest {posargs}\n",
    "source_module.py": "def compute():\n    return 1\n",
    "alpha_helper.py": "ALPHA = 'base'\n",
    "bravo_widget.py": "BRAVO = 'base'\n",
}

# The image's build-time `pre_install` edit, left UNCOMMITTED in the working tree — sphinx's
# `sed -i 's/pytest/pytest -rA/' tox.ini`, which is the real shape of `_BUILD_STATE`.
_BUILD_TOX = "[testenv]\ncommands = pytest -rA {posargs}\n"
# A second build-edited file, used only by the partial-overlap case.
_BUILD_ALPHA = "ALPHA = 'built'\n"

_AGENT_SOURCE = "def compute():\n    return 2\n"
_GOLD_SOURCE = "def compute():\n    return 42\n"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout


def _write(repo: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        (repo / name).write_text(body, encoding="utf-8")


def _read(repo: Path, name: str) -> str:
    return (repo / name).read_text(encoding="utf-8")


@dataclass
class _Rig:
    """A real git repo standing in for /testbed, plus the exec/classify plumbing around it."""

    repo: Path
    state: Path
    exec_fn: offline_replay.ContainerExec
    commands: list[str] = field(default_factory=list)
    classified: list[tuple[str, int]] = field(default_factory=list)

    def classify(self, combined: str, rc: int) -> VerifierResult:
        # Stand-in for `GraderParity`. Recording that it was reached at all is half the point: a
        # step that fails to reconcile must never get this far.
        self.classified.append((combined, rc))
        return VerifierResult(outcome="success", confidence=1.0, detail="stub adjudicator")

    def build_edit(self, files: dict[str, str]) -> None:
        """Put the image's build-time edit into the worktree, unstaged — never committed."""
        _write(self.repo, files)

    def capture_step_diff(self, edits: dict[str, str]) -> str:
        """What the AGENT's recorder would have captured with *edits* on top of the tree NOW."""
        # `git diff HEAD`, verbatim from `step_snapshots.DIFF_COMMAND` — so whatever is
        # uncommitted right now (the image's build edit included) lands inside the returned diff.
        # That inclusion IS the defect under test, and getting the command wrong here would make
        # every green below meaningless.
        before = {name: _read(self.repo, name) for name in edits}
        _write(self.repo, edits)
        diff = _git(self.repo, "diff", "HEAD")
        _write(self.repo, before)
        return diff

    def replay(self, diff: str, test_patch: str = "") -> VerifierResult:
        # `cat <selectors>` as the "test command": the run's combined output is then literally the
        # reconstructed tree, so a wrong tree is visible in the same channel the real classifier
        # reads. File-content assertions below are the primary check; this is the cross-check.
        return offline_replay.replay_step(
            diff, test_patch, "cat", sorted(_BASE), self.exec_fn, self.classify
        )


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    repo = tmp_path / "testbed"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "replay@example.invalid")
    _git(repo, "config", "user.name", "replay")
    _write(repo, _BASE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    # OUTSIDE the repo, mirroring the real `/tmp` location: `git clean -fdq` must not reach it.
    state = tmp_path / "build_state.patch"
    monkeypatch.setattr(offline_replay, "TESTBED", str(repo))
    monkeypatch.setattr(offline_replay, "_BUILD_STATE", str(state))
    monkeypatch.setattr(offline_replay, "_BUILD_STATE_TMP", f"{state}.tmp")

    commands: list[str] = []

    def _exec(command: str, stdin: str | None = None) -> tuple[str, int]:
        commands.append(command)
        proc = subprocess.run(  # noqa: S603
            ["bash", "-c", command], input=stdin, capture_output=True, text=True, check=False
        )
        return f"{proc.stdout}\n{proc.stderr}", proc.returncode

    return _Rig(repo=repo, state=state, exec_fn=_exec, commands=commands)


# ---------------------------------------------------------------------------
# The defect: a step diff that already carries the build edit.
# ---------------------------------------------------------------------------


def test_a_step_diff_that_already_carries_the_build_edit_now_replays(rig: _Rig) -> None:
    # THE MEASURED DEFECT. 2 963 of 3 109 steps (95.3%) went infra on trajectories whose first
    # capture is non-empty, against 317 of 8 247 (3.8%) where it is empty; within-instance on
    # astropy-13977, 108/129 vs 0/48. Cause: the build edit is uncommitted in the AGENT's container
    # too, so the captured step diff contains it, and re-applying it BEFORE the step diff made
    # `git apply -` meet an already-applied hunk. `git apply` is atomic, so the whole step diff
    # was rejected and a real measurement became an infra failure.
    rig.build_edit({"tox_config.ini": _BUILD_TOX})
    step_diff = rig.capture_step_diff({"source_module.py": _AGENT_SOURCE})
    # The fixture only reproduces the defect if the capture really swallowed the build edit.
    assert "tox_config.ini" in step_diff
    assert "pytest -rA" in step_diff

    result = rig.replay(step_diff)

    assert result.is_infra_failure is False
    assert result.outcome == "success"
    # Ground truth: the agent's edit AND the build edit present, everything else at base.
    assert _read(rig.repo, "source_module.py") == _AGENT_SOURCE
    assert _read(rig.repo, "tox_config.ini") == _BUILD_TOX
    assert _read(rig.repo, "alpha_helper.py") == _BASE["alpha_helper.py"]
    assert _read(rig.repo, "bravo_widget.py") == _BASE["bravo_widget.py"]
    # ...and the build edit is present exactly once, not applied twice.
    assert _read(rig.repo, "tox_config.ini").count("-rA") == 1


def test_the_pre_c2_ordering_really_did_reject_that_step_diff(rig: _Rig) -> None:
    # POSITIVE CONTROL ON THE FIXTURE ABOVE. Without this, a green in the previous test could mean
    # "fixed" or "the fixture never reproduced the bug". Here the pre-C2 reset command is issued
    # VERBATIM — its trailing re-apply is the only difference — and the identical step diff is
    # then rejected by git. That rejection is the 95.3% infra rate, reproduced in a temp repo.
    rig.build_edit({"tox_config.ini": _BUILD_TOX})
    step_diff = rig.capture_step_diff({"source_module.py": _AGENT_SOURCE})

    state, tmp, repo = rig.state, f"{rig.state}.tmp", rig.repo
    pre_c2_reset = (
        f"{{ [ -f {state} ] || {{ git -C {repo} diff > {tmp} && mv {tmp} {state}; }}; }} && "
        f"git -C {repo} checkout -- . && git -C {repo} clean -fdq && "
        f"{{ [ ! -s {state} ] || git -C {repo} apply {state}; }}"
    )
    _out, reset_rc = rig.exec_fn(pre_c2_reset)
    assert reset_rc == 0
    assert _read(repo, "tox_config.ini") == _BUILD_TOX  # re-applied BEFORE the step diff

    out, rc = rig.exec_fn(f"git -C {repo} apply -", stdin=step_diff)

    assert rc != 0  # the whole step diff rejected — atomically, including the agent's real edit
    assert "patch does not apply" in out
    assert _read(repo, "source_module.py") == _BASE["source_module.py"]  # nothing landed


# ---------------------------------------------------------------------------
# The legs that must stay exactly as they were.
# ---------------------------------------------------------------------------


def test_a_trajectory_whose_image_has_no_build_state_is_unaffected(rig: _Rig) -> None:
    # 8 247 of the corpus's steps sit on trajectories whose first capture is empty — the image
    # carries no build-time edit. For them the captured patch is 0 bytes, `[ ! -s ]` short-circuits,
    # and git is never invoked for the build state at all: a no-op on the majority path.
    step_diff = rig.capture_step_diff({"source_module.py": _AGENT_SOURCE})
    assert "tox_config.ini" not in step_diff

    result = rig.replay(step_diff)

    assert result.is_infra_failure is False
    assert rig.state.stat().st_size == 0  # captured, and legitimately empty
    assert _read(rig.repo, "source_module.py") == _AGENT_SOURCE
    assert _read(rig.repo, "tox_config.ini") == _BASE["tox_config.ini"]


@pytest.mark.parametrize("has_build_state", [False, True])
def test_each_step_starts_from_base_not_from_the_previous_steps_tree(
    rig: _Rig, has_build_state: bool
) -> None:
    # THE CORRUPTION THE `--3way` "FIX" WOULD HAVE INTRODUCED, and the reason C2 touches no index.
    # `--3way` implies `--index` and STAGES what it applies, so the next `git checkout -- .` —
    # which rebuilds from the index — starts the following step from the PREVIOUS step's tree.
    # Measured on astropy-13977: rc=0, no conflict marker, tree 4242665f where truth is ea8ccfbf.
    # Silent, and silent in BOTH legs — with no build state the guard skips and nothing warns —
    # so both are asserted here.
    if has_build_state:
        rig.build_edit({"tox_config.ini": _BUILD_TOX})
    first = rig.capture_step_diff({"alpha_helper.py": "ALPHA = 'step-one'\n"})
    second = rig.capture_step_diff({"bravo_widget.py": "BRAVO = 'step-two'\n"})

    assert rig.replay(first).is_infra_failure is False
    assert _read(rig.repo, "alpha_helper.py") == "ALPHA = 'step-one'\n"
    assert rig.replay(second).is_infra_failure is False

    assert _read(rig.repo, "bravo_widget.py") == "BRAVO = 'step-two'\n"
    assert _read(rig.repo, "alpha_helper.py") == _BASE["alpha_helper.py"]  # step one is GONE
    expected_tox = _BUILD_TOX if has_build_state else _BASE["tox_config.ini"]
    assert _read(rig.repo, "tox_config.ini") == expected_tox


def test_the_admissibility_base_leg_still_gets_the_build_state_back(rig: _Rig) -> None:
    # `replay_admissibility.check_instance` runs its base leg as `replay_step("", ...)`, so the
    # empty diff applies nothing and the reconcile is the ONLY thing that can put the build edit
    # back. Losing it here is what made 8 of 19 sphinx instances unmeasurable (no `-rA` ⇒ no
    # per-test status lines ⇒ the log parser extracts zero statuses).
    rig.build_edit({"tox_config.ini": _BUILD_TOX})

    result = rig.replay("")

    assert result.is_infra_failure is False
    assert _read(rig.repo, "tox_config.ini") == _BUILD_TOX
    assert _read(rig.repo, "source_module.py") == _BASE["source_module.py"]
    assert "pytest -rA" in rig.classified[0][0]  # visible in what the adjudicator was handed


def test_a_gold_shaped_diff_that_lacks_the_build_edit_gets_it_re_applied(rig: _Rig) -> None:
    # The gate's OTHER leg: the dataset's gold `patch` is authored against `base_commit` and does
    # NOT carry the image's build edit. So for it the `-R --check` presence test must FAIL and the
    # forward apply must run — the opposite branch from the step-diff case above, same command.
    gold_diff = rig.capture_step_diff({"source_module.py": _GOLD_SOURCE})
    assert "tox_config.ini" not in gold_diff
    rig.build_edit({"tox_config.ini": _BUILD_TOX})

    result = rig.replay(gold_diff)

    assert result.is_infra_failure is False
    assert _read(rig.repo, "source_module.py") == _GOLD_SOURCE
    assert _read(rig.repo, "tox_config.ini") == _BUILD_TOX


def test_a_real_test_patch_still_applies_on_top_of_the_reconciled_tree(rig: _Rig) -> None:
    # The reconcile is inserted between the step diff and the test_patch, so prove the test_patch
    # — the gold artefact that adds the FAIL_TO_PASS tests — still lands after it. It is captured
    # BEFORE the build edit exists, because the real one comes from the dataset row: authored
    # against `base_commit`, touching test files only, and never carrying the image's build edit.
    test_patch = rig.capture_step_diff({"bravo_widget.py": "BRAVO = 'tested'\n"})
    rig.build_edit({"tox_config.ini": _BUILD_TOX})
    step_diff = rig.capture_step_diff({"source_module.py": _AGENT_SOURCE})

    result = rig.replay(step_diff, test_patch=test_patch)

    assert result.is_infra_failure is False
    assert _read(rig.repo, "bravo_widget.py") == "BRAVO = 'tested'\n"
    assert _read(rig.repo, "source_module.py") == _AGENT_SOURCE
    assert _read(rig.repo, "tox_config.ini") == _BUILD_TOX


# ---------------------------------------------------------------------------
# Loud failure, idempotence, and the short-circuit.
# ---------------------------------------------------------------------------


def test_a_partial_build_state_overlap_fails_loudly_as_infra(rig: _Rig) -> None:
    # THE DEGRADATION CASE. The build state covers two files; the step diff carries only one of
    # them. `-R --check` fails (the other file's hunk is absent) and the forward apply fails (the
    # first file's hunk is already there), so the step cannot be reconstructed without guessing
    # which half is authoritative. It must go infra — never a silently wrong tree, and never a
    # pass or a fail. This is the behaviour that makes C2 safe rather than merely permissive.
    rig.build_edit({"tox_config.ini": _BUILD_TOX, "alpha_helper.py": _BUILD_ALPHA})
    _write(rig.repo, {"alpha_helper.py": _BASE["alpha_helper.py"]})  # hide half from the capture
    partial = rig.capture_step_diff({"source_module.py": _AGENT_SOURCE})
    rig.build_edit({"alpha_helper.py": _BUILD_ALPHA})  # ...but the IMAGE still has both
    assert "tox_config.ini" in partial
    assert "alpha_helper.py" not in partial

    result = rig.replay(partial)

    assert result.is_infra_failure is True
    assert result.outcome == "unknown"
    assert result.confidence == 0.0
    assert "build state did not reconcile" in result.detail
    assert "does not apply" in result.detail  # git's own words travel into the record
    assert rig.classified == []  # the adjudicator was never reached: unmeasured, not fabricated


def test_the_reconcile_is_idempotent_and_never_double_applies(rig: _Rig) -> None:
    # `git apply` has no fuzz, so a second forward apply of an already-applied hunk cannot match —
    # but that is a property of git, not of us, so pin it: repeated reconciles must leave exactly
    # one copy of the build edit and keep returning "reconciled".
    rig.build_edit({"tox_config.ini": _BUILD_TOX})
    assert offline_replay._reset_to_base(rig.exec_fn) is True
    assert _read(rig.repo, "tox_config.ini") == _BASE["tox_config.ini"]  # reset does NOT re-apply

    for _ in range(3):
        assert offline_replay._restore_build_state(rig.exec_fn) is None

    assert _read(rig.repo, "tox_config.ini") == _BUILD_TOX
    assert _read(rig.repo, "tox_config.ini").count("-rA") == 1


def test_an_absent_or_empty_build_state_never_invokes_git(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `[ ! -s ]` must short-circuit before git is reached, not merely tolerate a failure from it.
    # Proven by pointing TESTBED at a directory that is NOT a repo: any `git apply` there exits
    # non-zero, so a `None` return can only mean git was never invoked.
    not_a_repo = tmp_path / "elsewhere"
    not_a_repo.mkdir()
    monkeypatch.setattr(offline_replay, "TESTBED", str(not_a_repo))

    assert not rig.state.exists()
    assert offline_replay._restore_build_state(rig.exec_fn) is None  # absent

    rig.state.write_text("", encoding="utf-8")
    assert offline_replay._restore_build_state(rig.exec_fn) is None  # 0 bytes

    # ...and the same call against a NON-empty state in that non-repo does fail, which is what
    # makes the two assertions above evidence of a short-circuit rather than of a lenient git.
    rig.state.write_text("diff --git a/x b/x\n", encoding="utf-8")
    failed = offline_replay._restore_build_state(rig.exec_fn)
    assert failed is not None
    assert failed.is_infra_failure is True
