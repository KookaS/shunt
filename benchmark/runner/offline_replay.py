"""Offline snapshot replay (Mechanism B): derive real per-step outcomes from captured diffs."""

# For each captured step diff we rebuild the instance's base state in a container, apply the diff,
# apply the gold ``test_patch`` (which adds the FAIL_TO_PASS tests), run the tests it touched, and
# classify the output through the SAME `parse_test_outcome` the live tier2 verifier uses — so the
# per-step dedup key matches production. A container/apply/collection error is marked
# ``is_infra_failure``, never fabricated into a pass or a fail (real-only).
#
# WHAT IS RUN, AND THE CONSEQUENCE. The selector is `get_test_directives(instance)` — SWE-bench's
# own derivation, which reads FILE PATHS out of the test_patch and applies the django
# module-dotted-path transform. It is NOT the raw FAIL_TO_PASS id list. Passing raw F2P ids as
# positional args only ever worked for `pytest -rA` repos: sympy's `bin/test` given a bare
# `test_coth` selected zero tests and exited 0 (a fabricated green), and django's unittest loader
# given `test_x (auth_tests.test_basic.TestGetUser.test_x)` split on `.`, tried to import a module
# named `test_x (auth_tests`, and died — a fabricated red with a per-instance constant dedup key.
#
# WHAT THE VERDICT IS READ FROM. Running whole test FILES is only half the parity: the grader runs
# the same directives but adjudicates on the per-test statuses it parses out of the log, counting
# only FAIL_TO_PASS ∪ PASS_TO_PASS. A test in the file that is in neither list cannot affect the
# grade, yet it does set the run's exit code — so an exit-code verdict is a strictly BROADER
# quantity than the terminal grade. Measured: matplotlib-20676's gold-patch run is
# `1 failed, 34 passed`, and the one failure (`test_widgets.py::test_rectangle_selector`) is in
# neither list, so the grader resolves the instance while the exit code reads red.
#
# So the exit code is NOT the verdict here. `replay_step` takes a `classify` callable and the real
# pipeline passes `swebench_grading.GraderParity`, which delegates to swebench's own log parser and
# grading functions; the per-step outcome is now the same quantity as `header.terminal_resolved`.
# `classify` is a required argument on purpose — a default would let a caller silently fall back to
# the exit-code adjudicator the gate below is calibrated against.

from __future__ import annotations

import logging
import shlex
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from benchmark import corpus_lock
from benchmark.escalation import authenticity, schema
from benchmark.escalation.normalize.mini_swe_agent import restamp_trajectory, unstamp_trajectory
from benchmark.runner import (
    replay_admissibility,
    step_snapshots,
    swebench_grading,
    swebench_specs,
)
from benchmark.runner.step_snapshots import TESTBED
from shunt.verifiers.base import VerifierResult

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOG = logging.getLogger(__name__)

# How a run's (combined output, exit code) becomes a verdict. The real pipeline passes
# `swebench_grading.GraderParity`; unit tests inject a fake to drive `replay_step` without Docker.
Classifier = Callable[[str, int], VerifierResult]


class SnapshotsMissingError(RuntimeError):
    """The per-step scratch this trajectory was captured with is absent from THIS checkout."""


class ContainerExec(Protocol):
    """Run a shell command inside the replay container, returning (combined output, returncode)."""

    def __call__(self, command: str, stdin: str | None = None) -> tuple[str, int]: ...


def _infra(detail: str) -> VerifierResult:
    """A step whose state could not be reconstructed — real-only: never a pass or a fail."""
    return VerifierResult(outcome="unknown", confidence=0.0, detail=detail, is_infra_failure=True)


# Where the container keeps the working-tree modifications it was BUILT with, so the per-step
# replay can put them back. Outside /testbed, so `git clean -fdq` cannot remove it.
_BUILD_STATE: Final = "/tmp/shunt_build_state.patch"
# The capture writes HERE first and is moved into place only on success. A redirect creates its
# target before the command on the left of it runs, so `git diff > _BUILD_STATE` leaves a 0-byte
# file behind when git fails — and a 0-byte file is indistinguishable from the legitimate
# "this image carries no build-time edit" result, so `[ ! -s … ]` then skips the re-apply for ever
# while `[ -f … ]` suppresses every retry. `git diff` failing is realistic in these images: a
# `safe.directory` dubious-ownership refusal and a corrupt index both exit non-zero.
_BUILD_STATE_TMP: Final = f"{_BUILD_STATE}.tmp"

# How much of a failed reconcile's combined output is carried into the VerifierResult detail.
# Enough for git's "patch does not apply" line and the path it names; short enough that a rejected
# many-hunk patch cannot bloat the record written into the trajectory jsonl.
_DETAIL_TAIL: Final = 400


def _reset_to_base(exec_fn: ContainerExec) -> bool:
    """Capture the image's build-time edit once, then discard everything else back to the index."""
    # WHAT THE BUILD STATE IS. `git checkout -- .` resets to the index, and for some instances that
    # is NOT the state the image was built in. SWE-bench applies each version's `pre_install` at
    # build time, and for sphinx that is `sed -i 's/pytest/pytest -rA/' tox.ini` — left as an
    # UNCOMMITTED working-tree edit in some images and committed in others. Where it is
    # uncommitted, the reset silently removed `-rA`, so pytest printed no per-test PASSED/FAILED
    # lines at all and swebench's log parser extracted ZERO statuses. Measured: 8 of 19 sphinx
    # instances came back unmeasurable that way, every one of them an instance the real grading
    # harness resolves (5/5, 4/4, 3/3 cells...) — i.e. the gate would have cleared real
    # trajectories over an artefact of its own reset.
    #
    # THIS CAPTURE IS A BARE `git diff` — WORKTREE-VS-INDEX — ON PURPOSE. DO NOT "FIX" IT to
    # `git diff HEAD`. It is one half of a deliberate PAIR with the `git checkout -- .` below,
    # which rebuilds the worktree FROM THE INDEX. The capture must describe exactly what that
    # restore drops — the unstaged delta — and nothing more. `git diff HEAD` would also carry the
    # staged delta, which `checkout -- .` KEEPS, so re-applying it would double-apply that half.
    #
    # It is NOT the same command as the agent-side step capture, and the difference is deliberate
    # on both sides. `step_snapshots.DIFF_COMMAND` is `git diff HEAD`, because an agent that runs
    # `git add` would otherwise drop its own work out of the capture. Here the pairing constraint
    # is the opposite one, and the two never disagree in practice: this container is fresh from
    # the image and nothing in the replay path ever stages, so its index IS HEAD.
    #
    # ONLY THE FIRST CALL CAPTURES. This runs first inside `replay_step`, and the first
    # `replay_step` of an instance is the admissibility gate's base leg, whose tree is still the
    # pristine image. `[ -f ]` pins that first reading for the life of the container; a later
    # capture would see a previous step's edits and enshrine them as "build state".
    #
    # THE RE-APPLY IS DELIBERATELY NOT HERE — it runs after the step diff, in
    # `_restore_build_state`, which explains why.
    #
    # Every clause is joined with `&&`, never `;`: a `;` discards the exit status of the clause
    # before it, so a failed capture returned "reset ok" and the caller replayed the step against a
    # state it never verified. `A || { B; } && C` groups as `(A || B) && C` — the capture is skipped
    # when the file already exists, and its failure aborts the whole reset instead of vanishing.
    command = (
        f"{{ [ -f {_BUILD_STATE} ] || {{ git -C {TESTBED} diff > {_BUILD_STATE_TMP} && "
        f"mv {_BUILD_STATE_TMP} {_BUILD_STATE}; }}; }} && "
        f"git -C {TESTBED} checkout -- . && git -C {TESTBED} clean -fdq"
    )
    _out, rc = exec_fn(command)
    return rc == 0


def _restore_build_state(exec_fn: ContainerExec) -> VerifierResult | None:
    """Make the image's build-time edit present exactly once — None when reconciled, else infra."""
    # ORDERING IS THE FIX, NOT A MERGE STRATEGY. This re-apply used to live at the end of
    # `_reset_to_base`, i.e. BEFORE the step diff was applied. The build edit is uncommitted in the
    # AGENT's container too, so the agent-side `git diff HEAD` capture sees it and the step diff
    # ALREADY CONTAINS it. Re-applying first meant `git apply -` met an already-applied hunk, and
    # `git apply` is atomic: the WHOLE step diff was rejected and a real measurement became an
    # infra failure. Measured over the corpus: 2 963 of 3 109 steps (95.3%) went infra on
    # trajectories whose first capture is non-empty, against 317 of 8 247 (3.8%) where it is empty.
    # Within-instance control on astropy-13977 — same image, same selectors, same F2P/P2P — 0/48
    # infra across four trajectories with no build state, 108/129 across three with one.
    #
    # WHY A PRESENCE TEST. `git apply -R --check` succeeds exactly when the patch is already
    # applied, so "did the step diff carry the build edit?" is asked exactly rather than guessed.
    #
    # WHY NOT `--3way`, AND WHY NOT ANY `--index` VARIANT. `--3way` implies `--index`, and an
    # `--index` apply refuses whenever a file's worktree content differs from its index entry —
    # which is precisely the state `_reset_to_base` leaves, since the build state is a worktree
    # edit the index never saw (`error: pyproject.toml: does not match index`, reproduced against
    # the real astropy-13977 image). Worse, `--3way` STAGES what it applies, so the next
    # `git checkout -- .` — which rebuilds from the index — would start the following step from the
    # PREVIOUS step's tree. That corruption is silent: it emits rc=0 with no conflict marker, and
    # when the build state is empty the guard skips so nothing warns at all. Proven end-to-end on
    # astropy-13977: a step applied rc=0 and produced tree 4242665f where the truth is ea8ccfbf.
    # Plain `git apply` gets that step right; the "obvious fix" is what breaks it.
    #
    # `A || B || C` groups left-to-right: an absent-or-empty build state short-circuits at A (the
    # image carried no build-time edit — most instances); a build state the step diff already
    # carried stops at B; anything else is applied forward by C. Only C can leave a non-zero rc,
    # and that means a genuine PARTIAL overlap — the step diff carries some of the build edit but
    # not all of it — which cannot be reconciled without guessing which half is authoritative. It
    # is reported as infra instead, loudly, because a wrong tree measured silently is the failure
    # this whole function exists to prevent.
    command = (
        f"[ ! -s {_BUILD_STATE} ] || git -C {TESTBED} apply -R --check {_BUILD_STATE} || "
        f"git -C {TESTBED} apply {_BUILD_STATE}"
    )
    out, rc = exec_fn(command)
    if rc == 0:
        return None
    return _infra(f"build state did not reconcile (rc={rc}): {out.strip()[-_DETAIL_TAIL:]}")


def replay_step(
    diff: str,
    test_patch: str,
    test_cmd: str,
    test_selectors: list[str],
    exec_fn: ContainerExec,
    classify: Classifier,
) -> VerifierResult:
    """Rebuild one step's state in the container, run the patched tests, classify the outcome."""
    if not _reset_to_base(exec_fn):
        return _infra("container reset to base failed")
    if diff.strip():
        _out, rc = exec_fn(f"git -C {TESTBED} apply -", stdin=diff)
        if rc != 0:
            return _infra(f"step diff did not apply (rc={rc})")
    # AFTER the step diff, never before — the step diff already carries the build edit. This also
    # covers the two admissibility legs, which reach here through this same function: the base leg
    # passes an empty diff and the gold leg passes the dataset's gold patch, and NEITHER carries
    # the build edit, so for both of them the reconcile below simply re-applies it.
    unreconciled = _restore_build_state(exec_fn)
    if unreconciled is not None:
        return unreconciled
    if test_patch.strip():
        _out, rc = exec_fn(f"git -C {TESTBED} apply -", stdin=test_patch)
        if rc != 0:
            return _infra(f"test_patch did not apply (rc={rc})")
    ids = " ".join(shlex.quote(t) for t in test_selectors)
    combined, rc = exec_fn(f"cd {TESTBED} && {test_cmd} {ids}")
    return classify(combined, rc)


def replay_snapshots(
    *,
    snapshots: dict[int, str],
    test_patch: str,
    test_cmd: str,
    test_selectors: list[str],
    exec_fn: ContainerExec,
    classify: Classifier,
) -> dict[int, VerifierResult]:
    """Replay every captured per-step diff, returning the per-step VerifierResult keyed by step."""
    return {
        index: replay_step(
            snapshots[index], test_patch, test_cmd, test_selectors, exec_fn, classify
        )
        for index in sorted(snapshots)
    }


# ---------------------------------------------------------------------------
# Real-container plumbing — needs Docker + swebench + the datasets row. Exercised only in the
# owner's offline replay step (flagged); unit tests drive `replay_snapshots` with a fake exec_fn.
# ---------------------------------------------------------------------------


def swebench_test_command(repo: str, version: str) -> str:
    """The per-repo/version test command from SWE-bench's own spec map (django ≠ pytest)."""
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # noqa: PLC0415

    return str(MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"])


def swebench_test_directives(instance_id: str) -> list[str]:
    """The test targets SWE-bench's grader runs: file paths derived from the gold ``test_patch``."""
    # DELEGATED, never re-derived. `get_test_directives` owns the django transform (drop `.py`,
    # strip the `tests/` prefix, slashes → dots) and the non-test-extension filter. Re-implementing
    # either here would let this replay's selector drift from the grader's, which is the whole
    # class of bug the raw-FAIL_TO_PASS selector was.
    from swebench.harness.test_spec.python import get_test_directives  # noqa: PLC0415

    return [str(d) for d in get_test_directives(_dataset_row(instance_id))]


# The instance images declare no locale, and the conda env a login shell activates is often
# Python 3.6 — too old for PEP-538 locale coercion — so `sys.stdout.encoding` resolves to ASCII.
# django's `migrate` then dies with `UnicodeEncodeError: 'ascii' codec can't encode '…'`
# BEFORE a single test runs: measured on django-10880, the gold leg produced zero parseable test
# statuses, which the gate reads (correctly, given that input) as "this instance cannot be
# measured". The SWE-bench grading harness resolves that same instance in that same image — all 9
# of its collected cells are graded RESOLVED — so the ASCII stdout is a defect in THIS exec path,
# not a property of the instance, and without this the gate would clear 9 real trajectories.
# `PYTHONIOENCODING` is the narrow fix: it changes stdio encoding only, where `LC_ALL` would also
# move collation and formatting and so could change what the tests themselves do.
_EXEC_ENV: Final = ("PYTHONIOENCODING=utf-8",)


# A step diff is agent-written code, so a step CAN introduce an infinite loop — and an unbounded
# `docker exec` then stalls the whole pass with no marker, no signal exit and no demotion, which is
# unacceptable for a multi-hour unsupervised rebuild. SWE-bench's own harness caps every test run
# the same way. 30 min is well above the slowest observed instance (sympy/sphinx full-file runs are
# minutes) and well below "nobody notices until morning".
_EXEC_TIMEOUT_S: Final = 1800.0

# Third-party loggers the replay CLI quiets so the gate verdicts stay readable (see `_main`).
_QUIET_LOGGERS: Final = ("httpx", "urllib3", "filelock", "datasets", "fsspec")

# GNU `timeout`'s own exit code for "the command was killed for exceeding the limit". It is NOT in
# `parse_test_outcome`'s non-session set, so the exit code alone would be graded a capability RED —
# the demotion comes from the swebench marker below, which `GraderParity` refuses to grade on.
_TIMEOUT_EXIT: Final = 124


def _decode(stream: bytes | str | None) -> str:
    """Whatever partial output a TimeoutExpired carries, as text (POSIX hands back bytes/None)."""
    # `subprocess.run(text=True)` does NOT decode the partial output it attaches to TimeoutExpired
    # on POSIX: `stdout` comes back as `bytes` and `stderr` as `None`. Formatting those straight
    # into the combined log would write `b'...'` and `None` into the record.
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode("utf-8", "replace")


def docker_exec(container: str, timeout_s: float = _EXEC_TIMEOUT_S) -> ContainerExec:
    """A ContainerExec backed by ``docker exec`` on a container (stdin piped for patches)."""

    def _exec(command: str, stdin: str | None = None) -> tuple[str, int]:
        env = [arg for pair in _EXEC_ENV for arg in ("-e", pair)]
        argv = ["docker", "exec", "-i", *env, container, "bash", "-lc", command]
        try:
            proc = subprocess.run(
                argv, input=stdin, capture_output=True, text=True, check=False, timeout=timeout_s
            )
        except subprocess.TimeoutExpired as expired:
            # Killing the `docker exec` CLIENT does not stop the runaway process inside the
            # container — it would keep burning a core for the rest of the pass — so the container
            # itself is reaped. Every later exec against it then fails, so the remaining steps come
            # back `is_infra_failure` (unmeasured) rather than fabricated, which is the real-only
            # answer: the instance's replay is over, and it says so.
            subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
            _LOG.error(
                "docker exec in %s exceeded %.0fs and was killed; container reaped",
                container,
                timeout_s,
            )
            partial = f"{_decode(expired.stdout)}\n{_decode(expired.stderr)}"
            expiry = f"shunt: docker exec exceeded {timeout_s:.0f}s"
            return f"{partial}\n{swebench_grading.timeout_marker()}\n{expiry}", _TIMEOUT_EXIT
        return f"{proc.stdout}\n{proc.stderr}", proc.returncode

    return _exec


@contextmanager
def instance_container(image_ref: str, name: str) -> Iterator[str]:
    """Start a detached container from the instance image; always reap it (context manager)."""
    subprocess.run(
        ["docker", "run", "-d", "--name", name, image_ref, "sleep", "infinity"],
        check=True,
        capture_output=True,
    )
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)


_ROW_FIELDS: Final = ("instance_id", "repo", "base_commit", "patch", "test_patch")


def _dataset_row(instance_id: str) -> dict[str, str]:
    """The HF Verified row for an instance: gold ``patch`` + ``test_patch`` + repo/base_commit."""
    # One accessor for every gold field the replay needs. `swebench_test_directives` reads
    # repo+test_patch from it, the step replay reads test_patch, and the admissibility gate's
    # GOLD leg reads `patch` — so all three describe the SAME dataset revision by construction.
    # The fetch is PINNED to `DATASET_REVISION`, the same pin `build_challenges` uses: the gate's
    # cache key claims `spec.dataset_revision`, so the rows it is measured against must actually
    # come from that revision, or a drift in HF's `test` split would serve a stale cached verdict
    # for a row it was never measured on (the key is the contract; the fetch is the subject).
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(
        swebench_specs.DATASET_NAME,
        split=swebench_specs.DATASET_SPLIT,
        revision=swebench_specs.DATASET_REVISION,
    )
    for row in ds:
        if str(row["instance_id"]) == instance_id:
            return {field: str(row[field]) for field in _ROW_FIELDS}
    raise KeyError(f"instance {instance_id!r} not in {swebench_specs.DATASET_NAME}")


@dataclass(frozen=True)
class ReplayPlan:
    """Everything a replay of one instance needs, plus the cache key its verdict is valid under."""

    instance_id: str
    image_ref: str
    test_cmd: str
    test_patch: str
    gold_patch: str
    selectors: list[str]
    classify: swebench_grading.GraderParity
    gate_key: str


def _replay_plan(instance_id: str) -> ReplayPlan:
    """Resolve spec + dataset row + selectors + adjudicator, and derive the gate's cache key."""
    spec = swebench_specs.load_spec(instance_id)
    if spec is None:
        raise KeyError(f"no SWE-bench spec for {instance_id!r}; materialise it first")
    test_cmd = swebench_test_command(spec.repo, spec.version)
    row = _dataset_row(instance_id)
    selectors = swebench_test_directives(instance_id)
    fail_to_pass, pass_to_pass = tuple(spec.fail_to_pass), tuple(spec.pass_to_pass)
    return ReplayPlan(
        instance_id=instance_id,
        image_ref=spec.image_ref,
        test_cmd=test_cmd,
        test_patch=row["test_patch"],
        gold_patch=row["patch"],
        selectors=selectors,
        classify=swebench_grading.GraderParity(
            repo=spec.repo, fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass
        ),
        gate_key=replay_admissibility.gate_key(
            dataset_revision=spec.dataset_revision,
            image_ref=spec.image_ref,
            test_cmd=test_cmd,
            test_selectors=selectors,
            # F2P/P2P are now the SUBJECT of the adjudication, not just dataset provenance: a spec
            # re-materialised with different lists must re-measure, not serve the cached verdict.
            fail_to_pass=list(fail_to_pass),
            pass_to_pass=list(pass_to_pass),
        ),
    )


def clear_rejected(
    verdict: replay_admissibility.AdmissibilityVerdict, jsonl_path: Path
) -> Path | None:
    """An inadmissible instance ACTIVELY loses its stamps — skipping the write is not enough."""
    # The gate used to return before writing anything, which left the committed trajectory exactly
    # as the defective replay had stamped it: astropy-8872 came out of a rebuild byte-identical,
    # 100% blocking on exit-4, 284 fabricated steps intact. Rewriting the file with every stamp
    # cleared is what actually excludes it. The manifest is rewritten too — clearing changes
    # `content_sha256`, so an un-rewritten manifest would leave the corpus failing its own
    # Layer-1 integrity check.
    replay_admissibility.record_verdict(verdict, jsonl_path.parent)
    with corpus_lock.corpus_lock(jsonl_path.parent):
        cleared = unstamp_trajectory(schema.load_jsonl(jsonl_path))
        commit_trajectory(cleared, jsonl_path)
    _LOG.warning(
        "%s is inadmissible; cleared every per-step stamp in %s (%s)",
        verdict.instance_id,
        jsonl_path,
        verdict.reason,
    )
    return None


def clear_unreplayable(trajectory_id: str, jsonl_path: Path) -> None:
    """A trajectory whose live run captured NO per-step diff can never be verified — unstamp it."""
    # THE DISTINCTION THAT MAKES THIS SAFE. "No snapshot directory" has two causes that look
    # identical on disk, and only one of them justifies clearing:
    #   * the live run captured nothing (the recorder never attached, or the run had no step) —
    #     this trajectory is unreplayable FOR EVER, in every checkout;
    #   * this checkout simply lacks the gitignored scratch (fresh clone, pruned artifacts).
    # Clearing on filesystem absence alone would wipe the entire corpus on any fresh clone, so the
    # source of truth is the COMMITTED header instead: `snapshot_steps` is recorded at capture time
    # and travels with the data. Only `snapshot_steps == 0` clears; anything else raises, because
    # an un-replayable trajectory that is silently reported as done is how the stamping stage
    # ledgers a rebuild as complete while 7 trajectories keep unverified stamps.
    with corpus_lock.corpus_lock(jsonl_path.parent):
        traj = schema.load_jsonl(jsonl_path)
        commit_trajectory(unstamp_trajectory(traj), jsonl_path)
    _LOG.warning(
        "%s captured 0 per-step snapshots and can never be replayed; cleared its stamps in %s",
        trajectory_id,
        jsonl_path,
    )


def _handle_absent_snapshots(trajectory_id: str, jsonl_path: Path) -> None:
    """Clear a never-captured trajectory; refuse to proceed when the scratch is merely missing."""
    captured = schema.load_jsonl(jsonl_path).header.snapshot_steps
    if captured == 0:
        clear_unreplayable(trajectory_id, jsonl_path)
        return
    raise SnapshotsMissingError(
        f"{trajectory_id} recorded snapshot_steps={captured} at capture time but no per-step diffs "
        f"are present under {step_snapshots.snapshot_dir(trajectory_id)}. This checkout cannot "
        "replay it; re-collect or restore the scratch. Refusing to clear real stamps."
    )


def run_offline_replay(trajectory_id: str, instance_id: str, jsonl_path: Path) -> Path | None:
    """Full offline pass: snapshots → container replay → restamp the committed trajectory jsonl."""
    snapshots = step_snapshots.read_snapshots(trajectory_id)
    if not snapshots:
        _handle_absent_snapshots(trajectory_id, jsonl_path)
        return None
    # A PARTIAL scratch is worse than an absent one. `restamp_trajectory` leaves any step it has no
    # outcome for untouched, and the header hash + manifest are rewritten afterwards — so an
    # interrupted write or a pruned directory yields a file that passes its own Layer-1 check while
    # most of its steps still carry stamps from the adjudicator this change replaced. The committed
    # count is the only thing that can notice, so compare against it and refuse.
    captured = schema.load_jsonl(jsonl_path).header.snapshot_steps
    if captured is None:
        # No count on record (the corpus predates the field, or was never backfilled) means a
        # partial scratch cannot be told from a complete one — replaying what is present and
        # leaving the rest stamped mixes two adjudicators, which is exactly what the guard above
        # exists to prevent. Refusing is the real-only answer: do not restamp against unknown
        # provenance. `record_snapshot_provenance` (collection-host only) is the backfill.
        raise SnapshotsMissingError(
            f"{trajectory_id} records no snapshot_steps, so a partial scratch cannot be told "
            f"from a complete one. Refusing to replay {len(snapshots)} step(s) against unknown "
            "provenance; backfill snapshot_steps (record_snapshot_provenance) on the collection "
            "host before rebuilding."
        )
    if len(snapshots) != captured:
        raise SnapshotsMissingError(
            f"{trajectory_id} captured {captured} per-step diffs but only {len(snapshots)} are "
            f"present under {step_snapshots.snapshot_dir(trajectory_id)}; a partial replay would "
            "silently keep the rest of its old stamps. Restore the scratch or re-collect."
        )
    plan = _replay_plan(instance_id)
    out_dir = jsonl_path.parent
    # The instrument-validity gate is a property of the INSTANCE, not of one trajectory, so its
    # verdict is cached under `gate_key` (subject + instrument-source digest). A cached rejection
    # needs no container at all: the clearing below is a pure file rewrite.
    verdict = replay_admissibility.cached_verdict(instance_id, plan.gate_key, out_dir)
    if verdict is not None and not verdict.admissible:
        return clear_rejected(verdict, jsonl_path)
    with instance_container(plan.image_ref, f"shunt-replay-{trajectory_id}") as name:
        exec_fn = docker_exec(name)
        if verdict is None:
            verdict = replay_admissibility.check_instance(
                instance_id=instance_id,
                test_patch=plan.test_patch,
                gold_patch=plan.gold_patch,
                test_cmd=plan.test_cmd,
                test_selectors=plan.selectors,
                exec_fn=exec_fn,
                classify=plan.classify,
                fail_to_pass=plan.classify.fail_to_pass,
                gate_key_value=plan.gate_key,
            )
            replay_admissibility.record_verdict(verdict, out_dir)
        if not verdict.admissible:
            return clear_rejected(verdict, jsonl_path)
        outcomes = replay_snapshots(
            snapshots=snapshots,
            test_patch=plan.test_patch,
            test_cmd=plan.test_cmd,
            test_selectors=plan.selectors,
            exec_fn=exec_fn,
            classify=plan.classify,
        )
    with corpus_lock.corpus_lock(out_dir):
        traj = restamp_trajectory(schema.load_jsonl(jsonl_path), outcomes)
        commit_trajectory(traj, jsonl_path)
    _LOG.info("restamped %s with %d per-step outcomes", jsonl_path, len(outcomes))
    return jsonl_path


def commit_trajectory(traj: schema.Trajectory, jsonl_path: Path) -> None:
    """Write the trajectory and re-bind the manifest to it as ONE indivisible transaction."""
    # `authenticity.manifest` hashes every *.jsonl in the directory, so under N parallel workers
    # these two writes cannot be separable: worker A reads the corpus, worker B then writes its
    # trajectory AND its manifest, and A's later manifest write re-binds B's file to the hash it
    # had before. Reproduced with two real processes — `verify_manifest` reported
    # `manifest.hash_mismatch`, i.e. the corpus failing its own Layer-1 check. Callers already
    # holding the lock re-enter it; the pairing is enforced here so no caller can forget it.
    import json  # noqa: PLC0415

    with corpus_lock.corpus_lock(jsonl_path.parent):
        schema.dump_jsonl(traj, jsonl_path)
        payload = json.dumps(authenticity.manifest(jsonl_path.parent), indent=2, sort_keys=True)
        corpus_lock.atomic_write_text(
            jsonl_path.parent / authenticity.MANIFEST_NAME, payload + "\n"
        )


def _main() -> int:
    import argparse  # noqa: PLC0415

    from benchmark.escalation.live_capture import LIVE_DIR  # noqa: PLC0415

    # The stamping stage runs this module as a SUBPROCESS, and a child with no handler falls back
    # to `logging.lastResort`, which starts at WARNING. Every ADMISSIBLE gate verdict is logged at
    # INFO, so without this the one thing a multi-hour rebuild is being supervised FOR — did the
    # positive control recover on this instance — is dropped from the log while rejections (WARNING)
    # still appear. Same format as `benchmark.pipeline.main`, so parent and child lines interleave
    # readably in the single tee'd run log.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # ...but the dataset lookup's HTTP chatter is not signal, and it BURIES the signal: measured
    # over 8 replays, 51 `httpx` INFO lines to 6 real ones. Across a full 799-trajectory rebuild
    # that is ~5 000 lines of HTTP between the gate verdicts a supervisor is reading the log for.
    for noisy in _QUIET_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description="Offline snapshot replay → per-step outcomes.")
    ap.add_argument("trajectory_id", help="captured trajectory id (also the snapshot dir name)")
    ap.add_argument("--instance-id", required=True, help="SWE-bench Verified instance id")
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="trajectory jsonl to restamp (default: the live plane's <trajectory_id>.jsonl)",
    )
    args = ap.parse_args()
    jsonl = args.jsonl or (LIVE_DIR / f"{args.trajectory_id}.jsonl")
    result = run_offline_replay(args.trajectory_id, args.instance_id, jsonl)
    # None has two causes — the instance was inadmissible, or the trajectory captured no
    # snapshots — and they are not interchangeable; see the WARNING each logs. A checkout that
    # merely lacks the scratch does not return None at all: it raises SnapshotsMissingError.
    print(f"restamped {result}" if result else f"not stamped: {jsonl} (see the log line above)")
    return 0


__all__ = [
    "Classifier",
    "ContainerExec",
    "ReplayPlan",
    "SnapshotsMissingError",
    "clear_rejected",
    "clear_unreplayable",
    "commit_trajectory",
    "docker_exec",
    "instance_container",
    "replay_snapshots",
    "replay_step",
    "run_offline_replay",
    "swebench_test_command",
    "swebench_test_directives",
]


if __name__ == "__main__":
    raise SystemExit(_main())
