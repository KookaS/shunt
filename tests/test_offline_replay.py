"""Offline replay core, driven by a FAKE in-container exec (legitimate unit mocking — no Docker):
the isolation ritual, real-only infra handling, and a verdict that comes from the INJECTED
`classify` callable rather than the exit code."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark.escalation import schema
from benchmark.runner import offline_replay, replay_admissibility
from benchmark.runner.offline_replay import replay_snapshots, replay_step
from shunt.verifiers.parse import parse_test_outcome
from tests.escalation.factories import make_step, make_trajectory


class _FakeContainer:
    """Records every command and returns canned (output, rc) by command substring."""

    def __init__(self, test_output: str, test_rc: int, *, apply_rc: int = 0, reset_rc: int = 0):
        self.commands: list[str] = []
        self._test_output = test_output
        self._test_rc = test_rc
        self._apply_rc = apply_rc
        self._reset_rc = reset_rc

    def __call__(self, command: str, stdin: str | None = None) -> tuple[str, int]:
        self.commands.append(command)
        if "checkout --" in command:
            return "", self._reset_rc
        if "git -C" in command and "apply" in command:
            return "", self._apply_rc
        return self._test_output, self._test_rc


def _replay(container: _FakeContainer, diff: str = "diff --git a/x b/x\n") -> object:
    # These cases are about `replay_step`'s PLUMBING (reset → apply → run → classify) and the
    # real-only infra handling, so they inject the shared exit-code parser directly. The
    # grader-parity adjudicator the real pipeline passes has its own suite
    # (tests/test_swebench_grading.py); what matters here is that `classify` is what decides.
    return replay_step(
        diff,
        "test_patch_body",
        "python -m pytest",
        ["tests/t.py::test_a"],
        container,
        parse_test_outcome,
    )


def test_failure_outcome_carries_the_node_id_via_shared_parser() -> None:
    output = "tests/t.py::test_a FAILED\nAssertionError"
    container = _FakeContainer(output, 1)
    vr = _replay(container)
    assert vr.outcome == "failure"
    # Byte-identical to what the shared parser derives directly (dedup-key parity).
    assert vr.failing_check_id == parse_test_outcome(output, 1).failing_check_id
    assert vr.failing_check_id == "tests/t.py::test_a"


def test_success_outcome() -> None:
    vr = _replay(_FakeContainer("2 passed", 0))
    assert vr.outcome == "success"
    assert vr.failing_check_id is None


def test_ritual_resets_applies_diff_and_test_patch_then_runs() -> None:
    container = _FakeContainer("1 passed", 0)
    _replay(container)
    # reset-to-base precedes the two patch applies, which precede the test run. The applies are
    # counted by their exact stdin form, since the reset command now mentions `apply` too.
    assert "checkout --" in container.commands[0]
    assert container.commands.count("git -C /testbed apply -") == 2  # step diff + test_patch
    assert container.commands[-1].startswith("cd ")


def test_diff_that_does_not_apply_is_infra_never_a_pass_or_fail() -> None:
    container = _FakeContainer("irrelevant", 0, apply_rc=1)
    vr = _replay(container)
    assert vr.is_infra_failure is True
    assert vr.outcome == "unknown"


def test_container_reset_failure_is_infra() -> None:
    container = _FakeContainer("irrelevant", 0, reset_rc=1)
    vr = _replay(container)
    assert vr.is_infra_failure is True


def test_collection_error_output_classifies_as_infra() -> None:
    container = _FakeContainer("ERROR collecting tests/t.py\nImportError", 2)
    vr = _replay(container)
    assert vr.is_infra_failure is True


def test_replay_snapshots_maps_every_step_in_order() -> None:
    container = _FakeContainer("tests/t.py::test_a FAILED", 1)
    outcomes = replay_snapshots(
        snapshots={0: "d0", 2: "d2"},
        test_patch="tp",
        test_cmd="python -m pytest",
        test_selectors=["tests/t.py::test_a"],
        exec_fn=container,
        classify=parse_test_outcome,
    )
    assert sorted(outcomes) == [0, 2]
    assert all(o.outcome == "failure" for o in outcomes.values())


def test_empty_diff_skips_the_step_apply_but_still_runs() -> None:
    # A no-change step (base state) applies only the test_patch, then runs.
    container = _FakeContainer("1 passed", 0)
    replay_step("", "tp", "python -m pytest", ["tests/t.py::test_a"], container, parse_test_outcome)
    applies = [c for c in container.commands if c == "git -C /testbed apply -"]
    assert len(applies) == 1  # only the test_patch, no step diff


# ---------------------------------------------------------------------------
# A1: a replayed step that selected NOTHING is not a pass. sympy's `bin/test -C --verbose`
# given a bare unittest name matches no test, prints `0 passed`, and exits 0 — which stamped
# 4064 committed steps PASS on runs where nothing executed (79.2% of sympy trajectories came
# out all-green, 83% of them on tasks the SWE-bench grader marked UNRESOLVED).
# ---------------------------------------------------------------------------


def test_zero_tests_selected_is_never_a_pass() -> None:
    container = _FakeContainer("tests finished: 0 passed, in 0.00 seconds", 0)
    vr = _replay(container)
    assert vr.outcome == "unknown"
    assert vr.is_infra_failure is True
    assert vr.failing_check_id is None  # absence of a measurement, not a red either


# ---------------------------------------------------------------------------
# A2: pytest USAGE_ERROR (exit 4) means the session never validly ran — infra, not a red.
# 746 committed steps sat at exit 4; 394 were stamped blocking capability failures carrying a
# recurring `hash:` dedup key, which fires the recurrence trigger on data the agent never caused.
# ---------------------------------------------------------------------------


def test_pytest_usage_error_is_infra_not_a_capability_red() -> None:
    container = _FakeContainer("ERROR: not found: /testbed/a.py::test_x", 4)
    vr = _replay(container)
    assert vr.is_infra_failure is True
    assert vr.failing_check_id is None
    assert vr.outcome == "unknown"


# ---------------------------------------------------------------------------
# R1: the selector is SWE-bench's own `get_test_directives` (file paths out of the test_patch),
# never the raw FAIL_TO_PASS ids. Driven through a faked dataset row (legitimate unit mocking:
# no HF download) so the REAL swebench transform is what is under test.
# ---------------------------------------------------------------------------

_SYMPY_TEST_PATCH = """diff --git a/sympy/functions/elementary/tests/test_hyperbolic.py \
b/sympy/functions/elementary/tests/test_hyperbolic.py
--- a/sympy/functions/elementary/tests/test_hyperbolic.py
+++ b/sympy/functions/elementary/tests/test_hyperbolic.py
@@ -272,6 +272,7 @@ def test_coth():
+    assert coth(x).rewrite(tanh) == 1/tanh(x)
"""

_DJANGO_TEST_PATCH = """diff --git a/tests/auth_tests/test_basic.py b/tests/auth_tests/test_basic.py
--- a/tests/auth_tests/test_basic.py
+++ b/tests/auth_tests/test_basic.py
@@ -1,3 +1,4 @@
+# added by the gold test_patch
"""


def _fake_row(repo: str, test_patch: str, patch: str = "") -> dict[str, str]:
    return {
        "instance_id": "x",
        "repo": repo,
        "base_commit": "c",
        "patch": patch,
        "test_patch": test_patch,
    }


def test_selectors_come_from_the_test_patch_not_f2p(monkeypatch: pytest.MonkeyPatch) -> None:
    # sympy's F2P ids are BARE names ("test_coth") with no `::`. Passed positionally to
    # `bin/test` they select nothing; the directive is the test FILE the patch touched.
    pytest.importorskip("swebench")
    monkeypatch.setattr(
        offline_replay, "_dataset_row", lambda _iid: _fake_row("sympy/sympy", _SYMPY_TEST_PATCH)
    )
    directives = offline_replay.swebench_test_directives("sympy__sympy-13480")
    assert directives == ["sympy/functions/elementary/tests/test_hyperbolic.py"]
    assert all(d.endswith(".py") for d in directives)
    assert "test_coth" not in directives


def test_django_selectors_use_swebenchs_dotted_module_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Proves the transform is DELEGATED, not re-derived: `tests/auth_tests/test_basic.py`
    # becomes `auth_tests.test_basic` — what django's unittest runner can actually load.
    pytest.importorskip("swebench")
    monkeypatch.setattr(
        offline_replay, "_dataset_row", lambda _iid: _fake_row("django/django", _DJANGO_TEST_PATCH)
    )
    assert offline_replay.swebench_test_directives("django__django-16631") == [
        "auth_tests.test_basic"
    ]


# ---------------------------------------------------------------------------
# R4: the instance-level two-sided control. Some images cannot import the patched test module
# under ANY selector (astropy-8872, pylint-4661 confirmed), so fixing the selector is necessary
# but not sufficient. An instance that fails either control is not stamped AT ALL.
# ---------------------------------------------------------------------------


class _LegScriptedContainer:
    """Fake container returning a different (output, rc) for the BASE vs GOLD control leg.

    The legs are distinguished by whether a step diff was applied since the last reset — exactly
    how `replay_step` sequences them (reset → [step diff] → test_patch → run).
    """

    def __init__(self, base: tuple[str, int], gold: tuple[str, int]) -> None:
        self.commands: list[str] = []
        self._base, self._gold = base, gold
        self._patched = False

    def __call__(self, command: str, stdin: str | None = None) -> tuple[str, int]:
        self.commands.append(command)
        if "checkout --" in command:
            self._patched = False
            return "", 0
        if "git -C" in command and "apply" in command:
            if stdin == _GOLD_PATCH:
                self._patched = True
            return "", 0
        return self._gold if self._patched else self._base


_GOLD_PATCH = "diff --git a/src/x.py b/src/x.py\n+fixed\n"
_TEST_PATCH = "diff --git a/tests/test_x.py b/tests/test_x.py\n+assert True\n"


_GATE_F2P = ("tests/test_x.py::test_a",)


def _check(
    base: tuple[str, int],
    gold: tuple[str, int],
    classify: object = parse_test_outcome,
    instance_id: str = "repo__repo-1",
) -> object:
    return replay_admissibility.check_instance(
        instance_id=instance_id,
        test_patch=_TEST_PATCH,
        gold_patch=_GOLD_PATCH,
        test_cmd="python -m pytest",
        test_selectors=["tests/test_x.py"],
        exec_fn=_LegScriptedContainer(base, gold),
        classify=classify,
        fail_to_pass=_GATE_F2P,
    )


def test_admissible_when_base_fails_and_gold_passes() -> None:
    verdict = _check(base=("FAILED tests/test_x.py::test_a", 1), gold=("3 passed", 0))
    assert verdict.admissible is True


def test_rejects_instance_whose_base_state_already_passes() -> None:
    # The sympy polarity: without the fix the pipeline still reports a pass, so its greens are
    # unearned. `0 passed` at exit 0 is the concrete shape — it selected nothing.
    verdict = _check(base=("tests finished: 0 passed, in 0.00 seconds", 0), gold=("3 passed", 0))
    assert verdict.admissible is False
    assert "base leg" in verdict.reason


def test_rejects_instance_whose_gold_state_fails() -> None:
    # The django/astropy polarity: a known-correct fix does NOT register as a pass, because the
    # test module cannot be loaded at all — so every red this instance emits is uninformative.
    verdict = _check(
        base=("ERROR: not found: /testbed/a.py::test_x", 4),
        gold=("ERROR: not found: /testbed/a.py::test_x", 4),
    )
    assert verdict.admissible is False
    assert "gold leg" in verdict.reason


def _wire_replay(monkeypatch: pytest.MonkeyPatch, container: _LegScriptedContainer) -> None:
    """Point run_offline_replay at a fake container + fake dataset row (no Docker, no HF)."""
    import contextlib

    from benchmark.runner import step_snapshots, swebench_specs

    spec = swebench_specs.SwebenchSpec(
        instance_id="repo__repo-1",
        repo="sympy/sympy",
        base_commit="c",
        version="1.1",
        difficulty_stratum="easy",
        fail_to_pass=["test_coth"],
        pass_to_pass=[],
        image_ref="img",
        dataset_revision="rev",
    )
    monkeypatch.setattr(step_snapshots, "read_snapshots", lambda _tid: {0: "d0", 1: "d1"})
    monkeypatch.setattr(swebench_specs, "load_spec", lambda _iid: spec)
    monkeypatch.setattr(offline_replay, "swebench_test_command", lambda *_a: "python -m pytest")
    monkeypatch.setattr(
        offline_replay,
        "_dataset_row",
        lambda _iid: _fake_row("sympy/sympy", _SYMPY_TEST_PATCH, patch=_GOLD_PATCH),
    )
    monkeypatch.setattr(offline_replay, "swebench_test_directives", lambda _iid: ["tests/t.py"])
    monkeypatch.setattr(
        offline_replay,
        "instance_container",
        lambda _ref, name: contextlib.nullcontext(name),
    )
    monkeypatch.setattr(offline_replay, "docker_exec", lambda _name: container)


def _unstamped(step_index: int) -> object:
    """A step as the normalizer leaves it before any replay: never observed, not a pass."""
    # The factory derives `exit_code` from `success`, so it cannot express the real parser
    # default (None). `replace` restores it — that null is exactly what marks "not replayed".
    return replace(make_step(step_index=step_index, confirmed=False), exit_code=None)


def _fabricated(step_index: int) -> object:
    """A step carrying the stamp set the PRE-FIX replay wrote: a blocking exit-4 capability red."""
    return make_step(
        step_index=step_index,
        success=False,
        confirmed=True,
        failing_check_id="hash:deadbeefdeadbeef",
        exit_code=4,
        observation=f"obs-{step_index}",
        action=f"act-{step_index}",
        args=f"args-{step_index}",
        result=f"res-{step_index}",
    )


# `_wire_replay` fakes a sympy spec whose FAIL_TO_PASS is `["test_coth"]`, so the assembled path
# builds a REAL `GraderParity` for sympy and these leg outputs are in sympy's own `bin/test`
# dialect — `test_coth ok` / `test_coth F` is what its parser reads a status out of.
_SYMPY_F2P_PASSES = ("test_coth ok", 0)
_SYMPY_F2P_FAILS = ("test_coth F", 1)
_SYMPY_SELECTED_NOTHING = ("tests finished: 0 passed", 0)
# The F2P test PASSES with the fix removed, and something the grader does not score fails. On the
# whole-file exit code this reads FAILURE (rc=1) and the instance is admitted; per-test it is
# RESOLVED, so the destroyed-signal leg reports success and the instance is rejected.
_SYMPY_BASE_ALREADY_RESOLVES = ("test_coth ok\ntest_unrelated F", 1)


def _reject(monkeypatch: pytest.MonkeyPatch, jsonl: Path) -> Path | None:
    """Drive a full replay of an instance the gate rejects (BASE cannot fail → the sympy shape)."""
    _wire_replay(monkeypatch, _LegScriptedContainer(_SYMPY_SELECTED_NOTHING, _SYMPY_F2P_PASSES))
    return offline_replay.run_offline_replay("traj-1", "repo__repo-1", jsonl)


def test_inadmissible_instance_has_its_stale_stamps_actively_cleared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The load-bearing assertion, and the one the old skip-and-return could not make: the gate
    # short-circuited BEFORE writing, so a rejected instance kept whatever the defective replay
    # had stamped. astropy-8872 came back from a rebuild byte-identical, 100% blocking on exit-4.
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(
        make_trajectory([_fabricated(0), _fabricated(1), _fabricated(2)], snapshot_steps=2), jsonl
    )
    assert _reject(monkeypatch, jsonl) is None

    reloaded = schema.load_jsonl(jsonl)
    assert all(not s.confirmed for s in reloaded.steps[:-1])
    assert all(s.exit_code is None for s in reloaded.steps)
    assert all(s.failing_check_id is None and s.dedup_key is None for s in reloaded.steps)
    assert not any(s.blocking for s in reloaded.steps[:-1])
    # The terminal step keeps the SWE-bench GRADER's verdict — a different instrument, which this
    # gate says nothing about — but not the replay's exit code.
    assert reloaded.steps[-1].confirmed is True
    assert reloaded.steps[-1].exit_code is None


def test_clearing_preserves_every_non_stamp_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The agent's behaviour is not the instrument's verdict: clearing must not touch it.
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=2), jsonl)
    _reject(monkeypatch, jsonl)
    step = schema.load_jsonl(jsonl).steps[0]
    assert (step.action, step.args, step.observation, step.result) == (
        "act-0",
        "args-0",
        "obs-0",
        "res-0",
    )
    assert step.step_index == 0 and step.metadata == {"k": "v"}


def test_clearing_is_idempotent_and_keeps_the_manifest_consistent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A rebuild re-visits a rejected instance every run. Clearing writes the file, so the manifest
    # must be rewritten with it or the corpus fails its own Layer-1 hash check.
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=2), jsonl)
    _reject(monkeypatch, jsonl)
    once = jsonl.read_bytes()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (
        manifest["trajectories"]["t1"]["content_sha256"]
        == schema.load_jsonl(jsonl).header.content_sha256
    )
    _reject(monkeypatch, jsonl)
    assert jsonl.read_bytes() == once  # second pass changes not one byte


def test_a_rejection_is_recorded_under_its_gate_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The exclusion has to be auditable rather than silent — and the record doubles as the cache.
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=2), jsonl)
    _reject(monkeypatch, jsonl)
    verdicts = json.loads((tmp_path / replay_admissibility.VERDICT_FILENAME).read_text())
    record = verdicts["repo__repo-1"]
    assert record["admissible"] is False
    assert "base leg" in record["reason"]
    assert record["base"]["outcome"] == "unknown" and record["gold"]["outcome"] == "success"
    assert record["gate_key"]


def test_admissible_instance_still_stamps_its_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The other direction — the gate must not reject everything. A valid instrument replays.
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_unstamped(0), _unstamped(1)], snapshot_steps=2), jsonl)
    _wire_replay(monkeypatch, _LegScriptedContainer(_SYMPY_F2P_FAILS, _SYMPY_F2P_PASSES))

    assert offline_replay.run_offline_replay("traj-1", "repo__repo-1", jsonl) == jsonl
    assert schema.load_jsonl(jsonl).steps[0].confirmed is True


def test_base_leg_that_already_resolves_is_rejected_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The BASE-leg hole, through the assembled path: the F2P test PASSES with the fix REMOVED, so
    # the replay cannot tell fixed from unfixed and the instance must be rejected — even though an
    # unrelated failure makes the whole-file exit code say the leg "failed".
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=2), jsonl)
    container = _LegScriptedContainer(_SYMPY_BASE_ALREADY_RESOLVES, _SYMPY_F2P_PASSES)
    _wire_replay(monkeypatch, container)
    # The exit-code adjudicator would have called this leg a FAILURE and admitted the instance.
    assert parse_test_outcome(*_SYMPY_BASE_ALREADY_RESOLVES).outcome == "failure"

    assert offline_replay.run_offline_replay("traj-1", "repo__repo-1", jsonl) is None
    record = json.loads((tmp_path / replay_admissibility.VERDICT_FILENAME).read_text())
    assert record["repo__repo-1"]["admissible"] is False
    assert record["repo__repo-1"]["base"]["outcome"] == "success"


# ---------------------------------------------------------------------------
# The gate is a property of the INSTANCE, not of one trajectory. Recomputing it per trajectory
# ran 792 control pairs (1 584 container test runs) for 166 instances — 626 pairs, 1–31 h, of
# pure redundancy. The cache key must make a stale entry impossible to serve.
# ---------------------------------------------------------------------------


def _no_container(_ref: str, _name: str) -> object:
    raise AssertionError("a cached verdict must not start a container")


def test_a_cached_rejection_needs_no_container_at_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=2), jsonl)
    _reject(monkeypatch, jsonl)  # first pass: measured, recorded

    monkeypatch.setattr(offline_replay, "instance_container", _no_container)
    assert offline_replay.run_offline_replay("traj-2", "repo__repo-1", jsonl) is None
    # ...and it still clears, so a second trajectory of the same instance is not left stamped.
    assert all(s.exit_code is None for s in schema.load_jsonl(jsonl).steps)


def test_a_verdict_measured_by_a_different_instrument_is_never_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The whole point of the key: edit the classifier or the replay and every cached verdict is
    # invalidated automatically. A hand-bumped version constant is what goes stale silently.
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=2), jsonl)
    _reject(monkeypatch, jsonl)
    assert (
        replay_admissibility.cached_verdict("repo__repo-1", "some-other-instrument", tmp_path)
        is None
    )


def test_the_gate_key_moves_with_the_subject_as_well_as_the_instrument() -> None:
    common = {"dataset_revision": "rev0", "image_ref": "img", "test_cmd": "pytest -rA"}
    a = replay_admissibility.gate_key(**common, test_selectors=["tests/a.py"])
    b = replay_admissibility.gate_key(**common, test_selectors=["tests/b.py"])
    c = replay_admissibility.gate_key(
        dataset_revision="rev1",
        image_ref="img",
        test_cmd="pytest -rA",
        test_selectors=["tests/a.py"],
    )
    # F2P/P2P are now what the verdict is adjudicated AGAINST, so they are part of the subject.
    d = replay_admissibility.gate_key(**common, test_selectors=["tests/a.py"], fail_to_pass=["t"])
    e = replay_admissibility.gate_key(**common, test_selectors=["tests/a.py"], pass_to_pass=["t"])
    assert len({a, b, c, d, e}) == 5


def test_a_pre_cache_record_carries_no_key_and_is_re_measured(tmp_path: Path) -> None:
    # `gate_key=""` is the pre-cache shape; it must never match a computed key.
    verdict = replay_admissibility.adjudicate(
        "i",
        parse_test_outcome("FAILED a.py::t", 1),
        parse_test_outcome("1 passed", 0),
        ("a.py::t",),
    )
    replay_admissibility.record_verdict(verdict, tmp_path)
    assert verdict.gate_key == ""
    key = replay_admissibility.gate_key(
        dataset_revision="r", image_ref="i", test_cmd="c", test_selectors=[]
    )
    assert replay_admissibility.cached_verdict("i", key, tmp_path) is None


# ---------------------------------------------------------------------------
# The snapshot-less trajectory. 7 of 799 committed trajectories have no per-step diffs at all, so
# nothing replays them and nothing cleared them — a rebuild silently reported them done and the
# stamp ledger recorded them as completed under the current instrument. "No snapshot directory"
# has two causes that look identical on disk; only the COMMITTED `snapshot_steps` separates them.
# ---------------------------------------------------------------------------


def _no_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark.runner import step_snapshots

    monkeypatch.setattr(step_snapshots, "read_snapshots", lambda _tid: {})


def test_a_trajectory_that_captured_no_snapshots_is_cleared_without_a_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=0), jsonl)
    _no_snapshots(monkeypatch)
    monkeypatch.setattr(offline_replay, "instance_container", _no_container)

    assert offline_replay.run_offline_replay("traj-1", "repo__repo-1", jsonl) is None
    reloaded = schema.load_jsonl(jsonl)
    assert all(not s.confirmed for s in reloaded.steps[:-1])
    assert all(s.exit_code is None for s in reloaded.steps)
    # The manifest is rewritten with it, or the corpus fails its own Layer-1 hash check.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["trajectories"]["t1"]["snapshot_steps"] == 0
    assert manifest["trajectories"]["t1"]["content_sha256"] == reloaded.header.content_sha256


def test_a_checkout_that_merely_lacks_the_scratch_never_clears_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # THE failure mode this guard exists for: the snapshots are gitignored, so a fresh clone has
    # none. Clearing on filesystem absence would wipe all 799 trajectories. It must refuse LOUDLY
    # instead — a silent success is what lets the stamp ledger record a rebuild that never ran.
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=12), jsonl)
    before = jsonl.read_bytes()
    _no_snapshots(monkeypatch)

    with pytest.raises(offline_replay.SnapshotsMissingError):
        offline_replay.run_offline_replay("traj-1", "repo__repo-1", jsonl)
    assert jsonl.read_bytes() == before


def test_unknown_snapshot_provenance_refuses_rather_than_clearing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A header written before the field existed says None. That is "unknown", never "zero".
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0)], snapshot_steps=None), jsonl)
    _no_snapshots(monkeypatch)

    with pytest.raises(offline_replay.SnapshotsMissingError):
        offline_replay.run_offline_replay("traj-1", "repo__repo-1", jsonl)


def test_unknown_snapshot_provenance_also_refuses_a_present_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The mirror case: the scratch IS present, but the header records no count, so a partial
    # scratch cannot be told from a complete one. Replaying what is present and leaving the rest
    # stamped would mix two adjudicators into a file that passes its own Layer-1 check — the exact
    # defect the partial-scratch guard exists to prevent. Refuse rather than restamp blind.
    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=None), jsonl)
    before = jsonl.read_bytes()
    _wire_replay(monkeypatch, _LegScriptedContainer(_SYMPY_F2P_FAILS, _SYMPY_F2P_PASSES))

    with pytest.raises(offline_replay.SnapshotsMissingError):
        offline_replay.run_offline_replay("traj-1", "repo__repo-1", jsonl)
    assert jsonl.read_bytes() == before


def test_the_replay_exec_forces_a_utf8_stdio_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without this the instance conda env (often Python 3.6, pre PEP-538) resolves stdout to
    # ASCII, and django's `migrate` dies on its "Creating tables…" ellipsis before any test runs.
    # Measured on django-10880: zero parseable statuses, so the gate rejected an instance the
    # SWE-bench harness resolves in the same image (9/9 collected cells graded RESOLVED).
    seen: list[list[str]] = []

    class _Proc:
        stdout, stderr, returncode = "", "", 0

    monkeypatch.setattr(
        offline_replay.subprocess,
        "run",
        lambda argv, **_kw: (seen.append(argv), _Proc())[1],
    )
    offline_replay.docker_exec("c")("echo hi")
    assert "-e" in seen[0]
    assert "PYTHONIOENCODING=utf-8" in seen[0]
    assert seen[0].index("PYTHONIOENCODING=utf-8") < seen[0].index("c")  # before the container id


def test_a_base_leg_that_fails_on_a_non_f2p_test_is_rejected() -> None:
    # The zero-power admission: a P2P test passes at base by SWE-bench's construction, so a red one
    # is flakiness — and it satisfies "base is not resolved" while every F2P test already passes
    # WITHOUT the fix. The instance would then be admitted with no power on the axis that encodes
    # the bug, and its steps would track the flaky test.
    from shunt.verifiers.base import VerifierResult

    base = VerifierResult(
        outcome="failure", confidence=0.7, failing_check_id="t.py::test_unrelated_flaky"
    )
    gold = VerifierResult(outcome="success", confidence=0.8)
    verdict = replay_admissibility.adjudicate(
        "i", base, gold, ("t.py::test_bugfix_a", "t.py::test_bugfix_b")
    )
    assert verdict.admissible is False
    assert "not on any FAIL_TO_PASS test" in verdict.reason
    # ...and it IS admissible once the base leg fails on the bug itself.
    on_bug = replace(base, failing_check_id="t.py::test_bugfix_a")
    assert replay_admissibility.adjudicate(
        "i", on_bug, gold, ("t.py::test_bugfix_a", "t.py::test_bugfix_b")
    ).admissible
    # An EMPTY list means "this spec declares no signal", never "skip the check".
    assert not replay_admissibility.adjudicate("i", on_bug, gold, ()).admissible


def test_a_partially_present_scratch_refuses_rather_than_restamping_half_the_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `restamp_trajectory` leaves un-replayed steps untouched and the manifest is rewritten after,
    # so a partial scratch produces a file that passes its own Layer-1 check while most of its
    # steps still carry stamps from the adjudicator this change replaced.
    from benchmark.runner import step_snapshots

    jsonl = tmp_path / "t.jsonl"
    schema.dump_jsonl(make_trajectory([_fabricated(0), _fabricated(1)], snapshot_steps=2), jsonl)
    before = jsonl.read_bytes()
    monkeypatch.setattr(step_snapshots, "read_snapshots", lambda _tid: {0: "d0"})

    with pytest.raises(offline_replay.SnapshotsMissingError):
        offline_replay.run_offline_replay("traj-1", "repo__repo-1", jsonl)
    assert jsonl.read_bytes() == before


def test_the_replay_preserves_the_images_own_build_time_modifications() -> None:
    # SWE-bench's `pre_install` edits land in the working tree of some images (sphinx's
    # `sed -i 's/pytest/pytest -rA/' tox.ini`). A plain `git checkout -- .` removes them, and
    # without `-rA` pytest prints no per-test status lines at all, so the log parser extracts
    # ZERO statuses and the instance reads unmeasurable. Measured: 8 of 19 sphinx instances, every
    # one of them resolved by the real grading harness.
    container = _FakeContainer("1 passed", 0)
    _replay(container)
    reset = container.commands[0]
    assert "checkout -- ." in reset
    # Captured once, to a temp path, and moved into place only after git succeeded (see the
    # atomicity test below).
    assert "git -C /testbed diff > /tmp/shunt_build_state.patch.tmp" in reset
    assert "mv /tmp/shunt_build_state.patch.tmp /tmp/shunt_build_state.patch" in reset
    # ...and the RESET no longer re-applies it. That re-apply now runs after the step diff, which
    # already carries the build edit — applying it first rejected the whole step diff (`git apply`
    # is atomic) and turned a real measurement into an infra failure.
    assert "apply /tmp/shunt_build_state.patch" not in reset
    reconcile = next(c for c in container.commands if c.startswith("[ ! -s "))
    assert container.commands.index(reconcile) > container.commands.index("git -C /testbed apply -")
    assert reconcile == (
        "[ ! -s /tmp/shunt_build_state.patch ] || "
        "git -C /testbed apply -R --check /tmp/shunt_build_state.patch || "
        "git -C /testbed apply /tmp/shunt_build_state.patch"
    )


# ---------------------------------------------------------------------------
# The reset runs in a REAL shell, so its bugs are shell bugs and a canned fake cannot see them.
# These drive the real command string through `bash` with a stubbed `git` — no Docker, but the
# thing under test (redirect ordering, `;` vs `&&`) is exercised exactly as it is in a container.
# ---------------------------------------------------------------------------


def _bash_container(bin_dir: Path) -> offline_replay.ContainerExec:
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    def _exec(command: str, stdin: str | None = None) -> tuple[str, int]:
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
        proc = subprocess.run(  # noqa: S603
            ["bash", "-c", command], capture_output=True, text=True, env=env, check=False
        )
        return f"{proc.stdout}\n{proc.stderr}", proc.returncode

    return _exec


def _stub_git(tmp_path: Path, *, diff_rc: int) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    stub.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f'  *diff*) echo "fatal: detected dubious ownership" >&2; exit {diff_rc};;\n'
        "  *) exit 0;;\n"
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bin_dir


def test_a_failed_build_state_capture_is_reported_and_never_poisons_the_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REVERT PROBE: restore `git diff > $STATE; ` (redirect straight to the target, clause ended by
    # `;`) and this returns True with a 0-byte file present. Both halves matter: the `;` discarded
    # git's non-zero status so a failed reset reported "ok", and the redirect created the target
    # BEFORE git ran, so `[ ! -s ]` skipped the re-apply for ever while `[ -f ]` blocked every
    # retry. `git diff` failing here is realistic — safe.directory refusal, corrupt index.
    state = tmp_path / "build_state.patch"
    monkeypatch.setattr(offline_replay, "_BUILD_STATE", str(state))
    monkeypatch.setattr(offline_replay, "_BUILD_STATE_TMP", f"{state}.tmp")

    assert offline_replay._reset_to_base(_bash_container(_stub_git(tmp_path, diff_rc=128))) is False
    assert not state.exists()  # nothing to skip the re-apply, nothing to suppress the retry


def test_a_successful_capture_still_records_the_build_state_and_survives_re_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The other side of the probe above: when `git diff` works, the patch lands at the real path
    # (not the temp one), the reset reports success, and a second reset reuses the captured file.
    state = tmp_path / "build_state.patch"
    monkeypatch.setattr(offline_replay, "_BUILD_STATE", str(state))
    monkeypatch.setattr(offline_replay, "_BUILD_STATE_TMP", f"{state}.tmp")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    stub.write_text(
        '#!/bin/sh\ncase "$*" in\n  *diff*) echo "diff --git a/tox.ini b/tox.ini";;\n'
        "esac\nexit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    exec_fn = _bash_container(bin_dir)
    assert offline_replay._reset_to_base(exec_fn) is True
    assert state.read_text(encoding="utf-8").startswith("diff --git")
    assert not Path(f"{state}.tmp").exists()
    assert offline_replay._reset_to_base(exec_fn) is True


# ---------------------------------------------------------------------------
# The exec timeout. A step diff is agent-written, so a step CAN loop for ever; an unbounded
# `docker exec` then stalls a multi-hour unsupervised pass with no marker and no demotion.
# ---------------------------------------------------------------------------


def test_a_hung_exec_is_killed_demoted_to_infra_and_reaps_its_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REVERT PROBE: drop `timeout=` from the `subprocess.run` in `docker_exec` and this test never
    # returns — the stand-in below is a REAL process that outlives the harness, so the timeout that
    # fires is subprocess's own and not a raise the test staged. That hang IS the production bug.
    import subprocess  # noqa: PLC0415

    from benchmark.runner import swebench_grading  # noqa: PLC0415

    real_run = subprocess.run
    reaped: list[str] = []

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[:3] == ["docker", "rm", "-f"]:
            reaped.append(argv[3])
            return real_run(["true"], capture_output=True, text=True, check=False)
        return real_run(["sh", "-c", "echo collecting ...; sleep 600"], **kwargs)  # type: ignore[call-overload,no-any-return]

    monkeypatch.setattr(offline_replay.subprocess, "run", _fake_run)
    combined, rc = offline_replay.docker_exec("shunt-replay-hung", timeout_s=0.5)("pytest")

    assert rc == offline_replay._TIMEOUT_EXIT
    assert swebench_grading.timeout_marker() in combined  # the demotion signal, not a bare rc
    assert "collecting ..." in combined  # partial POSIX bytes decoded, not rendered as b'...'
    assert reaped == ["shunt-replay-hung"]  # the runaway cannot burn a core for the rest of the run
    # ...and the real adjudicator refuses to grade it: unmeasured, never a fabricated red. Without
    # the marker, rc=124 is outside `parse_test_outcome`'s non-session set and grades as a RED.
    verdict = swebench_grading.GraderParity(
        repo="matplotlib/matplotlib", fail_to_pass=("a.py::t",), pass_to_pass=()
    )(combined, rc)
    assert verdict.outcome == "unknown"
    assert verdict.is_infra_failure is True


# ---------------------------------------------------------------------------
# KNOWN_ARTIFACTS: rejections diagnosed as defects of the INSTRUMENT and deliberately left
# unfixed. A rejection clears real stamps, so the exclusion has to be readable from the committed
# record — and the annotation must never be able to change the verdict it annotates.
# ---------------------------------------------------------------------------

_KNOWN_REJECTED = ("django__django-15098", "psf__requests-5414")


@pytest.mark.parametrize("instance_id", _KNOWN_REJECTED)
def test_a_known_artifact_rejection_names_the_defect_in_its_committed_record(
    instance_id: str, tmp_path: Path
) -> None:
    # The gold-leg-fails polarity, which is how requests-5414 presents; django-15098 rejects on
    # the base leg. Either way the record must carry the diagnosis, not just the leg outcomes.
    verdict = _check(
        base=("ERROR: not found: /testbed/a.py::test_x", 4),
        gold=("ERROR: not found: /testbed/a.py::test_x", 4),
        instance_id=instance_id,
    )
    assert verdict.admissible is False
    assert verdict.known_artifact == replay_admissibility.KNOWN_ARTIFACTS[instance_id]

    replay_admissibility.record_verdict(verdict, tmp_path)
    record = json.loads((tmp_path / replay_admissibility.VERDICT_FILENAME).read_text())[instance_id]
    # Readable straight off the committed JSON — no code needed to interpret the exclusion.
    assert "KNOWN INSTRUMENT ARTIFACT" in record["known_artifact"]
    assert replay_admissibility.load_verdicts(tmp_path)[instance_id].known_artifact


def test_an_unannotated_instance_gets_the_identical_verdict_with_an_empty_note() -> None:
    # The note is descriptive only: same legs, same decision, and nothing to say about an
    # instance the registry does not name.
    legs = {
        "base": ("ERROR: not found: /testbed/a.py::test_x", 4),
        "gold": ("ERROR: not found: /testbed/a.py::test_x", 4),
    }
    annotated = _check(**legs, instance_id=_KNOWN_REJECTED[0])
    plain = _check(**legs, instance_id="repo__repo-1")
    assert annotated.admissible == plain.admissible is False
    assert annotated.reason.split(":", 1)[1] == plain.reason.split(":", 1)[1]
    assert plain.known_artifact == ""


def test_a_stale_artifact_entry_cannot_annotate_an_admissible_instance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The registry says "rejected on a known defect". If the instance is ADMISSIBLE the entry
    # describes a reality that changed, so it is dropped from the record and reported — a note
    # that outlived its verdict is exactly the drift a clearing gate must not ship.
    with caplog.at_level(logging.WARNING):
        verdict = _check(
            base=("FAILED tests/test_x.py::test_a", 1),
            gold=("3 passed", 0),
            instance_id=_KNOWN_REJECTED[0],
        )
    assert verdict.admissible is True
    assert verdict.known_artifact == ""
    assert "STALE KNOWN_ARTIFACTS entry" in caplog.text


def test_the_replay_cli_configures_its_own_logging_so_gate_verdicts_reach_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The stamping stage runs this module as a subprocess. With no handler the child falls back to
    # `logging.lastResort` (WARNING), which drops every ADMISSIBLE gate verdict — the majority of
    # what a multi-hour rebuild is supervised for — while rejections still print.
    configured: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: configured.update(kw), raising=True)
    monkeypatch.setattr(offline_replay, "run_offline_replay", lambda *a, **k: None)
    monkeypatch.setattr(
        "sys.argv", ["offline_replay", "traj-1", "--instance-id", "x", "--jsonl", str(tmp_path)]
    )
    assert offline_replay._main() == 0
    assert configured["level"] == logging.INFO


def test_the_replay_cli_quiets_http_chatter_but_not_its_own_verdicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Measured over 8 replays: 51 `httpx` INFO lines to 6 real ones. Across 799 trajectories that
    # is ~5 000 lines of HTTP burying the gate verdicts a supervisor reads the log FOR.
    monkeypatch.setattr(sys, "argv", ["offline_replay", "t", "--instance-id", "i"])
    monkeypatch.setattr(offline_replay, "run_offline_replay", lambda *a, **k: None)
    # basicConfig is a no-op under pytest (the root logger already has a handler), so assert on
    # the per-logger levels `_main` sets explicitly — the quieting must not be level-inherited.
    monkeypatch.setattr(logging, "basicConfig", lambda **_kw: None)
    for name in (*offline_replay._QUIET_LOGGERS, "benchmark.runner.replay_admissibility"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
    assert offline_replay._main() == 0
    for name in offline_replay._QUIET_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING
    # The replay's OWN loggers keep whatever the run configured — quieting is targeted, not global.
    assert logging.getLogger("benchmark.runner.replay_admissibility").level == logging.NOTSET
