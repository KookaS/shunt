"""Offline replay core: drive `replay_snapshots` with a FAKE in-container exec (legitimate unit
mocking — no Docker, no swebench) and assert the isolation ritual, real-only infra handling, and
that per-step outcomes come through the SHARED `parse_test_outcome` (dedup-key parity)."""

from __future__ import annotations

from benchmark.runner.offline_replay import replay_snapshots, replay_step
from shunt.verifiers.parse import parse_test_outcome


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
    return replay_step(
        diff, "test_patch_body", "python -m pytest", ["tests/t.py::test_a"], container
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
    joined = " || ".join(container.commands)
    # reset-to-base precedes the two applies, which precede the test run.
    assert container.commands[0].startswith("git -C") and "checkout --" in container.commands[0]
    assert joined.count("apply") == 2  # step diff + test_patch
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
        fail_to_pass=["tests/t.py::test_a"],
        exec_fn=container,
    )
    assert sorted(outcomes) == [0, 2]
    assert all(o.outcome == "failure" for o in outcomes.values())


def test_empty_diff_skips_the_step_apply_but_still_runs() -> None:
    # A no-change step (base state) applies only the test_patch, then runs.
    container = _FakeContainer("1 passed", 0)
    replay_step("", "tp", "python -m pytest", ["tests/t.py::test_a"], container)
    applies = [c for c in container.commands if "apply" in c]
    assert len(applies) == 1  # only the test_patch, no step diff
