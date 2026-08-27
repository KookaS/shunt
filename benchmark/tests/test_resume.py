"""Conversation resume: continue a throttled/aborted cell from its saved messages.

All stubbed (no live/paid/Docker/model calls); the loop is proven against the installed
mini-swe-agent DefaultAgent with a fake model/env.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark import config
from benchmark.runner import infer, step_snapshots


def _assistant(content: str, *, cost: float = 0.1) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "tc1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
        "extra": {
            "actions": [{"command": "ls", "tool_call_id": "tc1"}],
            "response": {"usage": {"cost": cost, "prompt_tokens": 10, "completion_tokens": 5}},
            "cost": cost,
        },
    }


def _tool(content: str = "obs") -> dict[str, Any]:
    return {"role": "tool", "content": content, "tool_call_id": "tc1", "extra": {"returncode": 0}}


def _resumable_conversation(n_assistant: int = 2) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(n_assistant):
        messages.append(_assistant(f"turn {i}"))
        messages.append(_tool(f"obs {i}"))
    return messages


def _spec() -> Any:
    """A duck-typed SwebenchSpec stand-in (Any so mypy accepts the _invoke_scaffold arg)."""
    return _Spec("iid")


class _Spec:
    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id


class _ScaffoldPatches:
    """The light monkeypatching that lets `_invoke_scaffold` run with no minisweagent imports."""

    @staticmethod
    def apply(monkeypatch: pytest.MonkeyPatch) -> None:
        from benchmark.escalation import live_capture

        monkeypatch.setattr(
            infer, "_load_instance", lambda iid: {"instance_id": iid, "problem_statement": "ps"}
        )
        monkeypatch.setattr(infer, "litellm_model_target", lambda m: ("model-string", {}))
        monkeypatch.setattr(live_capture, "make_trajectory_id", lambda *a: "tid")


def _redirect_message_list(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Route the scratch message-list lookup (bound at def time) into *root*."""
    monkeypatch.setattr(
        step_snapshots, "message_list_path", lambda tid, root=root: root / f"{tid}.json"
    )


def _write_message_list(root: Path, messages: list[dict[str, Any]]) -> None:
    path = step_snapshots.message_list_path("tid", root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"info": {}, "messages": messages, "trajectory_format": "x"}))


class TestResumeDetection:
    """A saved conversation is resumable iff it exists, is non-trivial, and ends cleanly."""

    def test_absent_file_is_not_resumable(self, tmp_path, monkeypatch) -> None:
        _redirect_message_list(monkeypatch, tmp_path)
        assert infer._resume_messages("tid") is None

    def test_resumable_conversation_is_loaded_with_exit_stripped(
        self, tmp_path, monkeypatch
    ) -> None:
        _redirect_message_list(monkeypatch, tmp_path)
        conversation = _resumable_conversation(n_assistant=2)
        conversation.append(
            {"role": "exit", "content": "boom", "extra": {"exit_status": "SomeError"}}
        )
        _write_message_list(tmp_path, conversation)
        loaded = infer._resume_messages("tid")
        assert loaded is not None
        assert len(loaded) == len(conversation) - 1  # the trailing exit message is dropped
        assert loaded[-1]["role"] == "tool"
        assert all(m["role"] != "exit" for m in loaded)

    def test_trivial_conversation_without_assistant_turn_is_not_resumable(
        self, tmp_path, monkeypatch
    ) -> None:
        _redirect_message_list(monkeypatch, tmp_path)
        _write_message_list(
            tmp_path, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        )
        assert infer._resume_messages("tid") is None

    def test_mid_turn_conversation_is_not_resumable(self, tmp_path, monkeypatch) -> None:
        # Ends on an assistant message whose actions were never observed — re-sending those
        # dangling tool_calls is not safe, so the cell must start fresh.
        _redirect_message_list(monkeypatch, tmp_path)
        conversation = _resumable_conversation(n_assistant=1)
        conversation.append(_assistant("pending action"))
        _write_message_list(tmp_path, conversation)
        assert infer._resume_messages("tid") is None

    def test_unreadable_file_is_not_resumable(self, tmp_path, monkeypatch) -> None:
        _redirect_message_list(monkeypatch, tmp_path)
        path = step_snapshots.message_list_path("tid", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        assert infer._resume_messages("tid") is None


class TestResumeGate:
    """resume.enabled is the switch; off = byte-identical fresh start, no detection at all."""

    def test_gate_defaults_to_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "get", lambda: {"live": {"step_limit": 150}})
        assert config.resume_enabled() is False

    def test_gate_enabled_when_configured(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "get", lambda: {"resume": {"enabled": True}})
        assert config.resume_enabled() is True

    def test_disabled_never_even_checks_for_a_saved_conversation(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "resume_enabled", lambda: False)
        monkeypatch.setattr(
            infer, "_resume_messages", lambda *a, **k: pytest.fail("detected while disabled!")
        )
        _ScaffoldPatches.apply(monkeypatch)
        seen: dict[str, object] = {}

        def _attempt(*a, **k):
            seen["resume"] = k["resume"]
            return infer.AgentPatch(patch="", in_tok=0, out_tok=0, calls=0, cost=0.0)

        monkeypatch.setattr(infer, "_invoke_scaffold_attempt", _attempt)
        infer._invoke_scaffold(_spec(), "m", "mini-swe-agent")
        assert seen["resume"] is None  # a fresh start even with a conversation on disk


class TestResumeSeeding:
    """The loaded history is passed to the agent unchanged and continued, never duplicated."""

    def test_seed_resume_copies_messages_verbatim(self) -> None:
        class _Agent:
            def __init__(self) -> None:
                self.messages: list[dict[str, Any]] = []
                self.run: Any = lambda task: {}

        conversation = _resumable_conversation(n_assistant=1)
        agent = _Agent()
        infer._seed_resume(agent, conversation, format_error_cls=Exception, interrupt_cls=Exception)
        assert agent.messages == conversation
        assert agent.messages is not conversation  # copied, never aliased
        assert callable(agent.run)

    def test_resume_loop_continues_installed_default_agent(self, tmp_path) -> None:
        # Proven against the INSTALLED scaffold: seeding messages and routing run() through the
        # resume loop continues from the saved conversation — system/user are NOT re-added, the
        # prior turns are preserved, the new turn is appended, and the dump holds the WHOLE cell.
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded

        class _Model:
            config: Any = {}

            def __init__(self) -> None:
                self.calls = 0

            def format_message(self, **kwargs: Any) -> dict[str, Any]:
                return {
                    "role": kwargs.get("role"),
                    "content": kwargs.get("content"),
                    "extra": kwargs.get("extra") or {},
                }

            def format_observation_messages(self, message, outputs, template_vars=None):
                return [_tool(f"obs {message.get('content')}")]

            def query(self, messages, **kwargs):
                self.calls += 1
                if self.calls >= 2:  # one real resumed step, then the budget stop
                    raise LimitsExceeded(
                        {
                            "role": "exit",
                            "content": "LimitsExceeded",
                            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                        }
                    )
                return _assistant("new resumed turn")

            def get_template_vars(self, **kwargs):
                return {}

            def serialize(self):
                return {"model": {}}

        class _Env:
            config: Any = {}

            def get_template_vars(self, **kwargs):
                return {}

            def execute(self, action, cwd=""):
                return {"output": "out", "returncode": 0}

            def serialize(self):
                return {"environment": {}}

        dump = tmp_path / "tid__m__d.json"
        conversation = _resumable_conversation(n_assistant=2)
        agent = DefaultAgent(
            _Model(),
            _Env(),
            system_template="sys {{task}}",
            instance_template="task {{task}}",
            step_limit=2,
            output_path=dump,
        )
        infer._seed_resume(
            agent, conversation, format_error_cls=FormatError, interrupt_cls=InterruptAgentFlow
        )
        result = agent.run("fix it")

        assert result.get("exit_status") == "LimitsExceeded"
        assert agent.messages[:2] == conversation[:2]  # system + instance preserved once
        assert agent.messages[2:6] == conversation[2:6]  # prior assistant + tool turns unchanged
        assert agent.messages.count({"role": "system", "content": "sys"}) == 1
        new_assistant = [
            m
            for m in agent.messages
            if m.get("role") == "assistant" and m.get("content") == "new resumed turn"
        ]
        assert len(new_assistant) == 1  # the resumed window appended exactly one new turn
        data = json.loads(dump.read_text())
        assert data["messages"] == agent.messages  # the dump is the FULL extended conversation


class TestResumeBudget:
    """Remaining step budget = full limit minus loaded assistant turns, floored at 1."""

    def test_step_limit_reduced_by_loaded_turns(self) -> None:
        assert infer._resume_step_limit(150, _resumable_conversation(n_assistant=3)) == 147

    def test_step_limit_floor_is_one(self) -> None:
        assert infer._resume_step_limit(150, _resumable_conversation(n_assistant=200)) == 1

    def test_zero_loaded_turns_keeps_full_budget(self) -> None:
        assert infer._resume_step_limit(150, []) == 150

    def test_reduced_limit_binds_the_resumed_run(self, tmp_path) -> None:
        # The resumed agent's config.step_limit (n_calls starts at 0 per fresh agent object) is
        # exactly the remaining budget: reduced by the loaded turns, cost check disabled so the
        # step ceiling is what stops the run — not an inflated spend.
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.exceptions import FormatError, InterruptAgentFlow

        class _Model:
            config: Any = {}

            def __init__(self) -> None:
                self.calls = 0

            def format_message(self, **kwargs: Any) -> dict[str, Any]:
                return {
                    "role": kwargs.get("role"),
                    "content": kwargs.get("content"),
                    "extra": kwargs.get("extra") or {},
                }

            def format_observation_messages(self, message, outputs, template_vars=None):
                return [_tool()]

            def query(self, messages, **kwargs):
                self.calls += 1
                return _assistant("t", cost=0.0)

            def get_template_vars(self, **kwargs):
                return {}

            def serialize(self):
                return {"model": {}}

        class _Env:
            config: Any = {}

            def get_template_vars(self, **kwargs):
                return {}

            def execute(self, action, cwd=""):
                return {"output": "out", "returncode": 0}

            def serialize(self):
                return {"environment": {}}

        agent = DefaultAgent(
            _Model(),
            _Env(),
            system_template="sys {{task}}",
            instance_template="task {{task}}",
            step_limit=infer._resume_step_limit(150, _resumable_conversation(n_assistant=2)),
            cost_limit=0.0,  # disable the cost ceiling so only the step ceiling stops the run
            output_path=tmp_path / "d.json",
        )
        agent.messages = [dict(m) for m in _resumable_conversation(n_assistant=2)]
        infer._resume_run(
            agent, "t", format_error_cls=FormatError, interrupt_cls=InterruptAgentFlow
        )
        prior = 2
        new = sum(1 for m in agent.messages if m.get("role") == "assistant") - prior
        assert new == 148


class TestResumeCheckoutReconstruction:
    """The checkout is rebuilt from the LAST cumulative snapshot; any failure → fallback."""

    def test_applies_only_the_last_cumulative_snapshot(self, monkeypatch) -> None:
        # Each step diff is `git diff HEAD` — CUMULATIVE — so applying snapshots 0..k-1
        # sequentially would double-apply and reject; the last snapshot alone carries the state.
        commands: list[str] = []

        class _Env:
            def execute(self, action, cwd=""):
                commands.append(action["command"])
                return {"output": "", "returncode": 0}

        monkeypatch.setattr(
            step_snapshots, "read_snapshots", lambda tid: {0: "d0", 1: "d1", 2: "d2"}
        )
        assert infer._restore_checkout(_Env(), "tid") is True
        assert len(commands) == 1
        assert "d2" in commands[0]  # only the LAST cumulative diff is applied
        assert "d0" not in commands[0] and "d1" not in commands[0]

    def test_apply_command_pipes_the_diff_via_heredoc(self, monkeypatch) -> None:
        captured: dict[str, str] = {}

        class _Env:
            def execute(self, action, cwd=""):
                captured["command"] = action["command"]
                return {"output": "", "returncode": 0}

        monkeypatch.setattr(step_snapshots, "read_snapshots", lambda tid: {0: "diff-body"})
        assert infer._restore_checkout(_Env(), "tid") is True
        assert f"git -C {step_snapshots.TESTBED} apply - <<'" in captured["command"]
        assert "diff-body" in captured["command"]
        assert infer._RESUME_PATCH_DELIMITER in captured["command"]

    def test_missing_snapshots_is_a_failure_not_a_clean_apply(self, monkeypatch) -> None:
        monkeypatch.setattr(step_snapshots, "read_snapshots", lambda tid: {})
        assert infer._restore_checkout(object(), "tid") is False

    def test_patch_apply_failure_is_a_failure(self, monkeypatch) -> None:
        class _Env:
            def execute(self, action, cwd=""):
                return {"output": "patch does not apply", "returncode": 1}

        monkeypatch.setattr(step_snapshots, "read_snapshots", lambda tid: {0: "d"})
        assert infer._restore_checkout(_Env(), "tid") is False

    def test_empty_last_snapshot_is_a_reconstruction_failure(self, monkeypatch) -> None:
        # An empty `git diff HEAD` does NOT prove the tree is at base state: an agent that ran
        # `git add -A && git commit` has moved HEAD onto its own work, so the capture is 0
        # bytes while the work sits on disk. Empty is "we cannot tell", and a resume that
        # restores nothing yet reports a trustworthy environment would continue a PAID
        # conversation against a base checkout. Refuse; the caller falls back to a fresh run.
        executed: list[str] = []

        class _Env:
            def execute(self, action, cwd=""):
                executed.append(action["command"])
                return {"output": "", "returncode": 0}

        monkeypatch.setattr(step_snapshots, "read_snapshots", lambda tid: {0: "", 1: ""})
        assert infer._restore_checkout(_Env(), "tid") is False
        assert executed == []  # nothing applied, and nothing claimed restored

    def test_refuses_snapshot_indexes_the_conversation_cannot_justify(self, monkeypatch) -> None:
        # The stale-state shape from the review chain: the loaded conversation has 10 assistant
        # turns (0..9 justify), but the snapshot dir still holds run A's files up to step 39. The
        # guard must refuse — a cumulative diff the loaded conversation never produced must NEVER
        # be applied, and the fallback to a fresh run fires instead.
        commands: list[str] = []

        class _Env:
            def execute(self, action, cwd=""):
                commands.append(action["command"])
                return {"output": "", "returncode": 0}

        monkeypatch.setattr(step_snapshots, "read_snapshots", lambda tid: {0: "d0", 39: "d39"})
        assert infer._restore_checkout(_Env(), "tid", loaded_turn_count=10) is False
        assert commands == []  # refused BEFORE any apply: no diff ever touched the tree

    def test_allows_indexes_within_the_loaded_conversation(self, monkeypatch) -> None:
        # A conversation of 10 turns can justify snapshot indices 0..9 — its own last cumulative
        # diff (index 9) is the correct reconstruction target and must be applied.
        commands: list[str] = []

        class _Env:
            def execute(self, action, cwd=""):
                commands.append(action["command"])
                return {"output": "", "returncode": 0}

        monkeypatch.setattr(step_snapshots, "read_snapshots", lambda tid: {0: "d0", 9: "d9"})
        assert infer._restore_checkout(_Env(), "tid", loaded_turn_count=10) is True
        assert len(commands) == 1 and "d9" in commands[0]

    def test_invoke_scaffold_falls_back_to_a_fresh_run(self, monkeypatch) -> None:
        # Reconstruction failure must NEVER poison the cell: the first (resume) attempt raises
        # ResumeFallbackError, the second attempt is a fresh start at the FULL step budget.
        _ScaffoldPatches.apply(monkeypatch)
        monkeypatch.setattr(config, "resume_enabled", lambda: True)
        monkeypatch.setattr(
            infer, "_resume_messages", lambda tid: _resumable_conversation(n_assistant=2)
        )
        attempts: list[dict[str, object]] = []

        def _attempt(*a, **k):
            attempts.append({"step_limit": k["step_limit"], "resume": k["resume"]})
            if k["resume"] is not None:
                raise infer.ResumeFallbackError("reconstruction failed")
            return infer.AgentPatch(patch="", in_tok=0, out_tok=0, calls=0, cost=0.0)

        monkeypatch.setattr(infer, "_invoke_scaffold_attempt", _attempt)
        infer._invoke_scaffold(_spec(), "m", "mini-swe-agent", step_limit=150)
        assert len(attempts) == 2
        assert attempts[0]["resume"] is not None
        assert attempts[1]["resume"] is None  # retried as a clean fresh run
        assert attempts[1]["step_limit"] == 150  # full budget restored on the fallback


class TestResumeSnapshotContinuation:
    """The recorder keeps indexing from where the loaded history ends, and captures append."""

    def test_recorder_continues_at_the_seeded_index(self) -> None:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded

        class _Model:
            config: Any = {}

            def __init__(self) -> None:
                self.calls = 0

            def format_message(self, **kwargs: Any) -> dict[str, Any]:
                return {
                    "role": kwargs.get("role"),
                    "content": kwargs.get("content"),
                    "extra": kwargs.get("extra") or {},
                }

            def format_observation_messages(self, message, outputs, template_vars=None):
                return [_tool()]

            def query(self, messages, **kwargs):
                self.calls += 1
                if self.calls >= 2:
                    raise LimitsExceeded(
                        {
                            "role": "exit",
                            "content": "LimitsExceeded",
                            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                        }
                    )
                return _assistant("new")

            def get_template_vars(self, **kwargs):
                return {}

            def serialize(self):
                return {"model": {}}

        class _Env:
            config: Any = {}

            def get_template_vars(self, **kwargs):
                return {}

            def execute(self, action, cwd=""):
                return {"output": "", "returncode": 0}

            def serialize(self):
                return {"environment": {}}

        agent = DefaultAgent(
            _Model(),
            _Env(),
            system_template="sys {{task}}",
            instance_template="task {{task}}",
            step_limit=2,
        )
        agent.messages = [dict(m) for m in _resumable_conversation(n_assistant=2)]
        recorder = infer._attach_snapshot_recorder(agent, _Env())
        infer._resume_run(
            agent, "t", format_error_cls=FormatError, interrupt_cls=InterruptAgentFlow
        )
        assert sorted(recorder.snapshots) == [2]  # continues after the 2 seeded assistant turns

    def test_capture_appends_snapshots_and_counts_all_on_disk(self, monkeypatch, tmp_path) -> None:
        # A resumed run's write must NOT wipe the prior run's diffs, and the committed header
        # count must equal what the offline replay finds (prior + tail) — otherwise it refuses.
        from benchmark.escalation import live_capture, schema

        snap_root = tmp_path / "snap"
        real_write = step_snapshots.write_snapshots
        real_read = step_snapshots.read_snapshots
        monkeypatch.setattr(
            step_snapshots,
            "write_snapshots",
            lambda tid, snap, root=snap_root: real_write(tid, snap, root),
        )
        monkeypatch.setattr(
            step_snapshots, "read_snapshots", lambda tid, root=snap_root: real_read(tid, root)
        )
        trajectory_id = live_capture.make_trajectory_id("repo__repo-1", "m", "default")
        prior = {0: "old0", 1: "old1"}  # prior run's step files already on disk
        real_write(trajectory_id, prior, snap_root)
        prior_dir = step_snapshots.snapshot_dir(trajectory_id, snap_root)

        real = live_capture.capture_live_trajectory
        monkeypatch.setattr(
            live_capture,
            "capture_live_trajectory",
            lambda *a, **kw: real(*a, **{**kw, "out_dir": tmp_path}),
        )
        tail = {2: "new2", 3: "new3"}
        patch = infer.AgentPatch(
            patch="d",
            in_tok=0,
            out_tok=0,
            calls=0,
            cost=0.0,
            messages=_resumable_conversation(n_assistant=4),
            snapshots=tail,
            resumed=True,
        )
        infer._capture_escalation_trajectory(patch, "repo__repo-1", "m", "default", resolved=False)

        written = next(tmp_path.glob("*.jsonl"))
        assert schema.load_jsonl(written).header.snapshot_steps == 4  # 2 prior + 2 tail
        assert sorted(p.name for p in prior_dir.glob("step_*.diff")) == [
            "step_0000.diff",
            "step_0001.diff",
            "step_0002.diff",
            "step_0003.diff",
        ]

    def test_fresh_capture_counts_only_this_run(self, monkeypatch, tmp_path) -> None:
        from benchmark.escalation import live_capture, schema

        snap_root = tmp_path / "snap"
        real_write = step_snapshots.write_snapshots
        real_read = step_snapshots.read_snapshots
        monkeypatch.setattr(
            step_snapshots,
            "write_snapshots",
            lambda tid, snap, root=snap_root: real_write(tid, snap, root),
        )
        monkeypatch.setattr(
            step_snapshots, "read_snapshots", lambda tid, root=snap_root: real_read(tid, root)
        )
        real = live_capture.capture_live_trajectory
        monkeypatch.setattr(
            live_capture,
            "capture_live_trajectory",
            lambda *a, **kw: real(*a, **{**kw, "out_dir": tmp_path}),
        )
        patch = infer.AgentPatch(
            patch="d",
            in_tok=0,
            out_tok=0,
            calls=0,
            cost=0.0,
            messages=_resumable_conversation(n_assistant=1),
            snapshots={0: "d0"},
            resumed=False,
        )
        infer._capture_escalation_trajectory(patch, "repo__repo-1", "m", "default", resolved=False)
        written = next(tmp_path.glob("*.jsonl"))
        assert schema.load_jsonl(written).header.snapshot_steps == 1


class TestResumeUsageAccounting:
    """_sum_usage over the WHOLE cell: the seeded history plus the resumed tail."""

    def test_sum_usage_covers_loaded_and_new_turns(self) -> None:
        loaded = [_assistant("prior", cost=0.5), _assistant("prior2", cost=0.7)]
        new = [_assistant("new", cost=0.9)]
        in_tok, out_tok, calls, cost = infer._sum_usage([*loaded, *new])
        assert calls == 3
        assert cost == pytest.approx(2.1)
        assert in_tok == 30 and out_tok == 15

    def test_agent_messages_after_resume_are_the_whole_cell(self, tmp_path) -> None:
        # The resumed agent holds loaded + new; _sum_usage therefore reports the entire cell's
        # spend, not just the tail — the results row reflects everything the cell ever burned.
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded

        class _Model:
            config: Any = {}

            def __init__(self) -> None:
                self.calls = 0

            def format_message(self, **kwargs: Any) -> dict[str, Any]:
                return {
                    "role": kwargs.get("role"),
                    "content": kwargs.get("content"),
                    "extra": kwargs.get("extra") or {},
                }

            def format_observation_messages(self, message, outputs, template_vars=None):
                return [_tool()]

            def query(self, messages, **kwargs):
                self.calls += 1
                if self.calls >= 2:
                    raise LimitsExceeded(
                        {
                            "role": "exit",
                            "content": "LimitsExceeded",
                            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                        }
                    )
                return _assistant("new", cost=0.9)

            def get_template_vars(self, **kwargs):
                return {}

            def serialize(self):
                return {"model": {}}

        class _Env:
            config: Any = {}

            def get_template_vars(self, **kwargs):
                return {}

            def execute(self, action, cwd=""):
                return {"output": "out", "returncode": 0}

            def serialize(self):
                return {"environment": {}}

        loaded = [_assistant("p1", cost=0.5), _assistant("p2", cost=0.7)]
        agent = DefaultAgent(
            _Model(),
            _Env(),
            system_template="sys {{task}}",
            instance_template="task {{task}}",
            step_limit=2,
        )
        agent.messages = [dict(m) for m in loaded]
        infer._resume_run(
            agent, "t", format_error_cls=FormatError, interrupt_cls=InterruptAgentFlow
        )
        in_tok, out_tok, calls, cost = infer._sum_usage(agent.messages)
        assert calls == 3  # 2 loaded + 1 new
        assert cost == pytest.approx(2.1)


class TestStaleSnapshotChain:
    """A discarded run's stale snapshots must never rebuild a tree a later resume never had."""

    def test_chain_never_applies_a_discarded_runs_diff(self, monkeypatch, tmp_path) -> None:
        import minisweagent.models
        import minisweagent.run.benchmarks.swebench
        from minisweagent.exceptions import LimitsExceeded

        # The window the next `_invoke_scaffold` call simulates: the diff the fake `git diff HEAD`
        # reports, the step cap, and the tag stamped on commands the fake env executes.
        current: dict[str, Any] = {"diff": "RUN-A-diff", "cap": 40, "window": "A"}
        applied: list[str] = []

        class _Model:
            config: Any = {}

            def __init__(self, cap: int) -> None:
                self.calls = 0
                self.cap = cap

            def format_message(self, **kwargs: Any) -> dict[str, Any]:
                return {
                    "role": kwargs.get("role"),
                    "content": kwargs.get("content"),
                    "extra": kwargs.get("extra") or {},
                }

            def format_observation_messages(self, message, outputs, template_vars=None):
                return [_tool(f"obs {message.get('content')}")]

            def query(self, messages, **kwargs):
                self.calls += 1
                if self.calls > self.cap:  # the window's step budget is exhausted
                    raise LimitsExceeded(
                        {
                            "role": "exit",
                            "content": "LimitsExceeded",
                            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                        }
                    )
                return _assistant(f"{current['window']} step {self.calls}")

            def get_template_vars(self, **kwargs):
                return {}

            def serialize(self):
                return {"model": {}}

        class _Env:
            config: Any = {}

            def get_template_vars(self, **kwargs):
                return {}

            def serialize(self):
                return {"environment": {}}

            def cleanup(self) -> None:
                pass

            def execute(self, action, cwd=""):
                command = action["command"]
                if "diff HEAD" in command:
                    return {"output": str(current["diff"]), "returncode": 0}
                if "apply" in command:
                    applied.append(f"{current['window']}:{command}")
                    # A's cumulative diff no longer applies to a base checkout (that is the
                    # reconstruction failure that makes window B fall back to fresh); B's does.
                    rc = 1 if "RUN-A-diff" in command else 0
                    return {"output": "", "returncode": rc}
                return {"output": "out", "returncode": 0}

        snap_root = tmp_path / "snap"
        msg_root = tmp_path / "msg"
        real_write = step_snapshots.write_snapshots
        real_read = step_snapshots.read_snapshots
        # Force every snapshot/message-list lookup into tmp_path regardless of the root argument
        # callers pass (clear_trajectory_scratch passes the real defaults), so the run never touches
        # the repo's real scratch dirs.
        monkeypatch.setattr(
            step_snapshots, "snapshot_dir", lambda tid, root=None, _snap=snap_root: _snap / tid
        )
        monkeypatch.setattr(
            step_snapshots,
            "message_list_path",
            lambda tid, root=None, _msg=msg_root: _msg / f"{tid}.json",
        )
        monkeypatch.setattr(
            step_snapshots,
            "write_snapshots",
            lambda tid, snap, root=snap_root: real_write(tid, snap, root),
        )
        monkeypatch.setattr(
            step_snapshots,
            "read_snapshots",
            lambda tid, root=snap_root: real_read(tid, root),
        )
        _ScaffoldPatches.apply(monkeypatch)
        monkeypatch.setattr(config, "resume_enabled", lambda: True)
        monkeypatch.setattr(
            minisweagent.run.benchmarks.swebench,
            "get_sb_environment",
            lambda config, instance: _Env(),
        )
        monkeypatch.setattr(minisweagent.models, "get_model", lambda **kw: _Model(current["cap"]))

        def _window(diff: str, cap: int, tag: str) -> infer.AgentPatch:
            current.update(diff=diff, cap=cap, window=tag)
            applied.clear()
            return infer._invoke_scaffold(_spec(), "m", "mini-swe-agent", cost_limit=5.0)

        # Window A: a FRESH run (nothing on disk yet), 40 steps, snapshots 0..39 + 40-turn list.
        patch_a = _window("RUN-A-diff", cap=40, tag="A")
        assert patch_a.resumed is False
        assert len(patch_a.snapshots) == 40
        step_snapshots.write_snapshots("tid", patch_a.snapshots)
        assert infer._assistant_turn_count(infer._resume_messages("tid") or []) == 40
        assert len(list(step_snapshots.snapshot_dir("tid").glob("step_*.diff"))) == 40

        # Window B: resume of A is ATTEMPTED; A's step-39 diff fails to apply (reconstruction
        # failure) -> fallback FRESH run, which clears A's stale files, then runs 10 steps.
        patch_b = _window("RUN-B-diff", cap=10, tag="B")
        assert patch_b.resumed is False  # the final attempt was the fallback fresh run
        assert len(patch_b.snapshots) == 10
        step_snapshots.write_snapshots("tid", patch_b.snapshots)
        # THE CLEARING FIX: A's stale step_0010..0039 files are gone; only B's 0..9 remain, so a
        # later resume can only ever pair B's conversation with B's own snapshots.
        assert sorted(p.name for p in step_snapshots.snapshot_dir("tid").glob("step_*.diff")) == [
            f"step_{i:04d}.diff" for i in range(10)
        ]
        assert infer._assistant_turn_count(infer._resume_messages("tid") or []) == 10

        # Window C: resume of B must rebuild B's tree (its last cumulative diff), never A's stale
        # step-39 diff. B's snapshots (0..9) are all within B's 10-turn conversation, so the
        # consistency guard passes and reconstruction applies index 9.
        patch_c = _window("RUN-B-diff", cap=5, tag="C")
        assert patch_c.resumed is True  # continued B's conversation consistently
        c_apply = [cmd for cmd in applied if cmd.startswith("C:")]
        assert c_apply  # the resume DID reconstruct the checkout
        assert all("RUN-B-diff" in cmd for cmd in c_apply)  # applied B's own cumulative diff
        assert not any("RUN-A-diff" in cmd for cmd in c_apply)  # A's stale diff never applied
        # C continued at the seeded index: its own tail starts where B's 10 turns end.
        assert sorted(patch_c.snapshots) == list(range(10, 15))
        step_snapshots.write_snapshots("tid", patch_c.snapshots)
        assert len(step_snapshots.read_snapshots("tid")) == 15  # B's 0..9 + C's 10..14

    def test_fresh_start_clears_leftover_message_list(self, monkeypatch, tmp_path) -> None:
        # The message-list half of the clearing: a fresh run must also delete a leftover partial
        # conversation from a discarded run, so a later resume cannot pair it against stale state.
        snap_root = tmp_path / "snap"
        msg_root = tmp_path / "msg"
        real_write = step_snapshots.write_snapshots
        monkeypatch.setattr(
            step_snapshots, "snapshot_dir", lambda tid, root=None, _snap=snap_root: _snap / tid
        )
        monkeypatch.setattr(
            step_snapshots,
            "message_list_path",
            lambda tid, root=None, _msg=msg_root: _msg / f"{tid}.json",
        )
        monkeypatch.setattr(
            step_snapshots,
            "write_snapshots",
            lambda tid, snap, root=snap_root: real_write(tid, snap, root),
        )
        _write_message_list(tmp_path, _resumable_conversation(n_assistant=4))
        step_snapshots.write_snapshots("tid", {0: "d0", 1: "d1", 2: "d2", 3: "d3"})
        step_snapshots.clear_trajectory_scratch("tid")
        assert not step_snapshots.snapshot_dir("tid").exists()
        assert not step_snapshots.message_list_path("tid").exists()


class TestResumeCostBudget:
    """Remaining USD budget = the declared cap minus what the loaded conversation already spent."""

    def test_cost_limit_reduced_by_loaded_spend(self) -> None:
        # 3 loaded turns at $1.00 each against a $4.00 cap leave $1.00.
        loaded = _resumable_conversation(n_assistant=3)
        for msg in loaded:
            if msg["role"] == "assistant":
                msg["extra"]["response"]["usage"]["cost"] = 1.0
        assert infer._resume_cost_limit(4.0, loaded) == pytest.approx(1.0)

    def test_exhausted_budget_floors_to_a_still_enabled_cap(self) -> None:
        # A cap of exactly 0.0 DISABLES the ceiling in mini-swe-agent 2.4.5
        # (agents/default.py: `0 < self.config.cost_limit <= self.cost`), so an exhausted
        # resume must floor to a positive value — never 0.
        loaded = _resumable_conversation(n_assistant=4)
        for msg in loaded:
            if msg["role"] == "assistant":
                msg["extra"]["response"]["usage"]["cost"] = 1.0
        remaining = infer._resume_cost_limit(4.0, loaded)
        assert 0.0 < remaining <= infer._MIN_RESUME_COST_LIMIT

    def test_zero_loaded_turns_keeps_full_budget(self) -> None:
        assert infer._resume_cost_limit(4.0, []) == pytest.approx(4.0)

    def test_a_disabled_cap_is_left_disabled(self) -> None:
        # cost_limit <= 0 means "no ceiling" upstream; a resume must not invent one.
        assert infer._resume_cost_limit(0.0, _resumable_conversation(n_assistant=3)) == 0.0

    def test_resumed_attempt_overlays_the_reduced_cap_and_the_full_timeout(
        self, monkeypatch
    ) -> None:
        # The SPEND defect end-to-end: the overlay a resumed cell hands the scaffold must carry
        # the REDUCED cost cap (a fresh DefaultAgent restarts self.cost at 0.0), the reduced step
        # budget, and the UNCHANGED wall-clock timeout (a fresh process needs a full window).
        captured: dict[str, Any] = {}

        class _StopError(Exception):
            pass

        def _overlay(model_string, model_kwargs, **kwargs):
            captured.update(kwargs)
            raise _StopError

        monkeypatch.setattr(infer, "_scaffold_config_overlay", _overlay)
        monkeypatch.setattr(infer, "_scaffold_model_kwargs", lambda *a: {})
        loaded = _resumable_conversation(n_assistant=3)
        for msg in loaded:
            if msg["role"] == "assistant":
                msg["extra"]["response"]["usage"]["cost"] = 1.0
        with pytest.raises(_StopError):
            infer._invoke_scaffold_attempt(
                {"instance_id": "iid", "problem_statement": "ps"},
                "model-string",
                {},
                "tid",
                "m",
                "default",
                timeout=1800,
                step_limit=150,
                cost_limit=4.0,
                resume=loaded,
            )
        assert captured["cost_limit"] == pytest.approx(1.0)
        assert captured["step_limit"] == 147
        assert captured["timeout"] == 1800


class TestResumeSnapshotStalenessIsTwoSided:
    """The snapshot set must match the loaded conversation EXACTLY — too old is as bad as new."""

    def test_refuses_snapshots_that_are_too_old(self, monkeypatch) -> None:
        # {0, 5} against a 40-turn conversation: applying step 5's cumulative diff would resume a
        # 40-turn conversation onto a tree 34 turns stale.
        commands: list[str] = []

        class _Env:
            def execute(self, action, cwd=""):
                commands.append(action["command"])
                return {"output": "", "returncode": 0}

        monkeypatch.setattr(step_snapshots, "read_snapshots", lambda tid: {0: "d0", 5: "d5"})
        assert infer._restore_checkout(_Env(), "tid", loaded_turn_count=40) is False
        assert commands == []
