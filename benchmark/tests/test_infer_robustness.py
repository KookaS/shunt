"""Live-collection robustness: permanent-error fast-fail + hard wall-clock backstop.

All stubbed (no live/paid/Docker): permanent 4xx aborts fast while 429 still retries, and
``agent.run`` is watchdog-bounded so an abandoned cell is recorded and collection continues.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Final

import litellm
import pytest
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.retry import retry

from benchmark import config
from benchmark.routing import censoring
from benchmark.runner import image_version, infer, run_matrix, swebench_harness, swebench_specs

_LOG: Final = logging.getLogger("test-infer-robustness")


class _FakeModel:
    """Stand-in carrying mini-swe-agent's default abort list (what get_model produces)."""

    def __init__(self) -> None:
        self.abort_exceptions = list(LitellmModel.abort_exceptions)


def _cpv() -> litellm.exceptions.ContentPolicyViolationError:
    return litellm.exceptions.ContentPolicyViolationError(
        message="your prompt was flagged as potentially violating our usage policy",
        model="gpt-5-mini",
        llm_provider="openai",
    )


def _rate_limit() -> litellm.exceptions.RateLimitError:
    return litellm.exceptions.RateLimitError(message="429", model="m", llm_provider="openai")


def _auth() -> litellm.exceptions.AuthenticationError:
    return litellm.exceptions.AuthenticationError(
        message="invalid api key", model="m", llm_provider="deepseek"
    )


class TestErrorTaxonomy:
    """Three classes: model-refusal (permanent) vs API-unusable (systemic) vs transient."""

    def test_auth_error_is_api_unusable(self) -> None:
        assert infer._is_api_unusable(_auth())  # invalid/empty key → API unusable

    def test_no_balance_message_is_api_unusable(self) -> None:
        # A provider that returns insufficient-balance as a generic body (Requesty/DeepSeek 402)
        # is still classified by message, not just by typed exception.
        for msg in ("Error 402: Insufficient Balance", "You exceeded your current quota"):
            assert infer._is_api_unusable(Exception(msg)), msg

    def test_rate_limit_is_not_api_unusable(self) -> None:
        assert not infer._is_api_unusable(_rate_limit())  # plain 429 is transient, not unusable

    def test_content_policy_is_not_api_unusable(self) -> None:
        assert not infer._is_api_unusable(_cpv())  # a refusal is a model failure, not systemic

    def test_reraise_auth_becomes_api_unusable(self) -> None:
        with pytest.raises(infer.ApiUnusableError):
            infer._reraise_classified("iid", "m", _auth())

    def test_reraise_content_policy_becomes_permanent(self) -> None:
        with pytest.raises(infer.PermanentModelError):
            infer._reraise_classified("iid", "m", _cpv())

    def test_reraise_unknown_error_propagates_unchanged(self) -> None:
        # An unclassified error is NOT swallowed into a fake failure — it re-raises as-is.
        with pytest.raises(ValueError):
            infer._reraise_classified("iid", "m", ValueError("something odd"))

    def test_harden_adds_auth_to_abort_list(self) -> None:
        model = _FakeModel()
        infer._harden_model_retries(model)
        # A dead key aborts on the first call (retrying it is pointless), like content-policy.
        assert litellm.exceptions.AuthenticationError in model.abort_exceptions
        assert litellm.exceptions.RateLimitError not in model.abort_exceptions  # still retried


class TestHardenRetries:
    """Permanent 4xx becomes non-retryable; transient stays retryable."""

    def test_adds_bad_request_without_mutating_class_list(self) -> None:
        model = _FakeModel()
        before = list(LitellmModel.abort_exceptions)
        infer._harden_model_retries(model)
        assert litellm.exceptions.BadRequestError in model.abort_exceptions
        # The shared class attribute must be untouched (per-instance change only).
        assert LitellmModel.abort_exceptions == before
        # A transient class must NOT be swept into the abort list.
        assert litellm.exceptions.RateLimitError not in model.abort_exceptions

    def test_no_attr_is_a_noop(self) -> None:
        infer._harden_model_retries(object())  # must not raise

    def test_content_policy_not_retried_after_hardening(self, monkeypatch) -> None:
        monkeypatch.setattr(time, "sleep", lambda *_: None)  # never actually back off
        model = _FakeModel()
        infer._harden_model_retries(model)
        calls = {"n": 0}
        with pytest.raises(litellm.exceptions.ContentPolicyViolationError):
            for attempt in retry(logger=_LOG, abort_exceptions=model.abort_exceptions):
                with attempt:
                    calls["n"] += 1
                    raise _cpv()
        assert calls["n"] == 1  # aborted on the first call — the whole point of the fix

    def test_default_list_would_retry_content_policy(self, monkeypatch) -> None:
        # Contrast: the un-hardened default IS the bug — the same error retries many times.
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "5")
        calls = {"n": 0}
        with pytest.raises(litellm.exceptions.ContentPolicyViolationError):
            for attempt in retry(logger=_LOG, abort_exceptions=list(LitellmModel.abort_exceptions)):
                with attempt:
                    calls["n"] += 1
                    raise _cpv()
        assert calls["n"] == 5

    def test_transient_still_retried_then_succeeds(self, monkeypatch) -> None:
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        model = _FakeModel()
        infer._harden_model_retries(model)
        calls = {"n": 0}
        result = None
        for attempt in retry(logger=_LOG, abort_exceptions=model.abort_exceptions):
            with attempt:
                calls["n"] += 1
                if calls["n"] < 3:
                    raise _rate_limit()
                result = "ok"
        assert result == "ok"
        assert calls["n"] == 3  # retried twice, then succeeded — retries preserved


class _FakeEnv:
    def __init__(self) -> None:
        self.reaped = 0

    def cleanup(self) -> None:
        self.reaped += 1


class TestRunAgentBounded:
    """agent.run is bounded by a thread-safe wall clock; the container is reaped on timeout."""

    def test_returns_result_on_success(self) -> None:
        class _Agent:
            def run(self, task: str) -> dict[str, Any]:
                return {"submission": "diff --git a b"}

        env = _FakeEnv()
        assert infer._run_agent_bounded(_Agent(), "t", env, timeout=5) == {
            "submission": "diff --git a b"
        }
        assert env.reaped == 0

    def test_times_out_reaps_and_raises(self) -> None:
        stop = threading.Event()

        class _Agent:
            def run(self, task: str) -> dict[str, Any]:
                stop.wait(30)  # blocks past the tiny timeout; released in finally
                return {}

        env = _FakeEnv()
        try:
            with pytest.raises(infer.AgentRunTimeoutError):
                infer._run_agent_bounded(_Agent(), "t", env, timeout=1)
            assert env.reaped == 1  # container reaped exactly once
        finally:
            stop.set()  # let the abandoned worker exit promptly


def _spec() -> Any:
    class _Spec:
        instance_id = "psf__requests-1142"
        image_ref = "swebench/x:latest"

    return _Spec()


class TestRunLiveCellRecordsFailures:
    """A permanent error / timeout yields a failed cell, never a raise that stops the run."""

    def _patch_common(self, monkeypatch, tmp_path) -> dict[str, int]:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        monkeypatch.setattr(swebench_specs, "load_spec", lambda iid: _spec())
        calls = {"harness": 0}

        def _no_harness(*a, **k):
            calls["harness"] += 1  # must never fire: we fail before the grader
            raise AssertionError("harness must not run for an errored cell")

        monkeypatch.setattr(swebench_harness, "run_harness", _no_harness)
        return calls

    def test_permanent_error_records_failed_cell(self, monkeypatch, tmp_path) -> None:
        calls = self._patch_common(monkeypatch, tmp_path)

        def _raise(*a, **k):
            raise infer.PermanentModelError("content policy")

        monkeypatch.setattr(infer, "generate_patch_live", _raise)
        out = infer.run_live_cell("psf__requests-1142", "m", work_dir=tmp_path, run_id="r")
        assert out["pass"] is False
        assert out["timeout_flag"] is False
        assert calls["harness"] == 0

    def test_timeout_records_timeout_cell(self, monkeypatch, tmp_path) -> None:
        calls = self._patch_common(monkeypatch, tmp_path)

        def _raise(*a, **k):
            raise infer.AgentRunTimeoutError("wall clock")

        monkeypatch.setattr(infer, "generate_patch_live", _raise)
        out = infer.run_live_cell("psf__requests-1142", "m", work_dir=tmp_path, run_id="r")
        assert out["pass"] is False
        assert out["timeout_flag"] is True
        assert calls["harness"] == 0

    def test_api_unusable_propagates_never_records_fake_failure(
        self, monkeypatch, tmp_path
    ) -> None:
        # THE motivating bug: a dead/empty-balance API must NOT be recorded as a cell pass=False
        # (that fabricates a failure and marches the ladder through every challenge). It PROPAGATES
        # so run_matrix aborts the run — contrast the PermanentModelError case above (recorded).
        self._patch_common(monkeypatch, tmp_path)

        def _raise(*a, **k):
            raise infer.ApiUnusableError("no balance")

        monkeypatch.setattr(infer, "generate_patch_live", _raise)
        with pytest.raises(infer.ApiUnusableError):
            infer.run_live_cell("psf__requests-1142", "m", work_dir=tmp_path, run_id="r")


class TestStepLimitAndBounds:
    """FIX A/B: step_limit is the primary bound; wall = passed timeout; watchdog > wall."""

    def test_overlay_carries_step_limit_and_wall(self) -> None:
        overlay = infer._scaffold_config_overlay("m", {}, timeout=1800, step_limit=70)
        assert overlay["agent"]["step_limit"] == 70  # PRIMARY model-speed-agnostic bound
        assert overlay["agent"]["wall_time_limit_seconds"] == 1800  # internal graceful wall

    def test_default_step_limit_matches_config(self) -> None:
        config.load("benchmark/benchmark.yaml")
        assert config.live_step_limit() == infer._DEFAULT_STEP_LIMIT

    def test_external_watchdog_strictly_greater_than_wall(self) -> None:
        # THE ordering invariant: the graceful internal wall must fire BEFORE the hard watchdog,
        # so a normal slow run terminates gracefully (usage captured), never gets thread-abandoned.
        for wall in (600, 1800, 3600):
            assert infer._external_watchdog_s(wall) > wall

    def test_generate_patch_live_forwards_step_limit_and_timeout(self, monkeypatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        captured: dict[str, int] = {}

        def _cap(spec, model, scaffold, arm, *, timeout, step_limit):
            captured.update(timeout=timeout, step_limit=step_limit)
            return infer.AgentPatch(patch="", in_tok=0, out_tok=0, calls=0, cost=0.0)

        monkeypatch.setattr(infer, "_invoke_scaffold", _cap)
        infer.generate_patch_live(_spec(), "m", timeout=123, step_limit=88)
        assert captured == {"timeout": 123, "step_limit": 88}

    def test_run_live_cell_forwards_step_limit(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        monkeypatch.setattr(swebench_specs, "load_spec", lambda iid: _spec())
        captured: dict[str, int] = {}

        def _cap(spec, model, *, arm, timeout, step_limit):
            captured.update(timeout=timeout, step_limit=step_limit)
            raise infer.AgentRunTimeoutError("stop before harness")

        monkeypatch.setattr(infer, "generate_patch_live", _cap)
        infer.run_live_cell("iid", "m", work_dir=tmp_path, run_id="r", timeout=222, step_limit=99)
        assert captured == {"timeout": 222, "step_limit": 99}

    def test_run_live_cells_threads_step_limit(self, monkeypatch) -> None:
        seen: dict[str, int | None] = {}

        def fake_cell(cid, model, **kw):
            seen["step_limit"] = kw.get("step_limit")
            return {"pass": True, "in_tok": 1, "out_tok": 1, "calls": 1, "real_cost": 0.0}

        monkeypatch.setattr(infer, "run_live_cell", fake_cell)
        monkeypatch.setattr(config, "models_missing_cache", lambda *a, **k: [])
        run_matrix.run_live_cells(
            [("t", "m", "default")],
            {},
            {"t": "h"},
            {"m": "v"},
            timeout=10,
            verbose=False,
            workers=1,
            step_limit=77,
        )
        assert seen["step_limit"] == 77


def _usage_msg(cost: float, in_tok: int = 10, out_tok: int = 5) -> dict[str, Any]:
    return {
        "extra": {
            "response": {
                "usage": {
                    "cost": cost,
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                }
            }
        }
    }


class TestPartialUsageHarvest:
    """FIX C: a hard-abandoned cell records the REAL partial spend, never a fabricated $0."""

    def test_bounded_timeout_harvests_partial_usage(self) -> None:
        stop = threading.Event()

        class _Agent:
            def __init__(self) -> None:
                self.messages = [_usage_msg(0.5), _usage_msg(0.7)]

            def run(self, task: str) -> dict[str, Any]:
                stop.wait(30)  # wedge past the tiny timeout
                return {}

        env = _FakeEnv()
        try:
            with pytest.raises(infer.AgentRunTimeoutError) as ei:
                infer._run_agent_bounded(_Agent(), "t", env, timeout=1)
            exc = ei.value
            assert exc.cost == pytest.approx(1.2)  # 0.5 + 0.7 from the in-progress messages
            assert exc.calls == 2
            assert exc.in_tok == 20 and exc.out_tok == 10
            assert env.reaped == 1
        finally:
            stop.set()

    def test_run_live_cell_timeout_records_partial_cost(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        monkeypatch.setattr(swebench_specs, "load_spec", lambda iid: _spec())

        def _raise(*a, **k):
            raise infer.AgentRunTimeoutError("wall clock", in_tok=20, out_tok=10, calls=2, cost=1.2)

        monkeypatch.setattr(infer, "generate_patch_live", _raise)
        out = infer.run_live_cell("iid", "m", work_dir=tmp_path, run_id="r")
        assert out["timeout_flag"] is True
        assert out["pass"] is False
        assert out["real_cost"] == pytest.approx(1.2)  # NOT zero — real spend recorded
        assert out["calls"] == 2

    def test_errored_outcome_defaults_to_zero_usage(self) -> None:
        # A permanent-model error (no harvestable messages) still records a clean $0 row.
        # stop_reason drives timeout_flag: an unsolved (uncensored) fail has timeout_flag False.
        out = infer._errored_outcome("iid", "m", stop_reason=censoring.UNSOLVED)
        assert out["real_cost"] == 0.0 and out["calls"] == 0
        assert out["timeout_flag"] is False
        assert out["stop_reason"] == censoring.UNSOLVED


class TestStopReasonOnSuccessPath:
    """run_live_cell maps (harness resolved + scaffold exit_status) to the right stop_reason."""

    def _run(self, monkeypatch, tmp_path, *, resolved: bool, exit_status: str) -> dict[str, object]:
        monkeypatch.setattr(swebench_specs, "load_spec", lambda iid: _spec())
        monkeypatch.setattr(
            infer,
            "generate_patch_live",
            lambda spec, model, arm="default", timeout=None, step_limit=None: infer.AgentPatch(
                patch="diff", in_tok=1, out_tok=1, calls=1, cost=0.1, exit_status=exit_status
            ),
        )
        report = tmp_path / "r.json"
        monkeypatch.setattr(
            swebench_harness,
            "run_harness",
            lambda **kw: swebench_harness.HarnessResult(
                {"psf__requests-1142": resolved}, report, {}, 0
            ),
        )
        monkeypatch.setattr(infer, "_capture_escalation_trajectory", lambda *a, **k: None)
        monkeypatch.setattr(image_version, "used_image_digest", lambda *a, **k: "")
        return infer.run_live_cell("psf__requests-1142", "m", work_dir=tmp_path, run_id="t")

    def test_solved(self, monkeypatch, tmp_path) -> None:
        out = self._run(monkeypatch, tmp_path, resolved=True, exit_status="Submitted")
        assert out["stop_reason"] == censoring.SOLVED
        assert out["pass"] is True
        assert out["timeout_flag"] is False

    def test_unsolved_when_finished_but_not_resolved(self, monkeypatch, tmp_path) -> None:
        out = self._run(monkeypatch, tmp_path, resolved=False, exit_status="Submitted")
        assert out["stop_reason"] == censoring.UNSOLVED
        assert out["timeout_flag"] is False

    def test_step_limit_is_censored(self, monkeypatch, tmp_path) -> None:
        # LimitsExceeded (step/cost limit) → censored step_limit; timeout_flag stays False.
        out = self._run(monkeypatch, tmp_path, resolved=False, exit_status="LimitsExceeded")
        assert out["stop_reason"] == censoring.STEP_LIMIT
        assert out["timeout_flag"] is False

    def test_wall_limit_is_censored_timeout(self, monkeypatch, tmp_path) -> None:
        # TimeExceeded → censored wall_limit; timeout_flag True (back-compat).
        out = self._run(monkeypatch, tmp_path, resolved=False, exit_status="TimeExceeded")
        assert out["stop_reason"] == censoring.WALL_LIMIT
        assert out["timeout_flag"] is True

    def test_resolved_wins_over_a_limit_exit_status(self, monkeypatch, tmp_path) -> None:
        # If the harness resolved, the cell is solved even if the scaffold recorded a limit exit.
        out = self._run(monkeypatch, tmp_path, resolved=True, exit_status="LimitsExceeded")
        assert out["stop_reason"] == censoring.SOLVED

    def test_abandoned_from_watchdog(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(swebench_specs, "load_spec", lambda iid: _spec())

        def _raise(*a, **k):
            raise infer.AgentRunTimeoutError("wall clock", cost=0.5, calls=3)

        monkeypatch.setattr(infer, "generate_patch_live", _raise)
        out = infer.run_live_cell("iid", "m", work_dir=tmp_path, run_id="r")
        assert out["stop_reason"] == censoring.ABANDONED
        assert out["timeout_flag"] is True
        assert out["real_cost"] == pytest.approx(0.5)


class TestCollectionContinuesPastFailure:
    """A failed cell is recorded (pass=False) and the batch keeps running the others."""

    def test_failed_cell_recorded_and_others_run(self, monkeypatch) -> None:
        hashes = {f"repo__task-{i}": "h" for i in range(1, 4)}
        versions = {"m": "v"}
        cells = [(f"repo__task-{i}", "m", "default") for i in range(1, 4)]

        def fake_cell(cid, model, **kw):
            if cid == "repo__task-2":  # the wedged cell, now abandoned as a timeout
                return infer._errored_outcome(cid, model, stop_reason=censoring.ABANDONED)
            return {"pass": True, "in_tok": 1, "out_tok": 1, "calls": 1, "real_cost": 0.0}

        monkeypatch.setattr(infer, "run_live_cell", fake_cell)
        monkeypatch.setattr(config, "models_missing_cache", lambda *a, **k: [])
        rows = run_matrix.run_live_cells(
            cells, {}, hashes, versions, timeout=10, verbose=False, workers=1
        )
        by_task = {r["challenge_id"]: r for r in rows}
        assert len(rows) == 3  # no cell lost — collection continued
        assert by_task["repo__task-2"]["pass"] is False
        assert by_task["repo__task-2"]["timeout_flag"] is True
        assert by_task["repo__task-1"]["pass"] is True
        assert by_task["repo__task-3"]["pass"] is True


def test_reraise_classified_redacts_secret_in_message() -> None:
    # A provider 401 quotes the key back; the reclassified exception must never carry it verbatim.
    secret = "sk-" + "SECRETKEY1234567890"
    exc = Exception(f"invalid api key {secret} was rejected")
    with pytest.raises(infer.ApiUnusableError) as excinfo:
        infer._reraise_classified("astropy__astropy-1", "kimi-k3", exc)
    message = str(excinfo.value)
    assert secret not in message
    assert "<redacted>" in message


def test_capture_records_the_snapshot_count_the_run_actually_produced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # The producer half of the unreplayable-trajectory guard. If the recorder never attached,
    # `patch.snapshots` is empty and the committed header must say 0 — the offline replay reads
    # that to tell "captured nothing, ever" from "this checkout lacks the gitignored scratch".
    from benchmark.escalation import live_capture, schema

    real = live_capture.capture_live_trajectory
    monkeypatch.setattr(
        live_capture,
        "capture_live_trajectory",
        lambda *a, **kw: real(*a, **{**kw, "out_dir": tmp_path}),
    )
    messages = [
        {"role": "assistant", "content": "ls", "extra": {"actions": [{"command": "ls"}]}},
        {"role": "tool", "content": "out", "extra": {"returncode": 0}},
    ]
    patch = infer.AgentPatch(
        patch="d", in_tok=0, out_tok=0, calls=0, cost=0.0, messages=messages, snapshots={}
    )
    infer._capture_escalation_trajectory(patch, "repo__repo-1", "m", "default", resolved=False)

    written = next(tmp_path.glob("*.jsonl"))
    assert schema.load_jsonl(written).header.snapshot_steps == 0
