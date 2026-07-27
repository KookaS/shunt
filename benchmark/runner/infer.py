"""Produce ``predictions.jsonl`` for the harness — gold (keyless) or live (gated).

gold emits each instance's dataset gold patch ($0 pipeline smoke); live runs one
fixed ``mini-swe-agent`` scaffold per (instance, model), key-gated so keyless never fabricates.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NoReturn

from benchmark import config
from benchmark.routing import censoring, integrity
from benchmark.runner import image_version, swebench_harness, swebench_specs
from shunt.proxy.redaction import redact_secrets

_LOG = logging.getLogger(__name__)

# Default INTERNAL graceful wall-clock ceiling (seconds) for one live agent run, passed as
# mini-swe-agent's own ``wall_time_limit_seconds``. Hitting it raises the scaffold's own
# ``LimitsExceeded`` (an ``InterruptAgentFlow`` subclass), so ``agent.run`` returns NORMALLY with
# ``agent.messages`` (and their real ``usage.cost``) intact — the cost is captured, not discarded.
# Generous by design: the PRIMARY per-cell bound is ``step_limit`` (model-speed-agnostic); the wall
# is only a secondary backstop for a slow or looping run.
_AGENT_WALL_LIMIT_S: Final[int] = 1800

# The EXTERNAL thread watchdog (``_run_agent_bounded``) fires this many seconds AFTER the internal
# wall. Because the internal wall is checked only BETWEEN steps, a single model call wedged in a
# retry loop or a network stall never returns to the step loop and the graceful limit never fires;
# the watchdog is the hard last resort for that. Keeping it strictly > the internal wall guarantees
# the graceful limit trips FIRST on a normal slow run, so a hard thread-abandon is rare.
_WATCHDOG_MARGIN_S: Final[int] = 300

# Default PRIMARY per-cell bound: the number of agent steps, independent of inference speed, so a
# slow model gets the SAME number of attempts as a fast one (wall-clock timing unfairly penalises
# slow-inference models). Default 70 is justified from the live trajectory corpus
# (benchmark/escalation/data/live): passing runs' step counts have p90=58 and p95=68, so 70 admits
# ~95% of observed passers with headroom while still bounding a runaway loop.
_DEFAULT_STEP_LIMIT: Final[int] = 70


def _external_watchdog_s(wall_limit_s: int) -> int:
    """External hard-watchdog ceiling — strictly greater than the internal graceful wall."""
    return wall_limit_s + _WATCHDOG_MARGIN_S


GOLD_MODEL_NAME: Final[str] = "gold"
LIVE_SCAFFOLD: Final[str] = "mini-swe-agent"

# Any of these present ⇒ live inference is permitted to attempt real model calls.
_KEY_ENV: Final[tuple[str, ...]] = (
    "DEEPSEEK_API_KEY",
    "REQUESTY_API_KEY",
    "XAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


class MissingApiKeysError(RuntimeError):
    """Raised when live inference is requested but no provider key is present."""


class HarnessInfraError(RuntimeError):
    """Raised when the harness fails to produce a report (Docker/image/timeout).

    An infra crash is NOT a model failure — the cell must stay MISSING and
    recompute, never cache as pass=False and poison the matrix/kill-gate.
    """


class PermanentModelError(RuntimeError):
    """A non-retryable model error (permanent 4xx — e.g. content-policy) during a live run.

    Deterministic in the prompt, so retrying only hangs the run. The cell is recorded as
    failed (pass=False) so the ladder escalates it, not left MISSING to recompute forever.
    """


class ApiUnusableError(RuntimeError):
    """A SYSTEMIC API failure no retry can fix (dead/empty key, no balance, provider down).

    NEVER a cell ``pass=False`` (that fabricates a fake fail): the cell stays MISSING, run aborts.
    """


class AgentRunTimeoutError(RuntimeError):
    """The live agent run hit the external hard watchdog and was abandoned.

    Carries the partial usage harvested from ``agent.messages`` before the reap, so the cell
    records the REAL spend it already incurred instead of a fabricated $0.
    """

    def __init__(
        self, message: str, *, in_tok: int = 0, out_tok: int = 0, calls: int = 0, cost: float = 0.0
    ) -> None:
        super().__init__(message)
        self.in_tok = in_tok
        self.out_tok = out_tok
        self.calls = calls
        self.cost = cost


def has_api_keys(env: dict[str, str] | None = None) -> bool:
    """True iff at least one known provider API key is set in the environment."""
    source = env if env is not None else os.environ
    return any(source.get(k) for k in _KEY_ENV)


def prediction_line(instance_id: str, model_name: str, patch: str) -> dict[str, str]:
    """One harness prediction record: instance + model + unified-diff patch."""
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def write_predictions(predictions: list[dict[str, str]], path: Path) -> Path:
    """Write predictions as JSONL (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    return path


# ---------------------------------------------------------------------------
# gold mode — $0 smoke, no keys
# ---------------------------------------------------------------------------


def gold_patches(instance_ids: list[str]) -> dict[str, str]:
    """Pull the gold ``patch`` for each instance id from the HF Verified dataset."""
    from datasets import load_dataset

    ds = load_dataset(swebench_specs.DATASET_NAME, split=swebench_specs.DATASET_SPLIT)
    wanted = set(instance_ids)
    found = {str(r["instance_id"]): str(r["patch"]) for r in ds if str(r["instance_id"]) in wanted}
    missing = wanted - found.keys()
    if missing:
        raise KeyError(f"instance ids not in dataset: {sorted(missing)}")
    return found


def build_gold_predictions(instance_ids: list[str]) -> list[dict[str, str]]:
    """Gold predictions: model = ``gold``, patch = the instance's gold diff."""
    patches = gold_patches(instance_ids)
    return [prediction_line(iid, GOLD_MODEL_NAME, patches[iid]) for iid in instance_ids]


# ---------------------------------------------------------------------------
# live mode — gated on API keys (built + unit-tested, not run here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentPatch:
    """A patch produced by the live agent scaffold, with measured usage + per-turn messages."""

    patch: str
    in_tok: int
    out_tok: int
    calls: int
    cost: float
    # mini-swe-agent's exit_status from the run() return (e.g. "Submitted", "LimitsExceeded",
    # "TimeExceeded"), the real signal mapping a non-resolved cell to its censored stop_reason.
    # Empty on any non-real path; run_live_cell maps it via routing.censoring.
    exit_status: str = ""
    # The full per-turn trajectory the scaffold held after run() — carried (never consumed by
    # routing) so a live run can also persist a real escalation-detector trajectory. Empty on any
    # non-real path, which the capture gate treats as "write nothing".
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Per-step code snapshots (git diff of the checkout keyed by assistant-turn index), captured
    # observe-only during the run for offline verified-outcome replay. Empty when unavailable.
    snapshots: dict[int, str] = field(default_factory=dict)


def generate_patch_live(
    spec: swebench_specs.SwebenchSpec,
    model: str,
    scaffold: str = LIVE_SCAFFOLD,
    env: dict[str, str] | None = None,
    arm: str = integrity.DEFAULT_REASONING,
    timeout: int = _AGENT_WALL_LIMIT_S,
    step_limit: int = _DEFAULT_STEP_LIMIT,
) -> AgentPatch:
    """Run the fixed agent scaffold on one instance/model/arm to produce a patch.

    Gated: raises ``MissingApiKeysError`` without keys (keyless never fabricates);
    scaffold import is lazy so the wiring is unit-testable without it installed.
    """
    if not has_api_keys(env):
        raise MissingApiKeysError(
            f"live inference for {spec.instance_id}/{model} needs one of {_KEY_ENV}"
        )
    return _invoke_scaffold(spec, model, scaffold, arm, timeout=timeout, step_limit=step_limit)


def litellm_model_target(model: str) -> tuple[str, dict[str, Any]]:
    """Map an internal model alias to a litellm ``(model_string, model_kwargs)`` pair.

    Route, base_url, and key env var all come from the registry's provider row.
    """
    info = config.load_pricing().get(model)
    if not isinstance(info, dict):
        raise KeyError(f"model {model!r} not in the model registry")
    route = str(info["route"])
    # A provider with its own litellm prefix (e.g. `deepseek/`) is dialled by
    # litellm directly, which reads that provider's key from the env by its
    # canonical name. A generic `openai/` surface needs base_url + key passed.
    if not route.startswith("openai/"):
        return route, {}
    key_env = str(info["api_key_env_var"])
    key = os.environ.get(key_env)
    if not key:
        raise MissingApiKeysError(f"routing {model!r} via {info['provider']} needs {key_env}")
    return route, {"api_base": str(info["base_url"]), "api_key": key}


def _cheapest_enabled_model() -> str:
    """The cheapest enabled benchmark model (the $0-probe target for the preflight)."""
    models = config.enabled_models()
    if not models:
        raise MissingApiKeysError("no enabled models to preflight")
    return models[0]


def preflight_api_check(model: str | None = None) -> bool:
    """Prove the API key works with ONE minimal real completion, before any container spins up.

    A dead/empty key or no-balance error raises ``ApiUnusableError`` (refuse the run); a
    transient blip (rate-limit / 5xx) is inconclusive and returns True — never refuse over one.
    """
    import litellm  # noqa: PLC0415

    target = model or _cheapest_enabled_model()
    model_string, model_kwargs = litellm_model_target(target)
    try:
        litellm.completion(
            model=model_string,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            **model_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 (health probe: classify, never fabricate a result)
        if _is_api_unusable(exc):
            raise ApiUnusableError(
                f"preflight health check failed for {target}: "
                f"{type(exc).__name__}: {redact_secrets(str(exc))}"
            ) from exc
        _LOG.warning(
            "preflight for %s hit a transient error (not refusing): %s",
            target,
            redact_secrets(str(exc)),
        )
    return True


@functools.lru_cache(maxsize=1)
def _dataset_instances() -> dict[str, dict[str, Any]]:
    """All Verified rows keyed by instance id (loaded once; used for problem statements)."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(swebench_specs.DATASET_NAME, split=swebench_specs.DATASET_SPLIT)
    return {str(row["instance_id"]): dict(row) for row in ds}


def _load_instance(instance_id: str) -> dict[str, Any]:
    """The dataset row (problem_statement, image name, …) for one instance id."""
    instances = _dataset_instances()
    if instance_id not in instances:
        raise KeyError(f"instance {instance_id!r} not in {swebench_specs.DATASET_NAME}")
    return instances[instance_id]


def _call_cost(extra: dict[str, Any], usage: dict[str, Any]) -> float:
    """One call's real cost: the provider's cache-aware usage.cost, else litellm's estimate."""
    # Prefer what the provider actually charged (Requesty's usage.cost is cache-aware and
    # authoritative) over litellm's client-side static-table computation; litellm is only the
    # fallback when the provider reports no cost. This keeps the kill-gate's cost basis on the
    # real cache-aware figure, never a pricing table that can shadow it.
    provider_cost = float(usage.get("cost", 0.0) or 0.0)
    if provider_cost > 0.0:
        return provider_cost
    return float(extra.get("cost", 0.0) or 0.0)


def _sum_usage(messages: list[dict[str, Any]]) -> tuple[int, int, int, float]:
    """Sum (in_tok, out_tok, calls, cost) over an agent's assistant messages."""
    in_tok = out_tok = calls = 0
    cost = 0.0
    for msg in messages:
        extra = msg.get("extra") or {}
        response = extra.get("response")
        if not response:
            continue
        calls += 1
        usage = response.get("usage") or {}
        cost += _call_cost(extra, usage)
        in_tok += int(usage.get("prompt_tokens", 0) or 0)
        out_tok += int(usage.get("completion_tokens", 0) or 0)
    return in_tok, out_tok, calls, cost


def _scaffold_model_kwargs(
    model: str, arm: str, base: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    """Model kwargs for one live call: base ← litellm target ← reasoning-arm params.

    Arm overlay is last, so distinct arms bill distinct requests (the arm_sampling premise).
    """
    arm_params = config.arm_api_params(model, arm)
    # Registry arm `api` blobs are free dicts; refuse one that would clobber the routing
    # target's auth/identity keys (silent auth breakage on a paid call). Boundary check.
    clash = arm_params.keys() & {"api_base", "api_key", "model_name"}
    if clash:
        raise ValueError(
            f"reasoning arm {arm!r} of {model!r} sets reserved request key(s) {sorted(clash)}"
        )
    return {**base, **target, **arm_params}


def _permanent_model_errors() -> tuple[type[BaseException], ...]:
    """Litellm exception types that are permanent MODEL errors — retrying can never succeed."""
    # BadRequestError (HTTP 400) is the base of ContentPolicyViolationError and the other
    # client-side 400s, none fixable by a retry. Rate-limit (429), timeout (408) and 5xx are
    # deliberately NOT here, so genuinely transient failures still retry.
    import litellm  # noqa: PLC0415

    return (litellm.exceptions.BadRequestError,)


def _api_unusable_error_types() -> tuple[type[BaseException], ...]:
    """Litellm exception types that mean the API is UNUSABLE, not the model is bad."""
    # 401 auth / 403 forbidden / exhausted budget — all abort the run. Provider-down (5xx) is
    # deliberately NOT here: a single 5xx retries; a persistent one hits the consecutive catch-all.
    import litellm  # noqa: PLC0415

    return (
        litellm.exceptions.AuthenticationError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.BudgetExceededError,
    )


# Message fragments (case-folded) that mark a no-balance / no-quota / payment error even when
# the provider wraps it in a generic type (Requesty/DeepSeek return these as 402s or 429s with
# an insufficient-balance body rather than a typed AuthenticationError).
_API_UNUSABLE_SIGNATURES: Final[tuple[str, ...]] = (
    "insufficient balance",
    "insufficient_quota",
    "insufficient funds",
    "insufficient credit",
    "no balance",
    "not enough balance",
    "exceeded your current quota",
    "payment required",
    "402",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "no api key",
)


def _is_api_unusable(exc: BaseException) -> bool:
    """True iff an exception means the API itself is unusable (dead key / no balance)."""
    if isinstance(exc, _api_unusable_error_types()):
        return True
    text = str(exc).lower()
    return any(sig in text for sig in _API_UNUSABLE_SIGNATURES)


def _abort_error_types() -> tuple[type[BaseException], ...]:
    """Errors the scaffold must ABORT on (not retry): permanent model + API-unusable."""
    # Retrying either is futile — a permanent 4xx is deterministic, a dead key never heals —
    # so both go on the abort list, surfacing on the first call instead of burning the retry budget.
    return (*_permanent_model_errors(), *_api_unusable_error_types())


def _harden_model_retries(model: Any) -> None:
    """Make the scaffold model abort (not retry) on permanent 4xx + API-unusable errors."""
    # mini-swe-agent retries any exception NOT in model.abort_exceptions (tenacity, up to
    # MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT with exponential backoff), and its default list omits
    # ContentPolicyViolationError / AuthenticationError — so a deterministic content-policy block
    # or a dead key is retried until the cap, wedging the run. Assign a NEW instance list (never
    # .append onto the shared class attribute), so the change is per-cell.
    current = getattr(model, "abort_exceptions", None)
    if not isinstance(current, list):
        return
    extra = [e for e in _abort_error_types() if e not in current]
    if extra:
        model.abort_exceptions = [*current, *extra]


def _reraise_classified(instance_id: str, model: str, exc: Exception) -> NoReturn:
    """Reclassify a scaffold error into the taxonomy, else re-raise it unchanged.

    API-unusable is checked FIRST: a dead key / empty balance must abort the run, never be
    mistaken for a permanent model refusal and recorded as a fake ``pass=False`` cell.
    """
    if _is_api_unusable(exc):
        raise ApiUnusableError(
            f"API unusable for {instance_id}/{model}: "
            f"{type(exc).__name__}: {redact_secrets(str(exc))}"
        ) from exc
    if isinstance(exc, _permanent_model_errors()):
        raise PermanentModelError(
            f"permanent model error for {instance_id}/{model}: "
            f"{type(exc).__name__}: {redact_secrets(str(exc))}"
        ) from exc
    raise exc


def _reap_container(env: Any) -> None:
    """Stop and remove the cell's Docker container (best-effort; never raises)."""
    cleanup = getattr(env, "cleanup", None)
    if not callable(cleanup):
        return
    try:
        cleanup()
    except Exception:  # noqa: BLE001 (reaping is best-effort; a failure must not mask the timeout)
        _LOG.exception("failed to reap container after agent run was abandoned")


def _run_agent_bounded(agent: Any, task: str, env: Any, timeout: int) -> dict[str, Any]:
    """Run ``agent.run(task)`` under a hard wall-clock ceiling (thread-safe watchdog)."""
    # Cells run in worker threads (run_matrix's pool), so no SIGALRM: a single-worker future
    # bounds the call. On timeout the container is reaped, AgentRunTimeoutError is raised, and
    # the stuck worker is abandoned (shutdown(wait=False) — it cannot wedge the outer pool).
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-run")
    future = pool.submit(agent.run, task)
    try:
        result: dict[str, Any] = future.result(timeout=timeout)
    except FuturesTimeoutError as exc:
        # Belt-and-suspenders: harvest the partial usage the abandoned agent already accrued
        # (a snapshot copy, so the still-running thread's mutation can't blow up the sum) BEFORE
        # reaping, so the errored row records real spend instead of a fabricated $0.
        in_tok, out_tok, calls, cost = _sum_usage(list(getattr(agent, "messages", [])))
        _reap_container(env)
        pool.shutdown(wait=False, cancel_futures=True)
        raise AgentRunTimeoutError(
            f"agent run exceeded {timeout}s wall clock; container reaped, cell abandoned",
            in_tok=in_tok,
            out_tok=out_tok,
            calls=calls,
            cost=cost,
        ) from exc
    pool.shutdown(wait=False)
    return result


def _scaffold_config_overlay(
    model_string: str, model_kwargs: dict[str, Any], *, timeout: int, step_limit: int
) -> dict[str, Any]:
    """The mini-swe-agent config overlay for one live cell (merged over the swebench default)."""
    # step_limit is the PRIMARY model-speed-agnostic bound; wall_time_limit_seconds is a generous
    # graceful secondary backstop set from the PASSED timeout. Both trip the scaffold's own
    # InterruptAgentFlow, so agent.run returns with usage intact (the cost is captured, not lost).
    return {
        "agent": {"wall_time_limit_seconds": timeout, "step_limit": step_limit},
        "model": {
            "model_name": model_string,
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        },
        "environment": {"environment_class": "docker"},
    }


def _invoke_scaffold(
    spec: swebench_specs.SwebenchSpec,
    model: str,
    scaffold: str,  # noqa: ARG001 (kept for signature stability; only mini-swe-agent is wired)
    arm: str = integrity.DEFAULT_REASONING,
    timeout: int = _AGENT_WALL_LIMIT_S,
    step_limit: int = _DEFAULT_STEP_LIMIT,
) -> AgentPatch:
    """Invoke mini-swe-agent (v2) for one instance/model/arm (only reached when keys exist)."""
    from minisweagent.agents import get_agent  # noqa: PLC0415
    from minisweagent.config import builtin_config_dir, get_config_from_spec  # noqa: PLC0415
    from minisweagent.models import get_model  # noqa: PLC0415
    from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: PLC0415
    from minisweagent.utils.serialize import recursive_merge  # noqa: PLC0415

    instance = _load_instance(spec.instance_id)
    model_string, model_kwargs = litellm_model_target(model)
    default_config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
    base_kwargs = default_config.get("model", {}).get("model_kwargs", {})
    overlay = _scaffold_config_overlay(
        model_string,
        _scaffold_model_kwargs(model, arm, base_kwargs, model_kwargs),
        timeout=timeout,
        step_limit=step_limit,
    )
    merged = recursive_merge(default_config, overlay)
    env = get_sb_environment(merged, instance)
    model_obj = get_model(config=merged.get("model", {}))
    _harden_model_retries(model_obj)
    agent = get_agent(model_obj, env, merged.get("agent", {}), default_type="default")
    recorder = _attach_snapshot_recorder(agent, env)
    try:
        info = _run_agent_bounded(
            agent, instance["problem_statement"], env, _external_watchdog_s(timeout)
        )
    except AgentRunTimeoutError:
        raise  # already reaped inside _run_agent_bounded; do not reclassify
    except Exception as exc:
        _reap_container(env)
        _reraise_classified(spec.instance_id, model, exc)
    messages: list[dict[str, Any]] = getattr(agent, "messages", [])
    in_tok, out_tok, calls, cost = _sum_usage(messages)
    return AgentPatch(
        patch=str(info.get("submission") or ""),
        in_tok=in_tok,
        out_tok=out_tok,
        calls=calls,
        cost=cost,
        exit_status=str(info.get("exit_status") or ""),
        messages=messages,
        snapshots=recorder.snapshots if recorder is not None else {},
    )


def _attach_snapshot_recorder(agent: Any, env: Any) -> Any:
    """Wrap the agent's per-step method to capture a git diff after each turn (observe-only)."""
    # Returns the recorder, or None if the installed scaffold exposes no overridable step / no
    # ``env.execute`` (the snapshot-mechanism build-time surface check). Any failure to attach is
    # swallowed — snapshot capture must never alter the paid run's outcome, cost, or agent tree.
    from benchmark.runner.step_snapshots import StepSnapshotRecorder  # noqa: PLC0415

    execute = getattr(env, "execute", None)
    original = getattr(agent, "step", None)
    if not callable(execute) or not callable(original):
        return None

    def exec_fn(command: str) -> str:
        # mini-swe-agent v2 env.execute takes an ACTION DICT ({"command": ...}) and returns
        # {"output": stdout, "returncode": ...}; passing a bare string raises inside execute.
        result = execute({"command": command})
        return str(result.get("output", "")) if isinstance(result, dict) else str(result)

    recorder = StepSnapshotRecorder(exec_fn)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        messages = getattr(agent, "messages", [])
        index = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant") - 1
        if index >= 0:
            recorder.capture(index)
        return out

    try:
        agent.step = wrapped
    except Exception:  # noqa: BLE001 (observe-only: never break a paid run to attach capture)
        _LOG.exception("could not attach snapshot recorder; continuing without per-step snapshots")
        return None
    return recorder


def _capture_escalation_trajectory(
    patch: AgentPatch, instance_id: str, model: str, arm: str, resolved: bool
) -> None:
    """Additively persist the real agent trajectory for the escalation plane (observe-only)."""
    # Off the routing decision path and post-hoc, so it is cache-safe and never alters the paid
    # outcome. The gate lives in `capture_live_trajectory` (empty messages ⇒ nothing written); any
    # failure here is swallowed so a capture bug can never poison a paid routing row.
    try:
        from benchmark.escalation.live_capture import (  # noqa: PLC0415
            capture_live_trajectory,
            make_trajectory_id,
        )
        from benchmark.runner.step_snapshots import write_snapshots  # noqa: PLC0415

        capture_live_trajectory(
            patch.messages, instance_id=instance_id, model=model, arm=arm, resolved=resolved
        )
        if patch.snapshots:
            # Persist per-step diffs to the gitignored scratch keyed the SAME way as the trajectory
            # jsonl, so the offline replay pass can pair them by trajectory id.
            write_snapshots(make_trajectory_id(instance_id, model, arm), patch.snapshots)
    except Exception:  # noqa: BLE001 (observe-only: capture must never break a paid outcome)
        _LOG.exception("escalation trajectory capture failed for %s/%s", instance_id, model)


def _errored_outcome(
    instance_id: str,
    model: str,
    *,
    stop_reason: str,
    usage: tuple[int, int, int, float] = (0, 0, 0, 0.0),
) -> dict[str, object]:
    """A failed-cell outcome (pass=False) for a permanent model error or abandoned timeout."""
    # Recorded (not left MISSING) so the ladder treats the cell as not-solved and escalates, and
    # it is never retried into the same hang. ``usage`` carries any real partial spend harvested
    # from agent.messages before the abandon — so the row reflects money actually spent, not $0.
    # ``timeout_flag`` is derived from ``stop_reason`` so the two never disagree.
    in_tok, out_tok, calls, cost = usage
    return {
        "task_id": instance_id,
        "model": model,
        "pass": False,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "calls": calls,
        "real_cost": cost,
        "timeout_flag": censoring.timeout_flag_for(stop_reason),
        "stop_reason": stop_reason,
        "image_digest": "",
    }


def run_live_cell(
    instance_id: str,
    model: str,
    work_dir: Path,
    run_id: str,
    namespace: str = swebench_harness.DEFAULT_NAMESPACE,
    timeout: int = _AGENT_WALL_LIMIT_S,
    arm: str = integrity.DEFAULT_REASONING,
    step_limit: int = _DEFAULT_STEP_LIMIT,
) -> dict[str, object]:
    """Full live cell: agent → patch → harness → outcome dict for results.csv.

    Gated on keys via ``generate_patch_live``. Returns the outcome shape
    ``run_matrix._build_row`` consumes (pass/in_tok/out_tok/calls/real_cost/...).
    """
    spec = swebench_specs.load_spec(instance_id)
    if spec is None:
        raise KeyError(f"no SWE-bench spec for {instance_id!r}; materialise it first")
    try:
        patch = generate_patch_live(spec, model, arm=arm, timeout=timeout, step_limit=step_limit)
    except AgentRunTimeoutError as exc:
        _LOG.warning("%s/%s abandoned: %s", instance_id, model, redact_secrets(str(exc)))
        return _errored_outcome(
            instance_id,
            model,
            stop_reason=censoring.ABANDONED,
            usage=(exc.in_tok, exc.out_tok, exc.calls, exc.cost),
        )
    except PermanentModelError as exc:
        _LOG.warning("%s/%s failed permanently: %s", instance_id, model, redact_secrets(str(exc)))
        # A permanent 4xx (content policy / bad request) is a genuine capability fail the agent
        # could not work around — an UNCENSORED unsolved, not a resource-limit stop.
        return _errored_outcome(instance_id, model, stop_reason=censoring.UNSOLVED)
    preds_path = write_predictions(
        [prediction_line(instance_id, model, patch.patch)],
        work_dir / f"predictions_{run_id}.jsonl",
    )
    result = swebench_harness.run_harness(
        predictions_path=preds_path,
        run_id=run_id,
        work_dir=work_dir,
        namespace=namespace,
        timeout=timeout,
    )
    if result.report_path is None:
        raise HarnessInfraError(
            f"harness produced no report for {instance_id}/{model} "
            f"(rc={result.returncode}); leaving cell MISSING"
        )
    # A nonzero harness returncode with a written report is NOT infra failure: some graded
    # test suites (e.g. matplotlib) exit nonzero when the FAIL_TO_PASS tests fail, which is a
    # legitimate pass=False — recording it resolves the cell instead of re-running it forever.
    # The report's resolved verdict is authoritative; a real infra failure yields no report.
    resolved = bool(result.resolved.get(instance_id, False))
    _capture_escalation_trajectory(patch, instance_id, model, arm, resolved)
    # Map the harness verdict + the scaffold's exit_status to the stop_reason: solved when
    # resolved, else a censored step/wall stop if the agent hit a limit, else a genuine unsolved.
    stop_reason = censoring.stop_reason_from_run(resolved=resolved, exit_status=patch.exit_status)
    return {
        "task_id": instance_id,
        "model": model,
        "pass": resolved,
        "in_tok": patch.in_tok,
        "out_tok": patch.out_tok,
        "calls": patch.calls,
        "real_cost": patch.cost,
        "timeout_flag": censoring.timeout_flag_for(stop_reason),
        "stop_reason": stop_reason,
        # Record the digest the harness ACTUALLY used so stored == produced.
        "image_digest": image_version.used_image_digest(spec.image_ref) or "",
    }
