"""Produce ``predictions.jsonl`` for the harness — gold (keyless) or live (gated).

gold emits each instance's dataset gold patch ($0 pipeline smoke); live runs one
fixed ``mini-swe-agent`` scaffold per (instance, model), key-gated so keyless never fabricates.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import time
import types
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NoReturn

from benchmark import config
from benchmark.routing import censoring, integrity
from benchmark.runner import image_version, scaffold_model, swebench_harness, swebench_specs
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
# slow-inference models). Justified from the measured HAZARD, not passers' percentiles (the old
# p90=58/p95=68 rationale was survivorship-biased — conditioning on cells that passed says nothing
# about cells still alive at step 70). Of the 92 corpus cells still running at step 70 under the
# old 250-step budget, 37 (40.2%) went on to pass; 35 of those 37 (95%) finished by step 150, so
# 150 is where censoring drops from a ~40% loss to a ~2% tail. Raised together with
# benchmark.yaml live.cost_limit so the budget increase cannot relabel cost censors as step censors.
_DEFAULT_STEP_LIMIT: Final[int] = 150


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
    "OPENROUTER_API_KEY",
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


class ResumeFallbackError(RuntimeError):
    """A resume attempt could not reconstruct a trustworthy environment.

    Raised internally to restart the cell as a clean fresh run — never to poison it.
    """


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

    ds = load_dataset(
        swebench_specs.DATASET_NAME,
        split=swebench_specs.DATASET_SPLIT,
        revision=swebench_specs.DATASET_REVISION,
    )
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
    # True iff this run CONTINUED a saved partial conversation (conversation resume) rather than
    # starting fresh. Carried so the trajectory capture can record the total per-step snapshot
    # count across the resumed run, which the offline replay pairs against the scratch.
    resumed: bool = False
    # CLIENT-side wall clock of the agent loop for THIS invocation: the seconds between the
    # scaffold being handed the task and it returning, so inference plus in-container tool
    # execution, and NOT the SWE-bench grading harness (which is not model work). None means
    # NOT MEASURED — never zero — and it is None on every non-real path and on a RESUMED run,
    # whose earlier process's seconds this process never saw and may not invent.
    wall_clock_s: float | None = None
    # Client-side duration of each SUCCESSFUL provider call, in call order, as measured at the
    # scaffold model seam (`scaffold_model.EnvKeyLitellmModel`). Empty means NOT MEASURED.
    call_latencies_s: tuple[float, ...] = ()


def generate_patch_live(
    spec: swebench_specs.SwebenchSpec,
    model: str,
    scaffold: str = LIVE_SCAFFOLD,
    env: dict[str, str] | None = None,
    arm: str = integrity.DEFAULT_REASONING,
    timeout: int = _AGENT_WALL_LIMIT_S,
    step_limit: int = _DEFAULT_STEP_LIMIT,
    cost_limit: float | None = None,
) -> AgentPatch:
    """Run the fixed agent scaffold on one instance/model/arm to produce a patch.

    Gated: raises ``MissingApiKeysError`` without keys (keyless never fabricates);
    scaffold import is lazy so the wiring is unit-testable without it installed.
    """
    if not has_api_keys(env):
        raise MissingApiKeysError(
            f"live inference for {spec.instance_id}/{model} needs one of {_KEY_ENV}"
        )
    return _invoke_scaffold(
        spec, model, scaffold, arm, timeout=timeout, step_limit=step_limit, cost_limit=cost_limit
    )


def _registry_row(model: str) -> dict[str, Any]:
    """The model registry's provider row for *model*."""
    info = config.load_pricing().get(model)
    if not isinstance(info, dict):
        raise KeyError(f"model {model!r} not in the model registry")
    return info


def _model_key_env_var(model: str) -> str | None:
    """The env var holding *model*'s credential, or None when litellm resolves it itself."""
    # A provider with its own litellm prefix (e.g. `deepseek/`) is dialled by litellm directly,
    # which reads that provider's key from the env by its canonical name; a generic `openai/`
    # surface needs base_url + key supplied by us. Deliberately tolerant of an unregistered
    # alias: this looks up a NAME, and the loud failure for an unknown model already happened in
    # `litellm_model_target`. Returning None cannot open a hole — `credential_free_model_block`
    # refuses a config that carries a credential with no env var named to re-supply it.
    info = config.load_pricing().get(model)
    if not isinstance(info, dict) or not str(info["route"]).startswith("openai/"):
        return None
    return str(info["api_key_env_var"])


def litellm_model_target(model: str) -> tuple[str, dict[str, Any]]:
    """Map an internal model alias to a litellm ``(model_string, model_kwargs)`` pair."""
    # Route, base_url and key env var all come from the registry's provider row. The returned
    # kwargs carry the RESOLVED key and are for a direct in-process `litellm.completion` only —
    # never hand them to the scaffold, which serialises its config to disk (scaffold_model.py).
    info = _registry_row(model)
    route = str(info["route"])
    key_env = _model_key_env_var(model)
    if key_env is None:
        return route, {}
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


def _scaffold_default_max_tokens() -> int | None:
    """The generation cap mini-swe-agent's own swebench overlay sends, if it declares one."""
    # The scaffold config is the ONLY place a cap the run actually sends can come from besides
    # the registry arm; read it rather than assuming. The extra is optional, so a missing
    # import is "no cap declared here", never a failure.
    try:
        from minisweagent.config import builtin_config_dir, get_config_from_spec  # noqa: PLC0415
    except ImportError:
        return None
    spec = str(builtin_config_dir / "benchmarks" / "swebench.yaml")
    kwargs = get_config_from_spec(spec).get("model", {}).get("model_kwargs", {})
    cap = kwargs.get("max_tokens")
    return int(cap) if cap is not None else None


def _effective_max_tokens(model: str) -> int | None:
    """The generation cap in force for *model*: what the run SENDS, else the declared fallback."""
    # Precedence mirrors _scaffold_model_kwargs (scaffold base <- registry arm), so this is the
    # number the request will actually carry. `live.serving.max_tokens` is the last resort: today
    # neither source declares a cap, and a guard that can only ever refuse is a wall, not a guard.
    arm = config.default_arm_ids([model]).get(model, integrity.DEFAULT_REASONING)
    arm_cap = config.arm_api_params(model, arm).get("max_tokens")
    if arm_cap is not None:
        return int(arm_cap)
    from benchmark.runner import serving_guard  # noqa: PLC0415

    return _scaffold_default_max_tokens() or serving_guard.serving_max_tokens()


def _preflight_serving_check(target: str, model_kwargs: dict[str, Any]) -> None:
    """Assert a LOCAL serving endpoint's resolved flags; no-op for hosted providers."""
    # A self-hosted server can silently delete the system prompt and tool schema mid-generation
    # and still answer HTTP 200 (ollama's `--context-shift --keep 4`), which no status-based
    # retry can see — so the flags are read and recorded here, before any container starts.
    from benchmark.runner import serving_guard  # noqa: PLC0415

    base_url = model_kwargs.get("api_base")
    try:
        # Inside the try: classification itself refuses on an endpoint it cannot decide
        # (`UndecidableEndpointError`), and that must abort the run, not escape as a crash.
        if not serving_guard.is_local_endpoint(base_url):
            return
        serving_guard.assert_serving_safe(base_url, max_tokens=_effective_max_tokens(target))
    except serving_guard.UnsafeServingError as exc:
        raise ApiUnusableError(f"unsafe local serving config for {target}: {exc}") from exc


def preflight_api_check(model: str | None = None) -> bool:
    """Prove the API key works with ONE minimal real completion, before any container spins up.

    A dead/empty key or no-balance error raises ``ApiUnusableError`` (refuse the run); a
    transient blip (rate-limit / 5xx) is inconclusive and returns True — never refuse over one.
    """
    # A local endpoint additionally has its resolved serving flags asserted (`serving_guard`).
    import litellm  # noqa: PLC0415

    target = model or _cheapest_enabled_model()
    model_string, model_kwargs = litellm_model_target(target)
    _preflight_serving_check(target, model_kwargs)
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

    ds = load_dataset(
        swebench_specs.DATASET_NAME,
        split=swebench_specs.DATASET_SPLIT,
        revision=swebench_specs.DATASET_REVISION,
    )
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
    model_string: str,
    model_kwargs: dict[str, Any],
    *,
    timeout: int,
    step_limit: int,
    cost_limit: float,
    trajectory_id: str,
    api_key_env_var: str | None,
) -> dict[str, Any]:
    """The mini-swe-agent config overlay for one live cell (merged over the swebench default)."""
    # step_limit is the PRIMARY model-speed-agnostic bound; wall_time_limit_seconds is a generous
    # graceful secondary backstop set from the PASSED timeout; cost_limit is the declared per-cell
    # USD cap (benchmark.yaml live.cost_limit — an undeclared key used to let the scaffold's own
    # 3.0 default govern silently). All three trip the scaffold's own InterruptAgentFlow, so
    # agent.run returns with usage intact (the cost is captured, not lost). output_path makes the
    # scaffold persist the FULL message list (DefaultAgent.run() saves it in a finally block) to
    # the gitignored scratch, keyed by trajectory id so it pairs with the escalation trajectory
    # jsonl and the per-step snapshot dir.
    from benchmark.runner.step_snapshots import message_list_path  # noqa: PLC0415

    return {
        "agent": {
            "wall_time_limit_seconds": timeout,
            "step_limit": step_limit,
            "cost_limit": cost_limit,
            "output_path": str(message_list_path(trajectory_id)),
        },
        # The model block is built credential-free ON PURPOSE: `output_path` above makes the
        # scaffold persist its own config verbatim, so a key inside `model_kwargs` would be
        # written to disk in plaintext once per cell. It carries the env var's NAME and the
        # value is injected per request. See `benchmark.runner.scaffold_model`.
        "model": scaffold_model.credential_free_model_block(
            model_string, model_kwargs, api_key_env_var=api_key_env_var
        ),
        "environment": {"environment_class": "docker"},
    }


# ---------------------------------------------------------------------------
# conversation resume — continue a throttled/aborted cell from its saved messages
# ---------------------------------------------------------------------------

# The heredoc delimiter used to pipe a snapshot diff through `env.execute` (which has no stdin
# plumbing). A QUOTED delimiter means bash performs NO interpolation on the diff body.
_RESUME_PATCH_DELIMITER: Final[str] = "SHUNT_RESUME_PATCH_7f3a9c"

# A saved conversation is only resumable from a clean boundary: the last message must be a tool
# observation (the model saw its result) or a user-role format-error retry. A trailing assistant
# message means an action was issued but never observed — resuming would re-send dangling
# tool_calls some providers reject, so those conversations start fresh.
_RESUME_SAFE_LAST_ROLES: Final[tuple[str, ...]] = ("tool", "user")


def _interrupt_messages(exc: BaseException) -> tuple[dict[str, Any], ...]:
    """The messages an interrupt/format exception carries (mini-swe-agent convention)."""
    messages = getattr(exc, "messages", ())
    return messages if isinstance(messages, tuple) else tuple(messages)


def _strip_exit_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop trailing terminal (role='exit') messages from a saved conversation."""
    stripped = list(messages)
    while stripped and stripped[-1].get("role") == "exit":
        stripped.pop()
    return stripped


def _assistant_turn_count(messages: list[dict[str, Any]]) -> int:
    """The number of assistant turns a saved conversation already contains."""
    return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")


def _resume_messages(trajectory_id: str) -> list[dict[str, Any]] | None:
    """The resumable prior conversation for a trajectory, or None (no resume / unsafe)."""
    # Detection: the scratch message list exists, is non-trivial (≥1 assistant turn), and ends at a
    # clean boundary. run_live_cell only ever invokes the scaffold on an ungraded (missing) cell,
    # so any saved conversation it finds is a partial one — a graded cell is never re-run.
    from benchmark.runner.step_snapshots import message_list_path  # noqa: PLC0415

    path = message_list_path(trajectory_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _LOG.warning("resume: unreadable message list for %s; starting fresh", trajectory_id)
        return None
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return None
    messages = [m for m in messages if isinstance(m, dict)]
    messages = _strip_exit_messages(messages)
    if _assistant_turn_count(messages) == 0:
        return None
    if messages[-1].get("role") not in _RESUME_SAFE_LAST_ROLES:
        _LOG.warning("resume: %s ended mid-turn; starting fresh", trajectory_id)
        return None
    return messages


def _resume_step_limit(step_limit: int, loaded: list[dict[str, Any]]) -> int:
    """Remaining step budget for a resumed cell — the full budget minus loaded turns, ≥1."""
    return max(1, step_limit - _assistant_turn_count(loaded))


# The smallest cap a resumed cell may be handed. NOT 0.0: mini-swe-agent 2.4.5 gates on
# `0 < self.config.cost_limit <= self.cost` (agents/default.py, DefaultAgent.query), so a cap of
# 0.0 DISABLES the ceiling entirely — handing an exhausted cell 0.0 would remove the cap it just
# blew through. A tiny positive cap is the exact analogue of `_resume_step_limit`'s floor of 1:
# the pre-call check passes once (self.cost starts at 0.0), so the cell gets at most ONE more
# model call and then exits LimitsExceeded. Refusing the resume outright is worse for spend —
# the fallback is a FRESH run at the FULL cap.
_MIN_RESUME_COST_LIMIT: Final[float] = 0.01


def _resume_cost_limit(cost_limit: float, loaded: list[dict[str, Any]]) -> float:
    """Remaining USD budget for a resumed cell — the full cap minus what the loaded turns spent."""
    # A resumed cell is a FRESH DefaultAgent: `self.cost` restarts at 0.0 and the per-cell cap is
    # re-armed from scratch, so passing `cost_limit` through unchanged lets one cell spend the cap
    # once per resume (unbounded across N resumes). Reduce it exactly as the step budget is
    # reduced. `_sum_usage` is the cost basis used everywhere else in this module — the provider's
    # cache-aware `usage.cost` where present — so the reduction is in REAL dollars.
    if cost_limit <= 0.0:
        return cost_limit  # the ceiling is disabled upstream; a resume must not invent one
    spent = _sum_usage(loaded)[3]
    return max(_MIN_RESUME_COST_LIMIT, cost_limit - spent)


def _apply_resume_diff(env: Any, diff: str) -> bool:
    """Apply one cumulative snapshot diff to /testbed inside the agent container (rc == 0)."""
    from benchmark.runner.step_snapshots import TESTBED  # noqa: PLC0415

    command = (
        f"git -C {TESTBED} apply - <<'{_RESUME_PATCH_DELIMITER}'\n{diff}\n{_RESUME_PATCH_DELIMITER}"
    )
    try:
        out = env.execute({"command": command})
    except Exception:  # noqa: BLE001 (reconstruction failure → fall back to a fresh start)
        _LOG.exception("resume: diff application command failed")
        return False
    return int(out.get("returncode", -1)) == 0


def _restore_checkout(env: Any, trajectory_id: str, loaded_turn_count: int | None = None) -> bool:
    """Reconstruct the checkout to match a saved conversation (last cumulative snapshot).

    Best-effort by design: only TRACKED working-tree edits are restored — untracked files and any
    agent ``git commit`` (which moves HEAD) are NOT (warned on the apply path).
    """
    # Each step diff is `git diff HEAD` — CUMULATIVE against the base, so the LAST captured
    # snapshot alone carries the full tracked-file state (applying 0..k-1 sequentially would
    # double-apply and reject). Any failure (missing scratch, patch does not apply) returns False
    # so the caller falls back to a clean fresh start — never resumes into a possibly-wrong tree.
    from benchmark.runner.step_snapshots import read_snapshots  # noqa: PLC0415

    snapshots = read_snapshots(trajectory_id)
    if not snapshots:
        _LOG.warning("resume: no per-step snapshots for %s; starting fresh", trajectory_id)
        return False
    # Consistency guard, TWO-SIDED. The invariant: `_attach_snapshot_recorder`'s wrapper captures
    # `index = <assistant messages> - 1` AFTER each step returns, so the k-th assistant turn writes
    # index k-1 and a conversation of N turns whose last turn executed is reconstructed by index
    # N-1 exactly. Anything HIGHER is stale residue from a longer, discarded run (its message list
    # was replaced by a shorter one) — a tree this conversation never had. Anything LOWER means the
    # set is incomplete (a swallowed `capture` exec error, a partial `clear_trajectory_scratch`, a
    # trailing format-error turn that never executed) — resuming would continue an N-turn
    # conversation against a tree frozen several turns back. Both are unresumable: refuse, and the
    # caller falls back to a fresh start.
    if loaded_turn_count is not None and max(snapshots) != loaded_turn_count - 1:
        _LOG.warning(
            "resume: snapshots for %s reach step %d but the loaded conversation's %d assistant "
            "turns require step %d — an incomplete or discarded run's set; starting fresh",
            trajectory_id,
            max(snapshots),
            loaded_turn_count,
            loaded_turn_count - 1,
        )
        return False
    diff = snapshots[max(snapshots)]
    _LOG.warning(
        "resume: reconstructing %s restores only TRACKED working-tree edits — untracked files "
        "the agent created and any agent `git commit`s (which move HEAD) are NOT restored; the "
        "tree is best-effort, not a byte-exact replay",
        trajectory_id,
    )
    if not diff.strip():
        # NOT "the tree is at base state". The snapshot is `git diff HEAD` (step_snapshots
        # DIFF_COMMAND), i.e. worktree-vs-HEAD in the AGENT's container — and an agent that ran
        # `git add -A && git commit` moves HEAD onto its own work, making that diff exactly 0 bytes
        # while the work sits on disk. Empty is therefore indistinguishable between "changed
        # nothing" and "committed everything", and this recorder captures nothing that separates
        # them. Restoring nothing and returning True would resume a PAID conversation against a
        # base checkout under the "trustworthy environment" contract; refuse instead — a fresh run
        # is cheap, a wrong tree is not.
        _LOG.warning(
            "resume: the last snapshot for %s is empty — indistinguishable between an unchanged "
            "tree and an agent that committed its work (HEAD moved); starting fresh",
            trajectory_id,
        )
        return False
    return _apply_resume_diff(env, diff)


def _resume_run(
    agent: Any,
    task: str,
    *,
    format_error_cls: type[BaseException],
    interrupt_cls: type[BaseException],
) -> dict[str, Any]:
    """Continue a pre-seeded conversation — DefaultAgent.run() minus its message reset."""
    # =============================================================================
    # MIRROR WARNING — mini-swe-agent 2.4.5 (DefaultAgent.run, agents/default.py)
    # -----------------------------------------------------------------------------
    # This loop is a hand-copy of DefaultAgent.run() adjusted for the resume case, and
    # pyproject.toml pins the scaffold only as `mini-swe-agent>=2.4` (unpinned). On ANY
    # mini-swe-agent upgrade, RE-VERIFY this mirror line-by-line against the installed
    # DefaultAgent.run() before trusting a resumed run — the loop (step, format-error
    # retry count + RepeatedFormatError exit, interrupt handling, finally-save, exit-role
    # stop) silently drifts otherwise. The ONLY intended differences from the original:
    # no `self.messages = []` reset and no system+instance re-add (the loaded history is
    # already seeded and must be preserved).
    # =============================================================================
    agent.extra_template_vars |= {"task": task}
    while True:
        try:
            agent.step()
            agent.n_consecutive_format_errors = 0
        except format_error_cls as exc:
            agent.n_consecutive_format_errors += 1
            if 0 < agent.config.max_consecutive_format_errors <= agent.n_consecutive_format_errors:
                agent.add_messages(
                    *_interrupt_messages(exc),
                    {
                        "role": "exit",
                        "content": "RepeatedFormatError",
                        "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                    },
                )
            else:
                agent.add_messages(*_interrupt_messages(exc))
        except interrupt_cls as exc:
            agent.add_messages(*_interrupt_messages(exc))
        except Exception as exc:  # noqa: BLE001 (mirrors DefaultAgent.run's fall-through)
            agent.handle_uncaught_exception(exc)
            raise
        finally:
            agent.save(agent.config.output_path)
        if agent.messages[-1].get("role") == "exit":
            break
    return agent.messages[-1].get("extra", {})


def _seed_resume(
    agent: Any,
    messages: list[dict[str, Any]],
    *,
    format_error_cls: type[BaseException],
    interrupt_cls: type[BaseException],
) -> None:
    """Pre-seed the agent's conversation and route ``run()`` through the resume loop."""
    # Copies, never aliases: the loaded dicts must not be mutated by the running agent.
    agent.messages = [dict(m) for m in messages]
    run = functools.partial(
        _resume_run, format_error_cls=format_error_cls, interrupt_cls=interrupt_cls
    )
    agent.run = types.MethodType(run, agent)


def _invoke_scaffold(
    spec: swebench_specs.SwebenchSpec,
    model: str,
    scaffold: str,  # noqa: ARG001 (kept for signature stability; only mini-swe-agent is wired)
    arm: str = integrity.DEFAULT_REASONING,
    timeout: int = _AGENT_WALL_LIMIT_S,
    step_limit: int = _DEFAULT_STEP_LIMIT,
    cost_limit: float | None = None,
) -> AgentPatch:
    """Invoke mini-swe-agent (v2) for one instance/model/arm (only reached when keys exist)."""
    from benchmark.escalation.live_capture import make_trajectory_id  # noqa: PLC0415

    instance = _load_instance(spec.instance_id)
    model_string, model_kwargs = litellm_model_target(model)
    trajectory_id = make_trajectory_id(spec.instance_id, model, arm)
    # A resumable saved conversation exists? (config-gated; a graded cell is never re-invoked, so
    # any conversation found here is a partial one.) Resume is attempted, never forced: a
    # reconstruction failure falls back to a clean fresh run at the FULL step budget.
    resume = _resume_messages(trajectory_id) if config.resume_enabled() else None
    if resume is not None:
        _LOG.warning(
            "resume: %s found a %d-turn saved conversation; continuing instead of restarting",
            trajectory_id,
            _assistant_turn_count(resume),
        )
    while True:
        try:
            return _invoke_scaffold_attempt(
                instance,
                model_string,
                model_kwargs,
                trajectory_id,
                model,
                arm,
                timeout=timeout,
                step_limit=step_limit,
                cost_limit=cost_limit,
                resume=resume,
            )
        except ResumeFallbackError:
            _LOG.warning(
                "resume: reconstruction for %s failed; restarting the cell as a fresh run",
                trajectory_id,
            )
            resume = None


def _invoke_scaffold_attempt(
    instance: dict[str, Any],
    model_string: str,
    model_kwargs: dict[str, Any],
    trajectory_id: str,
    model: str,
    arm: str,
    *,
    timeout: int,
    step_limit: int,
    cost_limit: float | None,
    resume: list[dict[str, Any]] | None,
) -> AgentPatch:
    """One full live attempt — continuing *resume* when provided, else a clean fresh start."""
    from minisweagent.agents import get_agent  # noqa: PLC0415
    from minisweagent.config import builtin_config_dir, get_config_from_spec  # noqa: PLC0415
    from minisweagent.exceptions import FormatError, InterruptAgentFlow  # noqa: PLC0415
    from minisweagent.models import get_model  # noqa: PLC0415
    from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: PLC0415
    from minisweagent.utils.serialize import recursive_merge  # noqa: PLC0415

    # A FRESH start owns the trajectory's scratch: clear the prior partial run's snapshot dir and
    # saved message list so stale higher-index diffs from a DISCARDED run cannot poison a later
    # resume (a resume pairs `snapshots[max(...)]` with the loaded conversation, and a stale max
    # from an older, longer run would rebuild a tree the loaded conversation never had). Fires on
    # the first run too — clearing an empty scratch is a no-op. Deliberately NOT on the resume
    # path, which needs the prior snapshots to reconstruct the checkout.
    if resume is None:
        from benchmark.runner.step_snapshots import clear_trajectory_scratch  # noqa: PLC0415

        clear_trajectory_scratch(trajectory_id)

    # A resumed cell must NOT get a fresh full step budget: the loaded assistant turns already
    # consumed part of it. Budget is floored at 1 so a resumed cell always gets at least one step.
    effective_step_limit = (
        _resume_step_limit(step_limit, resume) if resume is not None else step_limit
    )
    # The USD cap is re-armed on a fresh agent object exactly like the step count, so it is reduced
    # exactly like the step count. `timeout` deliberately is NOT: it feeds wall_time_limit_seconds
    # and the external watchdog, both of which bound THIS process's invocation (measured from a
    # freshly stamped `_start_time`). Wall-clock is not a budget the resume double-spends — the
    # prior run's seconds are gone, not re-consumable — and its job is to kill a HUNG process, which
    # needs a full window; shortening it would reap a healthy resumed run instead.
    declared_cost_limit = config.live_cost_limit() if cost_limit is None else cost_limit
    effective_cost_limit = (
        _resume_cost_limit(declared_cost_limit, resume)
        if resume is not None
        else declared_cost_limit
    )
    default_config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
    base_kwargs = default_config.get("model", {}).get("model_kwargs", {})
    overlay = _scaffold_config_overlay(
        model_string,
        _scaffold_model_kwargs(model, arm, base_kwargs, model_kwargs),
        timeout=timeout,
        step_limit=effective_step_limit,
        cost_limit=effective_cost_limit,
        trajectory_id=trajectory_id,
        api_key_env_var=_model_key_env_var(model),
    )
    merged = recursive_merge(default_config, overlay)
    # The overlay was built credential-free, but `recursive_merge` merges the scaffold's own
    # default `model_kwargs` UNDER it key-by-key — so the config the scaffold will serialise is
    # re-checked here, after the merge, rather than only before it.
    scaffold_model.assert_credential_free(merged.get("model", {}), what="merged scaffold config")
    env = get_sb_environment(merged, instance)
    # Reconstruct the filesystem to match the saved conversation BEFORE the agent is created, so a
    # wrong or missing tree aborts the resume (falling back to fresh) instead of running against it.
    # The loaded assistant-turn count gates the reconstruction: a snapshot index the conversation
    # cannot justify (stale files from a discarded run) must never be applied.
    if resume is not None and not _restore_checkout(
        env, trajectory_id, _assistant_turn_count(resume)
    ):
        _reap_container(env)
        raise ResumeFallbackError(
            f"resume reconstruction failed for {trajectory_id}; restarting fresh"
        )
    model_obj = get_model(config=merged.get("model", {}))
    _harden_model_retries(model_obj)
    agent = get_agent(model_obj, env, merged.get("agent", {}), default_type="default")
    _harden_output_write(agent)
    recorder = _attach_snapshot_recorder(agent, env)
    if resume is not None:
        _seed_resume(agent, resume, format_error_cls=FormatError, interrupt_cls=InterruptAgentFlow)
    started = time.perf_counter()
    try:
        info = _run_agent_bounded(
            agent, instance["problem_statement"], env, _external_watchdog_s(timeout)
        )
    except AgentRunTimeoutError:
        raise  # already reaped inside _run_agent_bounded; do not reclassify
    except Exception as exc:
        _reap_container(env)
        _reraise_classified(str(instance["instance_id"]), model, exc)
    # Cost/token accounting covers the WHOLE cell: agent.messages holds the seeded history plus the
    # new turns, and _sum_usage walks every assistant message's real usage — the row reflects the
    # entire cell, not just the resumed tail.
    wall_clock_s = time.perf_counter() - started
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
        resumed=resume is not None,
        # A RESUMED cell's total wall clock is unmeasurable here — the prior process's seconds
        # were never recorded — so it stays MISSING rather than being reported as the tail's
        # duration. The per-call latencies are unaffected: each is a complete, real
        # measurement of one round trip whether or not the conversation was resumed.
        wall_clock_s=None if resume is not None else wall_clock_s,
        call_latencies_s=tuple(getattr(model_obj, "call_latencies_s", ()) or ()),
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


def _harden_output_write(agent: Any) -> None:
    """Make a failed message-list dump never break the paid run (observe-only)."""
    # DefaultAgent.run() writes the FULL message list via ``self.save(self.config.output_path)``
    # in a finally block; a write failure there (disk full, permission) would otherwise propagate
    # and poison the run's outcome/cost/exit-status. Same observe-only contract as the snapshot
    # recorder and the escalation trajectory capture: capture must never alter a paid run. Any
    # failure to attach is swallowed too.
    original = getattr(agent, "save", None)
    if not callable(original):
        return

    def safe_save(path: Any, *extra: Any) -> dict[str, Any]:
        try:
            return original(path, *extra)
        except Exception:  # noqa: BLE001 (observe-only: a dump failure never breaks a paid run)
            _LOG.exception("message-list dump failed for a cell; continuing without it")
            return {}

    try:
        agent.save = safe_save
    except Exception:  # noqa: BLE001 (observe-only: never break a paid run to attach capture)
        _LOG.exception("could not harden the message-list output write; continuing")


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
        from benchmark.runner.step_snapshots import (  # noqa: PLC0415
            read_snapshots,
            write_snapshots,
        )

        trajectory_id = make_trajectory_id(instance_id, model, arm)
        # Persist per-step diffs to the gitignored scratch keyed the SAME way as the trajectory
        # jsonl, so the offline replay pass can pair them by trajectory id. A RESUME appends new
        # indices on top of the prior run's files (the recorder keys by assistant-turn, so the
        # continuation indices are disjoint) — a fresh run simply overwrites its own indices.
        if patch.snapshots:
            write_snapshots(trajectory_id, patch.snapshots)
        # The committed count must equal what the offline replay finds on disk: for a resumed run
        # that is ALL diffs for the trajectory (prior + this window's tail); for a fresh run it is
        # exactly what this run captured.
        snapshot_steps = (
            len(read_snapshots(trajectory_id)) if patch.resumed else len(patch.snapshots)
        )
        capture_live_trajectory(
            patch.messages,
            instance_id=instance_id,
            model=model,
            arm=arm,
            resolved=resolved,
            # Recorded even (especially) when it is 0 — that is the committed proof the offline
            # replay needs to tell "captured nothing" from "this checkout lacks the scratch".
            snapshot_steps=snapshot_steps,
        )
    except Exception:  # noqa: BLE001 (observe-only: capture must never break a paid outcome)
        _LOG.exception("escalation trajectory capture failed for %s/%s", instance_id, model)


def _collection_provenance(
    model: str, arm: str, *, step_limit: int, cost_limit: float
) -> dict[str, str]:
    """The collection-param provenance keys for a results row."""
    # Records the regime this cell was collected under so a later change to the caps, the
    # installed scaffold, the merged request kwargs, or the prompt templates is detected as
    # staleness (run_matrix._is_stale). Derived from the same integrity helpers that compute
    # the CURRENT expected anchors, so recorded and expected always agree at the same config.
    return {
        "step_limit": str(step_limit),
        "cost_limit": str(cost_limit),
        "scaffold_version": integrity.scaffold_version(),
        "sampling_hash": integrity.sampling_hash(model, arm),
        "prompt_hash": integrity.scaffold_prompt_hash(),
    }


def _latency_provenance(model: str, patch: AgentPatch) -> dict[str, str]:
    """The measured-latency and serving-provenance columns for one live results row."""
    # ONLY what this process actually measured. Every key here is a MEASUREMENT-OPTIONAL or
    # PROVENANCE-OPTIONAL column (`routing.integrity`), where a blank means MISSING FOREVER and
    # is NEVER read as zero — so an unmeasured field is omitted, never defaulted, and never
    # derived from a neighbouring quantity.
    #
    # Three deliberate omissions:
    #
    #   * `ttft_s` is ALWAYS absent. The scaffold calls `litellm.completion` non-streaming, so
    #     no first-token event exists to time; writing time-to-full-response under the TTFT
    #     name would publish a different quantity. Obtaining it needs a streaming scaffold.
    #   * an UNLABELLABLE timing is dropped rather than written. A latency whose `serving_mode`
    #     is unknown could later be pooled with its opposite (local batch-1 against a batched
    #     hosted API), which is exactly the defect the column exists to prevent, and the data
    #     would carry no trace of the mistake.
    #   * an ERRORED or ABANDONED cell records no timing at all (`_errored_outcome`): its wall
    #     clock is the watchdog's ceiling, a property of the limit rather than of the model.
    info = config.load_pricing().get(model)
    row: dict[str, str] = {}
    if isinstance(info, dict):
        row["provider"] = str(info.get("provider") or "")
        row["serving_mode"] = str(info.get("serving_mode") or "")
    timings: dict[str, str] = {}
    if patch.wall_clock_s is not None:
        timings["wall_clock_s"] = f"{patch.wall_clock_s:.6f}"
    if patch.call_latencies_s:
        mean_s = math.fsum(patch.call_latencies_s) / len(patch.call_latencies_s)
        timings["latency_per_call_s"] = f"{mean_s:.6f}"
    if timings and not row.get("serving_mode"):
        _LOG.warning(
            "no serving_mode for %r in the registry; dropping this cell's latency rather than "
            "writing a timing nothing can attribute to a serving stack",
            model,
        )
        timings = {}
    if timings:
        # Both numbers above are the client's own clock around the call, not a field the
        # provider reported. The two are different measurements and pooling them would be a
        # defect, so the row states which one it is.
        row["provider_latency_source"] = integrity.LATENCY_SOURCE_CLIENT
    row.update(timings)
    return {key: value for key, value in row.items() if value}


def _errored_outcome(
    instance_id: str,
    model: str,
    *,
    stop_reason: str,
    usage: tuple[int, int, int, float] = (0, 0, 0, 0.0),
    step_limit: int = _DEFAULT_STEP_LIMIT,
    cost_limit: float = 0.0,
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
        "step_limit": str(step_limit),
        "cost_limit": str(cost_limit),
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
    cost_limit: float | None = None,
) -> dict[str, object]:
    """Full live cell: agent → patch → harness → outcome dict for results.csv.

    Gated on keys via ``generate_patch_live``. Returns the outcome shape
    ``run_matrix._build_row`` consumes (pass/in_tok/out_tok/calls/real_cost/...).
    """
    spec_module = swebench_specs.spec_module_for(swebench_specs.manifest_source())
    spec = spec_module.load_spec(instance_id)
    if spec is None:
        raise KeyError(f"no SWE-bench spec for {instance_id!r}; materialise it first")
    actual_cost_limit = cost_limit if cost_limit is not None else config.live_cost_limit()
    try:
        patch = generate_patch_live(
            spec,
            model,
            arm=arm,
            timeout=timeout,
            step_limit=step_limit,
            cost_limit=actual_cost_limit,
        )
    except AgentRunTimeoutError as exc:
        _LOG.warning("%s/%s abandoned: %s", instance_id, model, redact_secrets(str(exc)))
        return _errored_outcome(
            instance_id,
            model,
            stop_reason=censoring.ABANDONED,
            usage=(exc.in_tok, exc.out_tok, exc.calls, exc.cost),
            step_limit=step_limit,
            cost_limit=actual_cost_limit,
        )
    except PermanentModelError as exc:
        _LOG.warning("%s/%s failed permanently: %s", instance_id, model, redact_secrets(str(exc)))
        # A permanent 4xx (content policy / bad request) is a genuine capability fail the agent
        # could not work around — an UNCENSORED unsolved, not a resource-limit stop.
        return _errored_outcome(
            instance_id,
            model,
            stop_reason=censoring.UNSOLVED,
            step_limit=step_limit,
            cost_limit=actual_cost_limit,
        )
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
        # The harness loads the SAME dataset split the spec came from (verified default;
        # the multimodal dev split for a multimodal config), so grading reads the right labels.
        dataset_name=spec_module.DATASET_NAME,
        split=spec_module.DATASET_SPLIT,
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
        **_collection_provenance(model, arm, step_limit=step_limit, cost_limit=actual_cost_limit),
        **_latency_provenance(model, patch),
    }
