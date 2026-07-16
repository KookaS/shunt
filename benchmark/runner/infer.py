"""Produce ``predictions.jsonl`` for the harness — gold (keyless) or live (gated).

gold emits each instance's dataset gold patch ($0 pipeline smoke); live runs one
fixed ``mini-swe-agent`` scaffold per (instance, model), key-gated so keyless never fabricates.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from benchmark import config
from benchmark.runner import image_version, swebench_harness, swebench_specs

# Wall-clock backstop for one live agent run (mini-swe-agent's own step_limit is the
# primary bound; this catches a runaway when per-call cost can't be priced, so the
# cost_limit never trips).
_AGENT_WALL_LIMIT_S: Final[int] = 1800

# OpenAI-compatible endpoint for requesty-routed models (key: REQUESTY_API_KEY).
_REQUESTY_BASE: Final[str] = "https://router.requesty.ai/v1"

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
    """A patch produced by the live agent scaffold, with measured usage."""

    patch: str
    in_tok: int
    out_tok: int
    calls: int
    cost: float


def generate_patch_live(
    spec: swebench_specs.SwebenchSpec,
    model: str,
    scaffold: str = LIVE_SCAFFOLD,
    env: dict[str, str] | None = None,
) -> AgentPatch:
    """Run the fixed agent scaffold on one instance/model to produce a patch.

    Gated: raises ``MissingApiKeysError`` without keys (keyless never fabricates);
    scaffold import is lazy so the wiring is unit-testable without it installed.
    """
    if not has_api_keys(env):
        raise MissingApiKeysError(
            f"live inference for {spec.instance_id}/{model} needs one of {_KEY_ENV}"
        )
    return _invoke_scaffold(spec, model, scaffold)


def litellm_model_target(model: str) -> tuple[str, dict[str, Any]]:
    """Map an internal model alias to a litellm ``(model_string, model_kwargs)`` pair.

    ``route`` in models.json is the litellm string; ``direct`` models read their own
    env key, ``requesty`` routes carry the shared base_url + REQUESTY_API_KEY.
    """
    info = config.load_pricing().get(model)
    if not isinstance(info, dict):
        raise KeyError(f"model {model!r} not in models.json")
    route = info.get("route")
    if not route:
        # No guessing: the requesty serving-provider prefix (e.g. fireworks/glm-5.2)
        # does not match the model's origin provider, so a derived route would be wrong.
        raise KeyError(f"model {model!r} has no explicit 'route' in models.json")
    route = str(route)
    access = info.get("access_via")
    if access == "direct":
        return route, {}
    if access == "requesty":
        key = os.environ.get("REQUESTY_API_KEY")
        if not key:
            raise MissingApiKeysError(f"requesty routing for {model!r} needs REQUESTY_API_KEY")
        return route, {"api_base": _REQUESTY_BASE, "api_key": key}
    raise NotImplementedError(f"no litellm route for {model!r} (access_via={access!r})")


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
    """Real cost of one call: litellm's computed cost, else the provider-returned one."""
    litellm_cost = float(extra.get("cost", 0.0) or 0.0)
    if litellm_cost > 0.0:
        return litellm_cost
    return float(usage.get("cost", 0.0) or 0.0)


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


def _invoke_scaffold(
    spec: swebench_specs.SwebenchSpec,
    model: str,
    scaffold: str,  # noqa: ARG001 (kept for signature stability; only mini-swe-agent is wired)
) -> AgentPatch:
    """Invoke mini-swe-agent (v2) for one instance/model (only reached when keys exist)."""
    from minisweagent.agents import get_agent  # noqa: PLC0415
    from minisweagent.config import builtin_config_dir, get_config_from_spec  # noqa: PLC0415
    from minisweagent.models import get_model  # noqa: PLC0415
    from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: PLC0415
    from minisweagent.utils.serialize import recursive_merge  # noqa: PLC0415

    instance = _load_instance(spec.instance_id)
    model_string, model_kwargs = litellm_model_target(model)
    default_config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
    base_kwargs = default_config.get("model", {}).get("model_kwargs", {})
    merged = recursive_merge(
        default_config,
        {
            "agent": {"wall_time_limit_seconds": _AGENT_WALL_LIMIT_S},
            "model": {
                "model_name": model_string,
                "model_kwargs": {**base_kwargs, **model_kwargs},
                "cost_tracking": "ignore_errors",
            },
            "environment": {"environment_class": "docker"},
        },
    )
    env = get_sb_environment(merged, instance)
    agent = get_agent(
        get_model(config=merged.get("model", {})),
        env,
        merged.get("agent", {}),
        default_type="default",
    )
    info = agent.run(instance["problem_statement"])
    messages: list[dict[str, Any]] = getattr(agent, "messages", [])
    in_tok, out_tok, calls, cost = _sum_usage(messages)
    return AgentPatch(
        patch=str(info.get("submission") or ""),
        in_tok=in_tok,
        out_tok=out_tok,
        calls=calls,
        cost=cost,
    )


def run_live_cell(
    instance_id: str,
    model: str,
    work_dir: Path,
    run_id: str,
    namespace: str = swebench_harness.DEFAULT_NAMESPACE,
    timeout: int = 1800,
) -> dict[str, object]:
    """Full live cell: agent → patch → harness → outcome dict for results.csv.

    Gated on keys via ``generate_patch_live``. Returns the outcome shape
    ``run_matrix._build_row`` consumes (pass/in_tok/out_tok/calls/real_cost/...).
    """
    spec = swebench_specs.load_spec(instance_id)
    if spec is None:
        raise KeyError(f"no SWE-bench spec for {instance_id!r}; materialise it first")
    patch = generate_patch_live(spec, model)
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
    if result.report_path is None or result.returncode != 0:
        raise HarnessInfraError(
            f"harness produced no valid report for {instance_id}/{model} "
            f"(report={result.report_path}, rc={result.returncode}); leaving cell MISSING"
        )
    return {
        "task_id": instance_id,
        "model": model,
        "pass": bool(result.resolved.get(instance_id, False)),
        "in_tok": patch.in_tok,
        "out_tok": patch.out_tok,
        "calls": patch.calls,
        "real_cost": patch.cost,
        "timeout_flag": False,
        # Record the digest the harness ACTUALLY used so stored == produced.
        "image_digest": image_version.used_image_digest(spec.image_ref) or "",
    }
