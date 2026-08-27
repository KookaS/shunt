#!/usr/bin/env python3
"""Judge probe — LLM-as-a-judge difficulty labels on the measured task set."""

# Asks several non-DeepSeek models to rate the difficulty of the SAME task set
# `report.py` scores, so their labels can be checked against real measured outcomes
# (the instrument-validity gate for a judge-labeling pipeline).
#
# REAL-ONLY BOUNDARY. Judge outputs are SYNTHETIC training-signal data, never
# benchmark measurements: this script never writes results.csv, never imputes a
# measured cell, and nothing it produces may be quoted as router performance. Raw
# responses go to a gitignored JSONL under `benchmark/routing/artifacts/`. The ONE
# committed exception is the DERIVED per-task projection
# `benchmark/routing/data/judge_difficulty.json`, written by
# derive_judge_difficulty.py (never this script): a difficulty number + measured
# judge cost per task, regenerable from these JSONLs so the committed figures that
# plot a difficulty strategy reproduce on a clean checkout.
#
# Run order follows the probe protocol: cheap judges first, the anchor last, one
# run per (task, judge), temperature ~0. ``--smoke`` runs ONE anchor call to prove
# requesty serves the anchor and the key loads before any real spend; if the
# anchor 404s or the key is absent, the live part stops and reports exactly that.

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from benchmark import config
from benchmark.routing.strategies import routing_text
from benchmark.runner.infer import MissingApiKeysError, litellm_model_target
from shunt.secrets import load_dotenv_file

# The judge pool. Order is the probe protocol's run order: cheap secondary judges
# first, the anchor last. Every judge routes via the requesty provider.
JUDGE_POOL: Final[tuple[str, ...]] = (
    "gpt-5-mini",
    "qwen3.7-plus",
    "gpt-5.6-sol",
    "claude-sonnet-5",
)
ANCHOR_JUDGE: Final[str] = "claude-sonnet-5"

# Runs per (task, judge). Round-2 protocol: R=2 for the cheap judges, R=3 for the
# anchor. gpt-5.6-sol is NOT cheap ($2.50/$15 live requesty listing) — at R=2 x 200
# tasks (~$3.5) it would blow the ~$4 round cap and starve the anchor, so it runs at
# R=1 (a full 200-task label set; no within-judge stability). gpt-5.6-terra is the
# cheap OpenAI-5.6 candidate judge ($1/$6); R=2 gives it run-to-run stability so its
# labels can be adopted as the knn_difficulty source if they match the anchor's.
_JUDGE_RUNS: Final[dict[str, int]] = {
    "gpt-5-mini": 2,
    "qwen3.7-plus": 2,
    "gpt-5.6-sol": 1,
    "gpt-5.6-terra": 2,
    "claude-sonnet-5": 3,
}

# Judges whose requesty one-shot surface cannot yield a label, with the measured reason.
# zai-glm-5.2: `openai/fireworks/glm-5.2` returns ONLY `reasoning_content` — content is
# never emitted, at any max_tokens (verified 400..4000), with every thinking-disable
# param (enable_thinking, thinking, reasoning_effort, reasoning) accepted-and-ignored.
# The live benchmark only ever got text from glm through multi-turn agent loops. Skipped
# loudly rather than fabricating labels from reasoning_content.
_JUDGE_UNAVAILABLE: Final[dict[str, str]] = {
    "zai-glm-5.2": (
        "requesty one-shot `openai/fireworks/glm-5.2` returns only reasoning_content, "
        "never content text (verified at max_tokens 400..4000 and with every "
        "thinking-disable param) — not usable as a one-shot judge"
    ),
}

COST_CAP_USD: Final[float] = 5.0
TEMPERATURE: Final[float] = 0.0
MAX_OUT_TOKENS: Final[int] = 400
DEFAULT_SEED: Final[int] = 42
# Which probe round wrote a record. Round 1 = bare difficulty prompt; round 2 = the
# metadata-augmented prompt (repo / issue title / failing-test path) plus per-judge
# runs. Metrics key on this so a round-1 record never mixes into a round-2 mean.
PROMPT_VERSION: Final[int] = 2
# Per-completion wall clock. A wedged upstream request otherwise blocks the probe
# for litellm's default (up to 10 min) with no output; 90s is long enough for a
# normal completion and short enough to fail loudly instead of stalling the run.
CALL_TIMEOUT_S: Final[int] = 90

# Per-judge temperature. The probe targets ~0 determinism, but gpt-5-family models
# only accept temperature=1 (litellm's client-side validation) — the API default for
# those models is 1 anyway, so passing it explicitly is the closest reachable setting.
_JUDGE_TEMPERATURE: Final[dict[str, float]] = {
    "gpt-5-mini": 1.0,
    "gpt-5.6-sol": 1.0,
    "gpt-5.6-terra": 1.0,
}

# Per-judge extra request params. gpt-5-family models think by default and burn the
# whole output budget on hidden reasoning (finish_reason=length, content=''); the probe
# is about the difficulty LABEL, not the model's reasoning mode, so pin reasoning off
# for the most deterministic, label-only response.
_JUDGE_EXTRA_PARAMS: Final[dict[str, dict[str, Any]]] = {
    "gpt-5-mini": {"reasoning_effort": "none"},
    "gpt-5.6-sol": {"reasoning_effort": "none"},
    "gpt-5.6-terra": {"reasoning_effort": "none"},
}

# Round-2 prompt: the core task text (routing_text: problem_statement, else the short
# description) plus task-level metadata a human would see WITHOUT the gold patch — the
# repo, an issue-title-style first line, and the failing-test reference (file::test
# path) parsed from the task description. The patch itself is the answer and is never
# fed to the judge.
_PROMPT_TEMPLATE: Final[str] = (
    "You are an independent judge rating a software-engineering task for difficulty.\n"
    "Rate how difficult this task is FOR A STRONG FRONTIER CODING AGENT "
    "(a top-of-the-line, state-of-the-art coding model). Do not compare models, "
    "and do not refer to any specific model or company.\n\n"
    "Difficulty scale:\n"
    "  1 = trivial, 2 = easy, 3 = moderate, 4 = hard, 5 = very hard\n\n"
    "Context:\n"
    "- Repository: {repo}\n"
    "- Issue title: {title}\n"
    "- Failing test reference: {module}\n\n"
    "Task description:\n{task_text}\n\n"
    "Reply with EXACTLY one JSON object of this shape (no prose before or after):\n"
    '  {{"difficulty": <int 1-5>, "reasoning": "<one short sentence>", '
    '"confidence": <float 0-1>}}'
)

_DIFFICULTY_RE = re.compile(r'"?difficulty"?\s*[:\s]\s*(\d)')
_CONFIDENCE_RE = re.compile(r'"?confidence"?\s*[:\s]\s*([01](?:\.\d+)?)')


def _task_context(task_id: str, task_meta: dict) -> dict[str, str]:
    """The prompt context for one task: core text + metadata WITHOUT the gold patch."""
    statement = str(task_meta.get("problem_statement") or "").strip()
    title = statement.splitlines()[0][:140] if statement else task_id
    description = str(task_meta.get("description") or "")
    module = description.split("resolve ", 1)[1].strip() if "resolve " in description else ""
    return {
        "repo": str(task_meta.get("repo") or ""),
        "title": title,
        "module": module,
        "task_text": routing_text(task_id, task_meta),
    }


def _classify_failure(exc: BaseException, judge: str) -> str:
    """One-line reason the live part must stop, classified per the probe protocol."""
    import litellm  # noqa: PLC0415

    if isinstance(exc, MissingApiKeysError):
        return f"KEY ABSENT: {exc} — stop the live part, leave the harness built for the owner"
    if isinstance(exc, litellm.exceptions.AuthenticationError):
        return f"KEY REJECTED (401) for {judge}: {exc} — stop the live part"
    if isinstance(exc, litellm.exceptions.BadRequestError):
        return (
            f"MODEL 404/BAD-REQUEST for {judge}: {exc} — requesty does not serve this id "
            "as configured (or the request is malformed); stop the live part"
        )
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return f"RATE-LIMIT for {judge}: {exc} — transient; retry later"
    return f"UNHANDLED for {judge}: {type(exc).__name__}: {exc}"


def _parse_judgement(text: str) -> tuple[int | None, float | None, str]:
    """Parse ``(difficulty, confidence, reasoning)`` from a model's reply."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]):
        if not candidate or candidate.count("{") == 0:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        difficulty: int | None = None
        confidence: float | None = None
        if data.get("difficulty") is not None:
            try:
                difficulty = int(data["difficulty"])
            except (TypeError, ValueError):
                difficulty = None
            if difficulty is not None and not 1 <= difficulty <= 5:
                difficulty = None
        if data.get("confidence") is not None:
            try:
                confidence = float(data["confidence"])
            except (TypeError, ValueError):
                confidence = None
        # A JSON block is authoritative even when a field is invalid: never let
        # the regex fallback resurrect an out-of-range value from a malformed block.
        if "difficulty" in data or "confidence" in data:
            return difficulty, confidence, str(data.get("reasoning") or "")
    d_match = _DIFFICULTY_RE.search(cleaned)
    d = int(d_match.group(1)) if d_match else None
    c_match = _CONFIDENCE_RE.search(cleaned)
    c = float(c_match.group(1)) if c_match else None
    return d, c, text


def _call_judge(judge: str, prompt: str) -> dict[str, Any]:
    """One judged completion: the parsed label plus the real usage for costing."""
    import litellm  # noqa: PLC0415

    model_string, model_kwargs = litellm_model_target(judge)
    temperature = _JUDGE_TEMPERATURE.get(judge, TEMPERATURE)
    messages = [{"role": "user", "content": prompt}]
    response = litellm.completion(
        model=model_string,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_OUT_TOKENS,
        timeout=CALL_TIMEOUT_S,
        **_JUDGE_EXTRA_PARAMS.get(judge, {}),
        **model_kwargs,
    )
    content = response["choices"][0]["message"].get("content") or ""
    return {"content": content, "usage": dict(response.get("usage") or {})}


def _call_cost(usage: dict[str, Any], pricing: dict) -> tuple[float, float]:
    """``(raw_cost, guard_cost)`` for one call; the guard ceiling is the list-price estimate."""
    in_tok = int(usage.get("prompt_tokens", 0) or 0)
    out_tok = int(usage.get("completion_tokens", 0) or 0)
    estimate = (
        in_tok * float(pricing.get("input_cost_per_1m", 0.0)) / 1e6
        + out_tok * float(pricing.get("output_cost_per_1m", 0.0)) / 1e6
    )
    provider_cost = float(usage.get("cost", 0.0) or 0.0)
    return (provider_cost if provider_cost > 0.0 else estimate), estimate


def _judge_once(judge: str, prompt: str, pricing: dict) -> dict[str, Any]:
    """One (judge, task, run) attempt: the parsed label + cost + raw response."""
    result = _call_judge(judge, prompt)
    difficulty, confidence, reasoning = _parse_judgement(result["content"])
    raw_cost, guard_cost = _call_cost(result["usage"], pricing)
    return {
        "difficulty": difficulty,
        "confidence": confidence,
        "reasoning": reasoning,
        "raw_content": result["content"],
        "raw_cost": raw_cost,
        "guard_cost": guard_cost,
        "parsed": difficulty is not None,
    }


def _arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Probe non-DeepSeek judges' difficulty labels on the measured task set"
    )
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to config YAML")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="One anchor call on one task, to prove serving + key before any real spend",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of tasks per judge (0 = all measured tasks)",
    )
    ap.add_argument(
        "--out-dir",
        default="benchmark/routing/artifacts",
        help="Directory for the gitignored judge_probe_<timestamp>.jsonl",
    )
    ap.add_argument(
        "--judges",
        nargs="*",
        default=None,
        help="Override the judge pool (default: gpt-5-mini qwen3.7-plus gpt-5.6-sol "
        "claude-sonnet-5, cheap first, anchor last)",
    )
    ap.add_argument(
        "--cost-cap",
        type=float,
        default=COST_CAP_USD,
        help=f"Abort once cumulative spend exceeds this (default ${COST_CAP_USD:.2f})",
    )
    ap.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids to judge (overrides the sampled measured set) — "
        "used to top up a partially-collected probe round with the exact missing tasks",
    )
    return ap


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    load_dotenv_file()
    args = _arg_parser().parse_args()
    config.load(args.config)

    matrix = config.load_matrix()
    results = matrix.get("results", {})
    if not results:
        print("No results yet — results.csv holds no rows. Refusing to judge on empty data.")
        return 1
    tasks = config.sample_tasks(sorted(results.keys()), seed=DEFAULT_SEED)
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        missing = [t for t in wanted if t not in set(results.keys())]
        if missing:
            print(f"Refusing: task id(s) not in results.csv: {sorted(missing)[:5]}")
            return 1
        tasks = [t for t in wanted]  # preserve the caller's order
    elif args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    challenges = config.load_challenges()
    task_meta = challenges.get("tasks", {})
    task_prompts = {
        tid: _PROMPT_TEMPLATE.format(**_task_context(tid, task_meta.get(tid, {}))) for tid in tasks
    }
    if not any(_task_context(tid, task_meta.get(tid, {}))["task_text"] for tid in tasks):
        print("Every selected task has an empty routing text; nothing to judge.")
        return 1

    if args.smoke:
        judges = [ANCHOR_JUDGE]
        tasks = tasks[:1]
        print(f"SMOKE: {ANCHOR_JUDGE} on 1 task ({tasks[0]})")
    elif args.judges:
        judges = args.judges
    else:
        judges = list(JUDGE_POOL)

    pricing = config.load_pricing()
    for judge in judges:
        if judge not in pricing:
            print(f"Judge {judge!r} is not a priced registry model. Aborting before spend.")
            return 1
        if judge in _JUDGE_UNAVAILABLE:
            print(f"SKIPPED judge {judge!r}: {_JUDGE_UNAVAILABLE[judge]}")
    judges = [j for j in judges if j not in _JUDGE_UNAVAILABLE]
    if not judges:
        print("Every requested judge is unavailable; nothing to run.")
        return 1

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judge_probe_{ts}.jsonl"
    cumulative = 0.0
    n_calls = 0
    aborted = False
    failed = False
    with out_path.open("w") as out_f:
        for judge in judges:
            runs = _JUDGE_RUNS.get(judge, 1)
            print(f"\n=== judge {judge} — {len(tasks)} tasks x {runs} runs ===")
            for tid in tasks:
                for run in range(1, runs + 1):
                    try:
                        attempt = _judge_once(judge, task_prompts[tid], pricing.get(judge, {}))
                    except Exception as exc:  # noqa: BLE001 (classify, never fabricate a label)
                        print(f"FAILED on {tid} run {run}: {_classify_failure(exc, judge)}")
                        failed = True
                        break
                    n_calls += 1
                    cumulative += attempt["guard_cost"]
                    record = {
                        "task_id": tid,
                        "judge": judge,
                        "run": run,
                        "prompt_version": PROMPT_VERSION,
                        "difficulty": attempt["difficulty"],
                        "confidence": attempt["confidence"],
                        "reasoning": attempt["reasoning"],
                        "raw_cost": attempt["raw_cost"],
                        "parsed": attempt["parsed"],
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    tag = f"d={attempt['difficulty']}" if attempt["parsed"] else "PARSE-FAIL"
                    print(
                        f"  {tid} r{run}  {tag}  cost=${attempt['guard_cost']:.6f} "
                        f"(cum ${cumulative:.4f})"
                    )
                    if cumulative > args.cost_cap:
                        print(
                            f"ABORT: cumulative spend ${cumulative:.4f} exceeds cap "
                            f"${args.cost_cap:.2f} after {n_calls} calls."
                        )
                        aborted = True
                        break
                if aborted or failed:
                    break
            if aborted or failed:
                break

    print(f"\n{len(judges)} judges x {len(tasks)} tasks, {n_calls} calls → {out_path}")
    print(f"Total billed (list-price estimate): ${cumulative:.4f}")
    return 1 if (aborted or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
