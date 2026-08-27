#!/usr/bin/env python3
"""Consistency probe — run-to-run output variance of deepseek-v4-flash per task."""

# Direct-LLM repetition of arXiv:2602.11619's "When Agents Disagree With
# Themselves" signal at the cheapest faithful grain Shunt can afford: R direct
# one-shot completions of a FIXED solution-outline prompt on the SAME problem
# statement, then measure the text spread across the R outputs. The measured
# pass/fail comes from results.csv (deepseek-v4-flash default arm, uncensored).
#
# REAL-ONLY BOUNDARY. These are REAL model outputs via the project's provider
# wiring (litellm + the registry's deepseek provider; keys loaded by
# `shunt.secrets.load_dotenv_file`, never read here). No mocks, no synthetic
# text, no imputed cells. Raw responses go to a gitignored JSONL under
# `benchmark/routing/artifacts/`. Nothing here writes results.csv.
#
# LIMITATION (state it in every report). This measures DIRECT-LLM output
# variance — a fixed prompt, no tool loop, no environment. The paper's signal is
# AGENTIC-TRAJECTORY divergence (distinct action sequences over a multi-step
# agent run). A null here is not a falsification of the paper; it only says the
# cheap one-shot proxy does/does not carry the signal.
#
# Run order: `--smoke` first (1 task x 3 runs) to prove serving + key + content
# before any real spend; then the full probe. Spend is guarded: the running
# list-price estimate is accumulated and the run aborts above ``COST_CAP_USD``.

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from benchmark import config
from benchmark.routing import censoring
from benchmark.routing.strategies import routing_text
from benchmark.runner.infer import MissingApiKeysError, litellm_model_target
from shunt.secrets import load_dotenv_file

# The single probed model. Direct provider (api.deepseek.com) via the registry.
MODEL: Final[str] = "deepseek-v4-flash"
# The reasoning arm whose measured pass/fail we correlate against. results.csv
# deepseek cells were collected on the default arm (high); probe the same arm.
ARM: Final[str] = "high"
# The paper's signal needs stochasticity; ~0.7 is the pre-registered setting.
TEMPERATURE: Final[float] = 0.7
# R repetitions per task. The paper uses 10; 5 is the cheap floor that still
# gives 10 pairwise text-similarity comparisons per task.
REPS: Final[int] = 5
# Output budget per completion. The model spends most of it on reasoning
# (content is empty on substantive tasks), so cap generously to leave room
# for a real answer after the reasoning trace.
MAX_OUT_TOKENS: Final[int] = 1500
# Running-spend ceiling for the whole probe (list-price estimate).
COST_CAP_USD: Final[float] = 5.0
# Per-completion wall clock — long enough for a normal completion, short enough
# to fail loudly instead of stalling the run on a wedged upstream request.
CALL_TIMEOUT_S: Final[int] = 120
DEFAULT_SEED: Final[int] = 42
# Target number of measured tasks to probe (stratified pass/fail).
N_TASKS: Final[int] = 40

# Positive-control tasks run through the SAME protocol, engineered to sit at
# opposite ends of the consistency axis. The metric must order them so: the
# deterministic control's pairwise similarity must exceed the open-ended
# control's (instrument-validity positive control, adjudicated in the metrics
# script). ``kind`` marks them so the metrics script can isolate them.
CONTROL_TASKS: Final[dict[str, dict[str, str]]] = {
    "ctl_deterministic": {
        "problem_statement": ("What is 2 + 2? Reply with only the number and nothing else."),
        "kind": "deterministic",
    },
    "ctl_open_ended": {
        "problem_statement": (
            "Write a creative short story about anything at all. "
            "Make it 150-250 words. Be as imaginative as you like."
        ),
        "kind": "open_ended",
    },
}

_PROMPT_TEMPLATE: Final[str] = (
    "You are a senior software engineer. Read the following software-engineering "
    "task and produce a detailed step-by-step solution outline describing how you "
    "would fix the issue. Be specific about the files and functions involved.\n\n"
    "TASK:\n{problem_statement}"
)

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    """Lowercase + whitespace-collapse a model output for similarity comparison."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity on normalized text, in [0, 1]."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 1.0
    return len(ta & tb) / len(union)


def mean_pairwise_similarity(texts: list[str]) -> float:
    """Mean pairwise Jaccard similarity across R outputs (R>=2 required)."""
    if len(texts) < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total += _jaccard_similarity(texts[i], texts[j])
            n_pairs += 1
    return total / n_pairs


def _classify_failure(exc: BaseException) -> str:
    """One-line reason the live part must stop, classified per the probe protocol."""
    import litellm  # noqa: PLC0415

    if isinstance(exc, MissingApiKeysError):
        return f"KEY ABSENT: {exc} — stop the live part, leave the harness built for the owner"
    if isinstance(exc, litellm.exceptions.AuthenticationError):
        return f"KEY REJECTED (401) for {MODEL}: {exc} — stop the live part"
    if isinstance(exc, litellm.exceptions.BadRequestError):
        return (
            f"MODEL 404/BAD-REQUEST for {MODEL}: {exc} — provider does not serve "
            "this id as configured; stop the live part"
        )
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return f"RATE-LIMIT for {MODEL}: {exc} — transient; retry later"
    return f"UNHANDLED for {MODEL}: {type(exc).__name__}: {exc}"


def _call_cost(usage: dict[str, Any], pricing: dict) -> tuple[float, float]:
    """``(raw_cost, guard_cost)`` for one call; the guard ceiling is list price."""
    in_tok = int(usage.get("prompt_tokens", 0) or 0)
    out_tok = int(usage.get("completion_tokens", 0) or 0)
    estimate = (
        in_tok * float(pricing.get("input_cost_per_1m", 0.0)) / 1e6
        + out_tok * float(pricing.get("output_cost_per_1m", 0.0)) / 1e6
    )
    provider_cost = float(usage.get("cost", 0.0) or 0.0)
    return (provider_cost if provider_cost > 0.0 else estimate), estimate


def _call_once(task_text: str, arm_params: dict[str, Any]) -> dict[str, Any]:
    """One completion: full output text + usage + finish_reason."""
    import litellm  # noqa: PLC0415

    model_string, model_kwargs = litellm_model_target(MODEL)
    messages = [{"role": "user", "content": _PROMPT_TEMPLATE.format(problem_statement=task_text)}]
    response = litellm.completion(
        model=model_string,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUT_TOKENS,
        timeout=CALL_TIMEOUT_S,
        **arm_params,
        **model_kwargs,
    )
    message = response["choices"][0]["message"]
    # deepseek-v4-flash emits the full reasoning trace in `reasoning_content` on
    # substantive tasks and often leaves `content` empty; the OUTPUT TEXT is the
    # concatenation — that is what a one-shot direct call actually produced.
    reasoning = str(getattr(message, "reasoning_content", None) or "")
    content = str(message.get("content") or "")
    return {
        "output_text": (reasoning + "\n" + content).strip(),
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "finish_reason": str(response["choices"][0].get("finish_reason") or ""),
        "usage": dict(response.get("usage") or {}),
    }


def _sample_tasks() -> list[str]:
    """Stratified sample of measured deepseek default-arm cells (pass/fail balanced)."""
    results = config.load_results()
    cells: list[tuple[str, bool]] = []
    for cid, per_model in results.items():
        per_arm = per_model.get(MODEL) or {}
        row = per_arm.get(ARM)
        if row is None:
            continue
        if censoring.is_censored(row):
            continue
        cells.append((cid, bool(row.get("pass"))))
    by_label: dict[bool, list[str]] = {False: [], True: []}
    for cid, passed in cells:
        by_label[passed].append(cid)
    import random

    rng = random.Random(DEFAULT_SEED)
    for group in by_label.values():
        rng.shuffle(group)
    half = max(N_TASKS // 2, 1)
    # Balanced target; if one class has fewer cells, fill the remainder from the
    # other so the sample is as balanced as the measured cells allow.
    taken = by_label[True][:half] + by_label[False][:half]
    if len(taken) < N_TASKS:
        rest = [t for t in by_label[True][half:] + by_label[False][half:] if t not in taken]
        rng.shuffle(rest)
        taken += rest[: N_TASKS - len(taken)]
    return taken


def _arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Probe deepseek-v4-flash run-to-run output variance on measured tasks"
    )
    ap.add_argument("--config", default="benchmark/benchmark.yaml", help="Path to config YAML")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="One task x min(3, reps) runs, to prove serving + key + content before spend",
    )
    ap.add_argument("--reps", type=int, default=REPS, help=f"Repetitions per task (default {REPS})")
    ap.add_argument(
        "--limit",
        type=int,
        default=N_TASKS,
        help=f"Cap the number of measured tasks (default {N_TASKS})",
    )
    ap.add_argument(
        "--out-dir",
        default="benchmark/routing/artifacts",
        help="Directory for the gitignored consistency_probe_<timestamp>.jsonl",
    )
    ap.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip the two positive-control tasks (default: include them)",
    )
    return ap


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    load_dotenv_file()
    args = _arg_parser().parse_args()
    config.load(args.config)

    pricing = config.load_pricing()
    if MODEL not in pricing:
        print(f"Model {MODEL!r} is not a priced registry model. Aborting before spend.")
        return 1

    tasks = _sample_tasks()
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]
    if not tasks:
        print(f"No uncensored measured {MODEL} {ARM}-arm cells; nothing to probe.")
        return 1

    challenges = config.load_challenges()
    task_meta = challenges.get("tasks", {})
    # Control tasks carry their own problem_statement; measured tasks route via
    # the committed problem text the routing layer embeds.
    task_texts: dict[str, str] = {}
    for tid in tasks:
        task_texts[tid] = routing_text(tid, task_meta.get(tid, {}))
    if not args.skip_controls:
        for cid, ctl in CONTROL_TASKS.items():
            task_texts[cid] = str(ctl["problem_statement"])
    empty = [t for t, text in task_texts.items() if not text.strip()]
    if empty:
        print(f"Empty task text for: {empty[:5]} — refusing to probe on empty input.")
        return 1

    if args.smoke:
        smoke_tasks = tasks[:1]
        reps = min(args.reps, 3)
        print(f"SMOKE: {MODEL} on 1 task ({smoke_tasks[0]}) x {reps} runs")
    else:
        smoke_tasks = list(tasks)
        reps = args.reps
        if not args.skip_controls:
            smoke_tasks += list(CONTROL_TASKS.keys())

    arm_params = config.arm_api_params(MODEL, ARM)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"consistency_probe_{ts}.jsonl"
    cumulative = 0.0
    n_calls = 0
    aborted = False
    failed = False
    with out_path.open("w") as out_f:
        for tid in smoke_tasks:
            kind = "control" if tid.startswith("ctl_") else "measured"
            runs: list[dict[str, Any]] = []
            for run in range(1, reps + 1):
                try:
                    attempt = _call_once(task_texts[tid], arm_params)
                except Exception as exc:  # noqa: BLE001 (classify, never fabricate)
                    print(f"FAILED on {tid}: {_classify_failure(exc)}")
                    failed = True
                    break
                raw_cost, guard_cost = _call_cost(attempt["usage"], pricing[MODEL])
                n_calls += 1
                cumulative += guard_cost
                runs.append(
                    {
                        "task_id": tid,
                        "kind": kind,
                        "run": run,
                        "output_text": attempt["output_text"],
                        "content_len": attempt["content_len"],
                        "reasoning_len": attempt["reasoning_len"],
                        "finish_reason": attempt["finish_reason"],
                        "raw_cost": raw_cost,
                        "guard_cost": guard_cost,
                    }
                )
                print(
                    f"  {tid} r{run} len={len(attempt['output_text'])} "
                    f"finish={attempt['finish_reason'] or '-'} "
                    f"cost=${guard_cost:.6f} (cum ${cumulative:.4f})"
                )
                if cumulative > COST_CAP_USD:
                    print(
                        f"ABORT: cumulative spend ${cumulative:.4f} exceeds cap "
                        f"${COST_CAP_USD:.2f} after {n_calls} calls."
                    )
                    aborted = True
                    break
            if failed or aborted:
                break
            for rec in runs:
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            sim = mean_pairwise_similarity([r["output_text"] for r in runs])
            print(f"  → {tid} {kind} n={len(runs)} mean_pairwise_sim={sim:.3f}")

    print(f"\n{len(smoke_tasks)} tasks x {reps} runs, {n_calls} calls → {out_path}")
    print(f"Total billed (list-price estimate): ${cumulative:.4f}")
    return 1 if (aborted or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
