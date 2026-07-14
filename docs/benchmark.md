---
title: Benchmark
description: How Shunt benchmarks model performance and evaluates routing strategies offline.
---

# Benchmark

Shunt has two related but independent evals. The first logs model runs from real use. The second evaluates routing strategies offline against those results — no extra API spend.

## What it measures

Model runs are logged to `benchmark/.runs/` whenever a session goes through Shunt. Each run captures the model, task prompt, stub file, held-out test, and a verdict (pass/fail).

The number that matters: **discrimination** — how many tasks separate models. If every model passes every task, routing is moot.

## Model pool

| Model | Tier | Input $/1M | Output $/1M |
|-------|------|-----------|------------|
| qwen3.5-plus | cheap | 0.11 | 0.66 |
| deepseek-v4-flash | cheap | 0.15 | 0.60 |
| Qwen3-Max | mid | 0.34 | 1.38 |
| zai-glm-5.2 | mid | 0.50 | 2.00 |
| kimi-k2.5 | mid | 0.01 | 2.90 |
| claude-opus-4-6 | frontier | 5.00 | 25.00 |

Spread: ~45x input, ~38x output between cheapest and most expensive.

## Routing benchmark

The routing benchmark evaluates strategies offline against a task × model performance matrix:

```bash
python3 benchmark/routing/run_eval.py
```

Output goes to `benchmark/routing/results.csv`. Metrics per strategy:

| Metric | Meaning |
|--------|---------|
| AvgPerf% | Tasks solved correctly |
| AvgPerf_ci_lower / AvgPerf_ci_upper | 95% bootstrap CI on AvgPerf% (resample tasks, B=1000) |
| TotalCost | Total backend model cost (USD) |
| Reward | `Σ(1.0 × passed − γ × cost)` per task (γ=0.1 default) |
| CumReg | `total(oracle_reward) − total(strategy_reward)` |
| CumReg_ci_lower / CumReg_ci_upper | 95% bootstrap CI on CumReg |
| rAcc | Fraction of tasks where strategy picked same model as oracle |
| Pareto | True if strategy is on the Pareto frontier (no other strategy has higher AvgPerf% AND lower TotalCost) |

Pareto frontier is computed across all evaluated strategies. A strategy is Pareto-optimal if no other strategy dominates it on both performance and cost.

## Pilot approach

The initial benchmark run is a **10-task pilot** using only cheap/mid models (qwen3.5-plus, deepseek-v4-flash, Qwen3-Max, zai-glm-5.2, kimi-k2.5). Budget cap: ~$10. The pilot runs a **full matrix** (all 5 models on all 10 tasks = 50 runs) to validate the eval harness and measure task discrimination before scaling.

Future scaling to 176 OOD tasks uses **cascade evaluation**: try cheapest model first, escalate on fail. This saves cost by not running expensive models on tasks where the cheap one succeeds.

## Benchmark execution

The benchmark runner is a Python orchestrator that:

1. Checks out each problem at a pinned commit
2. Spins up a Docker container per (problem, model) combination
3. Mounts source code read-only, provides writable sandbox
4. Runs the agent tool (opencode first, others later) with the model
5. Runs deterministic judge (pytest or key-match)
6. Records pass/fail, real cost (from API response), and estimated cost (from pricing × tokens)
7. Repeats for each (problem, model) combination

Containerization provides reproducibility, isolation, and parallelization. For the 10-task pilot, sequential execution is fine; parallel Docker containers scale later.

Cost double-counting boundary: only model API costs enter routing metrics. Evaluation/judging costs are excluded.

## CodeRouterBench

The canonical routing matrix lives at `benchmark/routing/matrices/coderouterbench100.json` (STORY-1.2 — not yet built). It will contain tasks selected from [CodeRouterBench](https://github.com/LanceZPF/agent-as-a-router) (OOD176) — real SWE-bench bugs where cheap models fail and frontier models pass.

Until the matrix exists, run with `--matrix path/to/your/matrix.json`. The default matrix is gated on STORY-1.2.

4 of 6 models have cached results from the ACRouter paper. Results for deepseek-v4-flash and zai-glm-5.2 are run in-house via sandboxed execution (STORY-4.1).

## Honest limits

- **Task selection bias**: If tasks come from SWE-bench (mostly Python bug fixes), the benchmark doesn't reflect the full distribution of real coding work. Documented limitation; addressed by adding diverse task sources in future iterations.
- **Timeout handling**: A timeout counts as a fail for that model on that task. The cascade escalates naturally (fail → next model). Timeout events are recorded in the result row for separate auditing.
- **Cost**: both real (from API response) and estimated (from pricing × token count) are stored. The eval can use either.
- **All tasks** use deterministic judges (pytest or key-match). No LLM-judged tasks — this rules out judge noise but limits task types.
- **Benchmark ≠ production**: the benchmark can reject bad routing strategies but can't prove a good one works in production. The Month-1 kill gate (beat fixed-Opus-with-caching) must be measured in production, not in the benchmark. A pilot with an open-source-heavy company or project is the medium-term goal to validate real-world routing value (see backlog).
