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

Output goes to `benchmark/routing/results.csv`. The table shows pass rate, total cost, reward, and efficiency for each strategy.

## CodeRouterBench

The canonical routing matrix lives at `benchmark/routing/matrices/coderouterbench100.json` (STORY-1.2 — not yet built). It will contain tasks selected from [CodeRouterBench](https://github.com/LanceZPF/agent-as-a-router) (OOD176) — real SWE-bench bugs where cheap models fail and frontier models pass.

Until the matrix exists, run with `--matrix path/to/your/matrix.json`. The default matrix is gated on STORY-1.2.

4 of 6 models have cached results from the ACRouter paper. Results for deepseek-v4-flash and zai-glm-5.2 are run in-house via sandboxed execution (STORY-4.1).

## Honest limits

- The task set grows organically from real work. N per capability varies.
- The 400s timeout caps slow models. A timeout is missing data, not a pass or fail.
- Cost is estimated from wall time and public pricing, not real token counts.
- All tasks use deterministic judges (pytest or key-match). No LLM-judged tasks — this rules out judge noise but limits task types.
