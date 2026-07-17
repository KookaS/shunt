---
title: Benchmark
description: How Shunt benchmarks model capability and evaluates routing strategies offline against verified SWE-bench outcomes.
---

# Benchmark

Shunt's benchmark answers one question: *which routing strategy maximizes reward
(performance − λ·cost)?* It runs in two stages. A live harness executes coding
challenges against each model and records verified pass/fail outcomes. A routing
evaluator then scores strategies offline against that outcome cache — no extra API
spend.

## Challenge source

The sole challenge source is **SWE-bench Verified** — real GitHub bug-fix tasks
with human-verified test sets. Each task is a minimal spec under
`benchmark/challenges/swebench_verified/{instance_id}.json` carrying the upstream
`repo`, `base_commit`, `version`, `FAIL_TO_PASS` / `PASS_TO_PASS` test sets, and a
pinned `dataset_revision`. Repo and patch content are pulled on demand by the
official harness — nothing is vendored.

The current suite is **10 instances** across 10 repos, spanning a spread of
difficulty strata, each with a verified prebuilt SWE-bench image. Provenance:
[`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified),
dataset revision `c104f840`.

## Model pool

Prices below are the **Requesty router listing** rates (as of mid-July 2026), in
USD per 1M tokens; each entry carries its own `price_as_of`, `price_note`, and
cache-read/write rate in `models.json`. Four tiers — cheap → mid → high → frontier.

| Model | Tier | Input $/1M | Output $/1M |
|-------|------|-----------:|------------:|
| deepseek-v4-flash | cheap | 0.14 | 0.28 |
| qwen3.7-plus | cheap | 0.32 | 1.28 |
| gpt-5-mini | mid | 0.25 | 2.00 |
| kimi-k2.5 | mid | 0.60 | 3.00 |
| zai-glm-5.2 | high | 1.40 | 4.40 |
| kimi-k3 | frontier | 3.00 | 15.00 |

Spread: ~21x input, ~54x output between the cheapest and the frontier model.
`models.json` is the single source of truth — the table above is a snapshot of it.
(claude-opus-4-6 is priced in `models.json` for provenance but is `enabled: false`
— excluded from runs; kimi-k3 is the frontier baseline.)

## Benchmark execution

The live harness runs each `(challenge, model)` cell as an isolated, reproducible
Docker job:

1. Resolve the challenge spec at its pinned `base_commit` and dataset revision.
2. Pull the challenge's prebuilt SWE-bench image (per-challenge, by manifest
   digest) — source mounted read-only, with a writable sandbox.
3. Run the coding agent with the target model against the task.
4. Run the deterministic judge (the spec's `FAIL_TO_PASS` / `PASS_TO_PASS` tests).
5. Record the verified pass/fail, real cost (from the API response), estimated
   cost (from `models.json` × token counts), and token usage.

Per-challenge images give reproducibility, isolation, and parallelization. Only
model API costs enter routing metrics; judging costs are excluded.

Outcomes are appended to `benchmark/routing/results.csv`. **This file
is populated by live matrix runs** (`run_matrix.py --live`), which need Docker and API keys.
The evaluator can backtest strategies against cached outcomes; if the cache is empty, it reports
coverage gaps rather than fabricating numbers.

## Routing evaluation

The routing evaluator is a backtest over the outcome cache. Install the harness
once, then run it:

```bash
pip install -e '.[dev,benchmark]'
python3 benchmark/routing/run_eval.py
```

It scores each strategy by looking up cached `(challenge × model)` cells. A
strategy whose decision needs an uncached cell is flagged (it can't be
backtested) rather than silently skipped. With an empty cache the evaluator
prints *"no results yet — run the live matrix"* and exits cleanly.

Metrics per strategy:

| Metric | Meaning |
|--------|---------|
| AvgPerf% | Tasks solved correctly |
| AvgPerf_ci_lower / AvgPerf_ci_upper | 95% bootstrap CI on AvgPerf% (resample tasks, B=1000) |
| TotalCost | Total backend model cost (USD) |
| Reward | `Σ(1.0 × passed − γ × cost)` per task (γ=0.1 default) |
| CumReg | `total(oracle_reward) − total(strategy_reward)` |
| CumReg_ci_lower / CumReg_ci_upper | 95% bootstrap CI on CumReg |
| rAcc | Fraction of tasks where strategy picked the same model as the oracle |
| Pareto | True if no other strategy has higher AvgPerf% AND lower TotalCost |

## Strategies

| Strategy | Description |
|----------|-------------|
| Oracle | Upper bound: cheapest model that passes each task |
| Always-Cheap | Route all to the cheapest model (derived from the pricing matrix) |
| Always-Frontier | Route all to the most expensive model |
| Random | Uniform random per task (mean over seeds) |
| kNN | Embed task → retrieve similar → cheapest capable model |
| kNN-cascade | kNN-informed try-verify-escalate |

## Honest limits

- **Task selection bias**: SWE-bench Verified is mostly Python bug fixes, so the
  benchmark doesn't reflect the full distribution of real coding work. Documented
  limitation; addressed by adding diverse task sources later.
- **Timeout handling**: a timeout counts as a fail for that model on that task and
  is recorded in the result row for separate auditing.
- **Cost**: both real (from the API response) and estimated (pricing × tokens) are
  stored; the evaluator can use either.
- **Deterministic judges only**: every task is judged by its test set — no
  LLM-judged tasks. This rules out judge noise but limits task types.
- **Pricing** is taken from the Requesty router listing (2026-07-15); each model
  records its rate, cache-read/write rate, and source in a `price_note` in
  `models.json`.
- **Benchmark ≠ production**: the benchmark can reject bad routing strategies but
  can't prove a good one works in production. The kill gate — beat a fixed-frontier
  baseline (the most expensive enabled model, currently kimi-k3) with caching at
  equal quality — must be measured on a real workflow, not in the benchmark.
- **Small sample, single run**: results are a pilot — few tasks (all Python), one
  stochastic run per cell (pass@1), and only ~15–20% of tasks carry routing
  headroom. See the benchmark harness README for the full limitations.

## Citation

```
@inproceedings{jimenez2024swebench,
  title     = {{SWE-bench}: Can Language Models Resolve Real-World GitHub Issues?},
  author    = {Jimenez, Carlos E. and Yang, John and others},
  booktitle = {ICLR},
  year      = {2024}
}
```
