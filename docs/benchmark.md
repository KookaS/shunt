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

The challenge suite is the full **500-instance** SWE-bench Verified set across 12
repos, spanning a spread of difficulty strata, each with a verified prebuilt
SWE-bench image. Live results cover a **nested partial subset** (set by
`sample_size`): the run order is diversity-first and nested, so raising the sample
`10 → 20 → 200 → 500` only adds tasks and reuses already-computed cells. Provenance:
[`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified),
dataset revision `c104f840`.

## Model pool

Prices below are the **Requesty router listing** rates (as of mid-July 2026), in
USD per 1M tokens; each entry carries its own `price_as_of`, `price_note`, and
cache-read/write rate in the model registry (`src/shunt/config/models.yaml`). Four tiers — cheap → mid → high → frontier.

| Model | Tier | Input $/1M | Output $/1M |
|-------|------|-----------:|------------:|
| deepseek-v4-flash | cheap | 0.14 | 0.28 |
| qwen3.7-plus | cheap | 0.32 | 1.28 |
| gpt-5-mini | mid | 0.25 | 2.00 |
| kimi-k2.5 | mid | 0.60 | 3.00 |
| zai-glm-5.2 | high | 1.40 | 4.40 |
| kimi-k3 | frontier | 3.00 | 15.00 |

Spread: ~21x input, ~54x output between the cheapest and the frontier model.
The model registry (`src/shunt/config/models.yaml`) is the single source of truth — the table above is a
snapshot of it. (claude-opus-4-6 is priced in the registry for provenance but is left out of
`benchmark/benchmark.yaml`'s `models` list — excluded from runs; the strongest enabled frontier model is the baseline.)

## Benchmark execution

The live harness runs each `(challenge, model, reasoning-arm)` cell as an isolated,
reproducible Docker job:

1. Resolve the challenge spec at its pinned `base_commit` and dataset revision.
2. Pull the challenge's prebuilt SWE-bench image (per-challenge, by manifest
   digest) — source mounted read-only, with a writable sandbox.
3. Run the coding agent with the target model against the task. The cell's
   reasoning arm is overlaid on the request (e.g. `reasoning_effort`,
   `thinking` on/off), so each arm bills a distinct call.
4. Run the deterministic judge (the spec's `FAIL_TO_PASS` / `PASS_TO_PASS` tests).
5. Record the verified pass/fail, real cost (from the API response), estimated
   cost (from the registry's prices × token counts), and token usage.

Which arms run is `p(arm|model)` exploration sampling: a model's default arm
always runs, and each extra arm runs on a deterministic, cost-skewed fraction of
challenges (hash-thresholded on the challenge id, so a re-run selects the identical
arms). Set `arm_sampling.enabled: false` in `benchmark/benchmark.yaml` to run
default-arm-only, or list models under `arm_sampling.default_only_models` to pin
just those (e.g. the expensive high/frontier tiers) to their default arm while the
rest keep exploring.

Per-challenge images give reproducibility, isolation, and parallelization. Cells
run concurrently with `--workers N` (each worker runs one SWE-bench container, so
raise it with an eye on host memory). Cells complete challenge-at-a-time — every
model (and sampled reasoning arm) for one challenge finishes before the next
challenge starts. `--max-cost USD` stops the run once cumulative real cost crosses
a ceiling, checked at challenge boundaries, so the run keeps a prefix of
**fully-covered** (comparable) challenges rather than many partially-covered ones.
Only model API costs enter routing metrics; judging costs are excluded.

Outcomes are appended to `benchmark/routing/results.csv`. **This file
is populated by live runs** (`python -m benchmark.runner.run_matrix --live`), which need
Docker and API keys.
Each cell is written to `results.csv` the moment it completes (an atomic
temp-file-then-`os.replace`), so a kill or crash only loses the handful of cells
still in flight — never the whole batch; a `--max-cost` stop cuts at a challenge
boundary, leaving no challenge partial.
The evaluator can backtest strategies against cached outcomes; if the cache is empty, it reports
coverage gaps rather than fabricating numbers.

### Result integrity

CI validates every `results.csv` row (`check_integrity.py`): identity anchors (spec
content hash, model version, reasoning-arm hash, image digest) must match the current
source, and
every derivable field is recomputed and cross-checked — the `cost` column against its
derivation rule, `real_cost` against a token-based plausibility floor (a real cost far
below the estimate — an expensive run billed as ~free — fails the build; unusually high
ratios warn), the challenge/model/arm against the registry, and basic plausibility (a
resolved cell must have emitted output and not be a timeout). A
corrupted or hand-edited row fails the build. This is an *internal-consistency* check:
it catches corruption and casual fabrication, not a determined forger reproducing every
invariant — stronger provenance (signed runs) and sampled re-execution are planned.

## Routing evaluation

The routing evaluator is a backtest over the outcome cache. Install the harness
once, then run it:

```bash
pip install -e '.[dev,benchmark]'
python3 -m benchmark.routing.run_eval
```

It scores each strategy by looking up cached `(challenge × model)` cells (the
evaluator uses each model's default reasoning arm). A
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
| External-Prior | SWE-bench leaderboard per-task difficulty prior; escalate on external p_solve signal |
| kNN-blended | kNN over our verified runs plus down-weighted external neighbours (off by default — embedding the external statements is slow) |

The embedding-based strategies are **offline evaluation strategies**, not live product behavior —
the proxy today forwards to a cheap default and calls none of them. The cascade
(try-verify-escalate) exists only here in the benchmark; it is not implemented on
the live request path.

### What the offline eval found about routing

Scored offline, the embedding-based routing strategies split by workload:

- On **QA and reasoning-style** tasks, the task embedding separates
  cheap-solvable from frontier-only work, so kNN has signal to route on.
- On the **agentic-coding** tasks this benchmark targets, it did **not** clear
  our viability bar. Ranking hard tasks from easy ones off the prompt embedding
  came out near chance, so a kNN router has little to exploit. The router is now
  wired into the live proxy (it decides the first turn). Outcomes can be manually
  recorded via `shunt flag`, but automatic capture is not yet wired, so the router
  typically cold-starts every session, and we do not yet claim a live advantage on
  a workload where the embedding signal did not support it.

### Evaluating the exploration policy without spending money

Exploration ships on ([configuration](configuration.md#tune-the-router)), so the
obvious question is what it costs. You can answer it from the committed data alone.
`results.csv` is a near-dense grid of *measured* (task, model) outcomes: 285 of the
49 × 6 cells are filled (96.9%), and 44 tasks have a result for all six models.
Scoring uses each model's default reasoning arm, which drops one more cell — so the
sub-grid the replay actually runs on is 43 tasks × 6 models (the largest fully dense
block, found greedily). On a fully
dense sub-grid, replaying a routing policy is exact rather than estimated: look up
the model the policy picks, read the outcome that was actually recorded for that
cell, average. Nothing is simulated and no request is sent.

```bash
python -m benchmark.routing.scripts.plot_exploration
```

This replays the shipped router — the same Thompson sampler, budget cap, and
conservative gate that run in the proxy — over the matrix, once with exploration
off and once with it on, and writes `routing/reports/exploration_replay.png` plus a
summary to stdout. Cells the policy routes to but the benchmark never ran are
skipped and counted, never filled in with a guess.

On the 43-task dense slice, averaged over 20 seeds: exploration costs **1.10× the
exploration-off bill** on average and **1.22× on the worst seed**, inside the ~1.4×
the default budget allows. The paired per-task difference is **−2.8 pp pass rate
(95% CI −6.5 to +0.3)** and **+$0.013 per task (95% CI −$0.000 to +$0.027)** — the
paired numbers are the ones to read, since the two arms' marginal intervals are far
too wide to separate at n=43.

Three caveats keep this honest. The replay's outcome matrix is **static**, so an
exploratory pull can never improve a later decision — this measures exploration's
cost with its learning benefit set to zero, which is the pessimistic half of the
ledger, not a verdict on whether exploration pays. The budget cap counts the
router's own confidence-weighted neighbourhood costs, not realized ones, so the
realized explore/exploit spend ratio can exceed `explore_budget_frac` on an unlucky
seed (0.85 against a 0.4 cap here) even though the cap is doing its job. And 43
tasks from one benchmark is a small, single-workload sample.

## Deciding the kill-gate on partial frontier coverage

Running the most expensive ("frontier") model on every task is costly, so Shunt can
collect frontier outcomes adaptively instead and estimate the fixed-frontier baseline
statistically. When that mode is on, the gate — *does routing match fixed-frontier
quality at lower cost?* — rests on four stated assumptions:

1. **Missing-at-random audit.** Cheap and mid-tier models run on every sampled task.
   The frontier model runs on every task where cheaper tiers *disagree* (the tasks that
   decide quality) plus a *uniformly random audit* of the rest. The audit is drawn by a
   deterministic salted hash of each task id, so its sampling probability is known and
   uniform — the precondition that makes the baseline estimate unbiased.
2. **Doubly-robust estimator.** The baseline pass-rate and cost are estimated with a
   prediction-powered (PPI++/AIPW) estimator that uses cheap+mid outcomes as covariates.
   The estimate is unbiased *regardless of how well cheap outcomes predict frontier
   outcomes* — a poor predictor only widens the interval. Validity comes from the random
   audit, not from the prediction being good.
3. **Measured non-monotonicity.** Stronger models sometimes fail tasks weaker models
   pass. Shunt measures that violation rate (with a confidence interval) on the audit
   stratum and reports it, rather than assuming it away.
4. **The gate is a paired contrast, not an absolute score.** At this task count the
   interval on the *absolute* frontier pass-rate is wide, so the decision rests on the
   *paired* comparison of routing versus fixed-frontier on the same tasks (a McNemar
   non-inferiority test with an anytime-valid stopping rule) — not on the absolute
   pass-rate. Per-instance oracle-relative regret, which needs every model on every
   task, is reported only where full coverage exists and is never the gate.

This decides quality non-inferiority and estimates baseline cost with honest intervals
from partial coverage. It cannot pin the absolute frontier pass-rate to a tight interval
at this task count, and it cannot decide the gate when routing's true edge over
fixed-frontier is near zero — a near-zero edge is itself the signal to stop.

**Running it.** The runner has two strategies, selected with `--strategy`:

- `cost_optimal` (**default**) — the adaptive collection above. A plain
  `python -m benchmark.runner.run_matrix` runs it: cheap+mid on every task, frontier
  only on disputed tasks plus a random audit. Savings over `full` are
  **scale-dependent** — the measured cheap↔frontier correlation is low (ρ²≈0.04), so the
  gain is modest at small task counts and the gate rests on the paired McNemar contrast
  plus the audit, not the covariate.
- `full` — the exhaustive every-enabled-model × every-sampled-challenge matrix
  (`python -m benchmark.runner.run_matrix --strategy full`). `full --live` with **no**
  `--max-cost` prompts for interactive confirmation before spending (uncapped live spend
  is dangerous); a non-interactive stdin aborts. `cost_optimal` keeps its own
  `constants_pinned` safety guard and needs no such prompt.

`python -m benchmark.runner.collect` is a **deprecated alias** for `--strategy
cost_optimal`. Key `cost_optimal` knobs live under `collect:` in `benchmark/benchmark.yaml`:
`phase_a_mode` (`single` = one representative model per tier, or `full` = every cheap+mid
model), `include_high` (add the high tier to the frontier phase), and the two sizing
constants `audit_fraction` (audit sampling probability π) and `noninferiority_margin`
(δ). Pin those two from the live `results.csv` (its cheap↔frontier correlation and
discriminating-task count) and set `constants_pinned: true` before any paid run, or the
interval is mis-sized.

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
  the model registry.
- **Benchmark ≠ production**: the benchmark can reject bad routing strategies but
  can't prove a good one works in production. The kill gate — beat a fixed-frontier
  baseline (the most expensive enabled model, currently kimi-k3) with caching at
  equal quality — must be measured on a real workflow, not in the benchmark.
- **Small measured sample, single run**: the suite is 500 tasks but live results
  cover only a nested partial subset so far (all Python), with one stochastic run
  per cell (pass@1), and only ~15–20% of tasks carry routing headroom. See the
  benchmark harness README for the full limitations.

## Citation

```
@inproceedings{jimenez2024swebench,
  title     = {{SWE-bench}: Can Language Models Resolve Real-World GitHub Issues?},
  author    = {Jimenez, Carlos E. and Yang, John and others},
  booktitle = {ICLR},
  year      = {2024}
}
```
