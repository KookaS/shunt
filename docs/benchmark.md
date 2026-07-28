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

## Configuration — which knob tunes what

The two stages have separate controls. The word "strategy" appears in **both** with
different meanings, so keep them apart (see the note below the table).

| Knob | Stage | Tunes |
|------|-------|-------|
| `models:` | collect | **Which models** are enabled for live runs. |
| `--strategy full` \| `cost_optimal` (CLI flag) | collect | **How** the live matrix is sampled — exhaustive vs adaptive frontier collection. A *collection mode*, unrelated to the `strategies:` block. |
| `arm_sampling.weights` | collect | **Reasoning-effort exploration within each model** (`nothink`/`high`/`max` …) — *not* model selection. Per-arm inclusion probabilities by cost rank; the default arm always runs. |
| `arm_sampling.default_only_models` | collect | Models pinned to their default reasoning arm (no effort sweep). |
| `collect.*` (`audit_fraction`, `noninferiority_margin`, `phase_a_mode` …) | collect | Knobs for the `cost_optimal` sampler only. |
| `sample_size`, `seed`, `n_default` | collect | **Which tasks** run and how many (nested order). |
| `strategies.enabled` | evaluate | **Which routing policies are scored offline** over the cache — `oracle`, `always_cheap`, `always_frontier`, `knn`, `knn_cascade`, `external_prior`. |
| `strategies.knn.*`, `knn_cascade.*` … | evaluate | Per-policy hyperparameters (`k`, `success_rate_threshold`, `max_tries`). |
| `routing.control_model` | evaluate | The fixed-frontier baseline the kill-gate is measured against. |

**The two "strategy" words.** `--strategy` (a CLI flag) chooses *how live data is
collected*; `strategies:` (a config block) lists *the routing policies scored on that
data*. A cheap-first cascade is a **policy you evaluate** (`knn_cascade`), never the
way data is collected — a cascade collector would never observe the frontier on easy
tasks and would bias the baseline (see [Deciding the kill-gate on partial frontier coverage](#deciding-the-kill-gate-on-partial-frontier-coverage)).

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
cache-read/write rate in the model registry (`src/shunt/config/models.yaml`). Listed cheapest-first by total price.

| Model | Input $/1M | Output $/1M |
|-------|-----------:|------------:|
| deepseek-v4-flash | 0.14 | 0.28 |
| qwen3.7-plus | 0.32 | 1.28 |
| gpt-5-mini | 0.25 | 2.00 |
| kimi-k2.5 | 0.60 | 3.00 |
| zai-glm-5.2 | 1.40 | 4.40 |
| kimi-k3 | 3.00 | 15.00 |

Spread: ~21x input, ~54x output between the cheapest and the frontier model.
The model registry (`src/shunt/config/models.yaml`) is the single source of truth — the table above is a
snapshot of it. (claude-opus-4-6 is priced in the registry for provenance but is left out of
`benchmark/benchmark.yaml`'s `models` list — excluded from runs; the strongest enabled frontier model is the baseline.)

## Benchmark execution

### One command: the pipeline

A single run collects **both** routing cells (`results.csv`) and escalation
trajectories, and each needs different downstream processing. `make benchmark`
(`python -m benchmark.pipeline`) is the **primary, one-command** way to run the whole
lifecycle end to end:

```bash
make benchmark ARGS="--live --max-cost 2"   # collect + process everything
make benchmark ARGS="--from report"         # recompute artifacts from existing data (no spend)
```

It composes four existing stages in order and prints one consolidated summary:

1. **collect** — `run_matrix` runs the outcome matrix (honours `--strategy`, `--live`,
   `--max-cost`, `--max-cost-overshoot`, `--workers`, `--timeout`, `--step-limit`,
   `--max-start-failures`, `--max-consecutive-failures`, `--check-images`).
   `--step-limit` (default from `benchmark.yaml` `live.step_limit`, 70) is the **primary**,
   model-speed-agnostic per-cell bound: every model gets the same number of agent steps
   regardless of inference speed (wall-clock timing unfairly penalises slow models). `--timeout`
   (default 1800s) is a **generous graceful** wall-clock backstop — it is the agent's own
   `wall_time_limit_seconds`, so hitting it (or the step limit) terminates the run *gracefully*
   with the real `usage.cost` captured. An external hard watchdog fires strictly later
   (`--timeout` + 300s) only as a last resort for a genuine single-call hang; a cell abandoned
   that way still records the partial spend it already incurred, never a fabricated `$0`.
   `--max-consecutive-failures` (default 5) aborts the whole run after that many consecutive
   cell failures of any cause; an unusable API (invalid/empty key, no balance) aborts
   immediately and is never recorded as a fake failure. `--max-start-failures` (default 5)
   aborts the run after that many consecutive container-start failures (image missing /
   registry throttled) instead of repeatedly hammering the registry; reset by any successful
   cell. A `--live` launch first runs a **preflight health check** — one minimal real
   completion against the cheapest enabled model — and refuses to start (exit 2) if the key
   is invalid, out of balance, or the provider is down, before any Docker container is
   created; a transient blip does not refuse. `--check-images` (off by default, NOT implied
   by `--live`) resolves image digests to detect environment drift; it makes a registry query
   per image (~sample_size tasks to Docker Hub), which rate-limits (429) on the swebench
   namespace and can defeat GHCR pre-staging.
2. **stamp** — `offline_replay` derives real per-step outcomes for the *newly collected*
   trajectories only (already-stamped ones are skipped), timeout-bounded per trajectory
   by `--replay-timeout` (default 120s). Skipped entirely on a simulated (non-`--live`)
   run — there are no new live trajectories.
3. **evaluate** — `escalation.run_eval` scores the escalation detector (metrics + plots).
4. **report** — `routing.report` regenerates the routing plots plus
   `capability_evidence.json`, `coverage_table.csv`, and `strategy_summary.csv`.

The final **summary** block prints the routing paired kill-gate line, the escalation
status (SKILL / NO_SKILL), the capability rank order with the strongest-vs-control
check, the band count, and the total real cost, followed by a per-stage ran/failed
ledger. A stage that fails never corrupts collected data or aborts the rest — the
ledger records which stages ran.

Flags: `--no-report` runs only collect; `--from {collect,stamp,evaluate,report}` starts
at a later stage (so a failed report never forces re-collection); `--replay-timeout`
bounds each stamp. Every stage prints a `=== [pipeline] stage: <name> ===` banner so a
supervising monitor can tell collection from reporting.

The four modules below stay independently runnable as **advanced / debug** entrypoints
(`make benchmark-live`, `make offline-replay`, `make escalation-eval`,
`make routing-report`) — reach for them to run or inspect a single stage.

**Always run the harness through the `benchmark` extra.** Use the Make targets
(`make benchmark`, `make benchmark-live ARGS="--live --max-cost 2"`, `make
offline-replay`, `make escalation-eval`, `make routing-report`) or `uv run --extra
benchmark python -m benchmark.pipeline …`. A bare `uv run` auto-syncs the venv to the
default locked deps and **strips** the extra (`mini-swe-agent`, `swebench`,
`matplotlib`), and the live scaffold imports `minisweagent` lazily per cell — so a
stripped venv fails every cell with `No module named 'minisweagent'`, burning time and
budget for zero data.

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

Which arms run is `p(arm|model)` exploration sampling — this tunes **reasoning
effort within a model**, not which model runs. A model's default arm always runs,
and each extra arm runs on a deterministic, cost-skewed fraction of challenges
(hash-thresholded on the challenge id, so a re-run selects the identical arms).
`weights` are per-arm inclusion probabilities by cost rank (`[0.6, 0.4, 0.25]` =
cheapest extra arm on 60% of challenges, next on 40%, priciest on 25%); the flatter
tail keeps the higher-effort arms — the ones the escalation effort-rung targets —
sampled often enough to evaluate. Set `arm_sampling.enabled: false` in `benchmark/benchmark.yaml` to run
default-arm-only, or list models under `arm_sampling.default_only_models` to pin
just those (e.g. the most expensive, highest-rank models) to their default arm while the
rest keep exploring.

Per-challenge images give reproducibility, isolation, and parallelization. Cells
run concurrently with `--workers N` (each worker runs one SWE-bench container, so
raise it with an eye on host memory). Cells complete challenge-at-a-time — every
model (and sampled reasoning arm) for one challenge finishes before the next
challenge starts. `--max-cost USD` stops the run once cumulative real cost crosses
a ceiling. Serial runs (`--workers 1`, the default) enforce it as a **hard per-cell
stop** — no further cell starts once the cap is crossed, so overrun is bounded to at
most one in-progress cell (its final challenge may be left partial). `--max-cost-overshoot
USD` (default `0`) relaxes that hard stop for the serial path only: it allows up to that
many extra dollars to **finish the challenge already in progress**, so a partially-collected
challenge is not discarded — no *new* challenge starts once spend crosses `--max-cost`.
Parallel runs check at challenge boundaries — a challenge's cells are already in flight
together — so they keep a prefix of **fully-covered** (comparable) challenges.
Only model API costs enter routing metrics; judging costs are excluded.

Outcomes are appended to `benchmark/routing/results.csv`. **This file
is populated by live runs** (`make benchmark-live ARGS="--live"`, i.e. `uv run
--extra benchmark python -m benchmark.runner.run_matrix --live`), which need
Docker and API keys.
Each cell is written to `results.csv` the moment it completes (an atomic
temp-file-then-`os.replace`), so a kill or crash only loses the handful of cells
still in flight — never the whole batch. A `--max-cost` stop is per-cell in serial
runs (the final challenge may be partial) and per-challenge in parallel runs.
The evaluator can backtest strategies against cached outcomes; if the cache is empty, it reports
coverage gaps rather than fabricating numbers.

Each row also records **why** the cell stopped in a `stop_reason` column
(`solved` · `unsolved` · `step_limit` · `wall_limit` · `abandoned`). The last three are
**censored** — the cell hit a resource limit, so its true pass/fail is unknown and it is
excluded from completeness crossovers and quality denominators rather than counted as a clean
fail (see the "Censored data" section in the design notes). `timeout_flag` is kept for
back-compat and now means exactly `stop_reason ∈ {wall_limit, abandoned}`; rows written before
the column derive a `stop_reason` on read.

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

### Data-integrity validator (write-time + pre-analysis)

`check_integrity.py` runs in CI, after data already exists. A separate validator
(`benchmark/routing/validate.py`) encodes what a *valid row* means as invariants that
fail loud at two earlier points, so invalid data is never silently recorded or analysed:

- **Accounting-hole (ERROR)** — a **paid** model (registry input/output price > 0) with
  `calls > 0` but `real_cost == 0`. This is the fingerprint of the miss that motivated
  the validator: a live run recorded `$0` frontier cells while real spend was ~$35. A
  genuinely free model (price 0) with zero cost is fine, and **censored** rows (a
  resource-limit stop whose cost may be legitimately unharvested) are exempt.
- **Ran-ness (ERROR)** — a genuine `unsolved` capability fail must have `calls > 0` (the
  agent must have actually run to fail).
- **Schema (ERROR)** — `stop_reason` in the vocabulary; `pass ⇔ stop_reason == solved`;
  `timeout_flag ⇔ stop_reason ∈ {wall_limit, abandoned}`.
- **Well-formedness (ERROR)** — `real_cost`/`in_tok`/`out_tok`/`calls` are finite and
  `≥ 0`; `computed_at` is an ISO-8601 UTC timestamp.
- **Suspicious (WARN)** — a `real_cost == 0, calls == 0` row that was neither run nor
  censored (a legacy/odd row). WARN never aborts.

**Write-time:** the runner validates every row as it is built (`run_matrix._build_row`);
any ERROR raises `DataIntegrityError` and aborts the run rather than persisting a poison
row — the $35 run would have stopped on its first frontier cell.

**Pre-analysis gate:** `uv run --extra benchmark python -m benchmark.validate_results`
scans `results.csv`, prints a report, and exits nonzero on any ERROR. The kill gate runs
it first and **refuses to run** on data with ERROR-severity violations (fail closed).

### Cost reconciliation

The write-time and pre-analysis gates above catch *poison rows*, but they can't tell
you whether the total the harness tracked matches what the provider actually billed.
Requesty exposes no balance API, so tracked cost can silently drift from the real bill
— a live run once spent ~$35 while the harness tracked $1.25. `reconcile-cost` closes
that loop: it sums the tracked `real_cost` over a window and compares it to a
ground-truth billed amount, alarms on drift, independently cross-checks each model's
tokens × posted price against its summed `usage.cost`, and flags `real_cost == 0` rows
that still made calls (accounting holes).

The billed amount is **owner-supplied** — read the total for the window off the
Requesty dashboard and pass it as `--billed`:

```bash
make reconcile-cost ARGS="--billed 34.80 --timestamp 2026-07-27T00:00:00 \
  --start 2026-07-25T00:00:00 --end 2026-07-27T00:00:00"
```

Drift beyond `--tolerance` (default 0.15), a per-model gap beyond `--gap-threshold`
(default 3×), or any accounting hole prints an alarm and **exits nonzero** — so it
gates a supervised run before you scale spend. Each reconciliation appends one row to
an append-only ledger. Omit `--billed` to skip the tracked-vs-billed leg and run only
the cross-check and hole scan.

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

### Every figure explains itself

You should never need this page open to read one of the benchmark's figures. Every PNG
under `benchmark/routing/reports/` and `benchmark/escalation/reports/` carries a footer
with a fixed five-section shape:

| Section | What it tells you |
|---------|-------------------|
| **READ** | what x and y actually are, what one mark represents, and what a good or bad pattern looks like |
| **GOAL** | what you are looking for, phrased spatially where the geometry allows ("aim top-left", "find the brightest cell") |
| **TERMS** | plain-language definitions of the jargon on *that* figure — regret, arm, stratum, prevalence, Wilson CI |
| **NOTE** | non-obvious facts, including numbers computed from the run itself |
| **LIMITS** | in red: what would mislead you — small samples, uneven coverage, hindsight-only reference lines, proxy metrics |

READ and GOAL are mandatory; the rest appear only when they have something to say. The
red LIMITS line is the one to read first — it is where the figure tells you what it
cannot support.

Anything that depends on the data (how many tasks were dropped as coverage gaps, whether
the frontier ran on a subset, whether a detector has no usable signal) is computed at
render time rather than written into the caption, so it cannot go stale as the data
grows. The mechanism is `benchmark/plot_frame.py`, and a lint gate (SH007) blocks any
figure that tries to skip it.

### Regret, and how to read the regret plot

Regret is the bandit/RL measure of decision quality: **how much worse off you are for not
having made the best possible choice.** It is always relative to an optimal baseline — here
the **Oracle**, which routes every task to the ideal model with perfect hindsight.

Per task, `regret = oracle_reward − strategy_reward`, in the benchmark's reward units
(`passed − γ × cost`, γ=0.1). Route where the oracle routed and the regret for that task is 0.
You incur it two ways:

- **Quality regret** — you routed to a model that *failed* a task the oracle solved.
- **Cost regret** — you solved it, but paid for a bigger model when a cheaper one would also
  have passed.

`CumReg` is that per-task gap summed over the task sequence, and
`benchmark/routing/reports/cumulative_regret.png` draws it as one climbing line per strategy.
Reading it:

- The oracle line is **flat at 0** by definition — it is the baseline, not a competitor.
- **Lower and flatter is better.** The **slope** is average regret per task; a steep line means
  the strategy makes costly-or-wrong choices consistently, not once.
- Coverage is uneven (the frontier only ran on a subset), so curves for different strategies
  can span different numbers of tasks. Compare **slope and shape** before the endpoint; the
  plot states how many tasks were dropped as coverage gaps.

The figure carries a one-paragraph version of this definition on the canvas, so it stands on
its own when read outside these docs.

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
- On the **agentic-coding** tasks this benchmark targets, the embedding-based
  difficulty signal did not clear the viability bar for cost-at-equal-quality
  relative to fixed-frontier-with-caching. Ranking hard tasks from easy ones off the
  prompt embedding came out near chance. The router is wired into the live proxy
  (it decides the first turn), outcomes are recorded automatically at session close
  (via off-wire test re-execution when configured), and the learning loop is live.
  On this particular workload, the embedding signal is not presently strong enough
  to justify routing below frontier, though outcomes continue to accumulate.

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
seed (1.29 against a 0.4 cap here) even though the cap is doing its job. And 43
tasks from one benchmark is a small, single-workload sample.

## Scoring every strategy on one task set — monotone-rank imputation

Running the most expensive ("frontier") model on every task is costly, so Shunt collects
frontier outcomes only where they are most informative. That leaves an **asymmetric
outcome matrix**: cheap and mid models ran on nearly every task, the frontier on only a
subset. Scored naively, each routing strategy would be graded on a *different* set of
tasks — apples to oranges. The default report fixes this by **completing the matrix in
memory** so every strategy is scored on one comparable task set. Full method:
[benchmark-design.md](benchmark-design.md).

**Capability is measured per model, not bucketed into fixed tiers.** Shunt derives a
**per-model capability rank** — a strict weakest-to-strongest order — directly from the
verified outcomes, so a cheaper model that measures stronger than a pricier one ranks above
it from the data, with no hand-tuned tier. The order comes from **pairwise dominance on
co-measured tasks** (which model wins more where both actually ran), never from each model's
raw pass-rate — the frontier ran only on the hard subset, so a raw-rate ranking would wrongly
sort it below cheap models scored on easy tasks. A model with too little data falls back to a
price-implied position (cheaper is assumed weaker) until it earns a measured rank. For the
narrative only, the report groups the ranked models into **ordinal bands** — band 1 (weakest)
to band N (strongest). The bands carry no semantic names: each is described purely by
metadata (its member models, price range, marginal-pass-rate range with CI, and the share of
tasks it is the weakest to solve), and the band count is data-driven — adjacent models whose
capability CIs overlap merge into one band. Bands are a grouping of the measured rank, never
the routing unit.

**The monotonicity assumption.** The rank is a capability ladder. The assumption: if a model
solves a task, every higher-ranked model solves it too. From each task's measured cells Shunt
reads the weakest model observed to pass and the strongest observed to fail, then fills the
gaps under that assumption — above a pass is imputed pass, below a fail is imputed fail. This
completion is **recomputed on every run and never written to `results.csv`**: the committed
data stays real-only, and imputation is a pure in-memory analysis layer that flags every cell
as measured or imputed so the two never blur.

**Why the assumption is conservative (this is load-bearing).** Completing the matrix
credits the always-frontier baseline with a *pass* on every task a weaker model already
solved — free quality at frontier cost. So if the assumption is ever wrong for a task, the
baseline's true quality is only *lower* than imputed, and the router's measured advantage
only *larger*. Imputation can understate routing's lead; it cannot flatter it. We impute to
keep exploration cheap, never to make the router look better than it is.

**We measure how often it breaks — we don't assume it away.** Stronger models sometimes
fail where weaker ones pass. Shunt reports that **monotonicity violation rate** as a
first-class, measured number: on the current data it holds about **90% of the time**
(19 violations over 192 multi-observed pairs — holds 90.1%, violation rate 0.099, 95% CI
[0.064, 0.149]). Where a real higher-ranked fail sits below a
real lower-ranked pass, the contradicted cells are kept as measured — never overwritten by an
imputed value — and the task is flagged. The report also ships a **sensitivity check**:
every conclusion is recomputed with the violating tasks excluded, and any result whose sign
or confidence-interval side flips is surfaced, not hidden.

**Honest about coverage.** When imputation is enabled (default), the completed matrix
excludes every **incomplete challenge** — one whose crossover is still bracketed by an
UNKNOWN band (an unclosed gap between the weakest observed fail and the strongest observed
pass). Only tasks with an **established crossover** (complete) feed the analysis, so every
strategy is scored on the same set with no guessing required. This makes `cost_optimal` and
`full` modes comparable to `ladder`, which collects gap-free data by design. On a partially-
collected `cost_optimal` run, excluded challenges are reported at evaluation time; fully
equal coverage across the whole suite is guaranteed by **ladder collection mode** below.

**What the report shows.** The headline is a **paired** cost/quality contrast — the router
versus fixed-frontier on the *same* completed task set — with its confidence interval, and
it stays honest when that interval crosses zero (equal quality is reported as equal, not
spun as a win). On the current coverage-incomplete data, the kNN-cascade strategy matches
fixed-frontier quality (**+2.1 pp, CI crosses zero → statistically equal**) at roughly
**15% lower cost** on the shared measurable set; the full-distribution figure waits on
ladder-mode collection.

**A population estimate, as a cross-check.** Alongside imputation, Shunt can estimate the
fixed-frontier baseline's pass-rate and cost directly from a *uniformly random audit* of
frontier outcomes, using a doubly-robust (PPI++/AIPW) estimator that treats cheap+mid
outcomes as covariates. Its validity rests on the random audit, not on cheap outcomes
predicting frontier ones — a poor predictor only widens the interval. It answers a
different question — the *population* pass-rate with an honest interval — and measures the
same violation rate on its audit stratum, so the two methods cross-check rather than
compete. At this task count the interval on the *absolute* frontier pass-rate is wide,
which is exactly why the gate rests on the paired contrast (a McNemar non-inferiority test
with an anytime-valid stopping rule), not on an absolute score. A near-zero paired edge is
itself the signal to stop.

**Running it.** The runner collects live data in one of three modes, selected with
`--strategy`:

- `cost_optimal` (**default**) — a plain `python -m benchmark.runner.run_matrix`: cheap+mid
  on every task, frontier only on tasks where cheaper models *disagree* plus a uniformly
  random audit. The cheapest way to a defensible baseline estimate; the measured
  cheap↔frontier correlation is low (ρ²≈0.04), so the gate rests on the paired contrast plus
  the audit, not the covariate.
- `ladder` — cheap-first, escalating per task weakest-to-strongest model only until the
  first one passes. Observes each task's crossover model exactly, giving gap-free equal
  coverage at minimum spend — the mode that makes the imputed matrix fully equal across the
  whole suite. It escalates up to `--workers` **different challenges concurrently** (the same
  fan-out mechanism and default as `cost_optimal`/`full`); *within* a challenge escalation
  stays serial cheap→strong, stopping at its first pass. Collection is **challenge-atomic**
  even under concurrency: before starting a challenge it predicts the worst-case cost to fully
  complete it (the median measured cost of its still-untested tiers), and a thread-safe
  worst-case **reservation** means concurrent challenges can never jointly cross `--max-cost`
  — a challenge that doesn't fit isn't started, so none is ever left half-collected. The set of
  cells collected is identical regardless of `--workers`; only order and wall-clock differ.
  Only challenges whose crossover is established (complete) feed the analysis.
  Optionally pass `--tasks-file <path.json>` (a JSON list of challenge ids) to run only a
  targeted subset (e.g. only the challenges whose crossover is still unknown); cached rungs
  are reused on resume.
- `full` — the exhaustive every-enabled-model × every-sampled-challenge matrix
  (`--strategy full`). `full --live` with **no** `--max-cost` prompts for interactive
  confirmation before spending (uncapped live spend is dangerous); a non-interactive stdin
  aborts. `cost_optimal` keeps its own `constants_pinned` safety guard and needs no such
  prompt.

`python -m benchmark.runner.collect` is a **deprecated alias** for `--strategy
cost_optimal`. Key `cost_optimal` knobs live under `collect:` in `benchmark/benchmark.yaml`:
`phase_a_mode` (`single` = one representative model from the lower-ranked models, or `full` = every lower-ranked
model), and the two sizing
constants `audit_fraction` (audit sampling probability π) and `noninferiority_margin` (δ).
Pin those two from the live `results.csv` and set `constants_pinned: true` before any paid
run, or the interval is mis-sized.

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
