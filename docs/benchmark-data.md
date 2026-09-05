---
title: Benchmark dataset
description: What the benchmark dataset is — how many SWE-bench Verified tasks are collected and usable, how censored cells and outliers are handled, the two collection algorithms, the hypothesis it tests, and how to run it end to end.
---

# Benchmark dataset

This page describes the **data** behind Shunt's benchmark: what is collected, what is
usable, and how it is produced. For what the benchmark *measures* and how routing
strategies are scored, see [Benchmark](benchmark.md) and
[Benchmark design](benchmark-design.md) — this page does not repeat them.

The escalation detector is scored on a *separate* trajectory corpus with its own provenance, defects and access mechanics: see [Escalation corpus data card](escalation-data-card.md).

## What the dataset is

The challenge source is **SWE-bench Verified**, 500 real GitHub bug-fix tasks with
human-verified test sets. We sample **200** of the 500 deterministically (a fixed
seed, so the sample is reproducible). Of those 200, **181 are fully collected and
usable** for analysis; the other **19 remain incomplete and are excluded**. In short:
**for now we run 181 of the 200 sampled** (SWE-bench Verified has 500 total). <!-- generated-by: benchmark.routing.summary:complete_scored_matrix -->

A run does not fill every cell of the task × model grid. On the current data (the census
the report writes to `benchmark/routing/reports/coverage_table.csv`):

| Cells | Meaning |
|------:|---------|
| **965** | real, observed cells (a model actually ran the task and was graded) |
| **410** | imputed cells (filled in memory under the monotonicity assumption, never paid for — see [below](#the-two-collection-algorithms)) |
| **25** | unknown (no bracketing observation, so the outcome can't be inferred) |

Seven models are compared, ranked by **measured** capability (which, at cold start
before any data, is the price order): `gpt-5-mini` < `kimi-k2.5` < `deepseek-v4-flash`
< `qwen3.7-plus` < `zai-glm-5.2` < `deepseek-v4-pro` < `kimi-k3`. Note that the measured
rank is *not* the price order — `deepseek-v4-pro` is the second-cheapest model in the
pool and ranks second-strongest. The rank is derived from the verified
outcomes themselves, not assumed — see [Benchmark design](benchmark-design.md#equal-coverage-scoring-monotone-rank-imputation) for how the rank is measured.

### The eighth model in the file, and why nothing scores it

The collected data holds an **eighth** model, `zai-glm-5.3-flash` (41 rows). It is in the
model registry but deliberately **not** in the benchmark's enabled list
(`models:` in `benchmark/benchmark.yaml`), and every figure, table and strategy on this
site is scoped to the enabled set — so it appears in no result here.

That is a decision, not an oversight. Its rows were collected as a probe during a window
when the model was priced at zero. A model without a settled, paid price cannot sit on a
capability ladder that a router chooses from: enabling it would make it a selectable rung
for the strategies that pick a model per task, and its apparent cost advantage would be a
pricing artifact rather than a measured one. It stays collected — deleting real
observations would be worse — and unscored, and it will only become a rung if it is
re-collected at a real price.

## Censored cells — why some tasks are held out

A cell is **censored** when its true pass/fail is *unknown*. A censored cell is
neither a pass nor a fail, so it is dropped from analysis rather than counted as a
failure. Two causes:

- **`step_limit`** — the agent hit its step budget (currently 150 steps) without
  finishing. We can't tell whether more steps would have solved the task, so the
  outcome is unknown, not a failure.
- **`wall_limit`** (legacy) — an earlier harness abandoned a cell at a 600-second
  wall-clock timeout and the outcome was lost.
- **`abandoned`** — a hard watchdog reaped a genuinely hung cell.

Censoring is why 16 of the 200 sampled tasks are excluded. A task becomes usable only
once its **crossover** is pinned — the point on the capability ladder where models go
from failing it to solving it. If that crossover depends on a censored cell, it can't
be pinned, so the task stays incomplete and is held out. The full treatment of
censoring is in [Benchmark design](benchmark-design.md#censored-data-why-a-cell-stopped).

These 16 are **left incomplete for now**: they are the hardest tail of the sample. The
strongest models hit the step limit on them, so those cells are censored and the
crossover can't yet be pinned — pinning it needs more measured runs. The excluded
challenge IDs are:

astropy-13398, astropy-13977, astropy-14369, astropy-14598, astropy-8707, astropy-8872,
django-11885, django-15629, django-16631, pylint-7080, scikit-learn-14710, sphinx-11510,
sphinx-9461, sympy-13878, sympy-17630, sympy-21596.

The list moves as coverage grows: a task leaves it the moment a measured cell pins its
crossover. Re-derive it rather than quoting it — it is what `complete_scored_matrix`
prints as the incomplete set on the committed corpus.

## Outliers — monotonicity violations

The analysis assumes a **capability ladder**: if a model solves a task, every
higher-ranked model solves it too. Real models aren't perfectly ordered, so this
occasionally breaks — a cheaper, lower-ranked model solves a task that a pricier,
higher-ranked one does not. Each such case is a **monotonicity violation**.

There are **15** violations in the collected data, across the **190** tasks where at
least two ranked models were actually observed, and **all 15 fall inside the usable
set**. Being usable does not make a
violation harmless, so the safeguard is not exclusion — it is that the analysis flags
every violation and runs a **sensitivity pass**, recomputing each conclusion with the
violating tasks removed. A violation therefore can never silently drive a result.
See [Benchmark design](benchmark-design.md#the-violation-rate-measured-90).

## Known defect: a local model can be silently mutilated behind a `200 OK`

This one is not about which cells are filled. It is about whether the cells that *are*
filled measured what they claim to, and it only affects runs pointed at a **self-hosted**
inference server.

A hosted provider serves a documented context policy. A self-hosted one serves whatever
flags its supervisor chose — and ollama spawns `llama-server` with `--context-shift
--keep 4`, overriding llama.cpp's own defaults (context shift *disabled*, `--keep 0`).
When a trajectory fills the window, half of it is deleted **from the front**, keeping four
tokens. The system prompt, the tool schema, the task instructions and the problem
statement all sit inside the discarded span. Generation then continues with no tools and
no instructions — and the request still returns **HTTP 200**. No status-based retry, no
abort list and no output validator sees it.

It is not hypothetical. On one 19-cell SWE-bench run: **337 shift events across 334 of
1001 requests**, producing **671k of 1.42M generated tokens (47%, roughly 7.8 GPU-hours)
from a mutilated context.** Those cells measure the server's truncation policy, not the
model, and the whole arm was discarded.

The fix is a pre-flight assertion (`benchmark/runner/serving_guard.py`) that runs before
any container starts. On a local endpoint it reads the serving process's own command line
and refuses the run unless **both** hold:

1. context shift is provably **disabled** — "not declared either way" refuses too; and
2. `n_ctx >= prompt_budget + max_tokens`, so the server can never be asked to generate
   past the end of its own window.

Hosted endpoints are exempt, because the concept does not apply to them — but the
exemption is only taken on a *decided* classification. An endpoint whose locality cannot
be decided refuses rather than skipping, because a fabricated pass is the one failure a
guard must not have.

**What the guard does not cover.** Read these before treating a green pre-flight as proof
the whole run was clean:

- **One model per run.** The pre-flight probes a single target — the cheapest enabled
  model — so a run that also serves a *second* local model has that second endpoint's
  serving configuration unchecked.
- **Pre-flight only.** It is asserted once, before the run. Nothing re-reads the serving
  process mid-run, so a server restarted with different flags halfway through is not
  detected.
- **It cannot see out of a container.** The check reads the process table of the machine
  it runs on. From inside a container, a server running on the *host* is invisible, and
  the guard refuses loudly with "no inference server process binds that port" rather than
  passing. This is also exactly ollama's topology — it listens on 11434 and spawns
  `llama-server` on a random high port — so point the benchmark at the `llama-server`
  port directly, or serve `llama-server` yourself.

## The two collection algorithms

Running every model on every task is exactly the cost this project exists to avoid, so
collection is deliberately partial. Two modes gather the data, and a third step fills
the gaps in memory for free.

- **cost-optimal** (default) — an adaptive, cheap-first collector. It runs the cheap
  and mid models on everything, then runs the frontier model only on tasks where the
  cheaper models *disagree* (plus a small uniform audit). This is the cheapest way to
  get routing-relevant signal, and it deliberately leaves gaps where the answer is
  already clear.
- **ladder** — cheap-to-strong escalation. It runs each task from the weakest model
  up, stopping at the first model that passes. This pins every task's crossover
  exactly with no gaps; it is the completeness collector, at the cost of more runs.

**Monotone imputation** then completes the matrix *in memory*: below an observed fail,
every weaker model is a fail; above an observed pass, every stronger model is a pass.
This fills the in-between cells for analysis without paying to run them. Nothing
imputed is ever written back to the committed data — the stored outcomes stay
real-only, and every filled cell is flagged as imputed. The precise rule and its
guarantees are in [Benchmark design](benchmark-design.md#equal-coverage-scoring-monotone-rank-imputation).

## The hypothesis this dataset tests

The dataset exists to **test a hypothesis, not to confirm one**:

> A cache-safe router that embeds each task and routes by nearest-neighbour past
> outcomes can match a strong single frontier model's solve-rate while spending less —
> handling routine work with cheaper models and escalating only the hard tail.

At the current sample size the verdict is **not settled**. The honest read of the
agentic-coding results so far is in [Benchmark](benchmark.md#what-the-offline-eval-found-about-routing)
and on the [home page](index.md#an-honest-result): on this workload the embedding
difficulty signal has not yet cleared the bar. This dataset is how that question gets
answered as coverage grows.

The **cost-at-equal-quality comparison** — router versus a strong single frontier model
at matched quality — has a much tighter coverage limit than the size of the usable set
suggests. It can only use tasks where **both** the router's chosen model **and** the frontier baseline
have real, measured cells at equal quality. Because the frontier model is the most
expensive and is run sparingly, the number of such paired tasks is small.

The numbers in the rest of this section are **pinned to one past snapshot** and are not
re-derived as the corpus grows: on the run of 2026-07-28, against the 177 usable tasks <!-- frozen-value: n=177, date=2026-07-28, run=cece0fd -->
the corpus then held, only about **20** were paired that way. That makes the **cost**
side of the comparison suggestive
rather than settled: it is a coverage limitation, not a result, and it resolves as more
paired frontier cells are collected.

The **quality** side of that same pinned sample is no longer unresolved. On the paired
bootstrap the router passes 16 of 20 against the frontier baseline's 19 of 20, a
pass-rate delta of **-15.0pp with a 90% CI of [-30.0, -5.0]**. The interval excludes
zero, so at 90% confidence the router is measurably **worse** on pass rate here — not
indistinguishable from the baseline. The saving is a saving at lower quality.

## How to run it end to end

Collection uses **real models, real embeddings, and real provider cost** — no proxies,
no synthetic outcomes. Live runs need Docker and provider API keys, and always go
through the `benchmark` extra (a bare `uv run` strips the harness dependencies). See
[Benchmark](benchmark.md#benchmark-execution) for the launch details and safeguards.

### Routing pipeline

`make benchmark` runs the whole lifecycle — **collect → columns → stamp → evaluate → report → figures** —
and prints one consolidated summary:

```bash
make benchmark ARGS="--live --max-cost 2"   # collect live data + process everything
make benchmark ARGS="--from report"         # recompute artifacts from existing data (no spend)
```

The report stage regenerates the routing plots into `docs/assets/figures/routing/` and the derived
CSVs into `benchmark/routing/reports/`. Each stage is also runnable on its own as a debug
entrypoint (`make benchmark-live`, `make offline-replay`, `make routing-report`).

### Escalation evaluation

The escalation detector has its own offline eval, producing its own metrics and plots
(no spend):

```bash
make escalation-eval
```

### Choosing a collection mode

The collection mode is a flag on the live matrix — `cost_optimal` (default) or
`ladder`:

```bash
make benchmark-live ARGS="--live --strategy ladder --max-cost 2"
```

Full mode reference, budget caps, and the row-integrity gates are in
[Benchmark](benchmark.md#benchmark-execution).
