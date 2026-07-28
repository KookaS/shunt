---
title: Benchmark dataset
description: What the benchmark dataset is — how many SWE-bench Verified tasks are collected and usable, how censored cells and outliers are handled, the two collection algorithms, the hypothesis it tests, and how to run it end to end.
---

# Benchmark dataset

This page describes the **data** behind Shunt's benchmark: what is collected, what is
usable, and how it is produced. For what the benchmark *measures* and how routing
strategies are scored, see [Benchmark](benchmark.md) and
[Benchmark design](benchmark-design.md) — this page does not repeat them.

## What the dataset is

The challenge source is **SWE-bench Verified**, 500 real GitHub bug-fix tasks with
human-verified test sets. We sample **200** of the 500 deterministically (a fixed
seed, so the sample is reproducible). Of those 200, **177 are fully collected and
usable** for analysis; the other **23 remain incomplete and are excluded**. In short:
**for now we run 177 of the 200 sampled** (SWE-bench Verified has 500 total).

A run does not fill every cell of the task × model grid. On the current data:

| Cells | Meaning |
|------:|---------|
| **752** | real, observed cells (a model actually ran the task and was graded) |
| **420** | imputed cells (filled in memory under the monotonicity assumption, never paid for — see [below](#the-two-collection-algorithms)) |
| **40** | unknown (no bracketing observation, so the outcome can't be inferred) |

Six models are compared, ranked by **measured** capability (which, at cold start
before any data, is the price order): `gpt-5-mini` < `kimi-k2.5` < `deepseek-v4-flash`
< `qwen3.7-plus` < `zai-glm-5.2` < `kimi-k3`. The rank is derived from the verified
outcomes themselves, not assumed — see [Benchmark design](benchmark-design.md#equal-coverage-scoring-monotone-rank-imputation) for how the rank is measured.

## Censored cells — why some tasks are held out

A cell is **censored** when its true pass/fail is *unknown*. A censored cell is
neither a pass nor a fail, so it is dropped from analysis rather than counted as a
failure. Two causes:

- **`step_limit`** — the agent hit its step budget (currently 70 steps) without
  finishing. We can't tell whether more steps would have solved the task, so the
  outcome is unknown, not a failure.
- **`wall_limit`** (legacy) — an earlier harness abandoned a cell at a 600-second
  wall-clock timeout and the outcome was lost.
- **`abandoned`** — a hard watchdog reaped a genuinely hung cell.

Censoring is why 23 of the 200 sampled tasks are excluded. A task becomes usable only
once its **crossover** is pinned — the point on the capability ladder where models go
from failing it to solving it. If that crossover depends on a censored cell, it can't
be pinned, so the task stays incomplete and is held out. The full treatment of
censoring is in [Benchmark design](benchmark-design.md#censored-data-why-a-cell-stopped).

These 23 are **left incomplete for now**: they are the hardest tail of the sample. The
strongest models hit the step limit on them, so those cells are censored and the
crossover can't yet be pinned — pinning it needs more measured runs. The excluded
challenge IDs are:

astropy-13398, astropy-13977, astropy-14369, astropy-14598, astropy-8707, astropy-8872,
django-15629, django-16631, pydata/xarray-3993, xarray-6599, xarray-7229, pylint-6386,
pylint-7080, scikit-learn-14710, sphinx-11510, sphinx-7590, sphinx-8548, sphinx-9461,
sympy-12489, sympy-13878, sympy-16597, sympy-17630, sympy-21596.

## Outliers — monotonicity violations

The analysis assumes a **capability ladder**: if a model solves a task, every
higher-ranked model solves it too. Real models aren't perfectly ordered, so this
occasionally breaks — a cheaper, lower-ranked model solves a task that a pricier,
higher-ranked one does not. Each such case is a **monotonicity violation**.

There are **19** violations in the collected data, and **15 of them fall inside the 177
usable tasks** (only 4 are among the excluded challenges). Being usable does not make a
violation harmless, so the safeguard is not exclusion — it is that the analysis flags
every violation and runs a **sensitivity pass**, recomputing each conclusion with the
violating tasks removed. A violation therefore can never silently drive a result.
See [Benchmark design](benchmark-design.md#the-violation-rate-measured-90).

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
at matched quality — has a tighter coverage limit than the 177 usable tasks suggest. It
can only use tasks where **both** the router's chosen model **and** the frontier baseline
have real, measured cells at equal quality. Because the frontier model is the most
expensive and is run sparingly, the number of such paired tasks is currently small
(**~20**) — far below 177. That makes the comparison **suggestive, not yet settled** at
this sample size. It is a coverage limitation, not a result; it resolves as more paired
frontier cells are collected.

## How to run it end to end

Collection uses **real models, real embeddings, and real provider cost** — no proxies,
no synthetic outcomes. Live runs need Docker and provider API keys, and always go
through the `benchmark` extra (a bare `uv run` strips the harness dependencies). See
[Benchmark](benchmark.md#benchmark-execution) for the launch details and safeguards.

### Routing pipeline

`make benchmark` runs the whole lifecycle — **collect → stamp → evaluate → report** —
and prints one consolidated summary:

```bash
make benchmark ARGS="--live --max-cost 2"   # collect live data + process everything
make benchmark ARGS="--from report"         # recompute artifacts from existing data (no spend)
```

The report stage regenerates the routing plots and CSVs under
`benchmark/routing/reports/`. Each stage is also runnable on its own as a debug
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
[Benchmark](benchmark.md#running-it).
