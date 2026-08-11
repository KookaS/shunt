---
title: Results
description: Measured routing and escalation results on Shunt's own benchmark, with every figure, every caveat, and every null reported as a null.
---

# Results

Everything on this page is measured on our own benchmark. Where a result is a
null, it is reported as a null. Nothing here uses a precomputed leaderboard
number.

The short version: cheap-first allocation with verified escalation reaches
always-frontier quality for a fraction of the cost, and — new this cycle — it
now does so on a **cache-safe** strategy rather than only on a blocked one.
`Session-Cascade`, which makes one decision per session and is what a default
install actually runs, costs **$28.71 cache-aware at 96.74%** on the 184-task
scoring path, against `Price-Cascade`'s $27.11 at the same pass rate and
Always-Frontier's $96.02 at 95.11%. On the harder, fully-measured 74-task set it <!-- frozen-value: n=74, date=2026-08-11, run=49b8362 -->
costs **$28.76 at 90.91%** against Always-Frontier's $37.63 at 86.36%. Details,
both sets, and every interval: [Routing at session
cadence](#routing-at-session-cadence).

Two things temper it. The margin is much smaller on measured cells than on
imputed ones — the four-fold saving above is largely an artifact of imputation,
and we now show both bases side by side rather than let you find that yourself.
And the machine learning still contributes nothing: the saving comes from
mechanism, not prediction. The escalation signal is real but sits in the
recurrence policy at the shipped threshold once the reproduction phase is
excluded, not in the prefix risk model (which reads no skill).

We do **not** claim the project's make-or-break gate is passed — that gate is
written about the owner's own coding-agent workflow, and everything here is
SWE-bench. The reasons are in [Routing results](#routing-results) and
[This is not the make-or-break gate](#this-is-not-the-make-or-break-gate).

## How to read this page

**Two models, two different jobs.** Shunt learns two separate things. Conflating
them makes every number ambiguous, so they stay apart throughout.

| | **Routing model** | **Escalation model** |
|---|---|---|
| Question | Which (model, effort) should *start* this task? | Is this attempt going to fail, escalate *now*? |
| When | Once, at the task boundary, before any tokens are spent | Mid-task, at decision boundaries |
| Input | The task text, embedded | The running trajectory: discussion, tool use, verified check results |
| Learns from | Task outcome, pass/fail | Whether this attempt ultimately failed |
| Today | k-nearest-neighbours over task embeddings | A recurrence rule over verified failing-check ids |
| Status | No measurable signal over the base rate yet | Policy: `OK_OFFLINE_ONLY` — real edge at the shipped threshold once the reproduction phase is excluded (eval-only); prefix model: `NO_SKILL` |
| Next | bigram / linear models, calibrated classifiers, better selection rules | calibrated risk scoring, structural loop features, late fusion |

**Rank and cost are different orderings.** We have conflated them ourselves, so,
precisely:

- **Cost** is the dollars a task actually consumed: `real_cost`, cache-aware,
  read from the provider's own usage accounting. Never a price-list estimate.
- **Rank** is a model's position in the registry, which is ordered by **price**.
  It is not a capability ordering.

The two come apart, which is the whole reason routing might be worth doing:

| model | price ($/Mtok) | measured pass rate | 95% CI |
|---|---:|---:|---|
| deepseek-v4-flash | 0.42 | 68.9% | 0.620–0.751 |
| qwen3.7-plus | 1.60 | 43.7% | 0.337–0.541 |
| gpt-5-mini | 2.25 | 54.5% | 0.476–0.613 |
| kimi-k2.5 | 3.60 | 49.6% | 0.408–0.584 |
| zai-glm-5.2 | 5.80 | 57.1% | 0.465–0.672 |
| kimi-k3 | 18.00 | 84.5% | 0.766–0.901 |

`deepseek-v4-flash` costs 5× less than `gpt-5-mini` and solves more. Price does
not buy capability monotonically. A model is only worth its price if it earns it
on *your* tasks. (These rates are each model's marginal rate over the tasks it
actually ran; coverage is adaptive, so they are not cross-comparable at face
value. On the 74 tasks **all six** models ran, the ordering barely survives — and
the cheapest model is the most flattered by adaptive coverage:

| model | pass rate, own coverage | pass rate, common 74 | pooling bias |
|---|---:|---:|---:|
| deepseek-v4-flash | 68.9% | 44.6% | +24.3pp |
| gpt-5-mini | 54.5% | 35.1% | +19.4pp |
| zai-glm-5.2 | 57.1% | 51.4% | +5.8pp |
| kimi-k2.5 | 49.6% | 43.2% | +6.3pp |
| kimi-k3 | 84.5% | 77.0% | +7.5pp |
| qwen3.7-plus | 43.7% | 41.9% | +1.8pp |

That 24.3pp is over four times the 5pp non-inferiority margin the kill gate is
judged at, so read any single-model rate as a marginal over its own task subset,
never as a comparison.)

## The dataset

The benchmark is a set of **challenges**: SWE-bench Verified tasks we run
ourselves. Each challenge is attempted by multiple **experiments**, one per
(model, reasoning-effort) **arm**, and each runs to a verified pass or fail
judged by the task's own tests.

One run yields two distinct data products:

1. **A pass/fail label per (challenge, arm)**, which supervises the **routing**
   model.
2. **A full trajectory log** (discussion, tool calls, verified check results),
   which supervises the **escalation** model.

Both come deterministically from real agent runs, so both models can be
re-evaluated **offline** against data already on disk. That is what makes
iteration cheap: a new routing rule or a new escalation detector can be tested
without spending another cent. Both are supervised learning problems. The
benchmark is the label factory.

### The assumption that fills the gaps

Running every arm on every challenge is expensive, so unmeasured cells are
filled under a **monotone capability ladder**: above a success is success, below
a failure is failure.

Our own paired test does not confirm it. Across 478 co-measured within-model arm
pair-observations, more reasoning effort is worth **+1.7pp**, exact McNemar
two-sided **p = 0.428**, which is indistinguishable from no effect. Monotonicity
is violated outright on **7.3%** of those pairs (35 of 478): the higher-effort
arm failed a task the lower-effort arm passed. All nine plotted arm pairs have
intervals straddling zero.

We flag this rather than lean on it, because it is load-bearing. It is what
fills every unmeasured cell, and every imputed cell is filled `pass=True`.

## Routing results

**The embedding numbers below have now been re-measured, and they got worse.** The kNN
strategies used to embed the manifest `description` — `<repo>@<commit12> - resolve
<test-node-id>`, median 106 characters — while the agent was handed the upstream
`problem_statement`. The router and the work it routed never saw the same text. The
manifest has been rebuilt with the real statements (median 1185 characters),
`routing_text()` prefers them, and every row below is recomputed on that basis.

Routing quality **fell**: kNN went from 81.71% to **78.26%**, which is inside noise of
Always-Cheap's 75.54%, and Tier-Classifier from 67.43% to **65.76%**. The 106-character
label was not merely uninformative — it was mildly *leaky*, because the repo name it
carried is a weak proxy for task difficulty. Given the correct input, the learned router
is not distinguishable from the trivial policy. The zero-ML rows (Oracle, Price-Cascade,
Always-Cheap, Always-Frontier) use no embeddings and are unchanged.

Seven **router-selection** strategies — each one a rule for picking the model a
task *starts* on — scored on the same 184 tasks (16 unscorable), from
[`strategy_summary.csv`](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/strategy_summary.csv).
An eighth scored strategy, `Session-Cascade`, is not a selection rule at all: it
models the escalation *layer* over whatever base routing chose, so it gets [its
own section](#routing-at-session-cadence) rather than a row here. Costs in this
table are naive per-task sums, cache-blind:

| strategy | passes | pass rate | 95% CI | total cost | avg cost/task | cumulative regret |
|---|---|---:|---|---:|---:|---:|
| Oracle (hindsight, not deployable) | 178 | 96.74% | 94.02–98.91 | $18.33 | $0.0996 | 0.00 |
| Price-Cascade (blocked, not deployable) | 178 | 96.74% | 94.02–98.91 | $27.11 | $0.1473 | 0.88 |
| kNN-cascade (blocked, not deployable) | 178 | 96.74% | 94.02–98.91 | $30.44 | $0.1655 | 1.21 |
| Always-Frontier | 175 | 95.11% | 91.85–97.83 | $96.02 | $0.5218 | 10.77 |
| kNN | 144 | 78.26% | 72.28–84.24 | $13.21 | $0.0718 | 33.49 |
| Always-Cheap | 139 | 75.54% | 69.02–81.52 | $1.50 | $0.0081 | 37.32 |
| Tier-Classifier (blocked, not deployable) | 121 | 65.76% | 58.70–72.28 | $11.53 | $0.0627 | 56.32 |

Only `Always-Frontier`, `kNN` and `Always-Cheap` name a strategy the router will
accept — `LIVE_STRATEGIES` in `src/shunt/router/policy.py` is the allowlist, and
`router.strategy` is validated against it at boot. The other four rows are real
measurements of things you cannot configure; `benchmark/routing/strategy_class.py`
carries each one's blocker and its path to live.

### The result that matters is which strategy gets there

`Price-Cascade` uses no embeddings, no nearest neighbours, and no training. It
tries the models in ascending price order and stops at the first one whose patch
passes. Setting aside the hindsight `Oracle`, it is the cheapest row in the table
whose quality interval overlaps Always-Frontier's — and it is **not purchasable
today**. Stopping at the first passing patch requires a verified outcome
mid-session; that is more than one decision per session and it breaks
cache-safety, so `price_cascade` is rejected at boot. The $27.11 @ 96.74%
operating point measures a mechanism, not a product capability.

The learned `kNN-cascade` costs **more** ($30.44 against $27.11) for the same
96.74%, and is blocked on the same cache-safety ground. The machine learning is
not paying for itself. What buys the quality back is **verified escalation** —
which is why the shipped router carries an escalation ladder rather than a
cascade, at a lower ceiling (see [escalation](escalation.md)).

Even the regret ordering is unresolved: Price-Cascade's bootstrap interval on
total regret is [0.56, 1.24] and kNN-cascade's is [0.80, 1.68]. They overlap.

### Measured versus projected

That table is still part projection. Of the dollars behind it:

| strategy | projected share of cost | projected passes |
|---|---:|---:|
| Always-Cheap | 3.3% | 10 of 139 |
| Oracle | 3.9% | 14 of 178 |
| Tier-Classifier | 24.6% | 36 of 121 |
| **Price-Cascade** | **27.7%** | 25 of 178 |
| **kNN-cascade** | **28.1%** | 31 of 178 |
| kNN | 36.1% | 30 of 144 |
| **Always-Frontier** | **44.9%** ($43.15 of $96.02) | 89 of 175 |

Every projected cell is filled `pass=True`. Imputation is not neutral: the
always-frontier baseline is charged full price on tasks a cheaper model
demonstrably solved, which is exactly where a router's apparent saving comes
from.

So we cut to the subset where **no cell on either path was projected**. On those
87 co-measured tasks:

- Always-Frontier: **$46.65 @ 92.0%**
- Price-Cascade: **$15.61 @ 93.1%**

Price-Cascade is $31.03 cheaper, with 1 versus 0 discordant task, McNemar exact
**p = 1.000**. The saving survives measurement, with no quality difference
resolved either way.

That 87-task cut is *pairwise*: it keeps the tasks where the **two strategies
being compared** both landed on measured cells, so it admits a different task set
for every pair. A stricter basis — every strategy measured on the same tasks —
now exists, and on it the saving is much smaller than a third of the baseline.
See [the correction](#the-correction-the-saving-is-much-smaller-on-measured-cells).

### Why we still do not call the gate passed

Three reasons, which we would rather state than have you find.

1. **The measured subset is opportunistic.** Those 87 tasks are what survived
   after removing every projected cell, not a pre-registered sample. Closing this
   with a designed run is the top priority.
2. **The two quality figures are not the same kind of number.** A cascade stops
   at the first attempt whose tests pass and is then scored on that same label,
   so 92.9% is a *best-of-N coverage* statistic while Always-Frontier's 91.7% is
   single-shot. We flagged exactly this pattern as a flaw in published work; it
   applies to us too. The **cost** axis is honest, because every attempt in the
   chain is billed. The quality axis flatters any retry strategy, ours included.
   We do not yet know how to remove this.
3. **The stopping oracle is not free in production.** In the benchmark, "did it
   pass" comes from the task's own test suite. On your machine that signal comes
   from your tests, which is the whole premise, but it is a real dependency.

### The plain kNN strategy is weaker still

It buys about 2.7pp of pass rate over always-cheapest for roughly 8.8× the cost,
and that margin sits far inside both intervals: [72.28, 84.24] against
[69.02, 81.52]. Indistinguishable at this sample size.

## Routing at session cadence

Both cascades above are cheap because they retry *inside* a task and read a
verified outcome after every attempt. That is precisely what the router rejects
at boot. So the interesting question was never "how cheap is a cascade" — it was
**how much of that survives when the ladder is paced the way Shunt actually runs
it.**

`Session-Cascade` answers it. One decision per session; the effort rung first
(same model, higher reasoning — the prompt cache survives), the rank rung only
once effort is exhausted; the climbed rank persists into the next session; every
attempt is billed. Nothing switches inside a cached turn, so it is cache-safe by
construction rather than by policy.

**It is the mechanism that ships.** `escalation.enabled: true` and
`escalation.rank_shortlist: 3` are the defaults in
`src/shunt/config/router.yaml`, so this ladder runs in a default install.
`benchmark/routing/strategy_class.py` nonetheless classifies the row **BLOCKED**,
and the blocker is narrow and worth stating exactly: `Session-Cascade` is not a
`router.strategy` value, because it is not a router. It models the escalation
**layer** sitting over whatever base routing chose, and no `router.strategy`
value can name a different layer. **The name is blocked; the mechanism is not.**
Do not read that row as something you cannot run — unlike `Price-Cascade`, you
are running it.

One modelling assumption, stated up front. The live counter escalates on
*recurrence of the same normalized failing-check id*, and `results.csv` records a
per-cell pass/fail with no failure identity. The replay therefore gives each task
one stable key, so every failure of a task recurs identically. That is the
assumption most favourable to escalation — the ladder climbs as fast as the
policy ever could — which makes every cost below a **lower bound** on live spend,
never an optimistic one.

### The two task sets, and why totals do not cross between them

The result is reported on two bases, chosen because they are biased in **opposite
directions**.

> **Read this before comparing anything.** Set A has 184 tasks and set B has 74
> (the count of tasks where all six enabled models were measured, re-derived from
> `benchmark/routing/results.csv`).
> Total costs are sums over tasks, so a set-A total and a set-B total are not
> comparable. Compare only *within* a set. Pass rates and orderings do carry
> across; dollar totals do not.

**Set A — the shipped scoring path, coverage-completed, 184 tasks.** 35% of its
cells are monotone-imputed. Its subset guard, verbatim:

> scored on 184/200 tasks selected by coverage; deepseek-v4-flash passes 74.1%
> here vs 12.5% on the 16 dropped (+61.6pp) — difficulty-biased, not a random
> sample

| strategy | pass rate | 95% CI | naive cost | naive 95% CI | cache-aware cost |
|---|---|---:|---|---:|---:|
| Oracle (hindsight — bound) | 96.74% | — | $18.33 | [11.05, 27.18] | $18.33 |
| Price-Cascade (blocked) | 96.74% | 94.02–98.91 | $27.11 | [17.92, 37.81] | $27.11 |
| **Session-Cascade, `rank_shortlist=3` (shipped)** | **96.74%** | 94.02–98.91 | $33.56 | [22.53, 46.14] | **$28.71** |
| kNN-cascade (blocked) | 96.74% | 94.02–98.91 | $30.44 | [20.39, 42.06] | $30.44 |
| Session-Cascade, `rank_shortlist=0` (pre-shortlist) | 96.57% | 93.71–98.86 | $48.19 | [29.32, 69.57] | $35.79 | <!-- frozen-value: n=180, date=2026-08-10, run=49b8362 -->
| Always-Frontier | 95.11% | 91.85–97.83 | $96.02 | [88.73, 104.19] | $96.02 |
| kNN (shipped router) | 78.26% | 72.28–84.24 | $13.21 | [9.15, 17.76] | $13.21 |
| Always-Cheap | 75.54% | 69.02–81.52 | $1.50 | [1.31, 1.71] | $1.50 |
| Tier-Classifier (blocked) | 65.76% | — | $11.53 | [8.85, 14.62] | $11.53 |

**Set B — raw, un-imputed, fully-measured tasks only: 74 scorable.** Every <!-- frozen-value: n=74, date=2026-08-11, run=49b8362 -->
cell here was actually run; the scorable count is re-derived from
`benchmark/routing/results.csv` (tasks where all six enabled models were measured),
never a hardcoded number. Its subset guard, verbatim:

> scored on 66/74 tasks selected by coverage; deepseek-v4-flash passes 50.0% here
> vs 0.0% on the 8 dropped (+50.0pp) — difficulty-biased, not a random sample

| strategy | pass rate | 95% CI | naive cost | naive 95% CI | cache-aware cost |
|---|---:|---|---:|---|---:|
| Price-Cascade | 90.91% | 83.33–96.97 | $27.10 | [19.49, 36.56] | $27.10 |
| **Session-Cascade, `rank_shortlist=3` (shipped)** | **90.91%** | 83.33–96.97 | $33.75 | [24.39, 44.23] | **$28.76** |
| Session-Cascade, `rank_shortlist=0` | 90.91% | 84.85–96.97 | $66.89 | [47.66, 89.68] | $49.34 | <!-- frozen-value: n=74, date=2026-08-11, run=49b8362 -->
| Always-Frontier | 86.36% | 80.30–93.94 | $37.63 | [30.57, 45.58] | $37.63 |
| kNN-cascade | 90.91% | 84.85–95.45 | $34.09 | [24.40, 46.95] | $34.09 |
| kNN | 54.55% | 42.42–66.67 | $12.34 | [6.45, 18.85] | $12.34 |
| Always-Cheap | 49.25% | 37.31–62.69 | $0.72 | [0.57, 0.88] | $0.72 |

Note the guards run in **opposite directions**. Set A drops the tasks the cheap
model almost never solves, so it is biased easy; set B drops tasks the cheap
model *never* solves, from a pool where it solves only half, so it is biased
hard. Always-Cheap reads 75.54% on one and 49.25% on the other, which is how far
apart the two selections are. **The ordering survives both. That is the
load-bearing point** — not either set's dollar figure.

### Cache-aware against naive

Both cost columns are reported because they are different quantities, and only
one of them is the bill.

`Session-Cascade` is the **only** strategy in either table the cache term moves.
That is not an accident of the model: it is the only strategy that re-serves the
*same* model on consecutive attempts, because its first escalation rung raises
reasoning effort rather than rank. Every other strategy here either never retries
or steps to a different model on each attempt, which forfeits the cached prefix,
so its naive and cache-aware totals are the same number.

The size of the term tracks how much same-model repetition each configuration
does. At `rank_shortlist=3` it removes 14% of the naive total on set A ($33.56 →
$28.71) and 15% on set B ($33.75 → $28.76). At `rank_shortlist=0`, which walks
every rank instead of jumping over the shortlist, the ladder is longer, repeats
more, and the term removes 26% and 26% respectively — a bigger discount on a much
worse total.

**Caching is scoped per task.** A task is one session, and the discount applies
to repeated attempts within it. Summing a whole run as a single cached sequence
would be wrong: consecutive tasks are different sessions with different prefixes,
and charging them at the cache-read rate would invent a discount no provider
offers. The per-model discount and input share behind these numbers are measured
from the registry and the corpus; the hit rate is assumed at 90% and flagged as
assumed wherever it is reported — see
[cache economics](routing.md#fig-cache-economics) for how far that assumption can
move the figure.

### What the paired test says

Total-cost intervals in the tables above are per-strategy and overlap freely,
which resolves almost nothing. The paired per-task bootstrap does better: it
compares strategies on the *same* task, so the task-difficulty variance that
inflates those intervals cancels. Set A, cache-aware, 2000 draws over the 175 <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->
shared tasks:

| comparison | Δ cost | 95% CI | reading |
|---|---:|---|---|
| `sl=3` vs Price-Cascade | +$1.37 | [+0.82, +2.00] | real, and small | <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->
| `sl=2` vs Price-Cascade | +$0.76 | [−0.06, +1.85] | not distinguishable | <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->
| `sl=3` vs kNN-cascade | −$1.23 | [−3.42, +0.73] | not distinguishable | <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->
| `sl=3` vs `sl=0` | −$13.30 | [−21.00, −6.42] | the shortlist pays | <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->
| `sl=3` vs Always-Frontier | −$66.12 | [−74.19, −57.90] | the headline | <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->
| `sl=2` vs `sl=3` | +$0.61 | [−0.42, +1.47] | not distinguishable | <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->

So: paying one decision per session instead of one per attempt costs $1.37 over
`Price-Cascade` on this set, and buys cache-safety and a mechanism you can
actually enable. Against `kNN-cascade` the two are not distinguishable. Against
Always-Frontier the gap is large and the interval is nowhere near zero.

The last row is why the shipped default stays at 3. `rank_shortlist=2` is not
distinguishable from 3 against either `Price-Cascade` or 3 itself, so there is no
evidence to move the default. We are not tuning on a difference we cannot
resolve.

Quality is a separate axis and it resolves less. On set A, `Session-Cascade` and
`Price-Cascade` read an identical 96.74%, and Always-Frontier's 95.11% sits
inside every one of those intervals. On set B the shipped ladder's 90.91%
[83.33, 96.97] point estimate is above Always-Frontier's 86.36% [80.30, 93.94],
but the intervals overlap heavily: **the two are not distinguishable on quality**,
and we do not claim the ladder passes more tasks. What set B shows is a cost
result at quality that is not resolved as different — and note that no paired
bootstrap was run on set B, so its cost gap is weaker evidence than set A's.

### The correction: the saving is much smaller on measured cells

This is the part we held back until this measurement existed, and it revises what
this page used to lead with.

Read the two sets against each other on the same comparison:

| basis | imputed share | Price-Cascade vs Always-Frontier | Session-Cascade `sl=3` (cache-aware) vs Always-Frontier |
|---|---|---|---|
| Set A (184 tasks) | 35% of cells | $27.11 vs $96.02 — **72% cheaper** | $28.71 vs $96.02 — **70% cheaper** | <!-- frozen-value: n=184, date=2026-08-11, run=49b8362 -->
| Set B (74 tasks) | none, 100% measured | $27.10 vs $37.63 — **28% cheaper** | $28.76 vs $37.63 — **24% cheaper** | <!-- frozen-value: n=74, date=2026-08-11, run=49b8362 -->

A four-fold saving becomes roughly a quarter. Both bases put the cascade family
below the fixed-frontier baseline, so the *direction* holds on measured data —
but the **magnitude is mostly imputation**, and it collapses in the direction you
would expect once you see the mechanism: every imputed cell is filled
`pass=True`, which charges Always-Frontier full price on tasks a cheaper model
demonstrably solved. That fill is exactly where a router's apparent saving comes
from, and set B removes it.

Set B is also a stricter basis than the 87-task subset in [Measured versus
projected](#measured-versus-projected) above. That one is *pairwise*
co-measured — it keeps tasks where the two strategies being compared both landed
on measured cells, which is a different, more permissive filter per comparison.
Set B requires every strategy's cells to be measured on the same tasks, so all
seven rows are scored on one honest basis. Where the two disagree about how much
survives, prefer set B.

We are stating this plainly rather than reporting the flattering basis and
letting a reader discover the other one.

### This is not the make-or-break gate

It is strong evidence *toward* it, and it is not it.

The project's kill gate is specific: beat fixed-frontier-with-caching, at equal
quality, **on the owner's own coding-agent workflow.** Everything above is
SWE-bench Verified — a different corpus, a different task distribution, and a
harness where the pass signal is the task's own test suite rather than a working
repository's. A result on SWE-bench does not discharge a gate written about
day-to-day agent traffic, and we are not going to let the two blur together
because the numbers came out well.

Three further reasons the gate stays open, unchanged by this measurement:

1. **Both task sets are coverage-selected, and both guards say so.** Neither is a
   pre-registered random sample; they are biased in opposite directions, which is
   why the ordering surviving both is the claim rather than either total.
2. **The quality axis still flatters any retry strategy.** A ladder scored on
   whether *some* attempt passed is a best-of-N statistic; Always-Frontier's is
   single-shot. The cost axis is honest because every attempt is billed. The
   quality axis is not, and we still do not know how to remove that.
3. **The shipped router's own selection still misses the bar.** The paired
   non-inferiority test on the kNN router against fixed-frontier is *inferior* by
   more than the pre-registered 5pp margin on all three evidence bases — see
   [the kill-gate figure](routing.md#fig-kill-gate). The ladder above sits over
   base routing; it does not repair it.

## Why the learned router adds nothing

Leave-one-task-out routing pass rate for the kNN router adds nothing over the
base rate: at k=2 it is **0.7880**, against Always-Cheap's 75.54%. It sits
inside the shuffled-outcome permutation null band of [0.7663, 0.8315] (null
mean 0.7946, 200 permutations).

Three more readings point the same way:

- **Against a fixed model it loses.** The best single always-one-model policy on
  this suite scores 0.9511. The router is 0.1631 below it. A router that cannot
  beat one fixed model is not routing.
- **Its neighbourhoods are no better than random.** Observed mean neighbourhood
  purity 0.919 sits inside a permutation null of [0.8839, 1.0000]. True chance
  purity is 0.9128 and the majority class alone
  is 0.9543, so a purity near 1.0 is what a constant router scores for free.
- **It is close to a constant function.** At k=20 the router sends 167 of 175 <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->
  tasks to `deepseek-v4-flash` and 8 to `qwen3.7-plus`, differing from
  always-cheapest on 8 tasks, for +$0.32 (+23%) and +1 pass.

Difficulty itself is present in the data (tasks are solved by 0 to 6 of the
enabled models) but is not separable from prompt embeddings here. PCA of the
shipped jina vectors shows hard and easy tasks intermixed, with PC1 and PC2
together carrying only 15.1% of the variance.

### How weak a routing signal could this suite have seen?

The null above is a null on a pipeline that has been *shown* to recover a signal
it is known to contain — the positive control and destroyed-signal null in
`benchmark/routing/instrument_control.py`, which both selection rules now clear.
That licenses one claim and no more: nothing was found **at the strength
probed**. The planted control signal is far stronger than any routing signal a
real corpus would carry, so it cannot tell you whether the null means "there is
nothing here" or "there is something here, below our floor".

`benchmark/routing/sensitivity.py` measures that floor. It re-assigns the real
outcome rows to the real tasks so that a controlled fraction of them line up
with a direction in the real embedding space, sweeps that fraction downward, and
reports the smallest effect the published test still flags at 80% power. Because
planting only re-assigns rows, the permutation null does not move: the bar is the
one quoted above. The floor is reported as the AUROC a perfect reader of the
planted signal would achieve, which is the same unit the escalation section uses.

The answer is not encouraging, and it is a result in its own right: the floor sits
far above any published task-difficulty detector, and under the transfer figure's
own selection-corrected best-over-*k* rule the test does not reach 80% power even
against a *perfectly* predictable corpus. **Read the routing null as "this suite
cannot resolve a plausible routing signal", not as "no routing signal exists."**
Run it yourself — it prints and writes nothing:

```sh
python3 -m benchmark.routing.sensitivity
```

**The one positive signal.** Tasks from the same source repository transfer
slightly better than across repos: a same-repo diagonal advantage of **+0.0330**
(0.7836 versus 0.7506) against a matched shuffled-outcome null of
[−0.0176, 0.0236], **z = +2.93** over 200 permutations. Small, real, and
repo-local rather than task-semantic. It is also fully explained by the embedded
string: the repo name was 14% of it (see the caveat opening this section), so this
is a measurement of the label, not of transfer. It does not survive as evidence
until re-measured on `problem_statement`.

## The oracle-gap decomposition

We decomposed the gap between the hindsight Oracle and the fixed-frontier
baseline to find how much of it is learnable at all. The answer changed the
roadmap.

| | cost | saving vs always-frontier | what it requires |
|---|---:|---:|---|
| Always-Frontier | $96.02 | — | nothing |
| **Price-Cascade** (blocked) | **$27.11** | **71.8%** | **no prediction — but mid-session verification, which is why it is not deployable** |
| Difficulty-only oracle | $14.25* | 83.9%* | perfect difficulty prediction |
| Oracle (exact model) | $18.33 | 80.9% | + hindsight token counts |

A **difficulty-only oracle**, one that always picks the cheapest model that
solves the task and ignores which specific model it is, agrees with the full
Oracle on **170 of 177 tasks (96%)** and costs only **$0.66** more. There is <!-- frozen-value: n=177, date=2026-07-29, run=f7ff37e -->
essentially no "one magic model for this task" effect to capture. The Oracle is
almost entirely *"use cheap when cheap works"*.

That splits the headroom in two. **Both shares below are of the headroom itself,
the $77.69 gap between Always-Frontier and the Oracle, not of the baseline's
total cost:**

- **~90% of the headroom is mechanically available.** Collectable today by
  trying models cheapest-first and stopping at the first verified pass. No model,
  no features, no training.
- **The remaining ~9% of the headroom requires predicting task difficulty**,
  and our kNN router does not predict it: leave-one-out accuracy never beats the
  base rate at any *k*. That is a result about the router as it stands, embedding
  the 106-character label — not about task embeddings, which have not been tested
  here (see the caveat opening [Routing results](#routing-results)).

To make the two denominators reconcilable: that ~9% residual is **~7% of
Always-Frontier's total cost** (about $7.0 of $96.02). Different denominator, same
dollars.

The honest conclusion is neither "routing works" nor "there is nothing here". It
is that the prize is real and large, almost all of it is mechanical, and the
learned part is currently worth nothing.

**Provenance caveat.** Unlike every other table on this page, this decomposition
came from a one-off analysis and is not yet regenerated by a committed script.
The Always-Frontier / Price-Cascade / Oracle costs above ARE the regenerated
`strategy_summary.csv` values (re-scored after the registry `default_arm`
change); the difficulty-only row (marked \*) and the agreement count and the
headroom split still carry the pre-change one-off numbers, which are internally
consistent with the old strategy summary and have not been re-derived. Until the
analysis ships as part of the pipeline, treat this section as the one place here
you cannot reproduce with a single command. Making it reproducible is queued.

## Escalation results

Status: **`OK_OFFLINE_ONLY` — through the recurrence policy, and the shipped threshold itself
once the reproduction phase is excluded.**

`OK_OFFLINE_ONLY` here means a statistical signal on the offline corpus: the best policy cell
clears its permutation null, and its precision interval clears the base rate.
It is not a shippable verdict. The `deployability` field (below) marks the
number **`OFFLINE-ONLY UPPER BOUND`**: the sweep scores one event per step while
production decides once per session, so this is a per-step signal the live
router does not run. "There is a signal" and "you can ship it" are different
sentences, and only the first is being asserted.

### The old evaluation could not have detected success

We rebuilt the escalation evaluation this cycle. The old label was *positional*,
defined as the last few steps of a failed run, so a content-free clock scored
AUROC 0.970 while a perfect task-level oracle capped at 0.757. Any detector
tuned against that was tuned against a clock.

### The shipped threshold is a coin flip — because it counts the reproduction phase

The shipped default (`escalate_after_n=2`, `stale_window=10`) fires on **723 of
723** trajectories: P(fail | fired) = **0.418** — exactly the
base rate, lift 1.00, no edge. This is not a tuning accident: every run, resolved or not,
starts by reproducing the bug, so the first one or two replayed steps are red on the target
F2P test and the counter trips immediately. The counter is counting the target bug at t=0 —
the agent's normal "fail, read the traceback, fix" loop — as if it were evidence the agent is
stuck. As-shipped, this is literally a coin flip, and no `escalate_after_n` tuning below the
run length fixes that.

### The same mechanism discriminates once the reproduction phase is excluded

The eval replays the identical recurrence rule in a second family (`count_from_first_edit`):
failures before the agent's first edit-like action are treated as the reproduction phase and
**not counted**. Measured over the same 723 stamped runs, that variant separates immediately
and strongly:

| cell | fires | P(fail\|fired) | lift | AUROC | len-only |
|---|---:|---:|---:|---:|---:|
| n=2 | 431/723 | 0.589 [0.508, 0.655] | 1.41 | 0.710 | 0.568 |
| n=3 | 354/723 | 0.638 [0.554, 0.703] | 1.53 | **0.722** | 0.576 |
| n=5 | 265/723 | 0.694 [0.612, 0.764] | 1.66 | 0.708 | 0.575 |
| n=10 | 182/723 | 0.808 [0.738, 0.870] | 1.93 | 0.702 | 0.560 |
| n=20 | 115/723 | 0.835 [0.756, 0.900] | 2.00 | 0.636 | 0.560 |

These rows are the `stale_window=1000` cells — a window wide enough to hold n recurrences, and
the family the *status verdict* is selected from. They are **not** the shipped configuration:
Shunt ships `escalate_after_n=2, stale_window=10`, and every committed figure draws that canonical
cell (edit-gated n=2: fires 431/723, P(fail|fired)=0.589, AUROC 0.710). The n=2 row is
identical across windows. The full 30-cell-per-family grid — every `escalate_after_n` from 1 to
50 at both windows — is on the sweep-table figures and in the report JSON.

The n=3 cell clears the family-wise permutation null (AUROC **0.722** against [0.5, 0.5499],
adjusted p = 0.0005 over 2000 permutations) **and** the length-stratified null (0.722 against [0.498, 0.563]) — so the
edge is recurrence beyond run length, not the length of the runs it selects. It fires on 354 of
723 runs — a useful fraction, not a tail. This is the honest read of the escalation idea:
**looking for repeated failures is the right approach; the shipped implementation was counting
the wrong failures.** The edit-gated family is eval-only (production has no per-step action
stream to gate on); closing that gap is a design question, not a data one.

### The as-shipped (reproduction-counted) family only separates at high thresholds

For completeness, the **as-shipped** family — the counter that counts every same-key failure
including the reproduction phase — over 723 stamped trajectories (152 distinct challenges, base
rate 0.418), the sweep varies `escalate_after_n` × `stale_window` (30 cells). The two knobs are
coupled: `_in_window` admits at most `stale_window` events, so reaching *n*
recurrences needs a window at least that wide, and the grid sweeps the window
over {10, 1000}. Its only edge sits at thresholds no one would ship (see the edit-gated family
above for the same mechanism at the shipped threshold):

- The shipped default (`escalate_after_n=2`, `stale_window=10`) fires on **723 of
  723** trajectories: P(fail | fired) = **0.418** — exactly the
  base rate, lift 1.00, no edge. It fires on essentially everything because reproduction
  failures recur at step 1–2.
- As the threshold rises, precision separates from the base rate: n=5 reads 0.423, n=8
  0.449, n=10 0.481, n=15 (`stale=1000`) **0.534** (lift
  1.28), n=20 **0.577** (lift 1.38), and n=30 (`stale=1000`)
  **0.701** (lift 1.68). Every cell reports its OWN marginal challenge bootstrap
  (the family-wise maxT correction is applied to the AUROC null only, never to a
  precision interval — a CI that excluded a cell's own point estimate would not be
  an interval for it). The n=30 cell's marginal interval is **[0.601, 0.782]**, so
  the numbers quoted against it are the ones the report prints.
- The n=30 cell clears the gate outright: AUROC **0.658** against the
  max-over-cells family-wise null 95% **[0.5, 0.5523]**, adjusted **p = 0.0005**.
  The harness's `OK_OFFLINE_ONLY` badge, however, is carried by the edit-gated n=3 cell above
  (AUROC 0.722 at stale_window=1000 — the best SKILLED cell, selected across both families and maxT-corrected, which is NOT the shipped cell the figures draw), not by this one; the gate itself is
  described on
  [the offline-eval page](escalation.md#evaluating-the-detector-offline).
- **About 40% of that excess over chance is run-length selection, and the report now says so.**
  Firing at n=30 requires ≥30 same-key failing steps, which requires a long run, and run length
  is outcome-correlated on this corpus. A pure "run length ≥ threshold" predictor scores AUROC
  **0.561** at the same flag count, and the cell still clears the **length-stratified null**
  (labels permuted within length bins, so the length→failure association survives): AUROC 0.658
  against **[0.536, 0.582]**, p = 0.0005. So the recurrence-specific signal is real but roughly
  about 40% of the raw 0.658-to-0.5 excess is the length of the runs the threshold selects. Both
  references — the length-only baseline and the length-stratified null — are reported on every
  swept cell (JSON `length_baseline_auroc` / `null_auroc_length_stratified`, and a column on the
  sweep table), because a cell that only clears the challenge-block null while matching its
  length baseline is selection, not recurrence.
- `stale_window` is not inert at high n: with the window at 10 the policy stops
  firing once n ≥ 12 — it takes a window at least that wide to collect the
  recurrence. Only the `stale=1000` rows reach the null-clearing edge.

### The prefix risk model is honestly `NO_SKILL`

The other half of the eval fits a continuous risk score from prefix-only
features. On this corpus it reads no signal:

- At depth **10** — the shallowest full-rank, leak-safe depth — prefix AUROC is
  **0.478** and the incremental over the prior floored at chance is **−0.022**,
  inside the family-wise null. This is a real negative, not a bug: the escalation
  signal does not live in a shallow prefix on this corpus.
- Depth 5 is dropped from the reported ladder: its design is rank-deficient (412
  of 414 admitted rows carry an identical feature vector, because in the first
  five replayed steps the agent is still reproducing the bug). Depth 20 is
  dropped too: at that depth the admission test selects failures (run length is
  outcome-correlated), so its near-nonzero incremental measured selection, not
  prefix evidence.
- The minimum detectable effect is ≈ **0.59**, so this corpus cannot resolve a
  weaker detector. The 723 scored runs cluster on only 152 distinct challenges,
  so settling the prefix question needs roughly four times the distinct
  challenges (152 → ~640); more runs per existing challenge buy almost nothing,
  because the clustering already inflates variance ~3×.

### The instrument is valid

The R0 gate passes: a planted, known-learnable signal is recovered by the
assembled pipeline (AUROC **1.000**), and a within-challenge shuffle of the
outcome labels collapses it to chance (**0.535**, inside the band). A second
positive control proves power at the effect size the claim rests on, not only
detectability: with the fired↔failed link made imperfect (70% of failed runs
fire, 40% of resolved runs fire spuriously) the best cell still clears the
family-wise null at AUROC **0.634**. A permanent
test (`tests/escalation/test_instrument_validity.py`) enforces both controls, so the
`NO_SKILL` verdict above is a null on an instrument shown to detect a signal,
not a null on a broken one.

### The retraction, still on record

- **Task identity alone**, which is what the routing model already knows at
  *t=0*, was once published here at **AUROC 0.886**. **We have retracted that
  number.** The prior gave each run the leave-one-out failure rate of *its own
  instance's other runs*, while the cross-validation split grouped by instance —
  so it was scored on labels from its own test fold. A router meeting a new task
  has no such siblings. That leaked quantity still appears in the harness, now
  explicitly labelled as *not* the baseline (it scores 0.804 here). Grouped
  honestly the deployable prior is **0.497**, i.e. no better than chance.
- Earlier prefix increments — **+0.144** at 5 decisions, **+0.061** floored, and
  **+0.076** at the best depth (20 decisions) — were inflated by the
  between-fold base-rate accounting artifact in the prior column and by depth
  selection. Under the corrected methodology the increment is **−0.022** at depth
  10. The comparator is floored at chance, `AUROC(prior + prefix) −
  max(AUROC(prior), 0.5)`, so an anti-predictive baseline cannot be beaten into
  an apparent finding.
- The null permutes labels **within a challenge**, not globally: a global shuffle
  destroys the challenge-level clustering of outcomes, which collapses the prior
  to chance under the null while the observation keeps a real one — the two arms
  then sit in different headroom regimes and the gate has no power. Permuting
  inside each challenge preserves every challenge's outcome multiset, so the
  prior is identical in both arms and only the prefix's contribution is nulled.

So the honest verdict has changed: the escalation signal is real, and it lives in
the recurrence rule at the shipped threshold once the reproduction phase is
excluded. As shipped the counter counts every same-key failure including the
reproduction phase — it fires on everything and reads the base rate — and the
prefix model remains `NO_SKILL`, but the recurrence mechanism, gated on failures
after the agent's first edit, now clears its gate at `escalate_after_n=3`.

### Two caveats that make it harder than it looks

**A data gap, reduced but not closed.** 253 of the committed corpus's trajectories once carried no
per-step outcomes, so the recurrence trigger structurally could not fire on them
— and three models (`kimi-k2.5`, `qwen3.7-plus`, `zai-glm-5.2`) sat at zero
coverage entirely. Because stamping coverage tracked capture date and capture
date correlates with model, model and coverage were confounded. Those runs have
since been re-stamped offline by container replay at zero API cost: 723 of the committed
corpus's trajectories now carry verified per-step outcomes. But coverage is NOT uniform:
the two models that once sat at zero (`qwen3.7-plus` 47/65, `zai-glm-5.2` 16/31)
still carry 3× the unstamped share of the other models, so stamping coverage still
tracks the same model-correlated axis as before — reduced, not eliminated. The 99
unstamped trajectories break down as: 23 whose captured state was
lost mid-run and whose steps the state-capture audit therefore marks *unmeasured*
rather than failed, 30 that carry no state-capture audit record at all (so whether
their capture was lost is unknown, and their stamps cannot be trusted), and 46
that the per-step stamping stage simply never reached. The model/coverage
confound is therefore still present and the prefix `NO_SKILL` verdict above stands
on the complete corpus with that caveat.

**The value is not identified.** Our logging policy never escalates, so
P(escalate) = 0 and the overlap condition that every off-policy estimator
requires fails. No stored trajectory contains an escalation that actually
happened. We currently cannot distinguish "escalation helps" from "escalation
hurts" from this data, at any confidence. The fix is ε-greedy randomisation at
flagged checkpoints with logged propensities.

## Figures

This page is the narrative — what we found and what it means. The figures
themselves are documented one by one, with how to read each axis and what each
one cannot support, in [Routing → Figures](routing.md#figures) and
[Escalation → Figures](escalation.md#figures). Each PNG carries only its claim,
its sample size, and — where a reader could be actively misled — one red line;
see [how the figures are built](benchmark.md#every-figure-explains-itself).

### Routing

Earlier versions of these figures encoded a 106-character `description` label
rather than the SWE-bench problem statement, and every embedding null was
reported as a coverage gap for that reason. That excuse is gone: the task
manifest was rebuilt with the real problem statements (median 1185 characters),
the router embeds them, and the null did not move. Predicting per-task
solvability over 190 tasks with ≥2 measured models, leave-one-out, real jina
embeddings, k=20:

| Input to the encoder | LOO R² |
|---|---|
| 106-char identifier label (old) | −0.0712 |
| Real problem statement (new) | −0.0662 |
| SWE-bench human difficulty tag | **+0.1876** |
| Shuffled-outcome null, 95% | [−0.1128, +0.0172] |

The embedding sits inside the null band on the correct input, while a three-level
human tag clears it on the same pipeline, the same n and the same null. That is a
working positive control beside a negative result, so this is a **falsification**,
not a coverage gap: on this corpus, embedding similarity carries no per-task
outcome signal. See [`embedding_signal.png`](routing.md#fig-embedding-signal).

Each figure is documented individually — how to read its axes, what to look for, and
what it cannot support — beside the mechanism it illustrates:

- **Routing** (14 figures): [routing.md → Figures](routing.md#figures)
- **Escalation** (6 figures): [escalation.md → Figures](escalation.md#figures)

The figures live under `docs/assets/figures/`, one subdirectory per half
(`routing/`, `escalation/`), inside the published docs tree, which is
why the pages above can link them relatively. A committed `figures.json` per half — beside
the code that writes it, in `benchmark/routing/` and `benchmark/escalation/` — records every
figure's full record and its input digest, and a lint gate (SH009) holds that manifest in
bijection with the sections above — so a retired figure cannot leave a stale description
behind it, and a documented figure cannot go missing.

## Where this leaves the project

Routing: the mechanism works, the model does not, and the gate is not passed —
but the routing null is a null on a suite whose minimum detectable effect sits
far above any plausible routing signal, so it bounds our resolution rather than
the idea. What changed this cycle is that the mechanism no longer has to be
quoted from a strategy the router refuses to run: the escalation ladder at
session cadence is cache-safe, ships enabled, and reaches the blocked cascades'
operating point for $1.37 more on the 175-task set ([session <!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->
cadence](#routing-at-session-cadence)). What also changed is the size of the
prize — on fully-measured tasks the cascade family is ~25% cheaper than
fixed-frontier rather than ~75%, and the difference was imputation. The next
moves are a designed measured run to replace both opportunistic subsets, sized
against that floor, and better routing models (bigram and linear, calibrated
classifiers, better selection rules) evaluated against the same nulls.

Escalation: the recurrence mechanism works, and the shipped implementation was
counting the wrong failures. As shipped, the counter counts the reproduction
phase — every run's first reds are the target bug at t=0 — so it
fires on 723/723 runs and reads the base rate: a coin flip. Gated on
failures after the agent's first edit, the same rule separates: at the shipped
threshold n=2 it reads AUROC 0.710 at P(fail|fired)=0.589 (fires 431/723), and
the family's best cell, n=3, reaches AUROC 0.722 at P=0.638 (354/723), both clearing
the family-wise and the length-stratified nulls. At the session cadence the
ladder's value is large: escalating to a frontier model after a cheap session
failed resolves 3.02× more tasks than a same-cost retry (observational). The prefix
risk model remains `NO_SKILL` (the corpus cannot resolve a shallow prefix detector
below AUROC ≈ 0.59), and the edit-gated variant is eval-only — production has no
per-step action stream to gate on. The remaining work is making the post-edit
gate real in the live capture path, more distinct challenges (~640), and ε-greedy
randomisation with logged propensities so the value question becomes identified at
all.

Related reading: [Benchmark](benchmark.md) for how the harness works,
[Benchmark design](benchmark-design.md) for why it is built this way, and
[Benchmark dataset](benchmark-data.md) for what is in it.
