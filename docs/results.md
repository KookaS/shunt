---
title: Results
description: Measured routing and escalation results on Shunt's own benchmark, with every figure, every caveat, and every null reported as a null.
---

# Results

Everything on this page is measured on our own benchmark. Where a result is a
null, it is reported as a null. Nothing here uses a precomputed leaderboard
number.

The short version: cheap-first routing with verified escalation reaches
always-frontier quality for a fraction of the cost ($20.46 against $87.04 over
the full suite; $13.42 against $43.72 on the measured-only subset), the machine
learning contributes nothing to that saving, and the escalation model does not
work. The saving comes from mechanism, not prediction.

We do **not** claim the project's make-or-break gate is passed. The reasons are
in [Routing results](#routing-results).

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
| Status | No measurable signal over the base rate yet | Recurrence policy active; prefix risk model: `NO_SKILL` |
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
| deepseek-v4-flash | 0.42 | 68.3% | 0.613–0.745 |
| qwen3.7-plus | 1.60 | 43.9% | 0.337–0.547 |
| gpt-5-mini | 2.25 | 54.5% | 0.476–0.613 |
| kimi-k2.5 | 3.60 | 50.9% | 0.419–0.598 |
| zai-glm-5.2 | 5.80 | 57.0% | 0.460–0.673 |
| kimi-k3 | 18.00 | 84.1% | 0.760–0.898 |

`deepseek-v4-flash` costs 5× less than `gpt-5-mini` and solves more. Price does
not buy capability monotonically. A model is only worth its price if it earns it
on *your* tasks. (These rates are each model's marginal rate over the tasks it
actually ran; coverage is adaptive, so they are not cross-comparable at face
value. The common-coverage view is in
[the descriptive figure](#per-model-descriptive-view).)

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

Our own paired test does not confirm it. Across 485 co-measured within-model arm
pair-observations, more reasoning effort is worth **+1.6pp**, exact McNemar
two-sided **p = 0.428**, which is indistinguishable from no effect. Monotonicity
is violated outright on **7.2%** of those pairs (35 of 485): the higher-effort
arm failed a task the lower-effort arm passed. All nine plotted arm pairs have
intervals straddling zero.

We flag this rather than lean on it, because it is load-bearing. It is what
fills every unmeasured cell, and every imputed cell is filled `pass=True`.

## Routing results

**Every embedding-based number below predates a fix and is pending re-measurement.**
The kNN strategies embed the manifest `description` —
`<repo>@<commit12> - resolve <test-node-id>`, median 106 characters, of which the repo
name is 14% and a per-task random commit prefix another 12% — while the agent is
handed the upstream `problem_statement`. The router and the work it routed never saw
the same text, so the neighbourhood was built over filenames and repo names rather
than task content. `routing_text()` now *prefers* `problem_statement`, but the key is
absent from all 500 committed challenge specs and all 500 manifest `tasks` entries, so
every row below was still computed on the `description` label. The figures, the
kNN/kNN-cascade/Tier-Classifier rows below, and the kill-gate verdict are regenerated
on a manifest that carries the statement before any of them is quoted again. The
zero-ML rows (Oracle, Price-Cascade, Always-Cheap, Always-Frontier) use no embeddings
and are unaffected.

Seven strategies, scored on the same 177 tasks (23 unscorable), from
[`strategy_summary.csv`](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/strategy_summary.csv):

| strategy | passes | pass rate | 95% CI | total cost | avg cost/task | cumulative regret |
|---|---:|---:|---|---:|---:|---:|
| Oracle (hindsight, not deployable) | 171 | 96.61% | 93.79–98.87 | $13.59 | $0.0768 | 0.00 |
| Price-Cascade | 171 | 96.61% | 93.79–98.87 | $20.46 | $0.1156 | 0.69 |
| kNN-cascade | 171 | 96.61% | 93.79–98.87 | $23.40 | $0.1322 | 0.98 |
| Always-Frontier | 170 | 96.05% | 93.22–98.87 | $87.04 | $0.4918 | 8.35 |
| kNN | 139 | 78.53% | 72.32–84.75 | $10.90 | $0.0616 | 31.73 |
| Always-Cheap | 137 | 77.40% | 71.19–83.62 | $1.36 | $0.0077 | 32.78 |
| Tier-Classifier | 120 | 67.80% | 60.45–74.58 | $9.38 | $0.0530 | 50.58 |

### The result that matters is which strategy gets there

`Price-Cascade` uses no embeddings, no nearest neighbours, and no training. It
tries the models in ascending price order and stops at the first one whose patch
passes. Of the deployable strategies whose quality interval overlaps
Always-Frontier's, it is the cheapest in the table.

The learned `kNN-cascade` costs **more** for the same 96.61%. The machine
learning is not paying for itself. What buys the quality back is **verified
escalation**.

Even the regret ordering is unresolved: Price-Cascade's bootstrap interval on
total regret is [0.43, 0.97] and kNN-cascade's is [0.60, 1.42]. They overlap.

### Measured versus projected

That table is still part projection. Of the dollars behind it:

| strategy | projected share of cost | projected passes |
|---|---:|---:|
| Always-Cheap | 5.3% | 14 of 137 |
| Oracle | 8.9% | 20 of 171 |
| Tier-Classifier | 28.1% | 37 of 120 |
| **Price-Cascade** | **31.1%** | 30 of 171 |
| **kNN-cascade** | **30.9%** | 35 of 171 |
| kNN | 42.1% | 37 of 139 |
| **Always-Frontier** | **48.5%** ($42.23 of $87.04) | 91 of 170 |

Every projected cell is filled `pass=True`. Imputation is not neutral: the
always-frontier baseline is charged full price on tasks a cheaper model
demonstrably solved, which is exactly where a router's apparent saving comes
from.

So we cut to the subset where **no cell on either path was projected**. On those
84 co-measured tasks:

- Always-Frontier: **$43.72 @ 91.7%**
- Price-Cascade: **$13.42 @ 92.9%**

Price-Cascade is $30.30 cheaper, with 0 versus 1 discordant task, McNemar exact
**p = 1.000**. The saving survives measurement at roughly a third of the
baseline's cost, with no quality difference resolved either way.

### Why we still do not call the gate passed

Three reasons, which we would rather state than have you find.

1. **The measured subset is opportunistic.** Those 84 tasks are what survived
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

It buys about 1.1pp of pass rate over always-cheapest for roughly 8× the cost,
and that margin sits far inside both intervals: [72.3, 84.8] against
[71.2, 83.6]. Indistinguishable at this sample size.

## Why the learned router adds nothing

Leave-one-task-out accuracy for the kNN router equals the base rate to four
decimals: **0.7740**, against Always-Cheap's 77.40%. It sits inside the
shuffled-outcome permutation null at every *k*. The best-over-*k*
selection-corrected value, 0.7966 at k=2, lands inside a null band of
[0.7797, 0.8363] (null mean 0.8078, z = −0.74, 200 permutations).

Three more readings point the same way:

- **Against a fixed model it loses.** The best single always-one-model policy on
  this suite scores 0.960. The router is 0.1638 below it. A router that cannot
  beat one fixed model is not routing.
- **Its neighbourhoods are no better than random.** Observed mean neighbourhood
  purity 0.9616 sits inside a permutation null of [0.9107, 1.0000] (null mean
  0.9790, z = −0.69). True chance purity is 0.9667 and the majority class alone
  is 0.9831, so a purity near 1.0 is what a constant router scores for free.
- **It is close to a constant function.** At k=20 the router sends 174 of 177
  tasks to `deepseek-v4-flash` and 3 to `qwen3.7-plus`, differing from
  always-cheapest on 3 tasks, for +$0.115 (+8.5%) and +0 passes.

Difficulty itself is present in the data (tasks are solved by 0 to 6 of the
enabled models) but is not separable from prompt embeddings here. PCA of the
shipped jina vectors shows hard and easy tasks intermixed, with PC1 and PC2
together carrying only 15.2% of the variance.

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
slightly better than across repos: a same-repo diagonal advantage of **+0.0248**
(0.7836 versus 0.7588) against a matched shuffled-outcome null of
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
| Always-Frontier | $87.04 | — | nothing |
| **Price-Cascade** | **$20.46** | **76.5%** | **nothing** |
| Difficulty-only oracle | $14.25 | 83.6% | perfect difficulty prediction |
| Oracle (exact model) | $13.59 | 84.4% | + hindsight token counts |

A **difficulty-only oracle**, one that always picks the cheapest model that
solves the task and ignores which specific model it is, agrees with the full
Oracle on **170 of 177 tasks (96%)** and costs only **$0.66** more. There is
essentially no "one magic model for this task" effect to capture. The Oracle is
almost entirely *"use cheap when cheap works"*.

That splits the headroom in two. **Both shares below are of the headroom itself,
the $73.45 gap between Always-Frontier and the Oracle, not of the baseline's
total cost:**

- **90.6% of the headroom is mechanically available.** Collectable today by
  trying models cheapest-first and stopping at the first verified pass. No model,
  no features, no training.
- **The remaining 9.4% of the headroom requires predicting task difficulty**,
  and our kNN router does not predict it: leave-one-out accuracy never beats the
  base rate at any *k*. That is a result about the router as it stands, embedding
  the 106-character label — not about task embeddings, which have not been tested
  here (see the caveat opening [Routing results](#routing-results)).

To make the two denominators reconcilable: that 9.4% residual is **7.9% of
Always-Frontier's total cost** ($6.87 of $87.04). Different denominator, same
dollars.

The honest conclusion is neither "routing works" nor "there is nothing here". It
is that the prize is real and large, almost all of it is mechanical, and the
learned part is currently worth nothing.

**Provenance caveat.** Unlike every other table on this page, this decomposition
came from a one-off analysis and is not yet regenerated by a committed script.
The figures are internally consistent and derive from the same
`strategy_summary.csv` as the rest, but until the analysis ships as part of the
pipeline, treat this section as the one place here you cannot reproduce with a
single command. Making it reproducible is queued.

## Escalation results

Status: **work in progress, and currently not working.**

### The old evaluation could not have detected success

We rebuilt the escalation evaluation this cycle. The old label was *positional*,
defined as the last few steps of a failed run, so a content-free clock scored
AUROC 0.970 while a perfect task-level oracle capped at 0.757. Any detector
tuned against that was tuned against a clock.

### On the corrected causal label

Over 727 trajectories with verified per-step outcomes:

- **Task identity alone**, which is what the routing model already knows at
  *t=0*, was once published here at **AUROC 0.886**. **We have retracted that
  number.** The prior gave each run the leave-one-out failure rate of *its own
  instance's other runs*, while the cross-validation split grouped by instance —
  so it was scored on labels from its own test fold. A router meeting a new task
  has no such siblings. That leaked quantity still appears in the harness, now
  explicitly labelled as *not* the baseline (it scores 0.846–0.859 here). Grouped
  honestly the deployable prior is **0.42–0.45**, i.e. no better than chance.
- **The increment is measured against the prior floored at chance**, i.e.
  `AUROC(prior + prefix) − max(AUROC(prior), 0.5)`. Without the floor, an
  anti-predictive baseline donates its own deficit to the detector: the
  previously published **+0.144** at 5 decisions is **+0.061** floored, so 57% of
  it was the broken comparator rather than prefix evidence.
- **The null permutes labels within a challenge, not globally.** A global shuffle
  destroys the challenge-level clustering of outcomes, which collapses the prior
  to chance under the null while the observation keeps a real one — the two arms
  then sit in different headroom regimes and the gate has no power. Permuting
  inside each challenge preserves every challenge's outcome multiset, so the
  prior is identical in both arms and only the prefix's contribution is nulled.
- The detector's **incremental** contribution is not measurable. An earlier
  evaluation reported +0.076 at the best depth (20 decisions) against a
  permutation null of [+0.036, +0.070], p = 0.015, but that figure was inflated
  by the between-fold base-rate accounting artifact in the prior column (see
  *"The increment is measured against the prior floored at chance"* above). Under
  the corrected methodology using StratifiedGroupKFold to stratify by challenge,
  the incremental signal collapses to approximately zero across all depths.
- That clearance is **not** skill, and the harness does not score it as such: the
  paired bootstrap over challenges puts the increment's 95% interval across zero,
  [−0.119, +0.129]. At the same depth (20 decisions, n = 228) the prefix-only risk
  model reads AUROC **0.519** (AUPRC 0.557 against a base rate of 0.540), and at 5
  decisions it reads **0.496** — chance. The gate requires the prefix null, the
  incremental null, and the bootstrap interval to all exclude no-information, and
  at no depth do all three.
- Policy precision runs **0.421** [0.346, 0.486] against a **0.421** base rate,
  lift 1.00×. It equals the base rate identically, because on this corpus the
  recurrence policy fires on *every* scored trajectory: 727 of 727 escalate, so
  there is no left-alone arm to compare against and recall is 1.0 by
  construction. Every swept configuration behaves the same way.
- Harness status: **`NO_SKILL`**.
- At the operating threshold the detector flags all 727 scored trajectories:
  306 true positives (failures caught), 421 false positives, 0 false negatives,
  0 true negatives.

So the honest verdict is the unsatisfying one: **we cannot yet tell.** This
evaluation can only resolve a detector at AUROC ≥ 0.59 (the measured minimum
detectable effect is 0.592 / 0.591 / 0.606 at 5 / 10 / 20 decisions), and the raw
features hint at ≈ 0.52 — inside the blind spot. The corpus is not what caps it:
the 727 scored runs cluster on only 152 distinct challenges, so the effective
sample is far smaller than the run count. Settling it needs roughly four times
the distinct challenges (152 → ~640); more runs per existing challenge buy almost
nothing, because the clustering already inflates variance ~3×. The feature ships
disabled until then.

### Two caveats that make it harder than it looks

**A data gap, now closed.** 253 of 799 trajectories once carried no per-step
outcomes, so the recurrence trigger structurally could not fire on them — and
three models (`kimi-k2.5`, `qwen3.7-plus`, `zai-glm-5.2`) sat at zero coverage
entirely. Because stamping coverage tracked capture date and capture date
correlates with model, model and coverage were confounded. Those runs have since
been re-stamped offline by container replay at zero API cost. All six models now
report **100% capture coverage**, and 727 of 799 trajectories carry verified
per-step outcomes. The 72 that do not break down as: 22 whose captured state was
lost mid-run and whose steps the state-capture audit therefore marks *unmeasured*
rather than failed, 7 that carry no state-capture audit record at all (so whether
their capture was lost is unknown, and their stamps cannot be trusted), and 43
that the per-step stamping stage simply never reached. The confound is gone — and,
as the section above records, closing it did not change the verdict.

**The value is not identified.** Our logging policy never escalates, so
P(escalate) = 0 and the overlap condition that every off-policy estimator
requires fails. No stored trajectory contains an escalation that actually
happened. We currently cannot distinguish "escalation helps" from "escalation
hurts" from this data, at any confidence. The fix is ε-greedy randomisation at
flagged checkpoints with logged propensities.

## Figures

Every benchmark figure carries its own READ / GOAL / TERMS / NOTE / LIMITS
footer on the canvas, so it stands alone without this page open. The figures
live under `benchmark/routing/reports/` and `benchmark/escalation/reports/`,
which sit **outside** the published docs tree, so this page **links** to them on
GitHub rather than embedding them. Nothing in `docs/` embeds benchmark images.
Read the red LIMITS line first; it is where each figure tells you what it cannot
support.

### Routing

Every embedding-based reading here was computed on the 106-character
`description` label, not the problem statement — see the caveat opening
[Routing results](#routing-results). The on-canvas footers now say so themselves:
every figure that reports a difficulty-separability null
(`embedding_routing_map`, `knn_pca_scatter`, `knn_transfer_curve`,
`chosen_arm_vs_difficulty`) names the string it encoded and marks that null a
coverage gap rather than a falsification, so a PNG lifted out of this page still
carries the caveat.

**[strategy_comparison.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/strategy_comparison.png)**
— pass rate against total cost, one mark per strategy, cost on a log axis. Aim
top-left: at or above the best-pass-rate guide and well left of the
frontier-cost line. Points are read straight from `strategy_summary.csv`, so
figure and table cannot disagree. Pareto membership here is relative to the
plotted set, and cost carries no interval.

**[cost_quality_equal.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/cost_quality_equal.png)**
— the kill-gate figure, two panels. Left: every strategy against Always-Frontier's
own Wilson band, blue where its interval overlaps that band. Right: the paired,
measured-only contrast on the 84 co-measured tasks, $13.42 @ 92.9% against
$43.72 @ 91.7%, McNemar p = 1.000. Read the right panel before believing the left
one. The left panel's dollars are roughly half projection; the right panel is an
opportunistic subset that was never designed to answer the gate.

**[measured_vs_imputed_cost.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/measured_vs_imputed_cost.png)**
— each strategy's total split into dollars a provider actually billed (solid) and
dollars projected for cells never run (hatched). The hatched fraction is how much
of the headline is inference. Where a bar is about half hatched, any saving
computed against it means "what we expect if the ordering holds", never "what we
measured". The projected segment carries no uncertainty at all.

**[cost_savings.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/cost_savings.png)**
— total cost per strategy, cheapest first, with each pass rate and its 95% Wilson
interval bracketed above the bar. Look for a short **solid** bar whose interval
still overlaps Always-Frontier's; a hatched bar is a hindsight oracle and not an
option. Bar height alone answers nothing, since the cheapest bar is usually the
worst router.

**[pareto_scatter.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/pareto_scatter.png)**
— the deployable Pareto frontier as a convex hull, with hindsight oracles drawn
as grey stars and excluded. The shaded region is what a router mixing two
deployable strategies can reach. The AIQ number is normalised by the widest cost
plotted, so it is readable within one figure and meaningless across figures.

**[cumulative_regret.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/cumulative_regret.png)**
— per task, the oracle's reward minus this strategy's, summed left to right.
Look for the lowest, flattest curve, then check the end brackets before believing
the ordering. Price-Cascade's total (0.69) is lowest, but its interval overlaps
kNN-cascade's, so the ranking is a leading hypothesis. The bracket bounds the
total only, never the curve at an intermediate index.

**[arm_monotonicity.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/arm_monotonicity.png)**
— the imputation assumption, tested. One row per within-model arm pair,
restricted to tasks where both arms actually ran. Look for a row whose interval
clears zero. None does. Pooled over 485 co-measured pair-observations the effect
is +1.6pp, McNemar p = 0.428, with 7.2% of pairs violating monotonicity.

**[knn_transfer_curve.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/knn_transfer_curve.png)**
— routing pass rate against *k*, leave-one-task-out (deployable) against three
reference lines: a memorisation curve that lets each task see itself, the best
constant policy, and the shuffled-outcome null band. The only claim this figure
can support is blue above the grey band **and** above the green line. Blue sits
inside the band at every *k*.

**[knn_cross_repo_transfer.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/knn_cross_repo_transfer.png)**
— rows are the tasks being routed, columns are the only repo the router may vote
over. A router that learned something generalisable looks flat. Read across a
row, never down a column, because repos differ in intrinsic difficulty. The
diagonal advantage is +0.0248 against a null band of [−0.0176, 0.0236],
z = +2.93.

**[neighborhood_purity.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/neighborhood_purity.png)**
— the task-by-task cosine similarity matrix over the real 768-d jina embeddings,
plus observed neighbourhood purity against its permutation null. The only reading
that supports the router is the blue marker above the grey band. Purity near 1.0
is meaningless on its own: a constant router scores it by construction, and so
does the null.

**[embedding_routing_map.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/embedding_routing_map.png)**
— 2-D PCA of the shipped jina embeddings, each task coloured by measured
`p_solve`. Look for separated dark and bright regions. None appear. PCA is linear
and unsupervised, so absence of clusters is suggestive rather than conclusive,
and the two components carry only 15.2% of the variance.

**[knn_pca_scatter.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/knn_pca_scatter.png)**
— the same space with colour as measured difficulty and marker shape as the
router's pick. Colour varies across the cloud, so difficulty *is* present. One
marker shape covers nearly the whole cloud: 98.3% of tasks get the same pick,
which is what a constant function looks like.

**[knn_cost_comparison.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/knn_cost_comparison.png)**
— the router against every always-one-model policy, with the fixed-frontier
kill-gate baseline drawn as a dashed cross. The router earns its existence only
by landing **outside** the constant-policy staircase. It does not: always
`deepseek-v4-flash` matches it on both axes for less.

**[model_allocation.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/model_allocation.png)**
— how many tasks the router sends to each model against what always-cheapest
would send. Where the two profiles coincide, the router is always-cheapest under
a different name. 174 of 177 coincide.

**[threshold_sweep_heatmap.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/threshold_sweep_heatmap.png)**
— held-out pass rate and frontier-share over the same *k* × threshold grid. Read
both panels together: a cell only counts as a routing result if it is bright on
the left while **not** saturated on the right. Bright-left plus bright-right
means the pass rate was bought with money, and a fixed policy would have done the
same for the same price.

**[capability_distribution.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/capability_distribution.png)**
— where the suite's difficulty lives, as the weakest capability band that solves
each task. 138 of 177 tasks (78%) fall in band 1, 6 (3%) are solved by nothing.
This is a **capability** ordering, not a price ordering: the cheapest model in the
registry sits in band 2, so left-hand mass does not by itself mean "routes
cheaply".

**[per_stratum_winrate.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/per_stratum_winrate.png)**
— mean reward per strategy within each difficulty stratum, every bar in a group
scored on the same tasks. If the tallest bar changes across groups, that
difference is the routing headroom. In strata 1, 2, 3, 4 and unsolvable the top
two intervals overlap, so no leader is resolved.

**[model_performance_descriptive.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/model_performance_descriptive.png)**
<a id="per-model-descriptive-view"></a> — per-model pass rate on the full
measured set against the 69-task common-coverage subset, plus mean real cost per
task on a log axis. **Rank models on the solid common-subset bars only.** Full-set
and common-subset rates differ by up to 23.4pp, a large pooling bias, because
coverage is adaptive.

**[arm_cost_quality_cloud.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/arm_cost_quality_cloud.png)**
— one marker per measured (model, arm) cell, average cost per task against pass
rate. Aim top-left. Hollow markers rest on fewer than 30 tasks and are excluded
from the hull. Each pass rate is a marginal over the tasks that cell happened to
run, and those subsets differ in difficulty.

**[model_complementarity_heatmap.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/model_complementarity_heatmap.png)**
— the raw task × (model, arm) outcome grid: green pass, red fail, grey never
sampled. Scan across a row for complementarity, meaning a task one column solved
and another did not. Grey is missing data, never a failure. Coverage is uneven by
design, so the "columns disagree" count understates real disagreement.

**[chosen_arm_vs_difficulty.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/chosen_arm_vs_difficulty.png)**
— what kNN-cascade selects, chosen-cell cost against how many sampled arms solved
the task. A router adapting to difficulty pushes cost down as you move right. The
cloud is flat. 43 tasks are missing because the chosen cell was never measured,
and they are not a random sample.

**[exploration_replay.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/exploration_replay.png)**
— the shipped exploration policy replayed exactly against the recorded matrix, no
live calls. Read the boxed paired difference, not the overlapping marginal
intervals: pass −2.6% [−3.8, −1.4], cost +$0.0021 per task, 1.27× the
exploration-off bill (worst seed 1.38×). The outcome matrix is static, so an
exploratory pull can never improve a later decision. This measures exploration's
cost with its learning benefit set to zero, which is the pessimistic half of the
ledger. The overhead is paired over the 137 tasks both arms scored: the
exploit-only arm drops 26 cells as unscorable and every one of them is
`qwen3.7-plus`, a model outside this dense slice, so the two arms do not cover
the same tasks and a ratio of their raw totals would have compared different
task sets. We published 1.65× from that unpaired ratio; it was ~30% too high.

**[embedding_compare.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/embedding_compare.png)**
— general-purpose (arctic) against code-specific (jina) neighbourhoods. The two
spaces agree on only 0.33 (k=5) and 0.41 (k=10) of their neighbour sets, so the
embedder is a real design decision. The right panel is a ceiling on kNN routing,
not routing accuracy: a model in the neighbourhood does not mean any threshold
rule picks it.

**[timing_comparison.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/timing_comparison.png)**
— API calls per task as a coarse latency proxy. We do not record wall-clock time.
Read the bars as ordinal, never as measured latency, and note that each left-hand
bar pools whatever arm mix that model ran.

### Escalation

**[permutation_null.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/permutation_null.png)**
— the headline null. Grey is the incremental-AUROC statistic recomputed under
shuffled outcome labels with the whole fitting pipeline re-run per shuffle;
dashed lines bound the central 95%. Labels are shuffled **inside each challenge**,
so every challenge keeps its outcome multiset and the two arms share the same
prior; only the prefix's contribution is nulled. For the detector to be doing
anything, the red line must sit clearly right of the upper dashed line. Under the
corrected methodology using StratifiedGroupKFold, the incremental signal is not
measurable — the point estimate is approximately zero across all depths. (An
earlier figure reported +0.076 at 20 decisions, but that was inflated by a
between-fold base-rate accounting artifact in the prior column; the corrected
stratification collapses the fold-prevalence spread from 0.141 to 0.011 and pins
E[incremental] to approximately zero, confirmed by a null-corpus regression test
on pure Gaussian noise.) The leaked leave-one-out prior (0.846) is still printed
on the panel as a diagnostic contrast and is explicitly **not** the baseline.
Read it against the bootstrap interval: resampling whole challenges puts the
increment at [−0.014, +0.145], across zero.

**[roc_curve.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/roc_curve.png)**
— the out-of-fold risk score. The grey band is the label-permutation null and the
dashed line through it is that null's **measured** centre; the faint dotted
diagonal at 0.5 is orientation only. Because the whole pipeline is refit on every
shuffle, its no-information AUROC is measured rather than assumed and need not
land on 0.5 — reading the curve against the diagonal is reading it against a line
the harness never measured. Look for the curve leaving the band toward the
top-left. Auxiliary to the PR view.

**[pr_curve.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/pr_curve.png)**
— precision against recall. Look for the curve well above the grey band. That
band is the **measured** permutation null — the AUPRC this same pipeline reaches
when the outcome labels are shuffled — and it, not prevalence, is the no-skill
reference: prevalence is the no-skill average precision for exchangeable rows,
while these rows cluster by challenge and the whole pipeline is refit per
shuffle. Both numbers are printed on the canvas, so the size of the assumption
that was dropped is readable. The curve hugs the band. AUPRC alone does not say
whether the model beats the router's t=0 task prior; the permutation-null figure
is where that is answered.

**[confusion_matrix.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/confusion_matrix.png)**
— counts at the operating threshold, each cell printed beside what a random
flagger at the same flag rate would produce and the difference between them. Want
the top-left count well above its bracketed counterpart. That single excess is
the whole figure: with the flag count fixed, the other three cells are the same
number with a sign, which is why the grid carries no colour channel. The footer
also states when the tie block at the cut overspent the base-rate flag budget —
`score >= threshold` admits a block of tied runs whole, so the realised flag rate
can exceed the rate the threshold was derived from. One arbitrary operating
point, not a sweep.

**[sweep_table.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/sweep_table.png)**
— the policy sweep over `escalate_after_n`. Read P(fail | fired) against the base
rate column: an interval containing the base rate is a configuration with no
measured value. Every configuration does: all show P(fail | fired) = 0.421
[0.346, 0.486] against a base rate of 0.421, because each fires on all 727 scored
trajectories (n = 1 or 2, stale window 5 or 10) or 726 of them (n = 3). The other
two knobs were measured inert on this corpus and are pinned, not swept.

**[trajectory_outcomes.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/trajectory_outcomes.png)**
— failure rate among runs the policy escalated against runs it left alone. Want
the escalated bar clearly above both the base rate and the other bar. On this
corpus there is no other bar: the policy escalates every scored trajectory, so
the escalated rate is the base rate 0.421 by construction and the comparison
carries no information. This is association only. No stored trajectory contains an
escalation that actually happened, so the figure cannot say what escalating would
have **changed**.

**[failure_capture_coverage.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/failure_capture_coverage.png)**
— the closed data gap, per model. Each bar is the share of that model's
trajectories that went through per-step verified-outcome stamping. Want every bar
at 1.0. All six now are: the three models that previously sat at zero were
re-stamped by offline container replay, so coverage no longer tracks capture date
and model is no longer confounded with it.

## Where this leaves the project

Routing: the mechanism works, the model does not, and the gate is not passed —
but the routing null is a null on a suite whose minimum detectable effect sits
far above any plausible routing signal, so it bounds our resolution rather than
the idea. The next moves are a designed measured run to replace the
opportunistic 84-task subset, sized against that floor, and better routing
models (bigram and linear, calibrated classifiers, better selection rules)
evaluated against the same nulls.

Escalation: the evaluation can only resolve a detector at AUROC ≥ 0.59, and the
raw features hint at ≈ 0.52 — inside the blind spot. We cannot yet tell whether
escalation works. The offline re-stamping is done — coverage is complete and the
verdict did not move — so the remaining work is more distinct challenges, and
ε-greedy randomisation with logged propensities so the value question becomes
identified at all.

Related reading: [Benchmark](benchmark.md) for how the harness works,
[Benchmark design](benchmark-design.md) for why it is built this way, and
[Benchmark dataset](benchmark-data.md) for what is in it.
