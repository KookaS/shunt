---
title: Results
description: Measured routing and escalation results on Shunt's own benchmark, with every figure, every caveat, and every null reported as a null.
---

# Results

Everything on this page is measured on our own benchmark. Where a result is a
null, it is reported as a null. Nothing here uses a precomputed leaderboard
number.

The short version: cheap-first routing with verified escalation reaches
always-frontier quality for a fraction of the cost ($21.12 against $88.61 over
the full suite; $15.61 against $46.65 on the measured-only subset), the machine
learning contributes nothing to that saving, and the escalation signal is real
but sits in the recurrence policy at the shipped threshold once the reproduction
phase is excluded, not in the
prefix risk model (which reads no skill). The saving comes from mechanism, not
prediction.

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

Our own paired test does not confirm it. Across 478 co-measured within-model arm
pair-observations, more reasoning effort is worth **+1.7pp**, exact McNemar
two-sided **p = 0.428**, which is indistinguishable from no effect. Monotonicity
is violated outright on **7.3%** of those pairs (35 of 478): the higher-effort
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

Seven strategies, scored on the same 175 tasks (25 unscorable), from
[`strategy_summary.csv`](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/strategy_summary.csv):

| strategy | passes | pass rate | 95% CI | total cost | avg cost/task | cumulative regret |
|---|---:|---:|---|---:|---:|---:|
| Oracle (hindsight, not deployable) | 169 | 96.57% | 93.71–98.86 | $14.37 | $0.0821 | 0.00 |
| Price-Cascade | 169 | 96.57% | 93.71–98.86 | $21.12 | $0.1207 | 0.68 |
| kNN-cascade | 169 | 96.57% | 93.71–98.86 | $22.74 | $0.1300 | 0.84 |
| Always-Frontier | 168 | 96.00% | 93.14–98.86 | $88.61 | $0.5063 | 8.42 |
| kNN | 143 | 81.71% | 76.00–87.43 | $13.34 | $0.0763 | 25.90 |
| Always-Cheap | 135 | 77.14% | 70.86–83.43 | $1.41 | $0.0080 | 32.70 |
| Tier-Classifier | 118 | 67.43% | 60.57–74.29 | $3.77 | $0.0216 | 49.94 |

### The result that matters is which strategy gets there

`Price-Cascade` uses no embeddings, no nearest neighbours, and no training. It
tries the models in ascending price order and stops at the first one whose patch
passes. Of the deployable strategies whose quality interval overlaps
Always-Frontier's, it is the cheapest in the table.

The learned `kNN-cascade` costs **more** for the same 96.57%. The machine
learning is not paying for itself. What buys the quality back is **verified
escalation**.

Even the regret ordering is unresolved: Price-Cascade's bootstrap interval on
total regret is [0.43, 0.96] and kNN-cascade's is [0.56, 1.17]. They overlap.

### Measured versus projected

That table is still part projection. Of the dollars behind it:

| strategy | projected share of cost | projected passes |
|---|---:|---:|
| Always-Cheap | 3.7% | 10 of 135 |
| Oracle | 4.8% | 14 of 169 |
| Tier-Classifier | 8.1% | 8 of 118 |
| **Price-Cascade** | **21.9%** | 23 of 169 |
| **kNN-cascade** | **22.3%** | 31 of 169 |
| kNN | 30.7% | 30 of 143 |
| **Always-Frontier** | **44.5%** ($39.44 of $88.61) | 85 of 168 |

Every projected cell is filled `pass=True`. Imputation is not neutral: the
always-frontier baseline is charged full price on tasks a cheaper model
demonstrably solved, which is exactly where a router's apparent saving comes
from.

So we cut to the subset where **no cell on either path was projected**. On those
87 co-measured tasks:

- Always-Frontier: **$46.65 @ 92.0%**
- Price-Cascade: **$15.61 @ 93.1%**

Price-Cascade is $31.03 cheaper, with 1 versus 0 discordant task, McNemar exact
**p = 1.000**. The saving survives measurement at roughly a third of the
baseline's cost, with no quality difference resolved either way.

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

It buys about 4.6pp of pass rate over always-cheapest for roughly 9.5× the cost,
and that margin sits far inside both intervals: [76.0, 87.4] against
[70.9, 83.4]. Indistinguishable at this sample size.

## Why the learned router adds nothing

Leave-one-task-out accuracy for the kNN router equals the base rate to four
decimals: **0.7714**, against Always-Cheap's 77.14%. It sits inside the
shuffled-outcome permutation null at every *k*. The best-over-*k*
selection-corrected value, 0.8171 at k=2, lands inside a null band of
[0.7771, 0.8343] (200 permutations).

Three more readings point the same way:

- **Against a fixed model it loses.** The best single always-one-model policy on
  this suite scores 0.960. The router is 0.1429 below it. A router that cannot
  beat one fixed model is not routing.
- **Its neighbourhoods are no better than random.** Observed mean neighbourhood
  purity 0.919 sits inside a permutation null of [0.8839, 1.0000]. True chance
  purity is 0.9128 and the majority class alone
  is 0.9543, so a purity near 1.0 is what a constant router scores for free.
- **It is close to a constant function.** At k=20 the router sends 167 of 175
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
| Always-Frontier | $88.61 | — | nothing |
| **Price-Cascade** | **$21.12** | **76.2%** | **nothing** |
| Difficulty-only oracle | $14.25* | 83.9%* | perfect difficulty prediction |
| Oracle (exact model) | $14.37 | 83.8% | + hindsight token counts |

A **difficulty-only oracle**, one that always picks the cheapest model that
solves the task and ignores which specific model it is, agrees with the full
Oracle on **170 of 177 tasks (96%)** and costs only **$0.66** more. There is
essentially no "one magic model for this task" effect to capture. The Oracle is
almost entirely *"use cheap when cheap works"*.

That splits the headroom in two. **Both shares below are of the headroom itself,
the $74.24 gap between Always-Frontier and the Oracle, not of the baseline's
total cost:**

- **~90% of the headroom is mechanically available.** Collectable today by
  trying models cheapest-first and stopping at the first verified pass. No model,
  no features, no training.
- **The remaining ~9% of the headroom requires predicting task difficulty**,
  and our kNN router does not predict it: leave-one-out accuracy never beats the
  base rate at any *k*. That is a result about the router as it stands, embedding
  the 106-character label — not about task embeddings, which have not been tested
  here (see the caveat opening [Routing results](#routing-results)).

To make the two denominators reconcilable: that ~9% residual is **~8% of
Always-Frontier's total cost** (about $6.9 of $88.61). Different denominator, same
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

The shipped default (`escalate_after_n=2`, `stale_window=10`) fires on **727 of
727** trajectories: P(fail | fired) = **0.421** — exactly the
base rate, lift 1.00, no edge. This is not a tuning accident: every run, resolved or not,
starts by reproducing the bug, so the first one or two replayed steps are red on the target
F2P test and the counter trips immediately. The counter is counting the target bug at t=0 —
the agent's normal "fail, read the traceback, fix" loop — as if it were evidence the agent is
stuck. As-shipped, this is literally a coin flip, and no `escalate_after_n` tuning below the
run length fixes that.

### The same mechanism discriminates once the reproduction phase is excluded

The eval replays the identical recurrence rule in a second family (`count_from_first_edit`):
failures before the agent's first edit-like action are treated as the reproduction phase and
**not counted**. Measured over the same 727 stamped runs, that variant separates immediately
and strongly:

| cell | fires | P(fail\|fired) | lift | AUROC | len-only |
|---|---:|---:|---:|---:|---:|
| n=2 | 435/727 | 0.593 [0.513, 0.659] | 1.41 | 0.711 | 0.570 |
| n=3 | 358/727 | 0.642 [0.559, 0.707] | 1.53 | **0.724** | 0.579 |
| n=5 | 269/727 | 0.699 [0.619, 0.768] | 1.66 | 0.711 | 0.579 |
| n=10 | 186/727 | 0.812 [0.745, 0.872] | 1.93 | 0.705 | 0.565 |
| n=20 | 117/727 | 0.838 [0.760, 0.903] | 1.99 | 0.638 | 0.565 |

Rows are the `stale_window=1000` cells (a window wide enough to hold n recurrences; the n=2 row is
identical across windows). The full 30-cell-per-family grid — every `escalate_after_n` from 1 to
50 at both windows — is on the sweep-table figures and in the report JSON.

The n=3 cell clears the family-wise permutation null (AUROC **0.724** against [0.500, 0.542],
adjusted p = 0.005) **and** the length-stratified null (0.724 against [0.498, 0.563]) — so the
edge is recurrence beyond run length, not the length of the runs it selects. It fires on 358 of
727 runs — a useful fraction, not a tail. This is the honest read of the escalation idea:
**looking for repeated failures is the right approach; the shipped implementation was counting
the wrong failures.** The edit-gated family is eval-only (production has no per-step action
stream to gate on); closing that gap is a design question, not a data one.

### The as-shipped (reproduction-counted) family only separates at high thresholds

For completeness, the **as-shipped** family — the counter that counts every same-key failure
including the reproduction phase — over 727 stamped trajectories (152 distinct challenges, base
rate 0.421), the sweep varies `escalate_after_n` × `stale_window` (30 cells). The two knobs are
coupled: `_in_window` admits at most `stale_window` events, so reaching *n*
recurrences needs a window at least that wide, and the grid sweeps the window
over {10, 1000}. Its only edge sits at thresholds no one would ship (see the edit-gated family
above for the same mechanism at the shipped threshold):

- The shipped default (`escalate_after_n=2`, `stale_window=10`) fires on **727 of
  727** trajectories: P(fail | fired) = **0.421** — exactly the
  base rate, lift 1.00, no edge. It fires on everything because reproduction
  failures recur at step 1–2.
- As the threshold rises, precision separates from the base rate: n=5 reads 0.426, n=8
  0.453, n=10 0.482–0.484, n=15 (`stale=1000`) **0.538** (lift
  1.28), n=20 **0.582** (lift 1.38), and n=30 (`stale=1000`)
  **0.706** (lift 1.68). Every cell reports its OWN marginal challenge bootstrap
  (the family-wise maxT correction is applied to the AUROC null only, never to a
  precision interval — a CI that excluded a cell's own point estimate would not be
  an interval for it). The n=30 cell's marginal interval is **[0.606, 0.788]**, so
  the numbers quoted against it are the ones the report prints.
- The n=30 cell clears the gate outright: AUROC **0.662** against the
  max-over-cells family-wise null 95% **[0.500, 0.549]**, adjusted **p = 0.005**.
  That clearance is why the harness reports `OK_OFFLINE_ONLY`; the gate itself is described on
  [the offline-eval page](escalation.md#evaluating-the-detector-offline).
- **About 40% of that excess over chance is run-length selection, and the report now says so.**
  Firing at n=30 requires ≥30 same-key failing steps, which requires a long run, and run length
  is outcome-correlated on this corpus. A pure "run length ≥ threshold" predictor scores AUROC
  **0.565** at the same flag count, and the cell still clears the **length-stratified null**
  (labels permuted within length bins, so the length→failure association survives): AUROC 0.662
  against **[0.538, 0.583]**, p = 0.005. So the recurrence-specific signal is real but roughly
  about 40% of the raw 0.662-to-0.5 excess is the length of the runs the threshold selects. Both
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
  weaker detector. The 727 scored runs cluster on only 152 distinct challenges,
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
after the agent's first edit, now clears its gate at `escalate_after_n=2`.

### Two caveats that make it harder than it looks

**A data gap, reduced but not closed.** 253 of 799 trajectories once carried no
per-step outcomes, so the recurrence trigger structurally could not fire on them
— and three models (`kimi-k2.5`, `qwen3.7-plus`, `zai-glm-5.2`) sat at zero
coverage entirely. Because stamping coverage tracked capture date and capture
date correlates with model, model and coverage were confounded. Those runs have
since been re-stamped offline by container replay at zero API cost: 727 of 799
trajectories now carry verified per-step outcomes. But coverage is NOT uniform:
the two models that once sat at zero (`qwen3.7-plus` 47/60, `zai-glm-5.2` 20/26)
still carry 3× the unstamped share of the other models, so stamping coverage still
tracks the same model-correlated axis as before — reduced, not eliminated. The 72
unstamped trajectories break down as: 22 whose captured state was
lost mid-run and whose steps the state-capture audit therefore marks *unmeasured*
rather than failed, 7 that carry no state-capture audit record at all (so whether
their capture was lost is unknown, and their stamps cannot be trusted), and 43
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
measured-only contrast on the 87 co-measured tasks, $15.61 @ 93.1% against
$46.65 @ 92.0%, McNemar p = 1.000. Read the right panel before believing the left
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
clears zero. None does. Pooled over 478 co-measured pair-observations the effect
is +1.7pp, McNemar p = 0.428, with 7.3% of pairs violating monotonicity.

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
diagonal advantage is +0.0330 against a null band of [−0.0176, 0.0236],
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
and the two components carry only 15.1% of the variance.

**[knn_pca_scatter.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/knn_pca_scatter.png)**
— the same space with colour as measured difficulty and marker shape as the
router's pick. Colour varies across the cloud, so difficulty *is* present. One
marker shape covers nearly the whole cloud: 95.4% of tasks get the same pick,
which is what a constant function looks like.

**[knn_cost_comparison.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/knn_cost_comparison.png)**
— the router against every always-one-model policy, with the fixed-frontier
kill-gate baseline drawn as a dashed cross. The router earns its existence only
by landing **outside** the constant-policy staircase. It does not: always
`deepseek-v4-flash` matches it on both axes for less.

**[model_allocation.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/model_allocation.png)**
— how many tasks the router sends to each model against what always-cheapest
would send. Where the two profiles coincide, the router is always-cheapest under
a different name. 167 of 175 coincide.

**[threshold_sweep_heatmap.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/threshold_sweep_heatmap.png)**
— held-out pass rate and frontier-share over the same *k* × threshold grid. Read
both panels together: a cell only counts as a routing result if it is bright on
the left while **not** saturated on the right. Bright-left plus bright-right
means the pass rate was bought with money, and a fixed policy would have done the
same for the same price.

**[capability_distribution.png](https://github.com/KookaS/shunt/blob/main/benchmark/routing/reports/capability_distribution.png)**
— where the suite's difficulty lives, as the weakest capability band that solves
each task. 109 of 175 tasks (62%) fall in band 1, 6 (3%) are solved by nothing.
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
intervals: pass −3.2% [−4.8, −1.8], cost +$0.0023 per task, 1.28× the
exploration-off bill (worst seed 1.48×). The outcome matrix is static, so an
exploratory pull can never improve a later decision. This measures exploration's
cost with its learning benefit set to zero, which is the pessimistic half of the
ledger. The overhead is paired over the 138 tasks both arms scored: the
exploit-only arm drops 27 cells as unscorable and every one of them is
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
— the headline null, now for the escalation **policy**. Grey is the maximum AUROC
the swept policy reaches under **block permutation** — whole challenge blocks
shuffled, so outcomes move between challenges while the global multiset is
preserved — with one shared shuffle scored at every swept cell and only the
largest kept — the family-wise (maxT) null across the whole sweep; dashed lines bound its
central 95%. The red line is the real, unshuffled AUROC at the cell that
separates best — on the current corpus the **edit-gated n=3** cell (post-first-edit
recurrence): **0.724**, clearly right of the upper dashed
bound **[0.500, 0.542]**, adjusted **p = 0.005**. The null is the gate — a point
estimate above chance is not skill on its own — and this clearance is one half of
the `OK_OFFLINE_ONLY` verdict: the precision interval must also clear the base rate (see the
confusion figure). A cell that never fires has AUROC 0.5 by definition and
contributes nothing to the max. The old figure nulled the prefix risk model's
incremental; that instrument now reports `NO_SKILL` on its own path, and this
panel plots the policy that carries the edge.

**[roc_curve.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/roc_curve.png)**
— the escalation **policy**'s ROC across the swept recurrence thresholds: one
point per `escalate_after_n` value that fired, each labelled with its n. As the
threshold rises the policy moves up the curve — more precision, less recall. A
SECOND curve (filled triangles) is the **edit-gated** family — the same recurrence
rule with failures before the agent's first edit excluded from the counter — and
it dominates the as-shipped curve (higher true-positive rate at the same
false-positive rate). The
faint dotted diagonal at 0.5 is chance and orientation only. The prefix risk
model's ROC is not drawn: its score is constant at the evaluated depths, so it
ranks nothing. Look for points leaving the diagonal toward the top-left. Auxiliary
to the PR view.

**[recurrence_roc.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/recurrence_roc.png)**
— the complete ROC of the recurrence "stuck depth" score. Each run gets a score
= the largest number of times the same failing-check id recurred in the shipped
window, and the curve sweeps that score over **all** thresholds — so it has a
point at every possible `escalate_after_n`, not just the swept grid, and its AUROC
bounds what *any* single threshold can achieve. Two curves: as-shipped (AUROC
**0.601**) and edit-gated (AUROC **0.781**). The shaded band is the score's own detection floor:
the central 95% of AUROCs the score reaches when terminal outcomes are shuffled across challenges
(**null 95% [0.472, 0.540]**) — both curves clear it, so the recurrence signal is real, and the
edit-gated AUROC clears it by the larger margin. The gap between them is the
reproduction phase masking the signal: the as-shipped score counts every run's
step-1-2 reproduction failures, so it hugs the diagonal until the score gets
large, while the edit-gated score (failures before the agent's first edit
excluded) separates immediately. The continuous edit-gated AUROC (0.781) exceeds
the best swept cell's operating-point AUROC (0.724 at edit-gated n=3) — the
threshold-free score ranks runs better than any single fixed threshold.

**[pr_curve.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/pr_curve.png)**
— precision against recall over the **swept escalation policy**: one point per
`escalate_after_n` value that fired — a discrete operating characteristic, not a
continuous score sweep. The dashed line is the corpus base failure rate — the
no-skill precision a random flagger reaches. As the threshold rises, precision
climbs from the base rate toward 0.88 (the as-shipped n=50 tail) while recall
falls; a point above the line whose interval excludes the base rate is a measured
operating point. The
**edit-gated** family is the second curve (filled triangles) and sits above and to
the right of the as-shipped one: n=2 already reads P(fail|fired)=0.593 (lift 1.41)
at recall 0.843, where the as-shipped n=2 reads the base rate at recall 1.0. The prefix
risk model's PR curve is not drawn (its score is constant at the evaluated
depths). This shows the detector's operating points, NOT what escalating would
have **changed**; the permutation-null figure is where a clearance is scored.

**[confusion_matrix.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/confusion_matrix.png)**
— the escalation **policy**'s 2×2 at the swept cell that separates best — on the
current corpus the **edit-gated n=3** cell (rows:
run failed / resolved; columns: flagged / not flagged). Each cell prints the
observed count, in brackets what a random flagger at the same flag rate would
produce, and the excess. Want the top-left count well above its bracketed
counterpart; that single excess is the whole figure — with the flag count fixed,
the other three cells are the same number with a sign. The cell fill is a
diverging heatmap on that excess (red above the random baseline, blue below,
white at it), so the two hues restate the one fact rather than carry four.
Unlike the old figure, the **"not flagged" column is
populated**: the previous empty 0/0 column was the degenerate prefix score
flagging everything, whereas the best-separating policy cell leaves runs alone.
One operating point, not a sweep — the PR/ROC figures and the sweep table show
the rest.

**[sweep_table.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/sweep_table.png)**
— the policy sweep over `escalate_after_n` × `stale_window` (30 cells). Read
P(fail | fired) against the base-rate column: an interval containing the base
rate is a configuration with no measured value. Each cell prints its OWN marginal
challenge-grouped bootstrap interval — the family-wise maxT correction applies to
the AUROC null only, never to a precision interval — so [0.606, 0.788] is
specifically the n=30 (`stale=1000`) cell's interval, and the separating facts
are the point estimates with their own intervals: n=15 (`stale=1000`) 0.538, n=20
0.582, n=30 0.706 — while the shipped
default (n=2, `stale=10`) fires on all 727 trajectories and reads the base rate
0.421. The
shipped-default row is highlighted (bold on a shaded background). Both knobs are
swept because they are coupled: reaching n recurrences needs a window at least
that wide, so the `stale=10` rows stop firing at n ≥ 12. The intervals are
challenge-level bootstraps.

**[edit_gated_sweep_table.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/edit_gated_sweep_table.png)**
— the SAME recurrence sweep replayed with the reproduction phase excluded
(`count_from_first_edit`): failures before the agent's first edit-like action are
not counted. Read P(fail | fired) and the AUROC against the len-only column. The
n=3 row fires on **358/727** runs at P(fail|fired)=**0.642** (lift 1.53) with AUROC
**0.724** against a run-length baseline of 0.579 — the as-shipped n=2 row (main
table) fires on 727/727 at the base rate. The gap between the two tables is the
reproduction phase masking the recurrence signal. This is an eval-only knob:
production has no per-step action stream to gate on, so it measures what a per-step
detector could do, not what ships.

**[trajectory_outcomes.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/trajectory_outcomes.png)**
— failure rate among runs the policy escalated against runs it left alone. Want
the escalated bar clearly above both the base rate and the other bar. On this
corpus there is no other bar: the policy escalates every scored trajectory, so
the escalated rate is the base rate 0.421 by construction and the comparison
carries no information. This is association only. No stored trajectory contains an
escalation that actually happened, so the figure cannot say what escalating would
have **changed**.

**[failure_capture_coverage.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/failure_capture_coverage.png)**
— the reduced data gap, per model. Each bar is the share of that model's
trajectories that went through per-step verified-outcome stamping. The bars are
stamped/total, not 1.0: deepseek-v4-flash 248/268 (0.925), gpt-5-mini 266/284
(0.937), kimi-k2.5 94/105 (0.895), kimi-k3 52/56 (0.929), qwen3.7-plus 47/60
(0.783), zai-glm-5.2 20/26 (0.769) — 72 of 799 trajectories carry no per-step
verified outcomes. (The `capture_rate` field, by contrast, is 1.000 for every
model BY CONSTRUCTION: the normalizer writes `success`, `failing_check_id` and
`blocking` in one assignment, so it is a different quantity and not the bar.) The
three models that previously sat at zero were
re-stamped by offline container replay, but coverage STILL tracks the
model-correlated axis: the two formerly-zero models remain the least stamped, so
model and coverage are still confounded on this corpus (see the figure's red
LIMITS line — that caveat is live, not vestigial).

**[session_cadence.png](https://github.com/KookaS/shunt/blob/main/benchmark/escalation/reports/session_cadence.png)**
— the value of the ladder at the cadence production actually runs. The corpus holds
several (model, arm) sessions per task, read here as repeated attempts: after a
CHEAP session failed, a FRONTIER session on the same task resolved it **56.8%** of
the time (21/37) against a same-cost cheap retry's **22.6%** (7/31) — a **2.5×**
lift, measured on the overlap subset (tasks with ≥2 cheap sessions AND a frontier
session, so the two arms are read on the same tasks). Observational, not causal:
the arms ran in parallel and coverage was adaptive, and the samples are small — the
intervals nearly touch but do not overlap ([0.409, 0.713] against [0.114, 0.398]).
It answers "is the ladder pointed the right way",
not "is the trigger well-tuned".

## Where this leaves the project

Routing: the mechanism works, the model does not, and the gate is not passed —
but the routing null is a null on a suite whose minimum detectable effect sits
far above any plausible routing signal, so it bounds our resolution rather than
the idea. The next moves are a designed measured run to replace the
opportunistic 87-task subset, sized against that floor, and better routing
models (bigram and linear, calibrated classifiers, better selection rules)
evaluated against the same nulls.

Escalation: the recurrence mechanism works, and the shipped implementation was
counting the wrong failures. As shipped, `escalate_after_n=2` counts the
reproduction phase — every run's first reds are the target bug at t=0 — so it
fires on 727/727 runs and reads exactly the base rate: a coin flip. Gated on
failures after the agent's first edit, the same rule separates just above the
shipped threshold (n=3: AUROC 0.724, P(fail|fired)=0.642, fires on 358/727; n=2
already reads AUROC 0.711 at P=0.593), clearing both
the family-wise and the length-stratified nulls. At the session cadence the
ladder's value is large: escalating to a frontier model after a cheap session
failed resolves 2.5× more tasks than a same-cost retry (observational). The prefix
risk model remains `NO_SKILL` (the corpus cannot resolve a shallow prefix detector
below AUROC ≈ 0.59), and the edit-gated variant is eval-only — production has no
per-step action stream to gate on. The remaining work is making the post-edit
gate real in the live capture path, more distinct challenges (~640), and ε-greedy
randomisation with logged propensities so the value question becomes identified at
all.

Related reading: [Benchmark](benchmark.md) for how the harness works,
[Benchmark design](benchmark-design.md) for why it is built this way, and
[Benchmark dataset](benchmark-data.md) for what is in it.
