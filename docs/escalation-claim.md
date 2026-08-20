---
title: The escalation claim
description: What Shunt does and does not assert about escalation at pre-alpha, the data and assumptions behind each measurement, and the seven pre-registered falsifiers with their verdicts — three of which fired.
---

# The escalation claim

This page exists so there is exactly one place to check what Shunt claims about
escalation, and one place to prove it wrong. Everything here is scoped to an
offline corpus committed to this repository. Nothing here is a claim about your
workload.

The claim we set out to make was *the escalation ladder works* — a quality claim.
The measurements refused it. We then tried to rest the case on **cost**. An
adversarial review of that fallback found the cost comparison unsound too. So the
honest position is the one below, and it is not a win.

## The claim

> **We make no claim that the shipped trigger, or the escalate arm it gates,
> beats always-frontier on quality or on cost. On this corpus the trigger is a
> null detector, the escalate arm loses to always-frontier on quality, and the
> full-policy cost comparison — now computed — is sound on its money axis and
> carries no information at all on its quality axis, so it cannot support a
> cost-at-equal-quality claim either.**

Escalation ships enabled anyway. That is a design choice — one decision per
session, cache-safe by construction, and a bounded spend ceiling — not a claim
that it has been shown to pay.

### What the measurements do support

Two things, and they are worth stating plainly before the refusals.

**Failure is learnable from a trajectory counter, offline.** The `edit_gated`
counter at the shipped knobs reaches AUROC 0.7103 against a run-length-only
baseline of 0.5682, and clears both a family-wise and a length-stratified null at
p = 0.0005 over 2000 draws. Stratifying by model and by challenge barely moves it
(0.7462 / 0.7095), so it is not a coverage confound. Beside it, the counter the
product runs sits exactly on the null: AUROC 0.500, p = 1.0. That pairing is a
positive result with its own negative control — *the signal exists; the shipped
counter is not the thing reading it.* Every key is in the table under
[the deployability refusal](#what-is-not-being-claimed), which is a separate and
narrower proposition: learnable is not the same as shippable.

**Escalating beats never escalating.** Against the always-cheap arm the paired
difference is +0.1847, 95% [+0.0063, +0.3750], interval excluding zero
(`session_value.png.comparisons.always_cheap.paired_difference_vs_escalate`). The
lower bound is a hairline and the arms are coverage-mismatched, so read it as
direction, not size. It is still the same statistic on the same resamples that we
quote against escalation for the always-frontier arm, and it would be dishonest to
report one and not the other.

**Not in scope of this page:** the `Session-Cascade` cost result in
[Results](results.md#routing-at-session-cadence) and the README. That row models
the escalation *layer* over the **routing** corpus's task sets and compares total
spend at a matched pass rate. This page is about the escalation *trigger* and the
*arm it escalates to*, on the escalation corpus's 48-instance overlap set. The two
are different measurements on different data; neither transfers to the other.

The one cost figure we will state, with its comparator and why the comparison
fails:

| Figure | Value | Provenance (`benchmark/escalation/reports/metrics.json`) |
|---|---:|---|
| overlap instances | 48 | `session_value.png.n_overlap_instances` |
| instances the escalate arm acts on | 30 | `session_value.png.cost.naive.escalate.n_tasks_acted_on` |
| escalate — USD per resolve above the always-cheap floor, on those 30 | 0.9069 | `session_value.png.cost.naive.escalate.usd_per_marginal_resolve` |
| — its 95% interval | [0.6313, 1.3442] | `session_value.png.cost.naive.escalate.usd_per_marginal_resolve_ci95` |
| the escalate arm's models | `zai-glm-5.2`, `kimi-k3` | `session_value.png.context.frontier_models` |

### Why there is no cost comparison here

Always-frontier's corresponding figure is 1.4851
(`session_value.png.cost.naive.always_frontier.usd_per_marginal_resolve`), and
0.907 against 1.485 looks like a result. It is not one, for three reasons that
are each sufficient on their own.

**The two ratios are computed on different task sets and different floors.** Both
divide spend by the arm's resolve gain over the always-cheap floor, and the code
reads that floor *on each arm's own covered instances*
(`benchmark/escalation/session_eval.py`, `_arm_costs`). Always-frontier covers all
48; the escalate arm covers the 30 where a cheap session failed — which is exactly
the subset where the cheap floor is lowest. The two denominators are therefore not
the same quantity, and that difference, not anything about escalation, is where
the gap between 0.907 and 1.485 comes from.

**The intervals overlap.** [0.6313, 1.3442] against
[1.0278, 2.3445] (`...always_frontier.usd_per_marginal_resolve_ci95`) share the
band [1.028, 1.344]. No paired difference between the two *ratios* is estimated
anywhere in the report — the full-policy block below pairs the arms' per-instance
*cost*, not their dollars-per-marginal-resolve — so this particular comparison has
no interval at all, and the same report's own rule is that two marginal intervals
are not a test of a difference.

**On the fired subset the ordering reverses — and that is only half of it.** On
the 30 instances where escalation fires, the two arms hold the *same* frontier
sessions with the *same* outcomes; the escalate arm additionally pays for the
cheap session that failed. So there, escalating costs strictly more than never
being cheap — though the premium is small, roughly the always-cheap per-task cost
of 0.008645 on top of 0.553978
(`session_value.png.cost.naive.{always_cheap,escalate}.cost_per_acted_task_usd`).
On the 18 instances where the trigger stays quiet the sign flips the other way:
escalate pays cheap prices there and always-frontier pays frontier prices. Neither
half is the answer; their sum is, and the section below computes it.

### The full-policy read, and why it still does not make a cost claim

The comparison that *would* decide the cost question is a full-policy read over
all 48 instances — escalate paying cheap on the instances it leaves alone and
cheap-plus-frontier on the ones it fires on, against always-frontier paying
frontier on all 48. `session_eval.py` now computes it
(`_full_policy_costs`, published under `session_value.png.cost_full_policy.<currency>`).
Every arm is read on the same 48 instances, the always-cheap floor is drawn on the
same resampled instances as the arm it is subtracted from, and the difference
against always-frontier is estimated inside each draw rather than compared between
two marginal intervals.

| Figure (naive currency) | escalate | always-frontier |
|---|---:|---:|
| USD per instance, all 48 | 0.348745 | 0.565547 |
| — its 95% interval | [0.247307, 0.465128] | [0.468099, 0.674234] |
| resolve rate | 0.8125 | 0.8125 |
| USD per marginal resolve above the always-cheap floor | 0.9069 | 1.4851 |
<!-- generated-by: benchmark.escalation.run_eval -->

Paired, escalate minus always-frontier costs **−0.216801 [−0.329188, −0.121492]**
per instance, an interval excluding zero
(`session_value.png.cost_full_policy.naive.escalate.paired_cost_difference_vs_always_frontier`).
The cache-aware currency returns the same figures for these two arms, because
neither repeats a model and the discount cannot bind.

**That is a sound money number and it is still not a cost claim.** The quality
axis of the same comparison carries *no information*: the two arms differ in
outcome on **0 of the 48 instances**
(`...paired_resolve_difference_vs_always_frontier.n_instances_outcome_differs`),
so the paired resolve difference is 0.0 by construction, not by measurement. On
the 30 instances where escalation fires the two arms *are* the same frontier
sessions; on the 18 where it stays quiet the escalate arm declines to buy
frontier only because the cheap attempt already resolved — so it scores 1 there
whatever a frontier session would have done. The escalate arm therefore cannot
score *below* always-frontier under this construction at any n: the comparison is
weakly ordered before the data arrives, which is a property of the design, not a
result. "38% cheaper at equal quality" is the sentence this table invites, and the
one it cannot support.

By contrast the arms that *can* separate do: `random_escalate` differs from
always-frontier on 10 of the 48 and reads −0.1146 [−0.2188, −0.0312] on resolve;
`cheap_retry` differs on 20 and reads −0.2292 [−0.3646, −0.1146]
(`session_value.png.cost_full_policy.naive.<arm>...`). That those intervals are
estimable while escalate's is degenerate is the clearest statement of what this
block can and cannot decide.

Two further reasons the block is context rather than a headline. Its per-instance
resolve rates are **not** this page's quality statistic — that stays the
session-pooled paired difference recorded under E1 below, a different estimand on
a different weighting, and the full-policy rates do not overturn it. And the arm priced here is the corpus's two most expensive models,
which is not the ladder the product ships.

## What is NOT being claimed

Four refusals, each forced by a measurement rather than by caution.

**We do not claim a cost advantage over always-frontier**, for the three reasons
above.

**We do not claim escalation improves quality.** On the same 48 instances,
always-frontier resolves *more* than escalating does, and the paired interval
excludes zero. Firing at random at the escalate arm's own rate is
indistinguishable from the trigger.

| Arm | Resolve rate | Paired difference vs escalate | 95% interval | Excludes zero |
|---|---:|---:|---|---|
| escalate | 0.6222 | — | — | — |
| always_frontier | 0.7302 | −0.1079 | [−0.1652, −0.0557] | yes |
| random_escalate | 0.6607 | −0.0385 | [−0.1515, +0.0843] | no |
| always_cheap | 0.4375 | +0.1847 | [+0.0063, +0.3750] | yes |
| cheap_retry | 0.2059 | +0.4163 | [+0.2393, +0.5808] | yes |

Provenance: `session_value.png.escalate.rate`, and for each other arm
`session_value.png.comparisons.<arm>.rate` and
`session_value.png.comparisons.<arm>.paired_difference_vs_escalate.{estimate,ci95,excludes_zero}`
(the `cheap_retry` row is `session_value.png.cheap_retry.rate` and
`session_value.png.paired_difference.*`).

The `cheap_retry` row is the flattering one, and it must never be quoted alone.
It is the weakest of the four baselines: it asks only whether a frontier model
beats a second run of the same cheap model, which is a low bar. It is also the
most coverage-mismatched — the retry arm acts on 27 instances against escalate's
30 (`session_value.png.cost.naive.cheap_retry.n_tasks_acted_on`) — so read it as
direction only, never as a size.

**We do not claim the shipped trigger detects anything.** Replayed over the
stamped corpus, the counter Shunt actually runs fires on every trajectory, so
its precision *is* the base rate and it has no quiet arm to compare against.

| Shipped counter, replayed | Value | Provenance |
|---|---:|---|
| trajectories stamped | 723 | `run.n_stamped` |
| fired | 723 | `operating_point.png.as_shipped.n_escalated` |
| P(fail \| fired) | 0.4177 | `operating_point.png.as_shipped.p_fail_given_fired` |
| corpus base failure rate | 0.4177 | `operating_point.png.as_shipped.base_failure_rate` |
| AUROC | 0.500 | `operating_point.png.as_shipped.null_auroc_familywise.observed` |
| family-wise null, p | 1.0 (2000 draws) | `operating_point.png.as_shipped.null_auroc_familywise.{p_value,n_permutations}` |

It is a null detector on this data. No number on this page is evidence that the
trigger picks the right sessions.

**We do not claim the separating counter is deployable.** A different counter —
`edit_gated`, which ignores failures before the agent's first edit-like action —
does separate outcomes at the same shipped knobs, and clears both its family-wise
and its length-stratified null.

| `edit_gated` at the shipped knobs | Value | Provenance |
|---|---:|---|
| AUROC (operating point) | 0.7103 | `operating_point.png.edit_gated.null_auroc.observed` |
| AUROC (score, pooled) | 0.7782 | `escalation_decision.png.auroc_edit_gated` |
| within-model / within-challenge | 0.7462 / 0.7095 | `corpus_and_coverage.png.stratified_auroc.{within_model,within_challenge}` |
| family-wise null, p | 0.0005 | `operating_point.png.edit_gated.null_auroc_familywise.p_value` |
| length-stratified null, p | 0.0005 | `operating_point.png.edit_gated.null_auroc_length_stratified.p_value` |
| run-length-only baseline | 0.5682 | `operating_point.png.edit_gated.length_baseline_auroc` |

Every result from it carries the stamp `OFFLINE-ONLY UPPER BOUND`
(`run.canonical_deployability.label`), for three reasons the eval states
mechanically rather than editorially (`run.canonical_deployability.reason`):

1. the counter reads `action`, a per-step field the live decision context does
   not carry, and no shipped knob can ask for that counting mode;
2. two of its features read fields the shipped `FailureEvent` never retains —
   `infra_rate` and `max_action_repeat_rate`
   (`run.canonical_deployability.unsupported_features`);
3. it is scored once per **step** while production decides once per **session**
   (`run.canonical_deployability.{cadence,production_cadence}`).

## The three scope limits that decide how to read all of it

**The escalate arm is not the shipped ladder.** The arm measured above is the two
most expensive models in the corpus (`session_value.png.context.frontier_models`).
The shipped ladder walks `qwen3.7-plus → gpt-5-mini → kimi-k3`
(`session_value.png.context.shipped_ladder_visits`, at
`escalation.rank_shortlist: 3`). So production bills two rungs before it reaches
the escalate arm at all, and it never reaches `zai-glm-5.2`. Read the claim as the
value of escalating *to that arm*, never as what a default install achieves.

That ordering is measured, and it is measured wrong. Against the cheap base
`deepseek-v4-flash`, on paired overlaps
(`benchmark/routing/reports/ladder_evidence.json`, `targets[]`):

| Rung | Price multiple | n | Δ resolve | p | Verdict | Ladder visits it? |
|---|---:|---:|---:|---:|---|---|
| `qwen3.7-plus` | 3.81× | 87 | +0.0345 | 0.51 | INDISTINGUISHABLE | yes |
| `gpt-5-mini` | 5.36× | 190 | −0.1684 | 0.0 | **NET-HARMFUL** | yes |
| `kimi-k2.5` | 8.57× | 121 | −0.0165 | 0.81 | INDISTINGUISHABLE | no |
| `zai-glm-5.2` | 13.81× | 84 | +0.1548 | 0.00098 | **NET-HELPFUL** | no |
| `kimi-k3` | 42.86× | 110 | +0.2364 | 3e-06 | **NET-HELPFUL** | yes |

`p` is `targets[].p_value` as the file reports it; `gpt-5-mini`'s is a rounded
zero from an exact paired-exchangeability test, not a claim of impossibility. The
price-ordered ladder pays for the one rung measured harmful and jumps over the
cheapest rung measured helpful. Reading and every caveat:
[the ladder-rungs figure](routing.md#fig-ladder-rungs).

**Every arm contrast on this page is coverage-mismatched.** The arms do not act
on the same instances: always-frontier acts on all 48, the escalate arm on 30, the
cheap-retry arm on 27
(`session_value.png.cost.<currency>.<arm>.n_tasks_acted_on`). Instances are
jointly *resampled* in the paired intervals, but each arm's rate is still read
over whatever sessions it holds in that draw — so a difference between two arms
with different coverage carries a selection term that no statistic on this page
removes. This is the single largest reason to treat every contrast here as
directional at best. It applies to the quality table above and to the cost figures
alike.

## The data

The corpus, its provenance, its censoring and its known biases are the
[escalation dataset card](escalation-data-card.md). In one line: 822 committed
trajectories (`run.n_trajectories`), 723 of them carrying per-step verified
outcomes (`run.n_stamped`), over 6 models, scored at digest
`370b4f954df89cdf` (`run.corpus_digest`).

The session-cadence numbers are read on all 822 trajectories, because a session
outcome comes off the run header and does not need per-step stamping. Every
per-step number is read on the 723.

Cost comes from the provider's billed `real_cost`
(`session_value.png.cost_provenance.source`). Every one of the 822 sessions joins
to a cost (`...join.join_rate` = 1.0, `n_joined` = 822 of `n_sessions` = 822).
That rate is per *session*; the underlying result table holds 1224 rows
(`...join.n_result_rows`), and no published key states the join grain, so read
the 100% as session coverage and nothing finer.

## Reproducing it

[Reproducing the escalation eval](escalation-reproduction.md) is the checklist:
clone, `git lfs pull`, `uv sync --extra benchmark`, `make escalation-eval`. The
run is deterministic and the contract is **bit-identical** — a correct run leaves
`benchmark/escalation/reports/metrics.json` unchanged in `git status`. The ladder
table comes from `benchmark/routing/reports/ladder_evidence.json`, seeded the same
way (`knobs.seed`).

If your run diverges, that is [falsifier E4](#pre-registered-falsifiers-and-their-verdicts)
and nothing on this page survives it.

## Assumptions

1. **Observational, not causal.** The arms ran in parallel and which tasks got
   frontier coverage was adaptive, not randomised. Our logging policy never
   escalates, so `P(escalate) = 0`, the overlap condition every off-policy
   estimator needs fails, and the value of the *policy* is not identified at any
   sample size.
2. **Parallel arms stand in for sequential sessions.** The overlap set's second
   cheap session did not run *because* the first failed and had no access to that
   failure. A real escalation decision does. This is the largest single gap
   between the measurement and the product.
3. **The two currencies are not a robustness check on these arms.** The
   cache-aware currency banks a discount when an arm *repeats a model*
   (`session_value.png.cost_provenance.currencies.cache_aware`); its 0.9 hit rate
   is assumed, not measured
   (`...cache_prices.<model>.hit_rate_provenance`). Neither the escalate arm
   (cheap then frontier) nor the always-frontier arm repeats a model, so the
   discount **cannot** bind on either — their `total_cost_usd` is bit-identical
   across the two currencies, while `cheap_retry`'s is not. Reporting "the same
   under both currencies" for these arms is arithmetic, not corroboration.
4. **Marginal resolves are counted against the always-cheap floor, read on each
   arm's own tasks.** That makes each arm's ratio internally coherent and makes
   ratios from *different* arms non-comparable, because the floor moves with the
   coverage. The escalate arm's tasks are the fired subset
   (`session_value.png.cost.naive.escalate.n_tasks_acted_on` = 30 of 48), which is
   precisely where the cheap floor is lowest. The `cost_full_policy` block does
   not have this defect — every arm there is read on all 48 — but it has the
   quality-axis one instead, and neither block is a cost claim.
5. **The flake guard is unmeasured.** Offline, every replayed failure is
   `confirmed=True` by construction, so the live re-run guard's effect on the
   policy is not measured and not shown to be nil.

## Limitations

- **n = 48.** Read the intervals, not the point estimates. Several of them are
  wide enough to admit very different worlds.
- **One corpus, one base model, SWE-bench-derived tasks.** Not your task mix.
- **Coverage is opportunistic.** Which model was run on which challenge was not
  assigned. A model measured on an easier overlap looks better for free.
- **Best-of-N versus single-shot.** An escalating arm is scored on the attempt
  that passed; the always-frontier arm is single-shot. The cost axis is honest —
  every attempt is billed — but the quality axes are not the same kind of number.
- **Per-model stamping is uneven** (`qwen3.7-plus` 47 of 65 trajectories,
  `zai-glm-5.2` 16 of 31 — `corpus_and_coverage.png.coverage[]`), which is what
  [falsifier E6](#pre-registered-falsifiers-and-their-verdicts) was written
  against. E6 was run and did not fire; the coverage ladder is published with it.
- **No difference statistic exists for the cost *ratios*.** The report now pairs
  the arms' per-instance cost against always-frontier, but nothing pairs two arms'
  USD-per-marginal-resolve, so any comparison drawn from *those* two figures is
  uninferable by construction.
- **The full-policy cost comparison has no quality axis.** Its escalate arm cannot
  score below always-frontier under its own construction — it declines to buy
  frontier only where the cheap attempt already resolved — so the two differ in
  outcome on 0 of 48 instances and only the money axis carries information.
- **The three scope limits above**, which are limitations, not footnotes.
 - **The prefix risk model is a null.** A shallow-prefix failure predictor reads
   `NO_SKILL` on this corpus at a minimum detectable effect of ≈ 0.59 AUROC — see
   [Results](results.md#escalation-results). That is a bound
   on our resolution, not a proof that no such detector exists.
 - **Predicted up-front difficulty is a null too — a go/no-go with its own null.**
   The only positive prediction signal ever recorded was *annotated* difficulty
   against the learning-to-defer label (AUROC 0.736, n=56, p=0.0035). The question
   that remained was whether *predicted* difficulty — computable at request time
   from the problem statement a router can actually see — retains it. The
   pre-registered go/no-go ran to its Stage-1 stop: the instrument cleared its
   admissibility gate (positive control 0.7915, shuffled-label null at chance, via
   the shared gate) and then failed to recover annotated difficulty from the
   real jina-code embeddings at all (Spearman ρ = 0.047, permutation p = 0.149
   over 2,000 within-population permutations, grouped CV by repo, n=500). Per the
   pre-registration the absence of a Stage-2 number *is* the finding: the
   up-front channel is null too, the fixed mechanical rule is the honest
   escalation model, and the learned-escalation workstream closes. Stage 0's free
   surface features (statement length, code-block count, traceback presence, file
    mentions) were also null against both the label and the annotation.
     Every number here is reproduced by the committed module
     `benchmark/escalation/difficulty_prediction.py` at seed 0 over the committed
     corpus and the shipped embedder (`uv run python -m
     benchmark.escalation.difficulty_prediction --out <path>`); the run's committed
     output is `benchmark/escalation/reports/difficulty_go_no_go.json`, and the
     module's own tests pin the stage order, the gate, and the derived verdict.

## Pre-registered falsifiers and their verdicts

Seven falsifiers were registered with the release plan, each naming the condition
that would force a retraction and the retraction it would force, before this page
was written. That is what makes this a claim rather than marketing: the conditions
under which we would have had to withdraw are published alongside the result.
Three of them fired, and three were worded badly enough to mislead — two would
have reported a pass on a literal reading, and a third named a population this
corpus does not contain. All three wording defects are recorded below rather than
quietly corrected.

| # | Falsifier, as written | Verdict | Consequence, as pre-registered |
|---|---|---|---|
| **E1** | If the `always_frontier` arm resolves at or above the escalate arm on the 48-instance overlap **and** the paired interval contains zero, the ladder has no measured advantage over never being cheap. | **TRIPPED** (see the wording defect below) | Do not publish "escalation works". State the result and let the case rest on cost. |
| **E2** | If the escalate arm's USD per marginal resolve is at or above always-frontier's under **both** the cache-blind and the cache-aware currency, the quality was bought by spending frontier money anyway. | **TRIPPED** (see the wording defect below) | Withdraw the cost advantage; do not rest the claim on cost either. |
| **E3** | If the `deepseek-v4-flash` → `gpt-5-mini` result reproduces at larger n, the shipped ladder's first rank rung is net-harmful on the product path. | **TRIPPED** | Scope the claim to escalate-to-frontier and state plainly that the price-ordered rung is measured harmful. Reconsider the ladder separately; do not patch it inside this release. |
| **E4** | If a fresh third-party clone does not reproduce `metrics.json` bit-identically, the instrument is not reproducible. | **NOT tripped, and open to you** | No claim at all until fixed. |
| **E5** | If the escalate-vs-cheap-retry interval crosses zero once instances are jointly resampled, the session-cadence lift over a cheap retry is a coverage artifact rather than a ladder effect. | **UNADJUDICATED** — the interval does exclude zero (`session_value.png.paired_difference.excludes_zero` = `true`, 48 jointly resampled instances), but joint resampling is not coverage matching, so this statistic cannot see the confound E5 names | Downgrade that sentence to direction-only at n=48. Applied on this page; the shipped config comments still carry the interval, and are qualified rather than stripped. |
| **E6** | If the `edit_gated` result drops inside its null when restricted to the fully-stamped models, it is an artifact of uneven per-model coverage. | **NOT tripped** — adjudicated directly, on a coverage ladder rather than the empty set the wording names (see below) | Retract the `edit_gated` sentence and record the retraction here. |
| **E7** | If a multi-session cheap-only probe yields fewer than 60 instances with at least 2 cheap sessions each, or its projected spend crosses the cap, the step-vs-session cadence gap stays open. | **UNADJUDICATED on published data** — the probe has not run, and the scoping count that would settle it is emitted by no committed entrypoint | Keep the `OFFLINE-ONLY UPPER BOUND` stamp. |

### E1 — the wording was defective, and we are treating it as tripped

E1 was written as a conjunction: always-frontier resolves **at or above** escalate
**and** the interval **contains zero**. What happened is that always-frontier
resolved *more* (0.7302 against 0.6222) and the interval **excludes** zero
(−0.1079, [−0.1652, −0.0557]). That is a worse outcome for the claim than the one
E1 anticipated — and a literal reading of the conjunction would report "not
tripped", because the second clause is false for the wrong reason.

We are recording that as a defect in the pre-registration, not as a pass. The
falsifier's intent was "the ladder has no measured quality advantage over never
being cheap"; the data says something stronger, that it has a measured
*disadvantage*. So E1 is **tripped** and its pre-registered consequence is
applied: no "escalation works" claim. The fallback it named — let the case rest on
cost — did not survive E2 either. Correctly worded, E1 should have read: *if
always-frontier's resolve rate is at or above escalate's, regardless of where the
interval sits*.

### E2 — the second defective wording, and why the cost case falls too

E2 compared the two arms' published USD-per-marginal-resolve figures. Literally
read, it is not tripped: 0.9069 is below 1.4851. But the two figures are computed
against floors read on different task sets, so the comparison E2 asked for was
never actually available from those two keys — the falsifier presumed a
comparability the instrument does not provide. Its "under **both** currencies"
clause was vacuous on top of that: neither arm repeats a model, so the cache
discount cannot bind on either, and the two currencies return one figure evaluated
twice.

Adjudicated on a common task set, the answer flips. On the 30 instances where
escalation fires, the escalate arm and the always-frontier arm hold the same
frontier sessions with the same outcomes, and the escalate arm additionally pays
for the cheap session that failed. Escalating therefore costs strictly more
there — the exact condition E2 was written to catch. So E2 is **tripped**, and its
pre-registered consequence is applied: the cost advantage is withdrawn, and the
page no longer rests on cost. Correctly worded, E2 should have read: *if the two
arms' cost per resolve, computed over a common set of instances, does not favour
escalate.*

**The full-policy read does not un-trip it.** Over all 48 instances on one
denominator the escalate arm is the cheaper of the two (−0.216801
[−0.329188, −0.121492] per instance), and that figure is sound arithmetic. It is
still not a cost advantage in the sense E2 was written to test: the two arms
differ in outcome on none of the 48, so the read prices two arms that cannot be
told apart on quality, and E2's own condition — costing strictly more on the
instances where escalation fires — remains true where it was measured. The
verdict stands as tripped, and the money figure is published as context under
the cost section above, not as the claim.

This is the same failure mode as E1 — a falsifier written against a published
number rather than against the quantity the number was supposed to stand for. Two
out of seven is a high rate, and it is the main process lesson from this release.

### E3 — the ladder's first rank rung is measured harmful

The result reproduced, at a larger n than the falsifier anticipated and with a
larger effect: `gpt-5-mini` at n=190, 4 helps against 36 hurts, −0.1684
[−0.2263, −0.1105], `p_value` 0.0, verdict `NET-HARMFUL`
(`ladder_evidence.json` `targets[]`). The pre-registered consequence is applied:
every statement on this page is scoped to escalate-to-frontier, the harmful rung
is stated plainly, and the ladder composition is
[left as future work](#what-would-take-this-past-pre-alpha) rather than patched
here.

### E4, E6, E7 — why "not tripped" is not the same as "confirmed"

**E4** named a specific blocker: the corpus is Git-LFS tracked and the fetch was
documented nowhere. That is closed — [the reproduction page](escalation-reproduction.md)
documents the fetch, the eval refuses to start on pointer stubs, and the
bit-identical contract is stated. What we cannot do is verify someone else's
clone. E4 is the falsifier a reader can fire; if your run diverges, say so.

**E6** asked for the `edit_gated` cell re-run on the fully-stamped models only.
It is now run, and it is **not tripped** — but the falsifier's wording had a
third defect worth recording: *no model on this corpus is fully stamped*. The
per-model stamped share runs 0.5161 to 0.9366
(`coverage_sensitivity.stamped_share_by_model`), so "the fully-stamped models"
names the empty set, and any single cut point picked instead would be a choice
made after seeing the data.

So `benchmark/escalation/coverage_sensitivity.py` removes the choice: the cut
points *are* the observed per-model shares, giving six nested strata from every
model down to the single best-covered one. Each stratum re-runs the whole swept
family and reads the canonical cell against **its own** family-wise, marginal and
length-stratified nulls, so a rung is judged inside its null rather than by a point
estimate compared to the unrestricted one.

| Models kept (share ≥) | n | share of rows kept | AUROC | Δ vs all models | run-length-only | clears its null |
|---|---:|---:|---:|---:|---:|---|
| all 6 (0.5161) | 723 | 1.000 | 0.7103 | — | 0.5682 | yes, p = 0.0005 |
| 5 (0.7231) | 707 | 0.978 | 0.7104 | +0.0001 | 0.5741 | yes, p = 0.0005 |
| 4 (0.8468) | 660 | 0.913 | 0.7168 | +0.0065 | 0.5519 | yes, p = 0.0005 |
| 3 (0.8814) | 566 | 0.783 | 0.7367 | +0.0264 | 0.5468 | yes, p = 0.0005 |
| 2 (0.9118) | 514 | 0.711 | 0.7171 | +0.0068 | 0.5598 | yes, p = 0.0005 |
| 1 (0.9366) | 266 | 0.368 | 0.6257 | −0.0847 | 0.5486 | yes, p = 0.0005 |
<!-- generated-by: benchmark.escalation.run_eval -->
(`coverage_sensitivity.strata[]`; the null each rung clears is its own
`null_auroc_familywise`, whose 97.5th percentile widens from 0.5492 to 0.5738 as
n falls.)

If the separation were coverage-driven it would decay along that ladder. It does
not: the AUROC is flat-to-rising for the first four rungs and clears its own null
at every one, including the strictest — a single model, 266 runs, 36.8% of the
rows, where the AUROC falls to 0.6257 and the null's ceiling rises to 0.5738 and
it still clears. Read the last rung with its power loss in mind, and note that it
also removes between-model variation entirely, so its drop is not attributable to
coverage alone. **E6 does not fire, and the `edit_gated` sentence stands with this
robustness row attached.** The earlier within-model / within-challenge proxy —
0.7462 and 0.7095 (`corpus_and_coverage.png.stratified_auroc`) against a null band
of [0.4735, 0.5499] — agrees with it.

**E7** is a kill condition on new spend, and the probe it guards has not been
run. Its go condition asks how many instances already carry two or more cheap
sessions; no committed entrypoint emits that count, so we do not quote one and we
do not score E7 off it. Either way the cadence gap stays open, which is the first
item below.

## What would take this past pre-alpha

Seven things, each with what it would take. This is a list of open problems, not a
roadmap with dates.

**1. Cadence — the real gap.** Every session-cadence number on this page reads
*parallel effort arms* as if they were *sequential sessions*. They are not:
session 2 in a real escalation launches **because** session 1 failed, and has
access to that failure. A cheap-repeat proxy cannot test that; only sequential
collection can. This is not a volume problem — the committed corpus already
carries more than enough instances with two or more cheap sessions to run the
probe at zero new spend. What it does not carry is a single *causally* sequential
pair. Closing this closes one of the three `OFFLINE-ONLY UPPER BOUND` causes.

**2. Deployability — two of the three causes are product-side.** Session cadence
closes cause 3. Causes 1 and 2 do not close with more data at all: `infra_rate`
and `max_action_repeat_rate` read fields the shipped `FailureEvent` never retains
(`run.canonical_deployability.unsupported_features`), and the `action` field the
`edit_gated` counter gates on is not in the live decision context. Those are
product changes — retain the fields, add the counting mode — and until they land,
no amount of collection makes the separating counter deployable. The encouraging
part: a `fail_rate`-only feature set at session cadence is already supported
(`run.canonical_deployability.supported_features`) and would return a
`DEPLOYABLE ESTIMATE` on existing data.

**3. Identification — turn on the randomisation that is already built.**
`escalation.exploration_epsilon` exists, defaults to `0.0`, and has never been
enabled ([configuration](configuration.md#auto-escalate-on-repeated-verified-failure)).
Under a deterministic policy `P(escalate) = 0` and the off-policy estimator can
only return `NOT_IDENTIFIED`. Running at ε in the 0.1–0.3 range with the realised
propensity logged is what turns "does escalation help?" into a question with an
interval attached. The explored arm is always HOLD, so it can only withhold an
escalation, never invent one — cache-safety is unaffected.

**4. Power — the corpus cannot resolve what we want to ask.** The prefix
instrument's minimum detectable effect is ≈ 0.59 AUROC, and closing that needs
roughly 640 distinct challenges against the 152 we have — more *distinct
challenges*, not more runs on the same ones
([Results](results.md#escalation-results)). Separately,
every outcome here is pass@1 from a single sample; at an escalation boundary,
where the decision hinges on whether *this* attempt failed, two or more samples
per cell are needed before a per-decision result means anything.

**5. Prediction economics — check the prize before building the predictor.** The
break-even discrimination for a prediction-based escalation trigger sits above the
best AUROC we have measured from the text channel, so on this corpus a text-only
predictor does not pay for itself. More important than either number: the share of
the total achievable saving that *prediction* could contribute is small next to
the share a *mechanism* contributes — whatever escalating at all turns out to be
worth, choosing *which* sessions to escalate is the smaller half of it. The
published analogue is on the
routing side, where the same decomposition is measured: about 90% of the headroom
is mechanical, which bounds the entire remaining prize for a *perfect* difficulty
predictor at roughly $7 on a $96 base
([Results](results.md#the-oracle-gap-decomposition)). The escalation-side
break-even and ceiling figures are internal and are not published here; treat
that half as a direction, not as a number.

**6. A cost estimator that can answer the question — built, and it refused.** Both
changes asked for here landed in `benchmark/escalation/session_eval.py`: a
**full-policy** read over all 48 instances on one denominator, and a **paired
difference** between the arms' per-instance cost on the same resamples. The money
answer is now inferable — escalate costs −0.216801 [−0.329188, −0.121492] per
instance against always-frontier. What the exercise exposed is that the *quality*
side of that comparison is empty: the two arms differ in outcome on none of the 48
instances, because wherever escalation fires they are the same sessions. So the
open problem is no longer the estimator; it is a corpus in which the two arms can
disagree — which is item 1 on this list. A paired difference between the two arms'
USD-per-marginal-resolve is still not estimated.

**7. Ladder composition — the rung set is measured wrong.** The ladder ranks by
price, and the price order is not the capability order: it buys `gpt-5-mini`
(NET-HARMFUL) and skips `zai-glm-5.2` (NET-HELPFUL, and 3× cheaper than the rung
it jumps to). A capability-ordered ladder is the obvious next step. The resolver
that would order rungs by measured outcome rather than by list price exists but
is not wired to the escalation path.

## Related

- [Error detection & auto-escalation](escalation.md) — how the mechanism works,
  every figure, every caveat
- [Escalation dataset](escalation-data-card.md) — what the corpus is
- [Reproducing the escalation eval](escalation-reproduction.md) — run it yourself
- [Results](results.md) — every measured routing, escalation, and inference number
