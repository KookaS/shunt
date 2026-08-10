---
title: Benchmark design
description: Two-part benchmark structure — live model execution and offline routing strategy evaluation.
---

# Benchmark design

The benchmark has two parts. `runner/` executes models against SWE-bench Verified instances and records the outcomes. `routing/` evaluates routing strategies offline against those recorded outcomes.

| Tree | Question | Output |
|---|---|---|
| `runner/` | Which models solve which tasks? | Verified per-cell pass/fail, cost, and tokens, written to `routing/results.csv` |
| `routing/` | Which routing strategy maximizes reward? | Per-strategy metrics across a task × model matrix |

`runner/` is the empirical source; `routing/results.csv` is the committed record it produces. `routing/` consumes that file — no dependency on runner infrastructure at eval time.

## Why split them

`runner/` answers a model-selection question: given N models, which is the cheapest that solves each task? This is the discrimination test. If every model passes everything, routing is pointless.

`routing/` answers a strategy-selection question: given a known task × model matrix, which algorithm (kNN, cascade, bandit, fixed) maximizes pass rate minus cost?

They share a `benchmark/` root because both evaluate model-decision capability. They stay separate because they have different runners, metrics, and output formats.

## Structure

```
benchmark/
  README.md                               Model-capability benchmark overview
  benchmark.yaml                          Enabled models, strategies, and run settings

  challenges/swebench_verified/           Instance specs (the sole challenge source)

  validate_results.py                     Pre-analysis data-integrity gate (fail closed, exits nonzero on ERROR)
  cost_reconcile.py                       Reconcile tracked real_cost vs owner-billed bill; cross-check + accounting-hole scan (exits nonzero on alarm)

  runner/                                 Live execution against the SWE-bench harness
    run_matrix.py                         Runs the (challenge x model x arm) matrix, upserts rows
                                          (validates every row at write-time — DataIntegrityError aborts)
    collect.py                            Adaptive collection (phase A + frontier tail)
    check_integrity.py                    Anchor/authenticity audit of the committed rows

  routing/                                Routing strategy evaluation
    results.csv                           THE committed source of truth (per-cell outcomes)
    validate.py                           Row-invariant validator (accounting-hole/ran-ness/schema/well-formed)
    data/                                 Curated read-only inputs
      challenges.json                     Challenge index + task metadata
    reports/                              Gitignored — derived strategy_summary.csv + plots
    strategies/
      __init__.py                         Strategy protocol
      oracle.py                           Best per-task (upper bound)
      fixed.py                            Always-cheap, always-frontier, random
      knn.py                              Embed task → retrieve neighbours → cheapest capable
      knn_cascade.py                      kNN-informed try-verify-escalate
      _template.py                        Skeleton for a new strategy
    run_eval.py                           Evaluate all strategies × tasks
    metrics.py                            Reward, regret, efficiency
```

## Strategy interface

```python
class Strategy(ABC):
    @property
    def name(self) -> str: ...

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        """Return the model name to route this task to."""
```

The evaluator iterates tasks, calls `select()` per strategy, looks up the outcome, and accumulates metrics.

## Metrics

| Metric | Formula | Meaning |
|---|---|---|
| AvgPerf% | `pass_count / total_tasks × 100` | % of tasks solved |
| TotalCost | `sum(cost of every billed attempt)` | Raw dollar cost, cache-blind |
| TotalCost_cacheaware | `TotalCost − Σ(cost × input_share × hit_rate × discount)` over consecutive same-model attempts | Dollar cost once repeat-model caching is priced. `input_share` is the cost-weighted input share of the model's MEASURED token mix; `discount` is `1 − cache_read_price/input_price` from the registry; only `hit_rate` is assumed |
| Reward | `1.0 × pass_rate − γ × total_cost` | Cost-aware utility |
| CumReg | `sum(oracle_reward − strategy_reward)` | Regret vs oracle — reward lost by not routing optimally |
| rAcc | fraction of tasks routed to the oracle's model | Routing accuracy |

γ defaults to 0.1, matching the `agent-as-a-router` cost-weight baseline.

**Regret**, borrowed from bandit/RL theory, is the decision-quality metric here: how much
worse off you are for not having made the best possible choice, always measured against an
optimal baseline (the Oracle, which routes with perfect hindsight). Per task it is
`oracle_reward − strategy_reward`; match the oracle's pick and it is 0. A strategy accumulates
regret two ways — **quality regret** (it failed a task the oracle solved) and **cost regret**
(it passed, but paid for a bigger model than the task needed). `CumReg` is that gap summed over
the task sequence, and `oracle_gap.png` reports the regret totals; the retired per-task line figure's slope was
average regret per task, so a lower, flatter curve means routing closer to optimal.

Cost is recorded from actual model API responses: the provider-returned cache-aware `usage.cost` when present (e.g. Requesty-routed models, including cache-aware rates), falling back to litellm's computed cost otherwise (e.g. direct routes litellm can price, such as deepseek). For offline eval, costs come from the cached `results.csv` (recorded during live benchmark matrix runs). Recording per-request API cost on the live proxy path is roadmap, not a current feature.

## Baselines

The two fixed baselines bracket the problem. Always-Frontier arrives but pays top
price on every task; Always-Cheap is cheap but grinds through the tail it cannot
solve. A router earns its place only by being cost-effective against both — which
is what the strategies below are scored on.

![The same workload under three policies. Always-frontier arrives and is the most costly, paying frontier price on every trip. Always-cheap arrives but not reliably — its hard tail stalls in traffic before reaching the goal. Shunt is cheap by default and pays frontier price only for the tail.](assets/route/versus.svg)

The figure is a conceptual framing of the three policies, not a result: cost is
shown as a price word rather than a number. The measured comparison is in
[Results](results.md).

| Strategy | Behavior |
|---|---|
| **Oracle** | Cheapest model that passes each task. Upper bound. |
| **Always-Cheap** | Always cheapest model (derived from pricing matrix). Lower bound — if a router can't beat this, it is pointless. |
| **Always-Frontier** | Always most expensive model (derived from pricing matrix). Maximum cost baseline. |
| **Random** | Random model per task (mean over N seeds). Null baseline. |

Additional strategies in `strategies/`: kNN, kNN-cascade, Price-Cascade (the zero-ML
price-ascending cascade — the floor a learned router has to beat), and Tier-Classifier.

## Equal-coverage scoring — monotone-rank imputation

The runner collects an **asymmetric** matrix on purpose: cheap and mid models run on
nearly every task, the frontier only on a subset, because running the most expensive model
everywhere is the cost the project exists to avoid. But routing strategies must be compared
on the *same* tasks, or a cheaper strategy just looks better by dodging the hard ones the
frontier column happens to cover. The default report resolves this by completing the matrix
in memory before scoring. This section states the method precisely.

### The monotonicity axiom

Order the models by a measured **capability rank** (below), weakest to strongest. The axiom:
for any task there is a single **crossover model τ** — every model at or above τ passes,
every model below τ fails. Plainly: *stronger than a success passes; weaker than a failure
fails.* This is the one assumption the completion rests on, and it is **measured, not
assumed** — see the violation rate below.

### The derived capability rank

Rather than sort models into a handful of hardcoded tiers, Shunt derives a **strict
per-model order from the verified outcomes**. The order is the **Copeland score over
co-measured tasks**: for each pair of models, over the tasks where *both* really ran, who
wins more (passes where the other fails); a model's score is how many models it dominates
minus how many dominate it. This is deliberately *not* each model's raw pass-rate — the
frontier ran only on the hard subset, so its raw rate is computed over harder tasks than a
cheap model's, and ranking on raw rates would sort the frontier *below* cheap models scored
on easy work. Each model also carries its **marginal pass-rate and a Wilson confidence
interval**, used as the reported stat and a confidence gate: a model with too few cells, too
wide an interval, or too few co-measured peers falls back to a **price-implied slot**
(cheaper assumed weaker) until it earns a measured rank. Ties break deterministically by
(pass-rate, price, name), so the order is reproducible across runs on the same data. The top
of the rank is asserted to equal the kill-gate control model; a disagreement is surfaced as
a loud finding rather than silently reordered. The imputation, the classifier strategy, and
the ladder collector all read this one derived order, and a per-model evidence artifact
(`capability_evidence.json`, regenerable) records the pass-rate, CI, rank, source, and price
behind each model's position, plus an **ordinal-band** grouping (band 1 = weakest … band N =
strongest) for the narrative. Bands have no semantic names — each is described only by its
metadata (member models, price range, pass-rate range with CI, `n_models`, and the share of
tasks it is the weakest to solve) — and the band count is data-driven: adjacent models whose
capability CIs overlap merge into one band.

### The completion rule (s\*/f\*/UNKNOWN)

Per task, from its **real** default-arm cells:

1. Read each observed model's outcome.
2. `s*` = the weakest model observed to **pass**; `f*` = the strongest model observed to
   **fail** (by capability rank).
3. Impute: model at rank `≥ s*` → pass; model at rank `≤ f*` → fail; a model strictly
   between `f*` and `s*` is left **UNKNOWN**. `τ = s*`.
4. Extend to every ranked model, so always-cheap, kNN, the cascades and always-frontier are
   all scorable on the same task set.
5. **Observed truth always wins** — a real cell is never overwritten, even when it
   contradicts the axiom.

The rule is collection-order-agnostic: it completes gap-free data (from ladder collection)
and random-order historical data alike, leaving an UNKNOWN band only where no bracketing
observation exists. Every imputed cell is tagged with the observed model that implied it;
real cells are tagged as themselves.

**Nothing imputed is ever persisted.** The completion returns an in-memory matrix;
`results.csv` is read, never written, and a test pins it byte-identical across a report run.
Imputation is recomputed fresh every time, so a stored cell can never drift from its own
definition.

### Cost imputation and its caveats

An imputed cell — pass or fail, since a failed attempt still bills — is priced from the
**per-model median measured `real_cost`** over that model's real cells, falling back to the
nearest ranked model's median (rank-neighbour) when a model was never measured. This is a point estimate **from measured
data**, never a fabricated proxy, and every such cell stays flagged so a reader can separate
measured from imputed spend. The median is taken over real **observations only**: a
**censored** cell (any resource-limit stop — `stop_reason` in `{step_limit, wall_limit,
abandoned}`; see below) or a zero-work cell (`calls == 0` and `real_cost == 0`) is a
non-observation, not a `$0` measurement, and is excluded so it cannot drag the per-model
median toward zero. Cost is the near-deterministic half of a cell (the task fixes
it), so the median is tight; the current report does not attach a confidence interval to
imputed cost, and because the gate is a **paired-quality** test, cost-point noise moves only
the cost axis of the plot, not the decision.

### The violation rate (measured ~90%)

Stronger models sometimes fail where weaker ones pass, so the axiom is not exact. When a
task has a real higher-ranked fail sitting below a real lower-ranked pass (`f* > s*`), the
region between them is left un-imputed and the task is recorded as a violation. The
**violation rate** — violating tasks over tasks with two or more observed models, with a
Wilson interval — is reported as a first-class number: on the current data monotonicity
holds about **90% of the time** (19 violations over 192 multi-observed pairs — holds
90.1%, violation rate 0.099, 95% CI [0.064, 0.149]). The report also recomputes every
conclusion with the violating tasks **excluded** (the sensitivity check) and flags any
result whose sign or CI side flips.

### Why the axiom is conservative

Completing the matrix credits the always-frontier baseline with a pass on every task a
weaker model already solved — free quality at frontier cost. If the axiom is wrong for some
task, the baseline's *true* quality is only lower than imputed, so the router's measured
advantage is only larger. A broken axiom **strengthens** the case for routing; it can never
inflate it. Imputation buys cheap exploration, not a flattering headline — and this is the
reason a sub-perfect axiom is safe to ship.

### Censored data (why a cell stopped)

A cell that stops because it hit a **resource limit** — the agent ran out of steps, hit its
graceful wall-clock ceiling, or was reaped mid-run by the hard watchdog — is **not** a
capability failure: its true pass/fail is *unknown*. Treating such a cell as a clean fail
biases both completeness (a challenge looks "all-tiers-fail / unsolvable" when its top tier
merely ran out of steps) and the kill-gate quality comparison (a censored frontier cell
understates the baseline's real quality).

Every produced row therefore records **why** it stopped in a `stop_reason` column, drawn from
a fixed vocabulary:

| `stop_reason` | Meaning | Censored? |
|---|---|---|
| `solved` | harness resolved the instance (`pass=True`) | no |
| `unsolved` | the agent genuinely finished (submitted / ran to completion) and the harness did not resolve — a real capability fail | no |
| `step_limit` | the agent hit its step (or cost) limit | **yes** |
| `wall_limit` | the agent hit its graceful `wall_time_limit_seconds` | **yes** |
| `abandoned` | the external hard watchdog reaped it mid-run | **yes** |

The signal is the scaffold's own exit status (mini-swe-agent's `LimitsExceeded` →
`step_limit`, `TimeExceeded` → `wall_limit`) plus the harness verdict; the watchdog path maps
to `abandoned`. `timeout_flag` is retained for back-compat and is now exactly
`stop_reason ∈ {wall_limit, abandoned}`. **Legacy rows** written before the column derive a
`stop_reason` on read (`solved` if pass, else `wall_limit` if `timeout_flag`, else `unsolved`).

A **censored** cell (`stop_reason ∈ {step_limit, wall_limit, abandoned}`) is treated as a
**non-observation** everywhere it would otherwise be mistaken for an observed fail:

- **Completeness / imputation.** A censored cell never establishes a crossover. A challenge
  whose crossover would depend on a censored top-tier cell is **incomplete** (unknown), not
  complete-all-fail — so it is excluded from analysis rather than counted as unsolvable. (A
  censored cell may still be *imputed* from a genuine observation on the same task under
  monotonicity — e.g. a weaker model passed, so the stronger censored cell is imputed pass.)
  This correctly **reduces** the complete-challenge count; on the current data 15 challenges
  (192 → 177 complete) revert to incomplete once censoring is respected.
- **Kill-gate / summary quality.** A censored cell is excluded from the pass-rate denominator
  (marked unscorable), so it never counts as a clean `pass=False` that understates a model's
  quality. Cost for censored cells is already excluded from the cost model (above).

### Unsolvable tasks

A task no model solves (`s*` undefined) counts as a fail — at its own cost — for **every**
strategy, including always-frontier. That keeps the task count `N` identical across
strategies (the whole point of equalizing coverage) and leaves always-frontier quality
well-defined. A toggle can exclude such tasks instead, but universal-fail is the default
because it keeps the denominator honest.

### Ladder collection (cheap-first)

To get gap-free equal coverage without running the frontier everywhere, the `ladder`
collection mode runs each task weakest-first and escalates only until the first model passes,
then stops. Under the axiom this observes τ exactly with **zero UNKNOWN gaps** at minimum
spend — everything above the first pass is imputed-pass, everything below is real-fail — and
it never runs the frontier on a task a weaker model already solved. It reuses the same
budget-capped executor and `collect_phase` primitive as the other modes, so cache-safety and
the `--max-cost` wall are unchanged.

Escalation runs up to `--workers` **different challenges concurrently** — the same
`--workers` fan-out (and default) `cost_optimal`/`full` use, not a new knob — while each
challenge escalates cheap→strong serially, stopping at its first pass (tier N+1 runs only if
tier N failed). Challenge-atomic budgeting survives concurrency through a thread-safe
worst-case **reservation**: before dispatching a challenge the coordinator, under a shared
lock, checks committed spend + the sum of in-flight reservations + this challenge's worst-case
against `--max-cost`, and only then reserves and dispatches (releasing the reservation when it
finishes). Because a reservation is the pessimistic worst-case, N challenges in flight can
never jointly overspend, and a challenge that doesn't fit waits for an in-flight drain to free
headroom rather than being skipped — so the **set of cells collected is identical to the
serial (`--workers 1`) run**; only order and wall-clock differ. Run-level aborts (API-unusable,
the shared consecutive-/start-failure catch-all) are cross-thread: the first worker to hit one
raises the existing `RunAbortError`, which latches a shared abort flag that halts new dispatch
and is re-raised after the in-flight challenges drain. Concurrent writes to `results.csv` are
serialized by a shared write lock.

### How strategies are scored on the completed matrix

The evaluator's shape is unchanged — it still iterates tasks, calls each strategy's
`select()`, looks up the outcome, and accumulates the [metrics](#metrics) above. Because the
completed matrix fills the cells, the set of tasks a strategy can't score collapses to the
genuine UNKNOWN band (empty after ladder collection), so every strategy is scored on the
same tasks automatically. The headline stops being an over-read "X% cheaper" over mismatched
sets and becomes the **paired** router-vs-frontier quality delta on the completed matrix
with an honest CI; oracle-relative regret stays a diagnostic, reported only where full
coverage exists.

### Reporting outputs

The completion adds a few report artifacts, all regenerable from `results.csv` plus
imputation:

| Output | What it shows |
|---|---|
| Capability-distribution histogram | τ per task bucketed over {cheap, mid, high, frontier, unsolvable} — what fraction of the suite models at each rank can solve |
| Violation-rate metric | `v̂` with its Wilson CI, printed to stdout and carried in the capability-distribution figure's footer |
| Per-stratum win-rates | grouping tasks by τ, which strategy wins on reward in each stratum (where routing helps, and where it can't) |
| Coverage table | real vs imputed vs UNKNOWN cell counts per ranked model — the audit of how much rests on imputation |

The cost/quality plots carry this disclosure in their footer NOTE section — the imputed
fraction, the measured violation rate, and the conservative-assumption caveat — so no
reader mistakes a completed matrix for a fully measured one.

## Relationship to src/shunt/

The strategies in `benchmark/routing/strategies/` are evaluation copies — they consume a known matrix and compute metrics offline. They are separate from `src/shunt/router/`, the decision module that is now called on the first turn by the live proxy and learns from verified outcomes recorded at session close. The offline kNN strategy is designed to mirror that module's algorithm, so that live behavior matches what the benchmark scored.
