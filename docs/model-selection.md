---
title: Model triage
description: Which models earn a slot in the router's live pool — the routine frontier with its Wilson-CI band, the escalation rung, and what gets dropped.
---

# Model triage: which models earn a live-pool slot

Every week a model is released, and every few weeks an incumbent is deprecated or
repriced. A pool has to react to that churn with something better than taste. This
page is the membership rule: a model earns a slot in the router's live pool when it
clears one of two evidence bars, and is dropped when it clears neither. The rule is
recomputed mechanically from committed data, so the pool is a consequence of the
measurement, not a decision someone made once.

## The two strata, and why the frontier is per-stratum

The pool serves two different jobs, and a model should be judged on the job it
actually does. The two strata have their own quality measure and their own frontier:

- **Routine stratum** — the tasks the cheap model would take. Quality is the
  marginal verified pass rate (a Wilson 95% CI) on the model's default reasoning
  arm.
- **Escalation stratum** — the tasks the cheap base, `deepseek-v4-flash`, failed.
  Quality is the change in resolve rate against that base on the paired overlap,
  with the exact paired-test p from the ladder evidence.

A single marginal-rate bar would get the pool wrong, and `zai-glm-5.2` is the
working example. On the routine stratum it is dominated: 57.1% at $5.80/Mtok sits
well below the frontier. A marginal-only rule drops it. But as an escalation rung it
is a different model — it resolves +0.1548 of the tasks the base failed, a net-helpful
pair that clears the test. It is in the pool because of the escalation stratum alone,
and the rule is built so a model that is *only* an escalation rung can still earn its
slot.

## The routine frontier and the Wilson-CI band

The reference for the routine stratum is the cost–performance frontier of the
**incumbent live pool**, taken per stratum. At any price `p`, the envelope `env(p)`
is the best marginal rate among live models priced at or below `p`. A model keeps
its slot when its CI upper bound reaches that envelope at its own price — overlap
with the frontier means it is inside the noise band. A model cheaper than every live
model defines the envelope's low end and keeps by construction. Below the band — too
slow, too expensive, or both — is a drop.

The frontier of what is already live is the reference for a reason. "The cheapest
model" is a tautology, not a standard: it says nothing about whether the pool's
second slot is earning its price. "The single most cost-effective model" is almost as
bad — it crowns one point and treats every other slot as failure, which is how a
pool loses a frontier rung the moment a cheaper model arrives. The live-pool frontier
is non-circular — the incumbent set is fixed before the new model is measured — and
it adapts on its own as the market improves, so the bar moves without a human
re-deciding it.

## The escalation bar

The escalation bar is a paired verdict, not a rate. A candidate rung is scored only
on the challenges where both it and the base have a default-arm outcome, and the
paired difference is tested with the exact paired-exchangeability test from the
ladder evidence. The model keeps its slot only when the verdict is **NET-HELPFUL**
(p < 0.05 on the helpful side). **NET-HARMFUL** and **INDISTINGUISHABLE** clear
nothing, so a flat rung — one that neither helps nor measurably hurts — still drops.

## The current table

| model | price $/Mtok | n | marginal rate | 95% CI | routine | esc Δ resolve | esc verdict | verdict |
|:---|---:|---:|---:|:---:|---:|---:|:---|:---|
| deepseek-v4-flash | 0.42 | 190 | 68.9% | 62.1–75.1 | KEEP | — | — | KEEP |
| zai-glm-5.2 | 5.80 | 84 | 57.1% | 46.5–67.2 | DROP | +0.1548 | NET-HELPFUL | KEEP |
| gemini-3.1-pro | 14.00 | 0 | — | — | — | — | — | UNMEASURED-EXCEPTION |
| kimi-k3 | 18.00 | 110 | 84.5% | 76.6–90.1 | KEEP | +0.2364 | NET-HELPFUL | KEEP |
| claude-opus-4-8 | 30.00 | 0 | — | — | — | — | — | UNMEASURED-EXCEPTION |
| gpt-5.6-sol | 17.50 | 0 | — | — | — | — | — | UNMEASURED-EXCEPTION |
| claude-fable-5 | 60.00 | 0 | — | — | — | — | — | UNMEASURED-EXCEPTION |
| qwen3.7-plus | 1.60 | 87 | 43.7% | 33.7–54.2 | DROP | +0.0345 | INDISTINGUISHABLE | DROP |
| gpt-5-mini | 2.25 | 200 | 54.5% | 47.6–61.3 | DROP | −0.1684 | NET-HARMFUL | DROP |
| kimi-k2.5 | 3.60 | 121 | 49.6% | 40.8–58.4 | DROP | −0.0165 | INDISTINGUISHABLE | DROP |

`n` is the routine default-arm cell count — the basis of the marginal rate and the
routine verdict. The escalation columns rest on a different n: the paired overlap
with the base, which is not the same number (`gpt-5-mini` shows n=200 here but n=190
on the [escalation ladder](escalation-claim.md)). Read each column against its own
basis.

This table is the regenerable report at `benchmark/routing/reports/triage_summary.csv`,
itself derived from the committed `results.csv`, the shipped registry in
`src/shunt/config/models.yaml`, and the ladder evidence. Recompute it with:

```bash
uv run python -m benchmark.routing.triage --out-dir benchmark/routing/reports
```

The CSV is gitignored: it is a build artifact, not a source. The per-model pass rates
behind the marginal column are in [Results](results.md#how-to-read-this-page), and
the per-rung escalation numbers behind the esc columns are in
[the ladder-rungs figure](routing.md#fig-ladder-rungs).

This committed pool is also the rule's positive control. The rule must reproduce
`deepseek-v4-flash` / `zai-glm-5.2` / `kimi-k3` = KEEP and
`qwen3.7-plus` / `gpt-5-mini` / `kimi-k2.5` = DROP — and it does, on every run.
`zai-glm-5.2` is the load-bearing case: dominated on the routine stratum, saved only
by the escalation one.

## Statuses and exceptions

Every verdict is one of five:

- **KEEP** — clears the routine band, or is a net-helpful escalation rung.
- **DROP** — clears neither.
- **INSUFFICIENT-DATA** — fewer than `K = 20` default-arm cells. No verdict, no slot
  decision either way.
- **UNMEASURED-EXCEPTION** — a benchmark-disabled model: a policy slot the committed
  corpus cannot measure, such as the research-estimated frontier tail. It is exempt
  by design, never a violation.
- **EXCEPTION** — a measured-DROP model kept under a named exception recorded in
  `benchmark/routing/triage_exceptions.yaml`. Advisory, surfaced by the gate, never a
  violation and never silent.

Named exceptions are allowed — a model may be kept for availability, or for a
capability the pool lacks. What is never allowed is a silent exception: it has to be
recorded in the triage report, visible next to the verdict, or it does not count.
The four `UNMEASURED-EXCEPTION` rows above are NOT named exceptions: that status is
automatic, because the committed corpus cannot measure a benchmark-disabled model.
A named exception is a separate, manual mechanism — it promotes a *measured* DROP to
verdict **EXCEPTION** via `benchmark/routing/triage_exceptions.yaml`, which holds no
entries today.

## Enforced mechanically

The rule is not advisory. The **SH015** pre-commit gate reruns the triage on every
commit and fails when a live-pool model — one listed in `router.yaml`'s `models:` —
verdicts as DROP, naming the model and its evidence so the fix is to remove it from
the pool or to collect evidence that clears a bar. A model measured DROP is not
deleted from the benchmark; it stays in the measured set and keeps being measured, so
a verdict is always revisable the day the evidence changes. A dropped model that
recovers can re-earn its slot; a dropped model that quietly stays dropped is what the
gate exists to prevent.

## Limits

Read the table with these in hand.

- **Marginal rates are over each model's own adaptive coverage, and are not
  cross-comparable at face value.** Each rate is over the tasks that model actually
  ran; coverage differs per model. Results carries the common-74 check — the 74 tasks
  every model ran — and the ordering barely survives it
  ([Results](results.md#how-to-read-this-page)). The triage rule leans on the CIs
  precisely because of this, but the band is still computed on those marginal rates.
- **A KEEP is per-stratum.** The escalation bar is the paired ladder verdict on the
  committed corpus, so a model can earn a KEEP on the stratum it serves while being
  dominated elsewhere — that is the `zai-glm-5.2` case. KEEP means *earns one of the
  two slots*, not *wins both*.
- **A green SH015 exit is a relative statement, not an attestation of quality.**
  When `results.csv` is absent or empty, every model is `INSUFFICIENT-DATA` and the
  gate is green by construction — so a passing run means *no measured live model is
  dominated relative to the frontier*, never that the pool is good. Four of the seven
  live slots (the research-estimated frontier tail) are benchmark-disabled and exempt
  by design; only the three measured models are ever actually judged.
- **Benchmark-disabled frontier models are unmeasured by policy.** Their prices are
  research estimates, not live listings, and the corpus does not run them. That is
  why they are `UNMEASURED-EXCEPTION` rather than KEEP or DROP — the rule refuses to
  grade what it cannot measure, and the exception note says so.

To add a model to the registry — and therefore into triage's candidate set — see
[Configuration → Add a model](configuration.md#add-a-model); the pool those models
fill is described in [Architecture](architecture.md#capabilities).
