---
title: Research log
description: What we implemented from the routing literature, what held up on our own data, and what did not.
---

# Research log

Shunt is built as research and development at once. We implement ideas from the
routing literature, then check whether they hold on our own trajectories. This
page records what survived that check and what did not.

Negative results are published here. A router that only reports its wins is not
measuring anything — it is advertising. Where a result cuts against our own
design, it is on this page too.

## ACRouter / Agent-as-a-Router

[arXiv 2606.22902](https://arxiv.org/abs/2606.22902). "ACRouter" and
"Agent-as-a-Router" are the same work — ACRouter is the named instantiation of
the Agent-as-a-Router framework in that paper. Citing both as independent
support double-counts one source. We did exactly that once, and it was wrong.

The paper proposes a Context→Action→Feedback loop: route a coding task, verify
the result, feed the verified outcome back into memory. That is close enough to
Shunt's thesis that we cited it as prior support. So we reproduced the headline
number from the result matrix committed in the authors' own repository.

**What the cascade does.** It tries cheap models in a fixed order and stops when
the task is resolved. It is then scored on that same resolution label. The
stopping rule is the metric. A cascade that halts the instant a grader says
"solved," and is then graded on whether it solved the task, cannot score below
the union of its chain — every task any chain member could solve is a task the
cascade stops on and banks.

Decomposing the published number over their 176 out-of-distribution tasks:

| Configuration | AvgPerf | Total cost |
|---|---:|---:|
| Union of the cheap chain (never escalate) | 71.02% | $46.07 |
| ACRouter as published | 73.30% | $86.72 |
| Always escalate on failure | 78.41% | $144.98 |

71.02 of the 73.30 points come free from oracle-stopping. The agentic gate — the
part that is supposed to be the contribution — adds **+2.28 points for +$40.65**,
and it is beaten on quality by the trivial rule "escalate whenever the cheap
model fails." It behaves as a cost knob on an oracle rather than as a router.

For scale: the strongest single cheap model in their own matrix scores 64.20% for
$18.27. The full cascade scores 73.30% for $86.72.

**The loop is not exercised.** The class implementing the advertised
Context→Action→Feedback loop has zero callers anywhere in the released
repository — no import, no script, no test. The shipped integration API does
write verified outcomes into memory, but its main escalation path never reads
that memory back; it walks a hardcoded chain, so the ordering after ten thousand
tasks is the ordering at task one. Neither headline number runs a loop.

**Our citation no longer stands.** We previously pointed at this paper as
evidence that closed-loop, execution-grounded routing works. It is not that
evidence. The framework may still be a reasonable proposal; the release does not
demonstrate it.

**One caveat we could not resolve.** Three different figures exist for what is
nominally the same out-of-distribution result: the paper reports 62.50, the
repository's committed table reports 73.30, and a genuinely sandbox-verified path
in the same repository reports 66.96 on an overlapping task set. We do not claim
which is right. A reader should know the spread is about eleven points.

To be fair to the authors: that sandbox path exists, it invokes a real grading
harness rather than a lookup table, and a source file in the repository states
the oracle-matrix problem plainly before solving it. Their baseline notes also
disclose where a proxy was substituted for a missing published artifact. That is
more transparency than most releases offer. The criticism is about which number
became the headline.

## The same critique applies to us

Our cascade routing strategies stop at the first attempt whose tests pass, and
are then scored on that same pass label. This is structurally the flaw we just
described in someone else's work, and we are not going to pretend otherwise.

Concretely: a cascade tries models cheapest-first and returns as soon as one
passes. Our reported cascade quality figures are therefore **best-of-N coverage
statistics, not single-shot quality**. They tell you what fraction of tasks
*some* model in the chain could solve, not how well the routing decision worked.

The cost axis is honest — every attempt in the chain is billed, including the
failures, and the fallback attempt after the shortlist is exhausted. Cost
comparisons between strategies mean what they appear to mean. The quality axis
flatters any retry strategy, ours included, and it flatters longer chains most.
Read the two axes accordingly, and treat a cascade's quality number as an upper
bound on what a single-shot router would deliver.

Borrowing a flaw while criticising it would be the worst available outcome, so
we name it here rather than leaving it for a reader to find.

## Rule-based semantic error detection

The same framework describes a self-consistency checker and an LLM-as-Judge among
its verification tools. Its released repository contains an implementation of
neither. There is no ablation, no number attributable to either, and nothing to
evaluate — a design that includes them, and a release in which they are absent.
We report that as a gap in the evidence, not as a refutation.

What *we* measured is the adjacent question: can pattern-matching over model
output stand in for a correctness label?

For **model prose**, no. Coding agents specification-game at roughly a **69%**
rate and misreport success while visible tests are failing. Regex over a model's
own narration of what it did is measuring the narration. That is why Shunt's
labels come only from executed tests, and why the constraint is enforced in the
data layer rather than left to convention: an outcome with no re-executed test
verdict never enters the index that routing reads.

The scope of that finding is narrow and worth stating precisely. Regex over
**executed test output** works fine and we still use it — extracting a test node
id as a dedup key, normalising hex digests and timestamps so a recurring failure
hashes stably, classifying infrastructure failures apart from real ones. Those
rules read a test runner's output, not a model's account of itself.

The claim is not "rules don't work." It is: **rules applied to model-authored
text do not work as a correctness label.**

We also have not measured code-shape heuristics — scoring output by whether it
has a function definition, a return statement, enough lines. We rejected that
class by argument from the reward-hacking evidence and from a ceiling on how much
confidence such a signal could carry, not from a head-to-head run. Saying "it
doesn't work" as a measured claim would require running it. We have not.

## What we are still testing

Shunt's escalation trigger — decide mid-task that a cheap model is not going to
get there — is partly settled. The recurrence policy carries a measured edge: as
shipped (counting every same-key failure) it only separates at thresholds far
above the shipped default, but gated on failures after the agent's first edit it
separates at (and just above) the shipped `escalate_after_n=2` (n=2 AUROC 0.711,
best cell n=3 AUROC 0.724, eval-only);
the prefix risk model performs near chance on our own trajectories (see the
dated entry below and the escalation page). These are the
candidate directions, listed as open questions rather than as a plan we have
committed to.

**Structural rules over trajectory events.** Not keyword matching, but bounded
repetition: the same action producing the same observation N times, an action
repeatedly producing an error, two states alternating. Loop detection is the one
rule-based sub-detector that reliably scores well in published evaluations, and
it needs no training data. Open question: does bounded repetition carry signal on
long agentic coding runs, where repeating a failing test is also what productive
iteration looks like?

**Regex over verified check identifiers.** Aggregating recurrences of a
normalised failing-check id — our current mechanism. On our corpus most
trajectories carry zero or one distinct check id, which leaves the recurrence
counter with little to count. Open question: is there a better key, or is the
whole recurrence framing wrong for this label?

**Calibrated risk scoring with small classifiers.** A shallow gradient-boosted
model or penalised logistic regression over a handful of structural features, with
Platt or Venn-Abers calibration and a conformally controlled firing threshold.
This is the family that best fits our data volume — a few hundred independent
tasks supports roughly ten parameters, not a sequence model. Open question: does
it beat the routing model's own task-level prior? We previously recorded that the
prior "already predicts outcome well from task identity alone"; that measurement
was leaking test-fold labels and has been retracted — grouped honestly the prior
is about 0.42–0.45. The open question stands on its own merits: a detector that merely
rediscovers task difficulty is not an escalation signal.

**N-gram or bigram models over trajectory events.** Count and TF-IDF
representations regularly beat semantic embeddings on small log-classification
datasets. On ours they came out weak — near the floor, well under structural
features. Open question: is that a property of the representation, or of shell
output being a poor lexical signal for eventual task resolution?

**Embedding-based detection.** Pool per-step embeddings over a sliding window
rather than re-embedding a growing prefix, then a shallow head. Our text-based
result above is the prior against it. Open question: does an embedder trained on
code recover signal the count-based methods missed?

**Late fusion of several weak signals.** The long-term shape: unit tests today,
then type checks, lint, and other verifiers, each a weak labelling function with
its own accuracy and coverage. The evidence favours calibrating each signal, then
a simple equal-or-lightly-weighted sum, then recalibrating the pool — learned
combination weights tend to cost more in estimation noise than they recover.
Open question: does fusion beat the single best signal by enough to justify
existing? If not, the fusion layer should be deleted and the single signal
shipped.

One constraint cuts across all of them. Our logging policy never escalates, so
the probability of the escalation action is zero everywhere in our data. Every
standard off-policy estimator requires overlap — some chance of observing the
action you want to evaluate. Without it the value of an escalation policy is not
merely noisy, it is **not identified**. No amount of re-analysis of the existing
trajectories will tell us whether escalation helps. The fix is randomised
escalation at flagged checkpoints with logged propensities, and it needs a live
run to collect.

## Escalation: the edge was real but sat at high recurrence thresholds (2026-08-02)

The escalation sweep once reported `NO_SKILL`, with the shipped default firing on every
trajectory. That was an artefact of the grid. The old sweep varied `escalate_after_n` over
n ∈ {1, 2, 3} only, which could measure the shipped-default mask — it fires on everything, so
precision equals the base rate — and nothing else. Extending the grid to n ∈ {2, 5, 8, 10, 15,
20, 30} × `stale_window` ∈ {10, 1000} (the two knobs are coupled: a recurrence needs a window
wide enough to hold n events) reveals a real, family-wise-corrected edge at high thresholds:
P(fail | fired) climbs from the base rate 0.421 to 0.538 at n=15 (lift 1.28), 0.582 at n=20
(lift 1.38) and 0.706 at n=30 (lift 1.68), with AUROC 0.662 clearing the max-over-cells
family-wise null 95% [0.500, 0.549] at adjusted p = 0.005. The escalation eval status is now
`OK_OFFLINE_ONLY` — through the policy half. The recurrence framing itself was not wrong; it just needed a
higher threshold than the shipped default.

The shallow prefix is informationless. Early steps are all bug-reproduction — the agent re-runs
the failing test before editing anything — so no prefix feature separates and the depth-5 design
is rank-deficient (412 of 414 rows identical). The deep-depth "signal" was length-selection:
admission at depth d requires `depth + MIN_WITHHELD` scorable steps, run length is
outcome-correlated, so the admission test selects failures and a rising incremental at depth
measures that selection rather than prefix evidence. The MIN_WITHHELD wall is validated — it is
what keeps the reported ladder (now depth 10, prefix AUROC 0.478, incremental −0.022) honest.
The R0 instrument gate passes: a planted signal is recovered by the assembled pipeline (AUROC
1.000) and a within-challenge label shuffle collapses it to chance (0.535, inside the band), so
the prefix null is a falsification, not a coverage gap.

Next directions: embedding features on step text fused with the recurrence signal — the policy
edge is real but threshold-only and per-step (production decides once per session, so it is an
offline-only upper bound below the cost break-even), and step content may carry more, earlier
signal — and more distinct challenges: the prefix instrument's minimum detectable effect is
≈ 0.59, and settling it needs roughly 640 challenges, not more runs per challenge. Tuning the
recurrence knob on the LIVE path is not a direct transfer of this result: at session cadence a
higher `escalate_after_n` means more *sessions* failing on the same check, not more steps.

## Escalation: the reproduction phase was the mask, and the grid is now dense (2026-08-04)

The 2026-08-02 finding above ("the recurrence framing just needed a higher threshold") was
re-audited and refined. The per-step stamp is a whole-spec gate — "is the F2P∪P2P target set
green at this step's tree state" — so same-key recurrence is a time-to-fix clock, and the
shipped counter was counting the *reproduction phase*: every run's first reds are the target bug
at t=0. Excluding failures before the agent's first edit-like action (`count_from_first_edit`,
eval-only — production has no per-step action stream) separates at and just above the shipped
threshold: the best cell is edit-gated n=3 — 358/727 runs, P(fail|fired)=0.642 (lift 1.53), AUROC
0.724, clearing both the family-wise null [0.500, 0.542] and a length-stratified null [0.498,
0.563] (p=0.005) — and n=2 already reads AUROC 0.711 (435/727, P=0.593, lift 1.41). The old
"high-threshold edge" (as-shipped n=30, AUROC 0.662) is roughly 40%
run-length selection. The grid is now dense — n ∈ {1,2,3,4,5,6,8,10,12,15,20,25,30,40,50} ×
{10, 1000} — so the PR/ROC figures trace the full precision/recall frontier; a continuous
"stuck-depth" score (`max_recurrence`) gives the complete ROC (as-shipped AUROC 0.601 vs
edit-gated 0.781). At the session cadence the ladder's value is large and observational: a
frontier session resolved 56.8% (21/37) of tasks a cheap session failed vs 22.6% (7/31) for a
same-cost cheap retry — 2.5× on the same-instance subset. An embedding "going-in-circles"
feature (fastembed on failing-step output) was explored in an earlier, UNCOMMITTED experiment and
carried ~0.70 AUROC alone; that figure was never reproduced by the committed pipeline — the
escalation path currently uses no embeddings at all — so it stands as a superseded hint, not a
measured result, and the claim is not repeated in the results page.

## Contributing

If you have trajectory data, a detector that beats the baselines above, or a
reproduction that contradicts something on this page, open an issue. A
contradicted result is more useful to us than a confirmed one, and this page is
where it will end up either way.
