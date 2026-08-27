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

## The escalation trigger was counting the wrong steps

Shunt's escalation trigger counts how many times the same normalised
failing-check id recurs, and escalates once that count reaches
`escalate_after_n`. The first honest thing to say about it is that the shipped
configuration is a null detector. At the shipped threshold it fires on all 723
scored runs, and among those the failure rate is 0.4177 against a corpus base
rate of 0.4177. A trigger that fires on everything reproduces the base rate by
construction. This is not specific to the shipped value: across the as-shipped
family at the shipped `stale_window`, every threshold up to n=10 reads AUROC
0.500-0.597, so no low `escalate_after_n` is measurably better than any other.

What is wrong is not the recurrence framing. It is what the counter counted.
The per-step verdict is a whole-spec gate — is the target test set green at this
step's tree state — so a run of same-key failures is a time-to-fix clock, and
every run opens with the target bug already red. The agent reproduces the bug
before it edits anything, so the first stretch of every trajectory is the same
failure repeating for reasons that say nothing about how the run will end. The
counter was measuring the reproduction phase.

Not counting failures that precede the agent's first edit-like action separates
at the threshold the product already ships. At the shipped `escalate_after_n=2`,
edit-gated counting fires on 431 of 723 runs; the failure rate among them is
**0.589** [0.508, 0.655] against **0.164** among the quiet runs, an AUROC of
**0.710** against a run-length-only baseline of 0.568. One step up, at n=3, it
fires on 354 of 723 at **0.638** [0.554, 0.703] against **0.210**
[0.151, 0.270], an AUROC of **0.722** against 0.576, and clears both a
family-wise null of [0.500, 0.549] and a length-stratified null of
[0.498, 0.565] at p = 0.0005 over 2000 challenge-block shuffles. Read as a
continuous score rather than a threshold, the same contrast is AUROC 0.601
as-shipped against 0.781 edit-gated. The sweep covers `escalate_after_n` ∈
{1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50} crossed with
`stale_window` ∈ {10, 1000}; the best cell in that grid, taken across both
counting families and corrected for the maximum, is n=3 — at the wider window,
AUROC 0.724. That best cell is EDIT-GATED, so it is not a reason to move the
shipped default: the product has no edit-gated counter, and under the counter it
does run, n=2 and n=3 are 0.500 and 0.501. The default stays a prior.

Three caveats, none of them small. The edit gate is **eval-only** — the live
router has no per-step action stream to gate on, so this measures what a
per-step detector could do, not what ships. The score is computed at step
cadence while production decides once per session. And it is **association
only**: no trajectory we have logged ever escalated, so nothing here shows that
escalating would have helped.

At session cadence there is a separate, observational number that does bear on
that. On the instances where both were attempted, a session at one of the two
most expensive models resolved 62.2% (28 of 45) of the tasks a cheap session had
failed, against 20.6% (7 of 34) for a same-cost cheap retry — a paired difference
of +0.416 [+0.239, +0.581] over the 48 shared instances. That is what escalating
to *those* models is worth if you can tell when to do it. It is not evidence that
the trigger tells you, and it is not the shipped ladder's result: the same
comparison run against the trivial arms shows the escalate arm losing to
always-frontier (−0.108 [−0.165, −0.056]) and tied with firing at random at the
same rate (−0.039 [−0.152, +0.084]).

The other half of the eval is a risk model over the run's opening steps, and it
reads nothing. At the shallowest depth this corpus can score full-rank, prefix
AUROC is 0.478 and the incremental over the task prior is −0.022. That is a
falsification rather than a coverage gap: on the same pipeline, a planted
known-learnable signal is recovered at AUROC 1.000 and a within-challenge label
shuffle collapses it to 0.535. The instrument works; the shallow prefix has
nothing in it. Its minimum detectable effect is ≈ 0.59, and closing that needs
roughly 640 distinct challenges rather than more runs on the ones we have.

One near-miss is worth recording so nobody rediscovers it as a result. An
embedding feature over failing-step output — is the agent going in circles —
carried around 0.70 AUROC alone in an early experiment that was never committed.
The committed pipeline has never reproduced it, and the escalation path uses no
embeddings at all today. It is a hint, not a measurement, and it is not repeated
on the results page.

## What we are still testing

The directions below are open questions, not a plan we have committed to.

**Structural rules over trajectory events.** Not keyword matching, but bounded
repetition: the same action producing the same observation N times, an action
repeatedly producing an error, two states alternating. Loop detection is the one
rule-based sub-detector that reliably scores well in published evaluations, and
it needs no training data. Open question: does bounded repetition carry signal on
long agentic coding runs, where repeating a failing test is also what productive
iteration looks like?

**Regex over verified check identifiers.** Aggregating recurrences of a
normalised failing-check id — our current mechanism. There is no shortage of
recurrences to count, but there is no diversity in the key: most trajectories
carry zero or one distinct check id, so the count is a clock on a single failure
rather than a signal read across several. Open question: is there a better key,
or is the whole recurrence framing wrong for this label?

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
rather than re-embedding a growing prefix, then a shallow head. Two results are
the prior against it: the count-based text representations above came out weak,
and on the routing half, task embeddings of the real problem statements sit
inside their own shuffled-outcome null while a three-level human difficulty tag
clears it on the same pipeline and the same sample. Open question: does an
embedder over step text recover signal those methods missed?

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

## Three claims we retracted after auditing our own benchmark

An audit of every committed figure found that three conclusions we were about to
sell you were **measurement artifacts**. We record it here rather than quietly
fixing it, because how a project handles its own bad results is the only evidence
you have about the rest of its numbers.

**We were not embedding the task — and now we are.** The router used to embed a
string built as `repo@sha — resolve <pytest node id>`: a median of 106 characters
containing a repo slug, a truncated commit hash and a test path. No problem
statement, no code, nothing about difficulty. 61% of each task's twenty nearest
neighbours shared its repo against a 10% chance rate — it was behaving as a path
detector. So *"prompt embeddings cannot separate task difficulty"* had never been
tested.

It has been now. The task manifest was rebuilt from the pinned SWE-bench revision
with the real `problem_statement` (median 1,185 characters), `routing_text()`
prefers it, and every committed routing figure and number is computed on that
basis. Two independent statistics, both in the committed pipeline:

| | embedding | human difficulty tag | shuffled null 95% |
|---|---:|---:|---|
| Neighbourhood Brier skill | −0.045 | **+0.038** | [−0.104, −0.012] |
| Leave-one-out R² on `p_solve` | −0.053 | **+0.076** | [−0.115, −0.001] |

The embedding sits inside the null on the correct input, while a crude three-level
human tag clears it on the same pipeline, the same n and the same null. That is a
**working positive control beside a negative result**, which is what makes this a
falsification rather than a coverage gap — the distinction we previously could not
claim. Task identity accounts for ~57% of outcome variance, so there is structure
there; this encoder does not reach it.

Routing quality *fell* when the input was corrected: kNN went from 81.71% to
**77.72%**, which is inside noise of always-cheap's 75.54%. The 106-character label
was not merely uninformative, it was mildly leaky — the repo name it carried is a
weak difficulty proxy. Given the right input, the learned router is not
distinguishable from the trivial policy. Figures:
[`embedding_signal`](routing.md#fig-embedding-signal),
[`knn_calibration`](routing.md#fig-knn-calibration).

**The detection floor still bounds what a null here can mean.** Running

```sh
python3 -m benchmark.routing.sensitivity
```

puts the weakest effect this test can detect at 80% power at **AUROC ≈ 0.88**, the
most sensitive of the eight configurations it sweeps. Published task-difficulty
detectors sit near 0.62. So the falsification is specific and bounded: on this
corpus, at this n, this encoder carries no routable signal — not that no encoder
could.

**Our escalation baseline was reading the test labels.** The comparison gave each
run the leave-one-out failure rate of *its own instance's other runs*, while the
cross-validation split grouped by instance — so the baseline was scored on labels
from its own test fold. A router meeting a new task has no such siblings. That is
where `AUROC 0.883` came from; computed honestly it is about 0.42–0.45. Since the
headline was `detector − baseline`, the detector was being asked to beat an
oracle, which is why it reported −0.000.

Worse, the permutation test that was supposed to catch this **could not fire.**
Shuffling labels globally collapses the baseline to chance, giving the null 0.5
of headroom while the observed statistic was arithmetically capped at 0.117 — and
the null's 97.5th percentile sat at 0.117. No detector, however good, could ever
have passed that gate. The null now permutes labels **inside each challenge**, so
every challenge keeps its outcome multiset and the baseline is identical under
the null and the observation; only the prefix's contribution is nulled.

**And the comparison was floored the wrong way — or rather, not at all.** The
honest prior scores *below* chance on this corpus (0.42 at 5 decisions), so
`combined − prior` was handing the detector the baseline's deficit as if it were
skill. Measured against `max(prior, 0.5)`, as it always should have been, the
headline increment at 5 decisions falls from **+0.144 to +0.061** — 57% of it was
the broken comparator. What remains does not clear the corrected null.

**Our escalation baseline was reading the fold identity.** The prior column,
`prior_from_splits`, gives every test row its train-fold base rate. Under the old
GroupKFold, that value is the exact arithmetic complement of the fold's own test
prevalence (measured corr = −1.0000 at every depth). The prior thus acted as a
fold-identity proxy: it added zero within-fold discrimination
(within-fold Spearman = 1.000000) while shifting the pooled AUROC. The
incremental headline at depth 20 was published as +0.076, p = 0.015. But six
columns of pure Gaussian noise over the real challenge group structure produced
E[incremental] = +0.042 at depth 20, with a maximum of +0.080 across 8 seeds —
the published +0.076 was *smaller* than what pure noise produced. StratifiedGroupKFold
now collapses the fold-prevalence spread from 0.141 to 0.011, and a null-corpus
regression test pins E[incremental] ~ 0. The pre-existing positive-control
fixture was structurally immune (one failed and one resolved run per challenge
makes every fold base rate exactly 0.5), which is why it never caught the defect.
A label-substitution positive control tested the real six features against model
identity (cheap vs frontier): AUROC 0.497/0.525/0.538 at depths 5/10/20, versus
0.428/0.491/0.543 against the real failure label — both measured before the
state-capture audit marked unverifiable steps as unmeasured. That the shipped
features cannot distinguish model identity — a fact unambiguously present in
trajectory patterns — shows the gate was a null generator, not an instrument.

So we are withdrawing *"the escalation model does not work"* and replacing it
with something less satisfying and more accurate: **we cannot yet tell** — at
least not for a shallow-prefix detector, which is why that half still reads
`NO_SKILL`; the escalation results now attribute the signal to the recurrence
rule at the **shipped** threshold once the reproduction phase is excluded
(eval-only; status `OK_OFFLINE_ONLY` at the edit-gated `escalate_after_n=3`,
AUROC 0.722, see [Results](results.md#escalation-results)).
The prefix evaluation could only ever have detected a detector at AUROC ≥ 0.59.
The raw features hint at ≈ 0.52 — squarely inside the blind spot. Resolving it
needs roughly four times the distinct challenges (152 → ~640); more runs per
existing challenge buy almost nothing, because the clustering already inflates
variance ~3×.

**What survives all of it:** the cascade result. It is untouched by every defect
above, because it uses no model, so there was no model to get wrong — and it is
no longer only a measurement. `Price-Cascade` at $27.11 against Always-Frontier's
$96.02 is still blocked at boot and still unrunnable, but the session-cadence
ladder reaches the same 96.74% for $28.71 cache-aware, is cache-safe by
construction, ships enabled, and is now nameable as
`router.strategy: session_cascade`. What the audit and the un-imputed basis together
sharpened is the *size*: on fully-measured tasks the saving is ~25%, not ~75%.
And ~90% of the headroom is mechanical, which bounds the entire remaining prize
for a perfect difficulty predictor at **about $7.0 on a $96.02 base**. That number
is the honest answer to "how much is routing intelligence worth here", and it is
small.

## Contributing

If you have trajectory data, a detector that beats the baselines above, or a
reproduction that contradicts something on this page, open an issue. A
contradicted result is more useful to us than a confirmed one, and this page is
where it will end up either way.
