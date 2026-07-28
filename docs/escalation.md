---
title: Error detection & auto-escalation
description: How Shunt detects a real, verified failure — never a model's self-report — and, when the same failure repeats, escalates one rung at the next session boundary. Off by default and opt-in.
---

# Error detection & auto-escalation

When the cheap model keeps failing the *same* verified check, Shunt can move up on
its own — raising the model's reasoning effort first, then its rank — without waiting
for you to intervene. This page explains how a failure is detected, what makes one
worth escalating, how the ladder climbs, the safety rails around it, and where it
does nothing.

**It ships OFF.** Auto-escalation is opt-in (`router.escalation.enabled: false` by
default). It is wired on the live routing path, so turning it on takes effect
immediately — but you turn it on deliberately, once you have a verified-outcome
signal for it to act on. The config knobs are in
[Configuration → Auto-escalate on repeated verified failure](configuration.md#auto-escalate-on-repeated-verified-failure).

## Detection — what counts as a failure

Escalation acts on **verified** failures only. A model's own claim that "the tests
pass" is never trusted — coding agents misreport — so the signal comes from a
non-model producer: your test suite, re-run off the wire.

- **Off-wire re-run.** At session close, if you have configured a `work_dir`, Shunt
  re-runs *that repo's* test suite off the request path and records the pass/fail as
  a verified outcome. The runner is auto-detected from the repo: **pytest** (Python),
  **jest**/**vitest** (a `package.json`), **`go test`** (a `go.mod`), or **`cargo
  test`** (a `Cargo.toml`). No `work_dir`, no framework, no signal — Shunt writes
  nothing and never guesses. See [Feedback](feedback.md#1-automatic--off-wire-test-execution-the-signal-that-matters).
- **Flake guard.** A test that fails then passes on unchanged state is a flake, not a
  regression. A failing run is re-run to confirm; if it does not reproduce, the result
  is abstained (it feeds neither the router nor escalation). Only a failure that
  reproduces every time is passed through.
- **Environment vs capability.** A missing module, a broken test-collection step, or a
  wrong interpreter (`ModuleNotFoundError`, a pytest collection error, `go: cannot
  find module`, an unresolved Rust import) is **infrastructural** — no bigger model
  fixes a missing dependency. Shunt classifies these as environment failures and they
  **never count toward escalation**. Only a genuine capability failure — the code ran
  and the assertions failed — is escalation-eligible.
- **A stable failure identity.** Each confirmed failure gets a **dedup key**: the
  failing test's node id where the runner prints one (`path::Test::case`), or a
  normalized hash of the failure detail otherwise — with timings, hex addresses,
  temp paths and timestamps stripped, so the *same* recurring failure hashes to the
  same key run to run. That key is what lets Shunt tell "the same problem again" from
  "a different failure."

Human feedback (`shunt flag <id> bad`) is the other verified source and carries the
same weight as a failing suite — a person confirming the result is ground truth.

## Triggers — when it escalates

One failure is not enough. Intermediate fail-then-fix is normal, so a single verified
failure never escalates. Shunt escalates only when the **same** verified failure
recurs:

- **The same key, `escalate_after_n` times** (default **2**) within `stale_window`
  decisions (default **10**). Two reds on the *same* failing check inside the window
  trip it.
- **Different failures don't aggregate.** Two verified failures with *different* keys
  are two different problems — that is the kNN store's job to learn from, not a signal
  to escalate. Only same-key recurrence counts.
- **A window that goes quiet retires.** A failure that does not recur within
  `stale_window` decisions is dropped from the counter, so an old, since-fixed problem
  cannot trigger later.
- **Success clears the slate.** When the whole suite goes green, every pending failure
  for that repo is retired — the router saw the problem resolve.
- **One escalation consumes its evidence.** After Shunt steps a rung, the failures it
  acted on are consumed. Climbing the *next* rung requires a genuine fresh
  recurrence — two more verified same-check reds — not the same two firing again.

Escalation is keyed per **task** (the repo / `work_dir`), so a repeated failure in one
project never escalates routing for another.

## The ladder — effort first, then rank

The default ladder is `effort_then_rank`, and it climbs **one rung per step, never
straight to the top**:

1. **Raise reasoning effort first.** The router bumps the *current model* up one
   reasoning arm (e.g. `medium` → `high`). It is the **same model**, so the provider's
   prompt-cache namespace is unchanged — this rung is cache-safe. The higher arm's
   request params are applied to the outbound call, overriding any the client sent.
2. **Then step a rank.** Only once the model's reasoning arms are exhausted — or if the
   model declares no reasoning arms at all — does the router step to the next model
   rank (the next-higher-price model). The new model starts at its *own* default
   arm, not mid-ladder.
3. **Hold at the top.** At the ceiling (top rank, top arm) escalation holds rather than
   thrashing.

Set `ladder: rank_only` to skip the effort rung and step ranks directly. The effort
rung needs a model that declares [reasoning arms](configuration.md#reasoning-effort-optional);
a model without them has no effort headroom and steps rank immediately.

## Safety — the rails

- **Never mid-cached-turn.** An escalation applies only at the **next session
  boundary**. Shunt never switches a model in the middle of a cached conversation,
  which would force a full-price re-read of the context. Cache-safety is preserved by
  construction.
- **Routing-collapse guard.** If the recent model-choice distribution is degenerate —
  the expensive tail dominates, or choice-entropy collapses — a routing-collapse alarm
  **suppresses further escalation** so the router cannot ossify onto costly models.
  The same signal is exposed at `GET /admin/loop-health`.
- **Escalated turns don't train the policy.** An escalation is imposed by the failure
  signal, not chosen by the policy, so an escalated turn is recorded as non-policy: its
  selection propensity and candidate scores are neutralized, and it opens a fresh label
  window. The learner never mistakes a forced escalation for a free policy win.
- **State survives a restart.** The failure log and per-task ladder position are
  snapshotted, so a restart resumes where it left off rather than forgetting a
  half-climbed ladder.

## Limitations — read before enabling

Be honest with yourself about where this does nothing:

- **Off by default, and for a reason.** With no verified-outcome history, every
  cheap-model failure would look escalation-worthy and nothing would be learned. Turn
  it on once you have a handful of labelled sessions (roughly 5–10) so the trigger has
  something real to act on.
- **No `work_dir`, no automatic signal.** Auto-escalation is inert until you point
  Shunt at a repo it can test. Without that, the only verified failures are the ones
  you enter by hand with `shunt flag <id> bad`.
- **No tests, no signal — the vibecode case.** A repo with no test suite produces no
  verified outcome, so auto-escalation does nothing there. It cannot escalate on a
  signal that does not exist.
- **The effort rung needs reasoning arms.** A model that declares none skips straight
  to a rank step; there is no effort headroom to climb.
- **Runs where Shunt sits beside your code.** The off-wire re-run needs the repo *and*
  its test toolchain on the same machine — a plain `shunt start` on your dev box. A
  slim container has neither unless you mount them; there, use `shunt flag` via
  `docker exec`. See [Feedback → by deployment](feedback.md#giving-feedback-by-deployment).
- **A pre-human-label mechanism, still being proven.** This is the day-one learning
  signal that works before any human-rating flow exists. Whether cheap-first routing
  plus verify-and-escalate beats always-frontier at equal quality is what Shunt is
  still validating; treat auto-escalation as opt-in until that holds for your own
  workflow.

## Turn it on

```bash
shunt start --config-override 'router.escalation.enabled=true'
```

or set it permanently in your `router.yaml`:

```yaml
router:
  escalation:
    enabled: true
    escalate_after_n: 2         # same-key verified failures before a step
    stale_window: 10            # a failure not recurring within N decisions retires
    ladder: effort_then_rank    # or rank_only
```

The knob reference, including the reserved `blocking_exit_code` field, is in
[Configuration](configuration.md#auto-escalate-on-repeated-verified-failure).

## Measuring whether escalation actually helps

Turning escalation on tells you nothing about whether it *worked*. A deterministic policy
escalates at every checkpoint it flags and nowhere else, so the logs contain no case where a
flagged checkpoint was left alone. There is no comparison to make — not a noisy one, none. Any
"escalation improved things" read off such logs is a difficulty proxy: escalation fires on the
runs that were already going badly.

`exploration_epsilon` fixes that by withholding the escalation at a small random fraction of
flagged checkpoints and **recording the probability that generated each choice**:

```yaml
router:
  escalation:
    enabled: true
    exploration_epsilon: 0.2    # withhold ~1 in 5 flagged escalations
    exploration_seed: 20260728  # optional; omit and shunt draws + records one
```

It is a **separate opt-in**: `enabled: true` on its own never randomizes anything, and
`exploration_epsilon: 0.0` (the default) is today's behaviour bit for bit.

What gets recorded, on the session's decision provenance, at flagged checkpoints only:

| Field | What it is |
|---|---|
| `checkpoint_id` | The recurring failing-check id that flagged this checkpoint |
| `action` / `policy_action` | The arm taken, and what the deterministic policy would have done |
| `propensity` | `P(action taken)` — `1-ε` for the escalation arm, `ε` for the withheld arm |
| `epsilon`, `seed` | Enough to re-derive the draw after the fact |
| `features` | The state as of the decision — failure counts, distinct keys, ladder headroom |
| `randomized` | `false` for a checkpoint with no rung left to climb; those are excluded, not weighted |

Three properties worth knowing:

- **It cannot cost you cache.** The explored arm is always *hold*. Randomizing can only
  withhold an escalation, never invent one, so it introduces no model switch the deterministic
  policy would not have made — and the directive still applies at the next boundary.
- **It is reproducible.** Given the seed, the sequence of draws is exact, so a logged
  propensity can be audited rather than trusted.
- **It costs quality while it runs.** You are deliberately declining some escalations. Turn it
  off outside collection windows.

Read the logs back with the estimator in `benchmark/escalation/ope.py`:

```python
from benchmark.escalation.ope import always_escalate, estimate_policy_value, rows_from_records
from shunt.db.store import OutcomeStore

store = OutcomeStore()
result = estimate_policy_value(rows_from_records(store.escalation_exploration_rows()))
print(result.status, result.dr_estimate, result.ci_low, result.ci_high)
```

It reports a doubly-robust estimate of a target policy's value with a 90% bootstrap interval —
**or** `status == "not_identified"` with every numeric field `None`, which is what you get from
logs that contain no randomization, only one realized arm, or no verified outcomes. That refusal
is deliberate: the estimator will not hand you a number the data cannot support.

## Evaluating the detector offline

Before you trust the trigger on live traffic, you can measure it — offline, at no API
cost — against stored agent trajectories. The `benchmark/escalation/` harness replays the
exact same decision the live router uses over per-decision trajectories and sweeps its
knobs, so you can see where it fires, how early, and at what precision.

Run it end-to-end:

```bash
make escalation-eval
# equivalently: uv run --extra benchmark python -m benchmark.escalation.run_eval
# figures land in benchmark/escalation/reports/; pass --plots-dir to write them elsewhere
```

The `--extra benchmark` flag (baked into the Make target) is required — it pulls the
eval deps (`matplotlib`, `swebench`, …); a bare `uv run` strips them.

It prints a JSON report and two metric tables to stdout, and renders the figures. The eval
has **two independent blocks**, because they answer different questions:

**1. The policy, graded per trajectory.** The shipped recurrence policy is replayed over each
stored run and asked the question the product actually asks: given it fired, is this run more
likely to fail? One row is one trajectory (not one prefix), `n_escalated` is per configuration,
and every rate carries a 95% Wilson interval next to the corpus base failure rate. Read
`P(fail | fired)` against `base`: an interval containing the base rate is a configuration with
no measured value, and an interval below it means firing predicts *success*.

**2. A prefix risk model, graded against the null.** A continuous risk score — the probability
this run ends unresolved — is fit from prefix-only features at fixed decision depths and
evaluated out-of-fold. Three numbers are reported and only the last one is meaningful:

- `prior` — the router's own t=0 knowledge (leave-one-out per-challenge failure rate). Task
  identity alone is a strong predictor, so any unconditional number mostly rediscovers task
  difficulty.
- `prefix` — out-of-fold discrimination from the prefix alone.
- `incremental` — `AUROC(prior + prefix) − AUROC(prior)`. **This is the number that decides
  whether escalation is worth anything.** Everything else overstates it.

What the rest of the report means:

- **`status` is gated by a permutation null, not by a point estimate.** The status is `OK`
  only when a depth's incremental AUROC sits above the 97.5th percentile of at least 200
  label shuffles, with the whole fitting pipeline re-run per shuffle. Otherwise it is
  `NO_SKILL` (with the number and the p-value in the reason) or `INSUFFICIENT_DATA`, and
  every figure states it in red under the plot. A point estimate above 0.5 is never treated
  as skill on its own.
- **Two structural anti-leak rules.** Features are read at a **fixed absolute depth**, never a
  fraction of the run — a fraction needs the total length, which is future information. And
  the **terminal step is excluded**, because the harness verdict is stamped onto it. Both are
  pinned by tests; see `benchmark/escalation/features.py`.
- **Grouped cross-validation by challenge.** A challenge never appears in both train and test,
  so the model cannot rediscover task identity through the fold boundary.
- **The sweep varies `escalate_after_n` only.** `stale_window` and `ladder` are pinned at their
  defaults because both were measured inert on the current corpus: the full 12-cell grid
  collapsed to 2 distinct score vectors. `escalate_after_n=1` is swept alongside the shipped
  default of 2 so the two are always reported side by side; the report **flags** a better cell
  in its notes and never changes the shipped default.
- **Runs with no per-step verified outcomes are excluded and counted.** A trajectory the
  stamping stage never processed carries parser defaults, not evidence; including it would feed
  the model a collection-date proxy. The exclusion count appears in the JSON and in every
  figure footer.
- **Figures.** PR curve, ROC (both tie-collapsed, so the drawn area equals the reported
  statistic, with the permutation null band shaded), the confusion matrix with a
  random-at-the-same-flag-rate baseline in each cell, the permutation-null histogram, lead time
  split by terminal outcome, the sweep as an interval table, trajectory outcomes against the
  base rate, and failure-capture coverage per model. Each carries a footer — what the axes are,
  what to look for, what the jargon means, and the honest limits — so a figure pasted elsewhere
  is still readable on its own.

**The harness runs on captured multi-step trajectories only.** The recurrence trigger cannot
fire on a length-1 stream, and a prefix model needs a prefix, so the eval reads
`benchmark/escalation/data/live/` (tracked via Git LFS, with a content-hash `manifest.json`
that fails CI if a label is tampered with). Point `--live-dir` elsewhere to score your own.

### Capturing your own trajectories (opt-in, encrypted, local-only)

To evaluate the detector on *your* traffic, Shunt can record full-content per-step
trajectories from live sessions. This is **off by default** and secure by construction:

```yaml
router:
  capture:
    work_dir: /path/to/repo    # the off-wire verifier's repo (see Feedback)
    full_content: true         # opt in to full-content capture
    trajectory_dir: null       # null ⇒ a local dir OUTSIDE the repo ($SHUNT_HOME/trajectories)
```

When enabled it needs the `capture` extra (`pip install 'shunt-router[capture]'`) and an
encryption key in `SHUNT_ESCALATION_KEY` (never commit it). Every free-text field is
**redacted** of secrets and then **encrypted at rest** before anything is written; capture
happens off the wire at the session boundary, never mid-turn, and never changes a routing
decision. The captured files stay **local and git-ignored** — only behaviour-only,
prose-free fields are ever eligible to be shared into a committable dataset.
