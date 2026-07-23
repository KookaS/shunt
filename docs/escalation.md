---
title: Error detection & auto-escalation
description: How Shunt detects a real, verified failure — never a model's self-report — and, when the same failure repeats, escalates one rung at the next session boundary. Off by default and opt-in.
---

# Error detection & auto-escalation

When the cheap model keeps failing the *same* verified check, Shunt can move up on
its own — raising the model's reasoning effort first, then its tier — without waiting
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

## The ladder — effort first, then tier

The default ladder is `effort_then_tier`, and it climbs **one rung per step, never
straight to frontier**:

1. **Raise reasoning effort first.** The router bumps the *current model* up one
   reasoning arm (e.g. `medium` → `high`). It is the **same model**, so the provider's
   prompt-cache namespace is unchanged — this rung is cache-safe. The higher arm's
   request params are applied to the outbound call, overriding any the client sent.
2. **Then step a tier.** Only once the model's reasoning arms are exhausted — or if the
   model declares no reasoning arms at all — does the router step to the next model
   tier (cheap → mid → high → frontier). The new model starts at its *own* default
   arm, not mid-ladder.
3. **Hold at the top.** At the ceiling (top tier, top arm) escalation holds rather than
   thrashing.

Set `ladder: tier_only` to skip the effort rung and step tiers directly. The effort
rung needs a model that declares [reasoning arms](configuration.md#reasoning-effort-optional);
a model without them has no effort headroom and steps tier immediately.

## Safety — the rails

- **Never mid-cached-turn.** An escalation applies only at the **next session
  boundary**. Shunt never switches a model in the middle of a cached conversation,
  which would force a full-price re-read of the context. Cache-safety is preserved by
  construction.
- **Routing-collapse guard.** If the recent model-choice distribution is degenerate —
  the expensive tier dominates, or choice-entropy collapses — a routing-collapse alarm
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
  to a tier step; there is no effort headroom to climb.
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
    ladder: effort_then_tier    # or tier_only
```

The knob reference, including the reserved `blocking_exit_code` field, is in
[Configuration](configuration.md#auto-escalate-on-repeated-verified-failure).
