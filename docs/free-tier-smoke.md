---
title: Free-tier smoke
description: The weekly, zero-cost CI check that routes one real completion through Shunt against an OpenRouter :free model and verifies the decision headers and captured session row.
---

# Free-tier smoke

The [live smoke](live-smoke-runbook.md) proves a real session against a real
provider, but it needs an interactive owner and spends a little money. The
**free-tier smoke** is the CI complement: a weekly, fully unattended run that
proves the same thing for **zero dollars** — one real completion through the
real proxy against a genuinely-$0 model, with the same machine-checkable
verification as the owner smoke.

It is a smoke, not a benchmark: one tiny completion, one model, no suite.

## What it proves

The [integration handshake](examples/integrations/README.md) proves *wiring*
against a fake upstream. The free-tier smoke proves the **real wire end to end
at zero cost**: a request reaches Shunt, Shunt routes it through a real provider,
the decision header rides back, and the session is captured in the outcome
store. Same pass criteria as the [live smoke](live-smoke-runbook.md#what-a-pass-means-the-machine-checkable-criteria),
minus the spend: HTTP 200 with a real answer, `X-Shunt-Decision` naming the
expected model with reason `always_cheap`, a session row with matching
`model_chosen`, a `decision_provenance` recording `model_chosen`,
`selection_rule_used` and `router_propensity`, and `cost_known=1` with a
recorded cost of **$0**.

## The free model, and why it is a separate config

The smoke routes through a **dedicated config directory**,
[`configs/free-tier/`](https://github.com/KookaS/shunt/blob/main/configs/free-tier/models.yaml),
selected with `SHUNT_CONFIG_DIR`. It contains a single model — currently
`openai/gpt-oss-20b:free` (verified `$0` in the public OpenRouter catalog on
2026-08-11) — priced at 0/0 per 1M tokens, plus a router policy that mirrors the
shipped one with `models:` listing only that model.

This separation is deliberate. A `$0` model in the **shipped** registry
(`src/shunt/config/models.yaml`) would rank cheapest-first and silently change
production routing, because Shunt ranks models by total list price ascending.
The free-tier config is never loaded by a normal Shunt boot — only the smoke
points `SHUNT_CONFIG_DIR` at it — so the smoke gets its `$0` model and
production routing is untouched.

## Guard rails

A genuinely-free run is enforced in the script itself, not by a schedule's good
intentions:

1. **Only a `:free` model can run.** Before anything is sent, the smoke asserts
   the served model's `model_id` ends in `:free`, its provider is OpenRouter,
   and its registry prices are 0/0. Any of those failing aborts with exit 2 — a
   model that could bill refuses to run.
2. **Non-interactive runs need an explicit opt-in.** `SHUNT_FREE_TIER_CI=1` (set
   by the CI job) is required for non-TTY operation, replacing the owner smoke's
   interactive `y/N` — the auto-approval is only ever earned by the `:free`
   assertion above.
3. **A missing key skips, never fails.** Without `OPENROUTER_API_KEY` the smoke
   exits 0 with a visible `SKIP` and writes nothing to the wire — nothing is
   sent, nothing is billed. In CI, a repo without that secret simply doesn't run
   the check yet.
4. **A hard $0 cost cap.** A recorded session cost above `$0` fails the smoke —
   a `:free` model is expected to bill nothing.

## When it runs

- **Weekly** (Monday 04:47 UTC) via `schedule`.
- **On demand** via `workflow_dispatch` (the workflow's "Run workflow" button).

It is deliberately **not** triggered by push or pull_request, and it is
**advisory** (`continue-on-error`): the run depends on a third-party API being
up, so a provider outage should surface as news in this job's artifact, not as a
red check on unrelated PRs. There is no `push:`/`pull_request:` trigger, so a
green PR never depends on OpenRouter being reachable.

## How to read the verdict

Each run uploads a verdict artifact named `free-tier-smoke` containing the run
directory. The structured verdict is `free-tier-smoke.json`:

| Field | Meaning |
|-------|---------|
| `status` | `PASS` · `FAIL` · `SKIP` |
| `expected_model` / `served_model` | The registry model that should have been routed, and the one the decision header named |
| `decision_reason` | The `X-Shunt-Decision` reason (`always_cheap`) |
| `session_id` | The `X-Shunt-Session-Id`; cross-check it in the outcome store |
| `recorded_cost_usd` | Must be `0` for a pass |
| `problems` / `warnings` | Every unmet pass criterion, and notes (e.g. `cost_known=0`) |

The run log (`free-tier-smoke.log`) and the routed server's log (`server.log`)
sit beside it. A `SKIP` verdict with a missing-key reason is not a failure — it
means the secret wasn't configured for this repo.

## Run it yourself

```bash
# Needs a real OpenRouter key in the environment (never a .env read).
export OPENROUTER_API_KEY=sk-or-...

# CI mode: non-interactive, auto-approved only for the $0 :free model.
SHUNT_FREE_TIER_CI=1 uv run python -m benchmark.runner.free_tier_smoke
```

Without the key the same command prints `SKIP` and exits 0 — nothing is sent.
