---
title: Live smoke
description: Run one real, cheap session through Shunt against a real provider and verify the decision headers and the captured session row (expected spend ~$0.10).
---

# Live smoke

The [integration handshake](https://github.com/KookaS/shunt/blob/main/examples/integrations/README.md)
proves *wiring*: every tool
reaches Shunt, Shunt routes, and the decision header rides back — but its upstream
is a fake, so no real model is ever called. This live smoke is the complementary
proof: **one real, cheap session through the shipped proxy against a real
provider**, with the decision headers and the capture record verified against the
outcome store.

It is a smoke, not a benchmark: one tiny completion, no suite, and a hard spend
cap. It answers one question — *does a real client roundtrip through Shunt to a
real model, and does the session get captured?*

## Cost

The smoke pins the router to `always_cheap`, so it calls the cheapest model in the
live registry (today `deepseek-v4-flash`). One completion of a ~20-token prompt
with `max_tokens=16` costs a fraction of a cent.

- **Expected spend:** ~**$0.10** — this is the budget envelope, not the actual
  bill; a single tiny completion usually lands well under a cent.
- **Hard cap:** `--max-cost` (default `$0.10`). If the recorded session cost
  exceeds it, the smoke fails.

## Guard rails — this cannot run by accident

Real spend is gated in the script itself, not by a README's good intentions:

1. **The spend flag.** `--live` (alias `--confirm`) is required. Without it the
   script exits `2` immediately, before anything starts.
2. **An interactive terminal.** stdin must be a TTY. CI, cron and scripts never
   are, so the smoke cannot fire unattended.
3. **An explicit confirmation.** After printing the plan (model, provider, key,
   cap) it asks `Continue? [y/N]` and refuses on anything but `y`/`yes`.
4. **Keys come from the environment.** The runner injects provider keys; the
   script never reads a `.env` file. The key for the expected model's provider
   (`DEEPSEEK_API_KEY` by default) must already be set, or the smoke refuses.

All four gates run before the server starts — a refusal spends nothing.

## Pre-flight (run these first)

Live-run discipline is two guards that prove different things; run both in order:

1. **No-spend simulated smoke** — the hermetic handshake against a fake upstream:
   `make e2e TOOL=curl`. This proves the wiring: a request reaches Shunt and a
   decision rides back, with nothing billed.
2. **Budget-capped live** — this script, with the cap. It is the guard the
   simulated smoke cannot be: it proves the *real* provider path end to end.

## Run it

From the shunt repo root:

```bash
# Inject the real key for the expected model's provider (never written to a file).
export DEEPSEEK_API_KEY=sk-...

# Run supervised and tee the output to a run log.
uv run python benchmark/runner/live_smoke.py --live 2>&1 \
  | tee benchmark/runner/artifacts/live-smoke/owner-run.log
```

`make live-smoke` is equivalent (`ARGS` passes extra flags, e.g. `ARGS="--live"`).

Flags: `--max-cost` (default `0.10`), `--port` (default `8080`), `--config-dir`
(smoke your own `models.yaml` + `router.yaml` instead of the packaged config), and
`--run-dir` (where run artifacts land).

## What a pass means (the machine-checkable criteria)

The script exits `0` only when **all** of these hold:

| Criterion | Checked against |
|-----------|-----------------|
| The routed completion returns HTTP 200 with a non-empty answer | response body |
| `X-Shunt-Decision` names the cheapest live model with reason `always_cheap` | response header |
| `X-Shunt-Session-Id` is present | response header |
| A session row with that id exists in the outcome store; `model_chosen` matches, and `decision_provenance` records `model_chosen`, `selection_rule_used: always_cheap` and `router_propensity` | `sessions` table at `SHUNT_DATA_DIR/outcomes.db` |
| The captured prompt carries the run's `shunt-live-smoke` marker — the row is this run's | store row |
| Recorded cost is known (`cost_known=1`) and within `--max-cost` | store row |

Any violation is reported as `FAIL: …` with the exact mismatch and the script exits
`1`. A completion that returns but never lands a session row is the critical
failure this smoke exists to catch — it is *spend without yield*, one request at a
time. A provider that reports no usage cost (`cost_known=0`) is a `WARNING`, not a
pass: the spend cannot be reconciled.

## Kill triggers

Any of these ends the run immediately (the script does it itself — no polling
needed):

- the server does not serve `/health` within 60 seconds, or dies during boot;
- the expected model is not listed by `GET /v1/models` (config or registry drift);
- a non-200 from the routed completion (auth, balance, provider down) — spent, no
  yield;
- the recorded cost exceeds `--max-cost`.

While supervising, the script's own 120-second request timeout and the cost cap
are the backstops; a hung run can be killed with `pkill -f live_smoke` and the
server with its normal shutdown signal.

## Interpreting the outcome

The full record of a run lives in its run directory (printed at the end):
`benchmark/runner/artifacts/live-smoke/<run>/` — `live-smoke.log` (the smoke's own
report), `server.log` (the routed server's output) and `live-smoke.json` (the
structured verdict). **Keep that run log** — it is the evidence for the outcome.

Record the outcome (PASS/FAIL, the session id, the recorded cost and the run-log
path) wherever live-run outcomes are tracked in the project, alongside the
benchmark's live runs.

## When it fails

- **Refused by a gate (exit 2)** — one of the four guard rails did not open; the
  message says which.
- **HTTP 401/403** — the injected key is wrong or the provider rejects it; check
  the key and the `--config-dir` registry.
- **Model mismatch** — the cheapest model's key is missing, so the router fell
  back; inject the right key (the key gate normally catches this first).
- **No session row** — the request roundtripped but nothing was stored: the
  capture loop is broken. Check `server.log` for the store error.
