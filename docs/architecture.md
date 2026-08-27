---
title: Architecture
description: What runs on Shunt's live request path today (router is called, cold-starts to cheap default) versus what's waiting for the outcome-learning loop.
---

# Architecture

**Status: pre-alpha.** This page describes the live request path — the router calls
`engine.decide()` on the first turn and chooses a model. Outcomes are recorded
automatically at session close via off-wire test execution (in a resolved `work_dir` —
by default Shunt's own launch repo), or manually via `shunt flag`. The learning loop is
integrated.

## What the live proxy does today

Shunt is a single process, localhost-bound. It accepts HTTP requests on two API
surfaces — OpenAI-compatible `/v1/chat/completions` and Anthropic `/v1/messages`
— translates between the wire formats, and forwards each request to a model chosen
by the router on the first turn. The router calls `engine.decide()` (embedding →
kNN over verified outcomes, or cold-start to a cheap default). At session close
(inactivity timeout), outcomes are recorded automatically by re-running the repo's
tests off the wire — the repo resolved from `capture.work_dirs`, `SHUNT_WORK_DIR` /
`--work-dir` / `capture.work_dir`, or the validated launch directory — or manually via
`shunt flag <session_id> good|bad`. The engine then learns from
verified outcomes, updating the `ConservativeGate` and exploration budget for
future decisions. That exploration state (the budget's cost cap and the gate's
banked slack) is persisted to the SQLite store, so a restart resumes it rather
than resetting the cap and slack to zero. It also exposes a `/v1/models` stub so clients that
auto-discover model lists don't 404, and returns an `X-Shunt-Decision` header
naming the model and reason.

That is the live path: translate, route via engine (deciding on embedded prompt
via kNN query of verified outcomes, with fallback to cheap default on cold-start),
forward to chosen model, stay cache-safe by never switching models mid-session,
and learn from verified session outcomes at close. There is no per-task model
choice or mid-session escalation.

```mermaid
graph TD
  A[Tool: Claude Code / opencode / aider] -->|ANTHROPIC_BASE_URL / OPENAI_BASE_URL| P
  subgraph Shunt[Shunt process · localhost:8080]
    P[proxy/ — FastAPI + OpenAI SDK] -->|calls on 1st turn| R[router/ — kNN decision]
    R -->|"cold-start (no outcomes yet)"| M[cheap default model]
    C[capture/ — off-wire verifier] -->|at session close| V[verifiers/ — auto-detect tests]
    V -->|append verified outcome| D[db/ — SQLite + HNSW]
    R -->|cold-start search| D
    D -->|update gate| R
  end
  Shunt --> E[Model API: Requesty, DeepSeek, etc.]
```

Solid = live path. The router chooses a model on the first turn (via kNN query of
the outcome database, or cold-start to cheap default). At session close, verified
outcomes are recorded automatically (via `capture/` + `verifiers/`), and the router
learns from them for subsequent sessions.

## Strategy and exploration

Which algorithm the router runs is one value, `router.strategy`, read from the
`router.yaml` packaged at `src/shunt/config/router.yaml`. Four strategies are
live-eligible: `session_cascade` (the default), `knn_semantic_cascade`, `always_cheap`, and
`always_frontier`. That list is `LIVE_STRATEGIES` in
`src/shunt/router/policy.py`, and it is the whole of it — every other strategy the
benchmark scores (`oracle`, `oracle_reward`, `random`,
`knn_semantic_cascade_withintask`, `price_cascade`, `knn_semantic_tier`,
`knn_difficulty`, `knn_difficulty_cascade`, `difficulty_band_cascade`) is rejected at boot.
The reasons differ: `oracle` and `oracle_reward` read the task's own verified
outcome and `random` is not a router at all; the two **within-task** cascades are
excluded on purpose and permanently, because a real quality cascade has to verify
**mid-session** and escalate, and that is not one cache-safe decision per session;
and `knn_semantic_tier` orders models by a capability rank fitted offline
from the outcome matrix, which the live path cannot compute. Each blocker and its
path to live is recorded in `benchmark/routing/strategy_class.py`.

The benchmark's `knn_semantic` row — the kNN selection rule with the ladder removed — is a
separate case, and it is **not** rejected at boot, because a `router.yaml` never
resolves to it. That row is a benchmark **control**, kept as the contrast that
isolates what the ladder buys; the strings `knn` and `knn_cascade` written in a config are
pre-rename spellings of `knn_semantic_cascade` and are migrated to it with a boot warning
(see below), so no install can select the ladder-less rule.

The two `*_cascade` ids are **presets**, not new selection rules: a base pick plus
[auto-escalation](escalation.md), so the router starts where the base pick lands
and climbs a rung at the next session boundary on a repeated verified failure.
`session_cascade` is the default and its base pick is `always_cheap`, so it always
starts at the cheapest model; `knn_semantic_cascade` swaps that for the kNN neighbourhood
rule. Selecting either with escalation disabled is a load error. `session_cascade`
is neighbour-independent, so under the shipped default the router never embeds,
exactly like the two fixed strategies — per-task routing is what `knn_semantic_cascade`
opts into.

`knn_semantic_cascade` was spelled `knn`, then `knn_cascade`, before the second rename.
That was never an accurate name: the kNN pick has participated in escalation since
escalation shipped on by default, so a default install has always run the ladder. Existing
configs are migrated automatically with a boot warning, and the aliases are kept for at
least one more minor release.

`session_cascade` is nonetheless a *separate* strategy from `always_cheap` rather than a
flag on it, because the two differ on one predicate — `participates_in_escalation`. `always_cheap` and
`always_frontier` are **pinned controls**: a verified failure may never move their pick,
since they are the baselines routing comparisons are read against. The cascade is the
opposite, and the engine branches on that predicate rather than on `consults_neighbors`,
which is False for all three and cannot tell them apart. Override the file by putting your own in
`$SHUNT_CONFIG_DIR`, or override single values with the `shunt start` flags — see
[configuration](configuration.md#tune-the-router).

The same file configures an exploration layer (Thompson sampling over the kNN
neighbourhood, bounded by a rolling exploration-cost budget), and it ships
enabled. Exploration fires once the router has verified outcomes to be uncertain
about. Verified outcomes accumulate automatically at session close (via off-wire
test execution in the resolved `work_dir`), or manually via `shunt flag`.
The knobs are live; exploration behaviour adapts as verified outcomes accumulate.

## Modules

| Module | Role | On the live path? |
|---|---|---|
| **proxy/** | HTTP server: `/health`, `/v1/chat/completions`, `/v1/messages`, `/v1/models` (stub), `/admin/loop-health` (read-only loop-health metrics, aggregates only — no prompts; **unauthenticated, like every route** — see [SECURITY.md](https://github.com/KookaS/shunt/blob/main/SECURITY.md)), streaming passthrough; calls router to decide model on first turn | **Yes** |
| **session/** | Session lifecycle: ID generation (the tool's conversation id when it sends one, else a source-IP + user-agent hash), inactivity timeout, model lock (keeps the session on one model — cache-safety) | **Yes** |
| **models/** | Provider config: model pool, price-derived capability rank, fallback chain | **Yes** (read at startup) |
| **router/** | Decision core: embed prompt via fastembed, kNN retrieval via hnswlib, selection rule → model chosen via outcome feedback or cold-start | **Yes** — called on first turn; learns from verified outcomes |
| **capture/** | Off-wire outcome capture: session-close triggers, work-dir resolver, coordinator, background worker | **Yes** — wired at session-close to run verifiers async |
| **verifiers/** | Async outcome verification: auto-detect and run the repo's test runner (pytest / jest / go test / cargo test / Maven / dotnet test / RSpec / PHPUnit / GTest / …) per project | **Yes** — called at session close by capture worker |
| **db/** | SQLite persistence for sessions, outcomes, HNSW index (append-only events + materialized view) | **Yes** — sessions persist on each turn; learning loop is live |

Every session's embedding is persisted, but only a session that carries a **recorded
outcome** joins the kNN index — a session with no outcome can never be a useful
neighbour, and indexing it anyway let ordinary traffic crowd the labelled sessions out
of the *k* nearest until selection quietly fell through to the cheapest model. A
session therefore becomes searchable when its outcome is recorded, not when it ends.

The router is called on the first turn to decide the session model, validated
**offline** on the SWE-bench Verified suite (see [benchmark.md](benchmark.md)). The
learning loop — automatic outcome capture at session close — is now wired. Outcomes
accumulate via off-wire test re-execution in the resolved `work_dir`, and
the router adapts over time. Cold-start sessions default to the cheap model until
verified outcomes build a neighbourhood for kNN to search.

## Repository layout

```
├── src/shunt/             Router package
│   ├── cli.py             CLI entry point (shunt start, doctor, explain, escalate, flag, reindex, inspect, version)
│   ├── proxy/             HTTP server: /health, /v1/chat/completions, /v1/messages, /v1/models
│   │                      (calls router to decide model; cold-starts to cheap default)
│   ├── router/            Decision core — embed → nearest-neighbour → selection rule
│   │                      (called on the first turn; learns from verified outcomes)
│   ├── capture/           Off-wire outcome capture at session close (work_dir resolver, coordinator, background worker)
│   ├── verifiers/         Async outcome verification (auto-detected tests, typecheck runner)
│   ├── db/                SQLite persistence for sessions, outcomes, index
│   ├── session/           Session lifecycle, inactivity timeout, model lock
│   ├── models/            Provider config, price-derived capability rank, fallback chain
│   ├── inspect/           Figure frame, layout contract and diagnostics over the live outcome store (`shunt inspect`, [inspect] extra)
│   │   └── inference/     Seven-figure inference family over the live store, driven by a figures.json manifest (`python -m shunt.inspect.inference`)
│   ├── analysis/          Off-policy evaluation (ope.py) and instrument admissibility (admissibility.py) over logged decisions
│   │                      (shipped rather than benchmark-side: src/shunt/ may not import benchmark/ — SH006 — and the rig image carries no benchmark/ tree)
│   └── config/            Shipped defaults: models.yaml registry, router.yaml policy
├── benchmark/             Offline model-capability and routing evaluation
├── docs/                  User documentation (MkDocs)
├── examples/providers/    Copy-paste registry config, one file per provider
├── examples/integrations/ Tool integration examples (CLI agents, frameworks, gateways)
└── tests/                 Test suite
```

## Capabilities

What the platform is built to support today.

- **Drop-in for any agent.** Speaks both the OpenAI and Anthropic wire formats
  and translates between them, so Claude Code, opencode, aider, Continue, Cline,
  Cursor, and Zed all connect with one line — plus agent frameworks (LangChain,
  Pydantic AI, LiteLLM) and no-code builders (n8n, Flowise).
- **A configurable model pool.** A provider registry ranked by price (cheapest →
  priciest), per-model enable/disable, and a fallback chain. You own the pool and
  the prices. See [configuration](configuration.md).
- **A decision core.** Task embedding → nearest-neighbour lookup → a
  cheapest-that-succeeds selection rule, plus pluggable strategies (fixed, kNN,
  cascade, tier-classifier, oracle).
- **Outcome verification.** Async, auto-detected test and typecheck verifiers
  grade a result at session close without blocking the response. Verified
  outcomes feed the next decision via the kNN index and exploration priors. See
  [feedback](feedback.md).
- **Cache-safety as a design center.** Decisions land at task and session
  boundaries, never mid-cached-turn, so normal operation never silently re-reads
  a cached conversation at full price. The one exception is an upstream failure:
  falling back to another model means that model must prefill the whole
  conversation, because a provider's cache is per-model and cannot be
  transferred. Shunt's job is to make that rare and deliberate, not to pretend it
  is free.
- **An offline benchmark.** Scores any routing strategy against a cache of
  verified outcomes — reward (quality minus cost), bootstrap confidence
  intervals, and a Pareto check against a perfect-oracle baseline. See
  [benchmark](benchmark.md) and [results](results.md).
- **Bring-your-own keys, zero telemetry.** Your provider accounts, your keys,
  localhost-bound by default. Nothing is phoned home, replayed, or resold.

## Running

The package is published; install it directly.

```bash
pip install shunt-router
shunt
```

Or with uv: `uv run shunt`. Or with Docker:

```bash
docker run -p 127.0.0.1:8080:8080 --env-file .env ghcr.io/kookas/shunt-router
```

Config: `SHUNT_PORT`, `SHUNT_HOST`. Provider keys are read from environment
variables (e.g. `DEEPSEEK_API_KEY`, `REQUESTY_API_KEY`) by the OpenAI SDK client;
each model's `base_url` and `api_key_env_var` come from the model config.

## Integration

Point your tool at Shunt (the router picks the session model on the first turn,
cold-starting to the cheap default until verified outcomes accumulate):

| Tool | Config |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8080` |
| opencode | `OPENAI_BASE_URL=http://localhost:8080/v1` |
| aider | `OPENAI_API_BASE=http://localhost:8080/v1` |
| n8n / LangChain | `baseURL: http://localhost:8080/v1` |

## Properties

- **Cache-safe**: forwards at session granularity, never switches model mid-turn
- **No telemetry**: any learning stays local to your SQLite store
- **Secure**: localhost-bind by default, no key logging
- **Runs on any laptop**: embeddings come from fastembed and the index is hnswlib, both
  CPU-only. A router that needed a big machine to save you money would defeat its own
  purpose. The `Dockerfile` builds hnswlib with `HNSWLIB_NO_NATIVE=1`, then runs `objdump`
  over the compiled extension and fails the build if an AVX-512 opcode was baked in — so a
  wheel that would crash on an older CPU never ships.
- **Apache-2.0**
