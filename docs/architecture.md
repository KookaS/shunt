---
title: Architecture
description: What runs on Shunt's live request path today (router is called, cold-starts to cheap default) versus what's waiting for the outcome-learning loop.
---

# Architecture

**Status: pre-alpha.** This page separates what is fully integrated in the live
request path from what is built but the learning loop is not yet wired to it —
because the router now calls `engine.decide()` on the first turn, but automatic
outcome capture is not yet wired (manual recording via `shunt flag` is available).

## What the live proxy does today

Shunt is a single process, localhost-bound. It accepts HTTP requests on two API
surfaces — OpenAI-compatible `/v1/chat/completions` and Anthropic `/v1/messages`
— translates between the wire formats, and forwards each request to a model chosen
by the router on the first turn. The router calls `engine.decide()` (embedding →
kNN over verified outcomes, or cold-start). Outcomes can be manually recorded via
`shunt flag <session_id> good|bad`, but automatic capture from test runs is not yet
wired, so the engine cold-starts every session to a cheap default (`qwen3.7-plus`)
held for the whole session. It also exposes a `/v1/models` stub so clients that
auto-discover model lists don't 404, and returns an `X-Shunt-Decision` header
naming the model and reason.

That is the whole live path: translate, route via engine (which cold-starts to
cheap default), stay cache-safe by never switching models mid-session. There is no
per-task model choice, no escalation, and no live outcome learning yet.

```mermaid
graph TD
  A[Tool: Claude Code / opencode / aider] -->|ANTHROPIC_BASE_URL / OPENAI_BASE_URL| P
  subgraph Shunt[Shunt process · localhost:8080]
    P[proxy/ — FastAPI + OpenAI SDK] -->|calls on 1st turn| R[router/ — kNN decision]
    R -->|cold-start (no outcomes yet)| M[cheap default model]
    V[verifiers/ — async backfill] -.->|built, NOT on the live loop| D[db/ — SQLite + HNSW]
    R -.-> D
  end
  Shunt --> E[Model API: Requesty, DeepSeek, etc.]
```

Solid = live (router is called). Dashed = present in the codebase and
unit-tested, but not yet on the live loop (outcome writing, learning feedback).

## Strategy and exploration

Which algorithm the router runs is one value, `router.strategy`, read from the
`router.yaml` packaged at `src/shunt/config/router.yaml`. Three strategies are
live-eligible: `knn` (the default), `always_cheap`, and `always_frontier`. The
benchmark-only strategies — `oracle`, `external_prior`, `random`, and
`knn_cascade` — are rejected at boot. `knn_cascade` is excluded on purpose: a real
quality cascade has to verify mid-session and escalate, and that is not one
cache-safe decision per session. Override the file by putting your own in
`$SHUNT_CONFIG_DIR`, or override single values with the `shunt start` flags — see
[configuration](configuration.md#tune-the-router).

The same file configures an exploration layer (Thompson sampling over the kNN
neighbourhood, bounded by a rolling exploration-cost budget), and it ships
enabled. It is mostly inert today. Exploration only fires once the router has
verified outcomes to be uncertain about. Manual outcome recording via `shunt flag`
can accumulate signal, but automatic capture from test runs is not yet wired, so
the router typically cold-starts every session and makes the cheap default decision.
The knobs are real; the behaviour they describe will engage once automatic
outcome capture lands.

## Modules

| Module | Role | On the live path? |
|---|---|---|
| **proxy/** | HTTP server: `/health`, `/v1/chat/completions`, `/v1/messages`, `/v1/models` (stub), streaming passthrough; calls router to decide model | **Yes** |
| **session/** | Session lifecycle: ID generation, inactivity timeout, model lock (keeps the session on one model — cache-safety) | **Yes** |
| **models/** | Provider config: model pool, capability tiers, fallback chain | **Yes** (read at startup) |
| **router/** | Decision core: embed prompt via fastembed, kNN retrieval via hnswlib, selection rule → cheapest capable model | **Yes** — called on the first turn to decide the session model (outcomes can be manually recorded, but automatic capture is not yet wired) |
| **verifiers/** | Async outcome verification: output mining, auto-detected tests | **No** — not yet wired into the live loop |
| **db/** | SQLite persistence for sessions, outcomes, HNSW index | Partial — sessions persist; the outcome/index learning loop is not live |

Every session's embedding is persisted, but only a session that carries a **recorded
outcome** joins the kNN index — a session with no outcome can never be a useful
neighbour, and indexing it anyway let ordinary traffic crowd the labelled sessions out
of the *k* nearest until selection quietly fell through to the cheapest model. A
session therefore becomes searchable when its outcome is recorded, not when it ends.

The router is now called on the first turn to decide the session model. It has been
validated **offline** on the SWE-bench Verified suite (see [benchmark.md](benchmark.md)).
Outcomes can be manually recorded via `shunt flag`, but automatic capture is not yet
wired, so the router typically cold-starts every session to the cheap default. Wiring
the automatic outcome-writing loop — gated on clearing the kill gate on a real workflow —
is the remaining learning integration step, and on the agentic-coding workload the
embedding-based difficulty signal has not yet cleared that bar.

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

Point your tool at Shunt (today every request forwards to the cheap default):

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
- **Apache-2.0**
</content>
