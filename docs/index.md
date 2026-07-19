---
title: Shunt
description: Pre-alpha cache-safe LLM proxy that today forwards to a cheap default; kNN outcome-based routing is designed and offline-validated but not yet live.
---

# Shunt

**Pre-alpha.** Shunt is a local, cache-safe proxy between your coding agent and
the model API. The goal is a router that sends routine work to a cheap model and
the hard tail to a frontier one, learning that line from your own passing tests.
**That routing is not live yet.** What runs today is the proxy: it speaks both
the OpenAI and Anthropic wire formats and forwards every request to a single
cheap default model.

```mermaid
graph LR
  A[Agent] -->|base_url| B[Shunt proxy]
  B -->|today: always| D[Cheap default model]
  B -.->|roadmap| C{kNN over verified outcomes}
  C -.-> D
  C -.-> E[Frontier model]
```

The solid path is what runs. The dashed path — per-task model selection over
verified outcomes — is designed and validated offline, but is **not** on the live
request path.

## An honest result

We tested the core idea (embed a task, find similar past tasks with known
outcomes, pick the cheapest model that succeeded) offline before shipping it.
On QA and reasoning-style workloads the embedding difficulty signal carries and
there is routing headroom. On the agentic-coding workload we actually target it
did **not** clear our viability bar: ranking hard tasks from easy ones off the
prompt embedding came out near chance. We publish that because it scopes the
project — it does not kill the cache-safe proxy or the verify-and-escalate path,
which does not depend on that signal, but it means we do not claim live
coding-task routing we cannot yet back with evidence.

## What runs today

- **A drop-in OpenAI/Anthropic-compatible proxy** — one env var and your agent
  talks to Shunt instead of the provider; Shunt translates between wire formats.
- **Cache-safe forwarding** — no mid-session model switch, so no silent full-price
  re-read of a cached conversation. With a fixed default there is nothing to
  switch; the future routing is being built to keep that guarantee.
- **A visible `X-Shunt-Decision` header** — names the model and the reason; today
  the reason is always the cold-start default.
- **Bring-your-own keys, zero telemetry** — nothing phoned home, replayed, or resold.

## Design center (what the roadmap is being built toward)

- **Cache-boundary-aware routing** — decisions at task/session boundaries only,
  never mid-cached-turn.
- **Pluggable, inspectable policy** — kNN over verified outcomes, no brittle rule
  tier; every decision surfaced in a header.
- **OpenAI ↔ Anthropic translation** — these two first, not 100+ providers.
- **Verifier + memory loop** — log `(task → model → verified outcome)` and learn
  from it; verification stays async/backfill, never on the hot path.
- **Secure by default** — localhost-bind, no exposed control plane, no key logging.

## Quickstart

The package is published; install it directly.

```bash
pip install shunt-router
shunt
```

Or with Docker:

```bash
docker run -p 8080:8080 ghcr.io/kookas/shunt-router
```

Point your tool at localhost:8080 (today, every request forwards to the cheap
default):

| Tool | Env var |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8080` |
| opencode | `OPENAI_BASE_URL=http://localhost:8080` |
| aider | `OPENAI_API_BASE=http://localhost:8080/v1` |
| n8n / LangChain | `baseURL: http://localhost:8080` |

## Contents

- [Architecture](architecture.md) — what runs live vs what is built but unwired
- [Configuration](configuration.md) — add provider keys and register models
- [Benchmark](benchmark.md) — run the offline model-capability and routing evals
- [Benchmark design](benchmark-design.md) — two-tree structure, strategy interface

## Status

Pre-alpha. The core hypothesis — cheap-first routing beats always-frontier at
equal quality on agentic coding — is unproven and, on the coding workload, the
embedding difficulty signal did not clear the bar. The kill gate (beat
fixed-frontier-with-caching at equal quality on a real workflow) has not been
run. If it does not hold, the router does not ship.

Apache-2.0. Import as `shunt` (`shunt-router` on PyPI — `shunt` is taken).
</content>
