# Integrations — using Shunt with your tools

Shunt is a drop-in proxy. Any tool that can point at a custom OpenAI- or
Anthropic-compatible base URL routes through it with **one line of config** — the
tool talks to Shunt exactly as it talked to the model API, and Shunt decides which
model answers. This directory has a copy-paste example per tool, plus a hermetic
handshake test that proves each one still works.

## The one thing you change

Shunt serves both wire formats on `http://127.0.0.1:8080`:

| Wire | Base URL to give the tool | Notes |
|------|---------------------------|-------|
| **OpenAI** (`/v1/chat/completions`) | `http://127.0.0.1:8080/v1` | keep the `/v1` |
| **Anthropic** (`/v1/messages`) | `http://127.0.0.1:8080` | **no** `/v1`; needs `anthropic-version` + `max_tokens` |

The API key is **ignored** — Shunt holds your real provider keys — but most clients
require a non-empty field, so pass any placeholder. Every response carries an
`X-Shunt-Decision` header naming the model and reason. There is also a
`GET /v1/models` stub so clients that auto-discover models don't 404.

## Tools

Each **CI-tested** tool has its own directory with a runnable handshake. **Docs-only**
tools are GUI/closed and can't run in headless CI — the directory documents the
config and you verify by hand.

### CI-tested (open, headless)

| Tool | Category | Wire | Notes |
|------|----------|------|-------|
| [curl](curl/) | raw HTTP | both | the ground-truth primitive |
| [openai-python](openai-python/) | raw SDK | openai | asserts `X-Shunt-Decision` |
| [anthropic-python](anthropic-python/) | raw SDK | anthropic | base URL without `/v1` |
| [claude-code](claude-code/) | CLI agent | anthropic | `ANTHROPIC_AUTH_TOKEN`, not the api-key var · best-effort |
| [opencode](opencode/) | CLI agent | both | OpenAI-compatible provider · best-effort |
| [aider](aider/) | CLI agent | openai | needs the `openai/` model prefix · best-effort |
| [continue](continue/) | editor CLI | openai | the `cn` CLI · best-effort |
| [langchain](langchain/) | framework | both | `ChatOpenAI` / `ChatAnthropic` |
| [litellm-sdk](litellm-sdk/) | framework | openai | mandatory `openai/` prefix |
| [pydantic-ai](pydantic-ai/) | framework | openai | `OpenAIProvider(base_url=…)` · best-effort |
| [litellm-proxy](litellm-proxy/) | gateway | openai | chaining: client → gateway → Shunt |
| [n8n](n8n/) | no-code | openai | LangChain node Base URL · best-effort |
| [flowise](flowise/) | no-code | openai | ChatOpenAI Base Path · best-effort |

**Best-effort** legs drive a real third-party CLI/service whose onboarding can
change; the CI matrix runs them `continue-on-error`, so a break never fails the
suite. Treat the copy-paste config in the tool's README as the source of truth.

### Docs-only (GUI/closed)

| Tool | Status |
|------|--------|
| [cline](cline/) | works fully (OpenAI-compatible provider); GUI-only |
| [zed](zed/) | works (`openai_compatible` in settings); GUI-only |
| [cursor](cursor/) | partial — base-URL override is Chat-only; agent is cloud-locked |
| [windsurf](windsurf/) | no base-URL override; not currently routable |

The [tool landscape](https://github.com/KookaS/shunt) is far wider — dozens more
CLI agents, editors, frameworks, gateways, and chat UIs integrate the same way.
The tools above are the representative, CI-verified set.

## How the handshake works

The test is a **dry run**: no real model is ever called, no key is needed, nothing
is billed. A per-tier fake upstream stands in for the providers, so a green
handshake proves the *wiring* — the tool reaches Shunt, Shunt routes and returns,
and the decision header rides back — not model quality.

Two layers:

1. **Always-on, hermetic** ([`tests/integrations/test_handshake.py`](../../tests/integrations/test_handshake.py)) —
   runs in the normal test job on every push, no Docker. Drives Shunt in-process
   against a live fake upstream across both wires.
2. **Opt-in Docker matrix** (`.github/workflows/integration-handshake.yml`) — one
   leg per tool directory that ships a `handshake.yaml`. Each leg starts the shared
   substrate ([`compose.base.yaml`](compose.base.yaml): Shunt + fake upstream) and
   runs the real tool against it.

### Run a handshake locally

```bash
# Build the Shunt image once (the compose base layers the fake registry onto it):
docker build -t shunt-router:handshake -f ../../Dockerfile ../..

# Then run any tool's leg — exit 0 means the roundtrip worked:
docker compose -f openai-python/compose.yaml up --build \
  --abort-on-container-exit --exit-code-from openai-python
```

### Add a tool

Create `examples/integrations/<tool>/` with:

- **`README.md`** — the copy-paste config (always).
- **`compose.yaml`** — `include: [../compose.base.yaml]` plus one service that drives
  Shunt at `http://shunt:8080`, exiting 0 on a routed completion.
- **`handshake.yaml`** — `tool`, `wire`, `service` (the verdict service), and
  `expected_tier` (a tier in [`fake_registry.yaml`](../../tests/integrations/fake_registry.yaml)).

Shipping a `handshake.yaml` is the **marker** that makes the tool CI-eligible — the
matrix globs `examples/integrations/*/handshake.yaml`, no central list to edit.
`tests/test_integrations_sync.py` checks every directory is well-formed. A docs-only
tool ships just a `README.md`.
