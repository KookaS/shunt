# Aider → Shunt

The Aider coding agent, pointed at Shunt on the OpenAI wire. Shunt picks the
model; Aider never learns which one.

## Point Aider at Shunt

```bash
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=dummy
aider --model openai/auto
```

The `openai/` prefix on the model is **mandatory** — it tells Aider to speak the
OpenAI wire to Shunt. `OPENAI_API_KEY` must be **non-empty** (Aider refuses to
start without it), but Shunt ignores it since it holds the real provider keys —
so any placeholder works. `auto` lets Shunt route.

Aider is a full agent with many moving parts (git, repo map, update and analytics
checks). Treat this leg as best-effort — it is the most fragile in CI; the
handshake disables git, auto-commits, update checks, and analytics to keep the
roundtrip minimal.

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from aider
```

Exit code 0 means the roundtrip worked against the fake upstream — no real model
was called. See [`../README.md`](../README.md) for how the shared harness works.
