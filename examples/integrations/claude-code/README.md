# claude-code → Shunt

[Claude Code](https://www.npmjs.com/package/@anthropic-ai/claude-code), the
Anthropic coding CLI, pointed at Shunt on the Anthropic wire. Shunt picks the
model; Claude Code never learns which one.

## Point Claude Code at Shunt

Claude Code reads its endpoint from the environment:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080      # NO /v1 on the Anthropic wire
export ANTHROPIC_AUTH_TOKEN=shunt-local              # any non-empty placeholder
claude -p "say hi" --output-format text
```

Two things matter:

- The Anthropic wire base URL has **no `/v1` suffix** — Claude Code appends the
  path itself.
- Use **`ANTHROPIC_AUTH_TOKEN`**, not `ANTHROPIC_API_KEY`. Setting the api-key
  variable makes the CLI send an `x-api-key` header, which Shunt rejects with a
  401. `ANTHROPIC_AUTH_TOKEN` sends a bearer token instead. The value is ignored
  — Shunt holds the real provider keys — but must be non-empty.

## Run the handshake (CI, best-effort)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from claude-code
```

Exit code 0 means Claude Code reached Shunt and got the routed completion against
the fake upstream — no real model was called.

**Best-effort.** This is a real third-party CLI: a future release may add an
onboarding or config step that a headless container can't satisfy. The CI matrix
runs this leg `continue-on-error`, so a break here never fails the suite. If it
goes red, treat the copy-paste config above as the source of truth and verify
manually. See [`../README.md`](../README.md) for how the shared harness works.
