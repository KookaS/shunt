# opencode → Shunt

[opencode](https://www.npmjs.com/package/opencode-ai), the open-source coding
agent, pointed at Shunt through an OpenAI-compatible provider. Shunt picks the
model; opencode never learns which one.

## Point opencode at Shunt

opencode registers custom providers in `opencode.json`. Add Shunt as an
OpenAI-compatible provider on the OpenAI wire:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "shunt": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Shunt",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "ignored"
      },
      "models": {
        "auto": { "name": "auto (Shunt-routed)" }
      }
    }
  }
}
```

Then select the provider/model when you run:

```bash
opencode run --model shunt/auto "say hi"
```

The OpenAI wire base URL keeps its **`/v1` suffix**. The API key is ignored —
Shunt holds the real provider keys — but must be non-empty. `model="auto"` lets
Shunt route.

## Run the handshake (CI, best-effort)

The compose file embeds the opencode config (the same JSON as above, with
`baseURL` set to `http://shunt:8080/v1` — the compose service name) and runs
opencode against it:

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from opencode
```

Exit code 0 means opencode reached Shunt and got the routed completion against the
fake upstream — no real model was called.

**Best-effort.** This is a real third-party CLI: a future release may add an
onboarding or config step that a headless container can't satisfy. The CI matrix
runs this leg `continue-on-error`, so a break here never fails the suite. If it
goes red, treat the copy-paste config above as the source of truth and verify
manually. See [`../README.md`](../README.md) for how the shared harness works.
