# continue → Shunt

The [Continue CLI](https://www.npmjs.com/package/@continuedev/cli) (`cn`), pointed
at Shunt through an OpenAI-compatible model. Shunt picks the model; Continue never
learns which one.

## Point Continue at Shunt

Continue reads models from a `config.yaml`. Register Shunt as an `openai`
provider on the OpenAI wire:

```yaml
name: shunt
version: 0.0.1
schema: v1
models:
  - name: auto
    provider: openai
    model: auto
    apiBase: http://127.0.0.1:8080/v1
    apiKey: dummy
    roles:
      - chat
```

Then run a prompt against it:

```bash
cn -p "say hi" --config ./config.yaml
```

The OpenAI wire base URL keeps its **`/v1` suffix**. The API key is ignored —
Shunt holds the real provider keys — but must be non-empty. `model: auto` lets
Shunt route.

## Run the handshake (CI, best-effort)

The compose file embeds the Continue config (the same YAML as above, with
`apiBase` set to `http://shunt:8080/v1` — the compose service name) and runs `cn`
against it:

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from continue
```

Exit code 0 means Continue reached Shunt and got the routed completion against the
fake upstream — no real model was called.

**Best-effort.** This is a real third-party CLI: a future release may add an
onboarding or config step that a headless container can't satisfy. The CI matrix
runs this leg `continue-on-error`, so a break here never fails the suite. If it
goes red, treat the copy-paste config above as the source of truth and verify
manually. See [`../README.md`](../README.md) for how the shared harness works.
