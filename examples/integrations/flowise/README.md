# Flowise → Shunt

[Flowise](https://flowiseai.com/) chatflows can route their LLM calls through
Shunt. Shunt picks the model; the chatflow never learns which one.

## The supported path: ChatOpenAI node

1. Drop a **ChatOpenAI** node into your chatflow.
2. Open its **Additional Parameters** and set **Base Path** =
   `http://127.0.0.1:8080/v1` (Flowise running on the host). If Flowise runs in
   Docker, use `http://shunt:8080/v1` (or your Shunt container's hostname) —
   `127.0.0.1` inside a container is the container, not the host.
3. Set the ChatOpenAI credential's **API Key** to any placeholder. Shunt holds the
   real provider keys and ignores this field, but Flowise requires it to be
   non-empty.
4. Set the model name to `auto` to let Shunt route.

Trigger the chatflow over the prediction API once it is saved:

```bash
curl http://127.0.0.1:3000/api/v1/prediction/<your-flow-id> \
  -H "Content-Type: application/json" \
  -d '{"question":"hi"}'
```

## Run the handshake (local or CI) — best-effort scaffold

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from client
```

This is a **best-effort scaffold**, not a cold-start pass. A Flowise chatflow must
be **seeded first** — the `client` leg POSTs to
`/api/v1/prediction/<flow-id>`, and with no imported flow there is nothing to
answer (replace `SEEDED_FLOW_ID` in `compose.yaml` with a real flow id once you
have one). The CI matrix therefore runs this leg `continue-on-error`; a red exit
reflects the missing seeded flow, not a Shunt failure. The supported integration
is the UI path above. See [`../README.md`](../README.md) for the shared harness.
