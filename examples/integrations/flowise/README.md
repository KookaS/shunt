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

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from client
```

The handshake **self-seeds** — the `client` leg fetches Flowise's own shipped
"Conversation Chain" template, points its ChatOpenAI node at `http://shunt:8080/v1`,
creates the chatflow via the API, then predicts and asserts the routed answer comes
back. It pins `flowiseai/flowise:2.2.8` (the last v2; v3 forces interactive account
registration, hostile to hermetic CI) and reaches the management API with the
`x-request-from: internal` header (no credentials set). The CI matrix still runs this
leg `continue-on-error` (`best_effort: true`) because it drives a heavy pinned server,
but a green exit is a genuine end-to-end proof. See [`../README.md`](../README.md) for
the shared harness.
