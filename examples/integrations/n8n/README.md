# n8n → Shunt

[n8n](https://n8n.io/) workflows can send their LLM calls through Shunt. Shunt
picks the model; the workflow never learns which one.

## The supported path: the OpenAI Chat Model node

Use the **OpenAI Chat Model** node — the one under **AI → Language Models**
(a LangChain sub-node). Do **not** use the plain OpenAI app-node: it has no Base
URL field, so it cannot be pointed at Shunt. This is the common trap.

1. Add an **OpenAI Chat Model** node to your workflow.
2. Create (or edit) its **OpenAI** credential and set:
   - **Base URL** = `http://127.0.0.1:8080/v1` (self-hosted n8n reaching Shunt on
     the host). If n8n itself runs in Docker, use `http://shunt:8080/v1` (or your
     Shunt container's hostname) instead — `127.0.0.1` inside a container is the
     container, not the host.
   - **API Key** = any placeholder. Shunt holds the real provider keys and ignores
     this field, but n8n requires it to be non-empty.
3. Set the node's **Model** to `auto` to let Shunt route.

> **Cloud n8n cannot reach `localhost`.** n8n Cloud runs off your machine, so it
> cannot see a Shunt bound to `127.0.0.1`. Use self-hosted n8n, or expose Shunt on
> a network address both can reach.

## Run the handshake (local or CI) — best-effort

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from n8n
```

This runs a workflow embedded in the compose file (a Manual Trigger → OpenAI Chat
Model node pointed at `http://shunt:8080/v1`) headless via `n8n execute`. It is a
**best-effort** leg: wiring the LangChain node's credential without the UI is
brittle, so the CI matrix runs it `continue-on-error` — a red exit here reflects
headless credential plumbing, not a Shunt failure. The supported integration is
the UI path above. See [`../README.md`](../README.md) for the shared harness.
