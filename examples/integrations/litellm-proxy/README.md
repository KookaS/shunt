# LiteLLM Proxy → Shunt

A [LiteLLM](https://docs.litellm.ai/) gateway placed **in front of** Shunt on the
OpenAI wire. This is a chain: your client talks to LiteLLM, LiteLLM forwards to
Shunt, and Shunt picks the model. Neither the client nor LiteLLM learns which one.

```
client ──▶ LiteLLM gateway ──▶ Shunt ──▶ provider
```

Chaining a gateway in front of Shunt is useful when you already run LiteLLM for
key management, budgets, or logging and want Shunt to make the routing decision
behind it. The direction also reverses cleanly: a LiteLLM gateway can sit
**behind** Shunt as one of the upstreams Shunt routes to. Shunt speaks the OpenAI
wire on both sides, so it drops into either slot.

## Point LiteLLM at Shunt

Give the gateway one model entry that forwards to Shunt. The `openai/` prefix
tells LiteLLM to speak the OpenAI wire; `api_base` is Shunt's `/v1` endpoint; the
key is ignored because Shunt holds the real provider keys.

```yaml
# litellm_config.yaml
model_list:
  - model_name: auto
    litellm_params:
      model: openai/auto
      api_base: http://shunt:8080/v1
      api_key: ignored
```

Run the gateway with `litellm --config litellm_config.yaml --port 4000`, then
point any OpenAI-compatible client at `http://<litellm-host>:4000/v1` with
`model: auto`.

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from client
```

Exit code 0 means the full chain worked against the fake upstream — the `client`
leg got `"content":"ok"` back through the LiteLLM gateway, and no real model was
called. See [`../README.md`](../README.md) for how the shared harness works.
