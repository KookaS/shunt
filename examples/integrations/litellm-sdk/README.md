# LiteLLM SDK → Shunt

The LiteLLM Python SDK, pointed at Shunt on the OpenAI wire. Shunt picks the
model; LiteLLM never learns which one.

## Point LiteLLM at Shunt

```python
import litellm

r = litellm.completion(
    model="openai/auto",
    api_base="http://127.0.0.1:8080/v1",
    api_key="ignored",
    messages=[{"role": "user", "content": "hi"}],
)
print(r.choices[0].message.content)
```

The `openai/` prefix on the model is **mandatory** — it tells LiteLLM to speak
the OpenAI wire to Shunt rather than guessing a provider from the bare name. The
API key is ignored (Shunt holds the real provider keys) but the SDK requires a
non-empty field, so pass any placeholder. `auto` lets Shunt route.

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from litellm-sdk
```

Exit code 0 means the roundtrip worked against the fake upstream — no real model
was called. See [`../README.md`](../README.md) for how the shared harness works.
