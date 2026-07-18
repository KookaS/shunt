# openai-python → Shunt

The official OpenAI Python SDK, pointed at Shunt on the OpenAI wire. Shunt picks
the model; the SDK never learns which one.

## Point the SDK at Shunt

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="ignored")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.choices[0].message.content)
```

The API key is ignored — Shunt holds the real provider keys — but the SDK
requires a non-empty field, so pass any placeholder. Use `model="auto"` to let
Shunt route; the routing choice comes back in the `X-Shunt-Decision` response
header, readable via `with_raw_response`.

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from openai-python
```

Exit code 0 means the roundtrip worked against the fake upstream — no real model
was called. See [`../README.md`](../README.md) for how the shared harness works.
