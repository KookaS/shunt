# anthropic-python → Shunt

The official Anthropic Python SDK, pointed at Shunt on the Anthropic wire. Shunt
picks the model; the SDK never learns which one.

## Point the SDK at Shunt

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080", api_key="ignored")
resp = client.messages.create(
    model="auto",
    max_tokens=64,
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.content[0].text)
```

The Anthropic wire base URL has **no `/v1` suffix** — the SDK appends the path
itself. The API key is ignored (Shunt holds the real provider keys) but the SDK
requires a non-empty field, so pass any placeholder. `max_tokens` is required.
Use `model="auto"` to let Shunt route; the routing choice comes back in the
`x-shunt-decision` response header, readable via `with_raw_response`.

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from anthropic-python
```

Exit code 0 means the roundtrip worked against the fake upstream — no real model
was called. See [`../README.md`](../README.md) for how the shared harness works.
