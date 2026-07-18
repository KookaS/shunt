# LangChain → Shunt

LangChain chat models, pointed at Shunt. Shunt picks the model; LangChain never
learns which one.

## Point LangChain at Shunt

OpenAI wire:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="auto", base_url="http://127.0.0.1:8080/v1", api_key="ignored")
print(llm.invoke("hi").content)
```

Anthropic wire — note the base URL has **no `/v1` suffix**:

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="auto", base_url="http://127.0.0.1:8080", api_key="ignored")
print(llm.invoke("hi").content)
```

The API key is ignored — Shunt holds the real provider keys — but the client
requires a non-empty field, so pass any placeholder. Use `model="auto"` to let
Shunt route. LangChain does not expose response headers, so the `X-Shunt-Decision`
routing header is not readable here; use a raw SDK leg if you need to inspect it.

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from langchain
```

Exit code 0 means the roundtrip worked against the fake upstream — no real model
was called. See [`../README.md`](../README.md) for how the shared harness works.
