# Pydantic AI → Shunt

A Pydantic AI agent, pointed at Shunt on the OpenAI wire. Shunt picks the model;
the agent never learns which one.

## Point Pydantic AI at Shunt

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    "auto",
    provider=OpenAIProvider(base_url="http://127.0.0.1:8080/v1", api_key="ignored"),
)
print(Agent(model).run_sync("say hi").output)
```

The API key is ignored — Shunt holds the real provider keys — but the provider
requires a non-empty field, so pass any placeholder. The `"auto"` model name lets
Shunt route.

Pydantic AI's model and provider API has shifted across releases, so treat this
wiring as best-effort — the import paths or constructor names may need adjusting
on an upgrade.

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from pydantic-ai
```

Exit code 0 means the roundtrip worked against the fake upstream — no real model
was called. See [`../README.md`](../README.md) for how the shared harness works.
