# curl → Shunt

The ground-truth primitive: a raw HTTP request through Shunt, no SDK. Use it to
confirm Shunt is up and to read the `X-Shunt-Decision` header directly.

## Point curl at Shunt

```bash
# OpenAI wire — the -D - dump surfaces X-Shunt-Decision
curl -s -D - http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer ignored" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}' \
  | grep -i x-shunt-decision

# Anthropic wire — anthropic-version + max_tokens are REQUIRED
curl -s -D - http://127.0.0.1:8080/v1/messages \
  -H "x-api-key: ignored" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}' \
  | grep -i x-shunt-decision
```

The API key is ignored — Shunt holds the real provider keys — but most clients
still require a non-empty field, so pass any placeholder.

## Run the handshake (local or CI)

```bash
docker compose -f compose.yaml up --build --abort-on-container-exit --exit-code-from curl
```

Exit code 0 means the roundtrip worked against the fake upstream — no real model
was called. See [`../README.md`](../README.md) for how the shared harness works.
