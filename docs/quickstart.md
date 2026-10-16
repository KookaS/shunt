---
title: Quickstart
description: Install Shunt from source or Docker, check the install with shunt doctor, start the proxy, and point your coding tool at it.
---

# Quickstart

The package is not yet published on PyPI; install from source:

```bash
git clone https://github.com/KookaS/shunt.git
cd shunt
cp .env.example .env          # then add your provider keys
pip install -e .              # or: uv sync — then `uv run shunt`
```

Or with Docker — the image is not published yet, so build it from the checkout.
`.env` carries your provider keys (copy `.env.example`), and the port is bound to
loopback because Shunt holds those keys and does not authenticate its own clients:

```bash
docker build -t shunt-router .
docker run -p 127.0.0.1:8080:8080 --env-file .env shunt-router
```

### Check the install first

```bash
shunt doctor
```

It reads configuration only — no provider call, no spend, and it will not download the
embedding model. It reports which provider keys resolve (presence, never the value),
how many models are routable, whether the embedding weights are already cached, whether
the bind address is free, and the one a fresh install usually gets wrong: whether
escalation is **armed** or merely *enabled and inert*. It exits non-zero only when the
router could not serve a request at all, so it is safe in a setup script.

Then start the proxy:

```bash
shunt
```

`uv run shunt`, `python -m shunt`, and the installed `shunt` command are equivalent —
each starts the proxy on `127.0.0.1:8080`. No uv? `pip install -e . && shunt` works
too. (Run from the repo root; the same `shunt <subcommand>` verbs — `doctor`, `flag`,
`reindex`, `explain`, `escalate`, `inspect` — apply.)

Point your tool at localhost:8080. The router picks the session model on the first
turn and locks it for the session; until enough outcomes accumulate — 20 verified, or
50 labelled of any tier — it cold-starts to the cheap default, which is why a fresh
install looks like one:

| Tool | Env var |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8080` |
| opencode | `OPENAI_BASE_URL=http://localhost:8080/v1` |
| aider | `OPENAI_API_BASE=http://localhost:8080/v1` |
| n8n / LangChain | `baseURL: http://localhost:8080/v1` |
