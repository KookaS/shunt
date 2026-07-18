# cursor → Shunt

> **Documented for completeness — not fully routable.** [Cursor](https://cursor.com)
> is closed-source, and its main coding surfaces are locked to Cursor's own cloud.
> Only part of it can be pointed at a localhost proxy like Shunt today; the rest
> cannot. This directory records what works and what doesn't so the picture is
> honest, not aspirational.

**Docs-only — verify manually.** Cursor is a GUI editor and can't be driven in
headless CI. There is no `compose.yaml` here.

## Point Cursor at Shunt (Chat / Plan only)

Cursor exposes an **Override OpenAI Base URL** toggle under
**Settings → Models**. Set it to Shunt on the OpenAI wire:

- **Override OpenAI Base URL** → `http://127.0.0.1:8080/v1`  (keep the `/v1` suffix)
- **API Key** → any non-empty placeholder — Shunt holds the real provider keys
  and ignores this field
- Add/select a model id Shunt routes, e.g. `auto`

## What actually routes

| Cursor surface | Routes through Shunt? |
|----------------|-----------------------|
| **Chat / Plan** | Yes — honours the OpenAI base-URL override |
| **Agent** | No — backend-locked to Cursor's cloud |
| **Tab** (autocomplete) | No — backend-locked to Cursor's cloud |

The base-URL override applies **only to Chat and Plan**. Cursor's Agent and Tab
features run against Cursor's own backend and cannot be redirected to a local
proxy, so Shunt can't route them. Treat Cursor as a **partial** integration: use
Shunt for Chat/Plan, and accept that Agent and Tab bypass it. See
[`../README.md`](../README.md) for the overall integration model.
