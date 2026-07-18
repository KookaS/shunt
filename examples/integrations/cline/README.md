# cline → Shunt

[Cline](https://github.com/cline/cline) (Apache-2.0), the VS Code coding agent,
pointed at Shunt through its OpenAI-Compatible provider. Shunt picks the model;
Cline never learns which one.

**Docs-only — verify manually.** Cline runs inside VS Code (a GUI), so it can't
be driven in headless CI. There is no `compose.yaml` here. The configuration
below is verified; confirm it in your editor.

## Point Cline at Shunt

In the Cline settings panel:

1. **API Provider** → **OpenAI Compatible**
2. **Base URL** → `http://127.0.0.1:8080/v1`  (OpenAI wire keeps the `/v1` suffix)
3. **API Key** → any non-empty placeholder (e.g. `ignored`) — Shunt holds the
   real provider keys and ignores this field, but the form requires it
4. **Model ID** → a Shunt-routed id such as `auto`

## Status

Works fully. Cline speaks the OpenAI Chat Completions wire, which Shunt serves,
so routing is transparent end to end. The only limitation is the harness: it's
GUI-only, so this directory documents the setup rather than running a CI
handshake. See [`../README.md`](../README.md) for the overall integration model.
