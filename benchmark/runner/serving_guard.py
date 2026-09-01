"""Pre-flight assertion on a LOCAL inference server's resolved serving configuration."""

# Hosted providers serve a documented context policy; a self-hosted one serves whatever flags the
# supervisor chose, and those flags can destroy the measurement while returning ``200 OK``.
# The concrete case this module exists for: ollama spawns ``llama-server`` with
# ``--context-shift --keep 4``, overriding llama.cpp's own defaults (context shift DISABLED,
# ``--keep 0``). When a trajectory fills the window, half of it is deleted from the FRONT, keeping
# four tokens — the system prompt, the tool schema, the task instructions and the problem statement
# all sit inside the discarded span. Generation then continues with no tools and no instructions,
# and the request still returns ``200``. No HTTP-status retry policy and no ``abort_exceptions``
# list can see that.
#
# Measured on a 19-cell SWE-bench run: 337 shift events, 334 of 1001 requests (33%) affected,
# 671k of 1.42M generated tokens (47%, ~7.8 GPU-hours) produced from a mutilated context. One
# ``ps`` of the spawned command line would have caught it beforehand — which is exactly what this
# module does, before any container starts.
#
# Two conditions are asserted for a local endpoint:
#
# 1. context shift is provably DISABLED, and
# 2. ``n_ctx >= prompt_budget + max_tokens`` — so the server can never be asked to generate past
#    the end of its own window.
#
# Both refuse LOUDLY. A router can catch a context error and escalate; it cannot catch a 200 OK
# carrying nonsense. Hosted endpoints are exempt: the concept does not apply to them and the guard
# is a no-op — but the exemption is only ever taken on a DECIDED classification, and the serving
# config is only ever read off a process that declares the endpoint's own port. "Could not tell"
# raises on both axes, because a fabricated PASS is the one failure mode a guard cannot have.

from __future__ import annotations

import ipaddress
import json
import logging
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import urlsplit

from benchmark import config

_LOG = logging.getLogger(__name__)

# Locality is DECIDED, not looked up in a list of five spellings: a name allowlist silently
# classified `192.168.1.50`, `http://ollama:11434` (the compose rig's own form) and `127.0.0.2`
# as hosted, which disables this guard with no log line. The rules below are the decidable ones;
# anything outside them refuses (`UndecidableEndpointError`) rather than skipping.
#
# Hostname suffixes reserved for names inside this machine or deployment: mDNS (`.local`),
# RFC 6761 (`.localhost`), RFC 8375 (`.home.arpa`), and the docker/k8s/site conventions.
_LOCAL_SUFFIXES: Final[tuple[str, ...]] = (
    ".local",
    ".localhost",
    ".internal",
    ".lan",
    ".home.arpa",
    ".intranet",
)

# A base_url whose transport is a filesystem socket cannot leave this machine.
_UNIX_SCHEMES: Final[frozenset[str]] = frozenset({"unix", "http+unix", "unix+http"})

# Ports a scheme implies when the URL spells none, so an endpoint always has a port to match a
# server process against.
_SCHEME_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}

# The inference server EXECUTABLES whose command line carries the context policy, matched
# against argv[0]'s basename exactly (see `is_server_command`). ollama does not serve inference
# itself — it spawns llama-server — so the child is what must be read.
_SERVER_PROCESS_NAMES: Final[tuple[str, ...]] = ("llama-server", "llama-cpp-server")

# Where the resolved configuration is asserted into the run record. Gitignored (regenerable);
# the authoritative copy for a supervised run is the WARNING line in the tee'd run log.
SERVING_RECORD: Final[Path] = (
    Path(__file__).resolve().parent.parent / "routing" / "artifacts" / "serving_config.json"
)


class UnsafeServingError(RuntimeError):
    """The local serving endpoint is configured in a way that would corrupt the measurement."""


class UndecidableEndpointError(UnsafeServingError):
    """The endpoint's locality could not be decided, so the guard refuses instead of skipping."""

    # A subclass, so every caller that already fails closed on ``UnsafeServingError`` fails closed
    # here too: "could not tell whether this is local" must never take the hosted no-op path.


@dataclass(frozen=True)
class ServingConfig:
    """The resolved serving parameters of one local inference server."""

    base_url: str
    command: str
    n_ctx: int | None
    context_shift: bool | None
    n_keep: int | None


def _named_host_is_local(host: str) -> bool:
    """Decide a non-IP hostname: local when it names something inside this deployment."""
    # A SINGLE-LABEL name is local by construction: the public DNS namespace has no single-label
    # hosts, so `ollama`, `llama`, `gpu-box` can only be a docker-compose service, a k8s short
    # name or a /etc/hosts entry — which is exactly how the containerised rig here addresses it.
    name = host.rstrip(".").lower()
    return name == "localhost" or name.endswith(_LOCAL_SUFFIXES) or "." not in name


def _ip_is_local(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, base_url: str) -> bool:
    """Decide a literal IP: loopback/private/link-local/unspecified is local, global is hosted."""
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified:
        return True
    if ip.is_multicast or ip.is_reserved:
        raise UndecidableEndpointError(
            f"endpoint {base_url} names the multicast/reserved address {ip}, which is neither a "
            "host on this machine nor a provider; refusing rather than guessing its locality."
        )
    if ip.is_global:
        return False
    raise UndecidableEndpointError(
        f"endpoint {base_url} names {ip}, which is neither private nor globally routable "
        "(e.g. carrier-grade NAT / 100.64.0.0/10, as a VPN overlay hands out). Whether its "
        "serving flags are inspectable here cannot be decided, so the guard refuses rather "
        "than silently skipping the context-shift assertion."
    )


def is_local_endpoint(base_url: str | None) -> bool:
    """True iff *base_url* is served from this machine or deployment (flags inspectable).

    Raises ``UndecidableEndpointError`` rather than defaulting to hosted when it cannot decide.
    """
    if not base_url:
        return False
    parts = urlsplit(base_url)
    if parts.scheme.lower() in _UNIX_SCHEMES or ".sock" in base_url:
        return True  # a filesystem socket cannot leave this machine
    host = parts.hostname
    if not host:
        raise UndecidableEndpointError(
            f"endpoint {base_url!r} parses to no hostname (a missing scheme spells one as a "
            "path), so it cannot be classified local or hosted; write it as a full URL."
        )
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return _named_host_is_local(host)
    return _ip_is_local(ip, base_url)


def _argv(command: str) -> list[str]:
    """The command line split into arguments, falling back to whitespace on an unbalanced quote."""
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def is_server_command(command: str) -> bool:
    """True iff *command* IS an inference server — argv[0]'s basename, matched exactly."""
    # Identification is by the executable, never by a substring of the line. A substring test
    # adopted `/opt/start_llama-server.sh` (a wrapper script), `grep llama-server`, an editor
    # with the unit file open, and `python3 /opt/llama-server/serve.py` — then read THEIR flags
    # as the serving configuration and returned PASS. What a wrapper passes to the real server
    # is not what the wrapper's own command line says.
    argv = _argv(command)
    if not argv:
        return False
    return PurePosixPath(argv[0]).name in _SERVER_PROCESS_NAMES


def _server_command_lines() -> list[str]:
    """Full command lines of every inference-server process visible on this host."""
    try:
        out = subprocess.run(  # noqa: S603 (fixed argv, no shell, no user input)
            ["ps", "-eo", "args="],  # noqa: S607 (ps is resolved from PATH by design)
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [stripped for line in out.splitlines() if is_server_command(stripped := line.strip())]


def _int_flag(argv: list[str], *names: str) -> int | None:
    """The integer value of the first of *names* present in *argv*, as ``-x V`` or ``--x=V``."""
    for name in names:
        for index, token in enumerate(argv):
            if token == name:
                value = argv[index + 1] if index + 1 < len(argv) else None
            elif token.startswith(f"{name}="):
                value = token.split("=", 1)[1]
            else:
                continue
            try:
                return int(value) if value is not None else None
            except ValueError:
                return None
    return None


def parse_server_command(command: str) -> tuple[int | None, bool | None, int | None]:
    """``(n_ctx, context_shift, n_keep)`` as spelled on a llama-server command line."""

    # ``context_shift`` is None when NEITHER flag appears: llama.cpp's compiled-in default is
    # disabled, but a supervisor may have been built against an older default, so an absent flag
    # is *unknown* rather than *safe*. Only an explicit ``--no-context-shift`` proves it off.
    argv = _argv(command)
    if "--no-context-shift" in argv:
        shift: bool | None = False
    elif "--context-shift" in argv:
        shift = True
    else:
        shift = None
    return _int_flag(argv, "-c", "--ctx-size"), shift, _int_flag(argv, "--keep")


def endpoint_port(base_url: str) -> int | None:
    """The TCP port *base_url* connects to, taking the scheme's default when none is spelled."""
    parts = urlsplit(base_url)
    return parts.port if parts.port is not None else _SCHEME_PORTS.get(parts.scheme.lower())


def server_port(command: str) -> int | None:
    """The port a server command line binds, read from its own argv (None when undeclared)."""
    # Read from argv, never as a substring of the whole line: `--port 9999` appearing ANYWHERE
    # in the text used to satisfy a match for a different endpoint.
    return _int_flag(_argv(command), "--port")


def _match_by_port(base_url: str, commands: list[str]) -> str:
    """The one server process that binds *base_url*'s port; refuse loudly on anything else."""
    port = endpoint_port(base_url)
    if port is None:
        raise UnsafeServingError(
            f"local endpoint {base_url} spells no port and its scheme implies none, so no "
            "server process can be attributed to it; give the endpoint an explicit port."
        )
    matched = [c for c in commands if server_port(c) == port]
    if len(matched) == 1:
        return matched[0]
    seen = sorted({str(server_port(c)) for c in commands})
    if not matched:
        raise UnsafeServingError(
            f"local endpoint {base_url} is on port {port}, but no inference server process "
            f"binds it (the {len(commands)} visible bind {seen}). This is EXACTLY ollama's "
            "topology: it listens on 11434 and spawns llama-server on a random high port, so "
            "the child's flags cannot be attributed to the endpoint under measurement. Point "
            "the benchmark at the llama-server port directly, or serve llama-server yourself. "
            "Refusing rather than reading an unrelated process's context policy."
        )
    raise UnsafeServingError(
        f"local endpoint {base_url} matches {len(matched)} inference server processes on "
        f"port {port}; cannot attribute a serving configuration to the endpoint."
    )


def resolve_serving_config(base_url: str) -> ServingConfig:
    """Read the resolved flags of the local inference server behind *base_url*.

    Attribution is by the port the process ITSELF declares; an unmatched process is refused,
    never accepted as the endpoint's — a mis-attributed PASS is worse than no guard.
    """
    commands = _server_command_lines()
    if not commands:
        raise UnsafeServingError(
            f"local endpoint {base_url} names no inspectable inference server on this host "
            f"(looked for {list(_SERVER_PROCESS_NAMES)} in `ps -eo args=`). The resolved "
            "context policy cannot be read, so it cannot be asserted; refusing rather than "
            "assuming it is safe."
        )
    command = _match_by_port(base_url, commands)
    n_ctx, shift, n_keep = parse_server_command(command)
    return ServingConfig(
        base_url=base_url, command=command, n_ctx=n_ctx, context_shift=shift, n_keep=n_keep
    )


def _record(resolved: ServingConfig, *, prompt_budget: int | None, max_tokens: int | None) -> None:
    """Assert the resolved flags into the run record — the log line and a JSON artifact.

    WARNING level so the line lands in the tee'd run log a supervised ``--live`` run keeps
    (``docs/benchmark-live-run-runbook-2026-07.md``), which is where a post-hoc audit looks.
    """
    payload = {**asdict(resolved), "prompt_budget": prompt_budget, "max_tokens": max_tokens}
    _LOG.warning("resolved local serving config: %s", json.dumps(payload, sort_keys=True))
    try:
        SERVING_RECORD.parent.mkdir(parents=True, exist_ok=True)
        SERVING_RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except OSError:  # the log line is the record that matters; a scratch write must not abort
        _LOG.exception("could not persist the resolved serving config to %s", SERVING_RECORD)


def _declared_int(key: str) -> int | None:
    """One positive integer from the declared ``live.serving`` block; None when absent."""
    # Config is a boundary, so the type is checked rather than assumed: a mistyped or
    # non-positive value must read as UNDECLARED and refuse, never coerce into a bogus margin.
    value = dict(config.live_config().get("serving") or {}).get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def serving_prompt_budget() -> int | None:
    """Declared ``live.serving.prompt_budget`` — the largest prompt a cell may present."""
    return _declared_int("prompt_budget")


def serving_max_tokens() -> int | None:
    """Declared ``live.serving.max_tokens`` — the fallback generation cap for the arithmetic."""
    # Used only when the scaffold sends no explicit cap of its own. Declared rather than
    # defaulted in code so the number a run was validated against is recorded where the run is
    # configured, and so the guard is SATISFIABLE without editing the model registry.
    return _declared_int("max_tokens")


def assert_serving_safe(
    base_url: str | None, *, max_tokens: int | None, prompt_budget: int | None = None
) -> ServingConfig | None:
    """Refuse a local endpoint that can silently corrupt the measurement; None if not local.

    Raises ``UnsafeServingError`` on context shift enabled or unproven, on an undeclared
    generation budget, and on ``n_ctx < prompt_budget + max_tokens``.
    """
    if not is_local_endpoint(base_url):
        return None
    assert base_url is not None  # noqa: S101 (narrowing after is_local_endpoint)
    budget = serving_prompt_budget() if prompt_budget is None else prompt_budget
    resolved = resolve_serving_config(base_url)
    _record(resolved, prompt_budget=budget, max_tokens=max_tokens)

    if resolved.context_shift is not False:
        spelled = "enabled" if resolved.context_shift else "not declared either way"
        raise UnsafeServingError(
            f"local endpoint {base_url} serves with context shift {spelled} "
            f"(--keep {resolved.n_keep}). Head-eviction deletes the system prompt and the tool "
            "schema mid-generation and still returns HTTP 200, so the corruption is invisible "
            "to every status-based retry. Re-serve with an explicit --no-context-shift "
            "(ollama: PARAMETER num_keep <size of the fixed prefix>, or serve llama-server "
            "directly)."
        )
    if budget is None or max_tokens is None:
        missing = "prompt_budget" if budget is None else "max_tokens"
        raise UnsafeServingError(
            f"local endpoint {base_url} cannot be validated: no {missing} is in force, so "
            "n_ctx >= prompt_budget + max_tokens is uncheckable. Declare it under "
            f"`live.serving.{missing}` in benchmark/benchmark.yaml (e.g. "
            f"`live:\n  serving:\n    {missing}: 26000`), rather than letting an undeclared "
            "default govern a measured run."
        )
    if resolved.n_ctx is None:
        raise UnsafeServingError(
            f"local endpoint {base_url} declares no context size on its command line "
            f"({resolved.command!r}); pass an explicit -c/--ctx-size."
        )
    required = budget + max_tokens
    if resolved.n_ctx < required:
        raise UnsafeServingError(
            f"local endpoint {base_url} serves n_ctx={resolved.n_ctx}, below the "
            f"prompt_budget({budget}) + max_tokens({max_tokens}) = {required} the run needs. "
            f"Any turn whose prompt exceeds {resolved.n_ctx - max_tokens} tokens can be asked "
            "to generate past the end of the window."
        )
    return resolved


__all__ = [
    "SERVING_RECORD",
    "ServingConfig",
    "UndecidableEndpointError",
    "UnsafeServingError",
    "assert_serving_safe",
    "endpoint_port",
    "is_local_endpoint",
    "is_server_command",
    "parse_server_command",
    "resolve_serving_config",
    "server_port",
    "serving_prompt_budget",
]
