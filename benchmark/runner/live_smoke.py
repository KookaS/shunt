#!/usr/bin/env python3
"""Owner-runnable live smoke: one real cheap session through shunt.

Boots the real proxy, sends one tiny completion, verifies the decision headers +
captured session row; `--live` + interactive confirmation are required.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from shunt.models.config import ModelConfig, ModelPool, default_registry_path
from shunt.router.policy import RouterPolicy, load_router_policy, packaged_policy_path

EXPECTED_SPEND_USD: Final[float] = 0.10
DEFAULT_MAX_COST_USD: Final[float] = 0.10
DEFAULT_PORT: Final[int] = 8080
BOOT_TIMEOUT_S: Final[float] = 60.0
REQUEST_TIMEOUT_S: Final[float] = 120.0
SESSION_WAIT_S: Final[float] = 15.0
PROMPT_MARKER: Final[str] = "shunt-live-smoke"
SMOKE_PROMPT: Final[str] = f"{PROMPT_MARKER}: reply with the single word OK."
STRATEGY: Final[str] = "always_cheap"
# 16 left a reasoning `:free` lane with no room to emit VISIBLE text — every token
# went to reasoning and `content` came back empty while the body parsed fine. 64
# gives a reasoning model room for both the scratchpad and the one-word answer.
SMOKE_MAX_TOKENS: Final[int] = 64
# Transient in-band refusals (upstream worker at capacity) are retried with
# exponential backoff rather than failed as a wiring bug. Five attempts span
# 2+4+8+16 = 30s, which clears a short 16/16 worker wall without stalling CI.
COMPLETION_MAX_ATTEMPTS: Final[int] = 5
COMPLETION_BACKOFF_S: Final[float] = 2.0
_TRANSIENT_ERROR_TYPES: Final[frozenset[str]] = frozenset(
    {"provider_unavailable", "rate_limit", "server_error", "timeout"}
)
_TRANSIENT_ERROR_CODES: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})


# ── spend gates (pure — unit-tested) ─────────────────────────────────────────


def live_flag_refusal(live: bool) -> str | None:
    """Refuse unless the caller accepted the spend flag."""
    if live:
        return None
    return "refusing to run: --live/--confirm is required (this spends real money)"


def tty_refusal(isatty: Callable[[], bool]) -> str | None:
    """Refuse when stdin is not an interactive terminal (CI, cron, scripts)."""
    if isatty():
        return None
    return "refusing to run: stdin is not an interactive TTY; the live smoke must run supervised"


def confirm_refusal(confirm: Callable[[str], str | None]) -> str | None:
    """Refuse unless the owner types y to the spend plan."""
    prompt = (
        f"One real completion through Shunt (strategy={STRATEGY}). "
        f"Expected spend ~${EXPECTED_SPEND_USD:.2f}; capped by --max-cost "
        f"(default ${DEFAULT_MAX_COST_USD:.2f}). Continue? [y/N] "
    )
    answer = confirm(prompt)
    if answer is None:
        return "refusing to run: no confirmation input (EOF); the live smoke must run supervised"
    if answer.strip().lower() not in ("y", "yes"):
        return "refusing to run: confirmation declined"
    return None


def key_refusal(key_env: str, key_value: str | None) -> str | None:
    """Refuse unless the provider key is already in the environment."""
    if key_value:
        return None
    return (
        f"refusing to run: ${key_env} is not set in the environment. The runner "
        "injects provider keys; this script never reads a .env file."
    )


def refusal_reason(
    *,
    live: bool,
    isatty: Callable[[], bool],
    confirm: Callable[[str], str | None],
    key_env: str,
    key_value: str | None,
) -> str | None:
    """The first unmet gate as a message; None when every gate passes."""
    reason = live_flag_refusal(live)
    if reason is not None:
        return reason
    reason = tty_refusal(isatty)
    if reason is not None:
        return reason
    reason = confirm_refusal(confirm)
    if reason is not None:
        return reason
    return key_refusal(key_env, key_value)


def _confirm_line(prompt: str) -> str | None:
    """One input() line; None on EOF so a non-interactive caller cannot pass."""
    try:
        return input(prompt)
    except EOFError:
        return None


# ── config surface ───────────────────────────────────────────────────────────


def resolve_config(config_dir: str | None) -> tuple[Path, Path]:
    """The (registry, policy) paths for the smoke's config surface."""
    if config_dir is not None:
        base = Path(config_dir)
        return base / "models.yaml", base / "router.yaml"
    return default_registry_path(), packaged_policy_path()


def build_pool(registry_path: Path, policy_path: Path) -> ModelPool:
    """The live pool, restricted exactly as the server restricts it at boot."""
    pool = ModelPool.load(str(registry_path))
    policy: RouterPolicy = load_router_policy(policy_path)
    pool.restrict_to_live(policy.models)
    return pool


def cheapest_live_model(pool: ModelPool) -> ModelConfig:
    """The model ``always_cheap`` will serve: lowest total list price in the live set."""
    ranked = pool.ranked_models()
    if not ranked:
        raise ValueError("the live model set is empty — nothing to smoke")
    return ranked[0]


# ── server ───────────────────────────────────────────────────────────────────


@dataclass
class RunningServer:
    """A launched shunt server plus its run artifacts."""

    proc: subprocess.Popen[str]
    port: int
    data_dir: Path
    log_path: Path

    def stop(self) -> None:
        """Terminate the server, escalating to kill after a grace period."""
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)


def _drain_server_output(proc: subprocess.Popen[str], log_path: Path) -> None:
    """Copy the server's stdout to its run log and echo it to stderr (tee-able)."""
    if proc.stdout is None:
        return
    with log_path.open("a", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", file=sys.stderr)  # noqa: T201 - harness CLI output


def server_env(port: int, data_dir: Path, run_dir: Path, config_dir: str | None) -> dict[str, str]:
    """The server's environment: injected keys inherited, config surface pinned."""
    env = dict(os.environ)
    env["SHUNT_HOST"] = "127.0.0.1"
    env["SHUNT_PORT"] = str(port)
    env["SHUNT_DATA_DIR"] = str(data_dir)
    env["SHUNT_ROUTER_STRATEGY"] = STRATEGY
    env["SHUNT_EXPLORATION_ENABLED"] = "0"
    # SHUNT_MODEL_CONFIG_PATH is a registry override with its own resolution order;
    # scrub it so a stale value cannot boot a foreign pool while the script asserts
    # against its own. --config-dir drives SHUNT_CONFIG_DIR instead.
    env.pop("SHUNT_MODEL_CONFIG_PATH", None)
    if config_dir is None:
        # An empty SHUNT_CONFIG_DIR falls through to the packaged config, matching
        # resolve_config's packaged paths and defeating dev-machine overrides.
        empty = run_dir / "empty-config"
        empty.mkdir(parents=True, exist_ok=True)
        env["SHUNT_CONFIG_DIR"] = str(empty)
    else:
        env["SHUNT_CONFIG_DIR"] = config_dir
    return env


def start_server(port: int, data_dir: Path, run_dir: Path, config_dir: str | None) -> RunningServer:
    """Launch ``python -m shunt start`` under the smoke env, logging its output."""
    env = server_env(port, data_dir, run_dir, config_dir)
    log_path = run_dir / "server.log"
    proc = subprocess.Popen(
        [sys.executable, "-m", "shunt", "start"],
        env=env,
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    threading.Thread(
        target=_drain_server_output,
        args=(proc, log_path),
        name="live-smoke-server-log",
        daemon=True,
    ).start()
    return RunningServer(proc=proc, port=port, data_dir=data_dir, log_path=log_path)


def _urlopen(url: str, timeout: float) -> tuple[int, bytes] | None:
    """GET *url* -> (status, body); None when the server is unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError):
        return None


def server_healthy(port: int, timeout: float = 2.0) -> bool:
    """True once the server's /health answers 200."""
    result = _urlopen(f"http://127.0.0.1:{port}/health", timeout)
    return result is not None and result[0] == 200


def model_listed(port: int, model_name: str, timeout: float = 5.0) -> bool:
    """True if the expected model is routable per the free GET /v1/models."""
    result = _urlopen(f"http://127.0.0.1:{port}/v1/models", timeout)
    if result is None or result[0] != 200:
        return False
    try:
        data = json.loads(result[1].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    ids = [str(row.get("id", "")) for row in data.get("data", []) if isinstance(row, dict)]
    return model_name in ids


def wait_healthy(server: RunningServer, timeout: float, report: Callable[[str], None]) -> bool:
    """Wait for /health, failing fast if the server process dies during boot."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_healthy(server.port):
            return True
        if server.proc.poll() is not None:
            report(f"shunt server exited during boot (rc={server.proc.returncode})")
            for line in _tail(server.log_path, 20):
                report(f"    {line}")
            return False
        time.sleep(0.25)
    report(f"shunt server did not serve /health within {timeout:.0f}s (log: {server.log_path})")
    return False


def _tail(path: Path, n: int) -> list[str]:
    """The last *n* lines of a file, or [] when absent."""
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()[-n:]


# ── the one real request ─────────────────────────────────────────────────────


def post_completion(port: int, timeout: float) -> tuple[int | None, dict[str, str], str]:
    """Send one tiny completion; (status, headers, body) — None status = unreachable."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": "auto",
            "stream": False,
            "max_tokens": SMOKE_MAX_TOKENS,
            "messages": [{"role": "user", "content": SMOKE_PROMPT}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": "Bearer smoke-placeholder",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            headers = {key.lower(): value for key, value in resp.headers.items()}
            return resp.status, headers, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            {key.lower(): value for key, value in exc.headers.items()},
            exc.read().decode("utf-8", "replace"),
        )
    except (urllib.error.URLError, OSError) as exc:
        return None, {}, str(exc)


def _inband_error(body: str) -> tuple[bool, str] | None:
    """``(is_transient, message)`` for an HTTP-200 body carrying an error, else None.

    OpenRouter reports a provider-side refusal inside a 200 body — ``choices`` is
    null and an ``error`` object carries the cause — rather than as an HTTP status.
    A missing key or header surfaces as a non-200 status and never reaches here.
    """
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("choices"):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    metadata = error.get("metadata")
    error_type = str(metadata.get("error_type") or "") if isinstance(metadata, dict) else ""
    code = error.get("code")
    message = str(error.get("message") or "")
    transient = (
        error_type in _TRANSIENT_ERROR_TYPES
        or (isinstance(code, int) and code in _TRANSIENT_ERROR_CODES)
        or "ResourceExhausted" in message
    )
    return transient, (message or f"in-band {error_type or code} error")


def transient_provider_error(body: str) -> str | None:
    """The message of a retryable in-band provider refusal, or None.

    A capacity refusal (``ResourceExhausted`` / ``provider_unavailable``) is
    transient provider news, not a content or wiring failure, so the caller may
    retry it; anything else returns None and fails on its own merits.
    """
    found = _inband_error(body)
    if found is None or not found[0]:
        return None
    return found[1]


def post_completion_with_retry(
    port: int, timeout: float, report: Callable[[str], None]
) -> tuple[int | None, dict[str, str], str]:
    """One completion, retrying a transient in-band provider refusal with backoff.

    Non-200 responses and reachable parseable bodies return immediately; only a
    transient in-band refusal (e.g. an upstream worker at capacity) is retried,
    bounded at ``COMPLETION_MAX_ATTEMPTS``. The final attempt's body is returned
    so a persistent refusal is still reported — as a provider condition.
    """
    delay = COMPLETION_BACKOFF_S
    result = post_completion(port, timeout)
    for attempt in range(2, COMPLETION_MAX_ATTEMPTS + 1):
        transient = transient_provider_error(result[2])
        if result[0] is None or transient is None:
            return result
        report(
            f"transient provider refusal (attempt {attempt - 1}/{COMPLETION_MAX_ATTEMPTS}): "
            f"{transient}; retrying in {delay:.0f}s"
        )
        time.sleep(delay)
        delay *= 2
        result = post_completion(port, timeout)
    return result


# ── verification (machine-checkable pass criteria) ───────────────────────────


def header_value(headers: dict[str, str], name: str) -> str | None:
    """Case-insensitive HTTP header lookup, or None when absent.

    Header names are case-insensitive, and Starlette lowercases them on the wire
    (``x-shunt-decision``), so a plain-dict lookup on the canonical spelling misses
    every one of them.
    """
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def decision_from_headers(headers: dict[str, str]) -> tuple[str, str] | None:
    """Split ``X-Shunt-Decision`` into ``(model, reason)``, or None when absent."""
    raw = header_value(headers, "X-Shunt-Decision")
    if not raw:
        return None
    model, _, reason = raw.partition("; reason=")
    return model.strip(), reason.strip()


def verify_headers(
    status: int | None,
    headers: dict[str, str],
    expected_model: str,
    reason: str,
) -> list[str]:
    """The decision-header pass criteria; empty list == pass."""
    problems: list[str] = []
    if status != 200:
        return [
            f"routed completion returned HTTP {status}; expected 200 "
            "(auth, balance, or provider failure)"
        ]
    decision = decision_from_headers(headers)
    if decision is None:
        problems.append("response carried no X-Shunt-Decision header")
    else:
        model, got_reason = decision
        if model != expected_model:
            problems.append(
                f"decision header served {model!r}; expected the cheapest live model "
                f"{expected_model!r}"
            )
        if got_reason != reason:
            problems.append(f"decision reason is {got_reason!r}; expected {reason!r}")
    if not header_value(headers, "X-Shunt-Session-Id"):
        problems.append("response carried no X-Shunt-Session-Id header")
    return problems


def content_problems(body: str) -> list[str]:
    """The completion body must be a real, non-empty model answer."""
    found = _inband_error(body)
    if found is not None:
        kind = "transient provider failure" if found[0] else "provider returned an in-band error"
        return [f"{kind}: {found[1]}"]
    try:
        payload = json.loads(body)
        content = payload["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ["routed completion returned no parseable text content"]
    if not content:
        return ["routed completion returned empty text content"]
    return []


def _read_session_row(data_dir: Path, session_id: str) -> dict[str, Any] | None:
    """Read one session row from the outcome store (read-only connection)."""
    db_path = data_dir / "outcomes.db"
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
    except sqlite3.DatabaseError:
        return None
    return dict(row) if row is not None else None


def read_session_row(data_dir: Path, session_id: str, timeout: float) -> dict[str, Any] | None:
    """Poll for the session row (the response can return before the store flush)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = _read_session_row(data_dir, session_id)
        if row is not None:
            return row
        time.sleep(0.25)
    return None


def verify_capture(
    row: dict[str, Any] | None,
    expected_model: str,
    max_cost: float,
    marker: str,
) -> tuple[list[str], list[str]]:
    """The capture-record pass criteria: (problems, warnings); empty problems == pass."""
    problems: list[str] = []
    warnings: list[str] = []
    if row is None:
        return ["no session row was captured in the outcome store (yield-without-capture)"], []
    provenance: dict[str, Any] | None = None
    try:
        provenance = json.loads(str(row.get("decision_provenance") or "{}"))
        if not isinstance(provenance, dict):
            provenance = None
    except json.JSONDecodeError:
        problems.append("captured decision_provenance is not valid JSON")
    served = str(row.get("model_chosen") or "")
    if served != expected_model:
        problems.append(f"captured model_chosen is {served!r}; expected {expected_model!r}")
    if provenance is not None:
        if provenance.get("model_chosen") != expected_model:
            problems.append(
                f"captured provenance model_chosen is {provenance.get('model_chosen')!r}; "
                f"expected {expected_model!r}"
            )
        if provenance.get("selection_rule_used") != STRATEGY:
            problems.append(
                f"captured selection_rule_used is {provenance.get('selection_rule_used')!r}; "
                f"expected {STRATEGY!r}"
            )
        if provenance.get("router_propensity") is None:
            problems.append("captured provenance has no router_propensity")
    if marker not in str(row.get("prompt_text") or ""):
        problems.append(
            "captured prompt_text does not carry the smoke marker — the row is not this run's"
        )
    raw_cost_known = row.get("cost_known")
    cost_known = 1 if raw_cost_known is None else int(raw_cost_known)
    cost = float(row.get("cost") or 0.0)
    if cost_known == 0:
        warnings.append(
            "provider reported no usage cost (cost_known=0) — spend cannot be reconciled"
        )
    elif cost > max_cost:
        problems.append(f"recorded cost ${cost:.4f} exceeds the --max-cost cap ${max_cost:.2f}")
    return problems, warnings


# ── CLI ──────────────────────────────────────────────────────────────────────


class _Reporter:
    """Print each line to stdout and append the same line to the run log."""

    def __init__(self, log_path: Path) -> None:
        self._log = log_path.open("a", encoding="utf-8")

    def write(self, line: str) -> None:
        """Emit *line* to stdout and the run log."""
        print(line)
        self._log.write(line + "\n")
        self._log.flush()

    def close(self) -> None:
        """Close the run log."""
        self._log.close()


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        "--confirm",
        action="store_true",
        dest="live",
        help="Accept real spend (alias --confirm). Required — the smoke is otherwise inert.",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        help=f"Hard cap on the recorded session cost in USD (default {DEFAULT_MAX_COST_USD}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Shunt listen port (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory holding models.yaml + router.yaml to smoke instead of the packaged config.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Parent directory for run artifacts (default benchmark/runner/artifacts/live-smoke).",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the smoke; 0 = pass, 1 = the smoke failed, 2 = refused by a gate."""
    parser = argparse.ArgumentParser(
        prog="live_smoke",
        description=(
            "Run one real, cheap session through Shunt against a real provider and verify "
            "the decision headers and the captured session row. Owner-runnable only: "
            "--live/--confirm plus an interactive confirmation are required."
        ),
    )
    _add_args(parser)
    args = parser.parse_args(argv)

    registry_path, policy_path = resolve_config(args.config_dir)
    try:
        pool = build_pool(registry_path, policy_path)
        expected = cheapest_live_model(pool)
    except Exception as exc:  # noqa: BLE001 - a bad config must abort before any spend
        print(f"aborted: cannot resolve the live model set: {exc}", file=sys.stderr)
        return 2
    key_env = expected.api_key_env_var

    reason = refusal_reason(
        live=args.live,
        isatty=sys.stdin.isatty,
        confirm=_confirm_line,
        key_env=key_env,
        key_value=os.environ.get(key_env),
    )
    if reason is not None:
        print(f"aborted: {reason}", file=sys.stderr)
        return 2

    run_root = (
        Path(args.run_dir)
        if args.run_dir
        else Path(__file__).resolve().parent.parent / "runner" / "artifacts" / "live-smoke"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"

    report = _Reporter(run_dir / "live-smoke.log")
    problems: list[str] = []
    warnings: list[str] = []
    served = expected.name
    decision_reason: str | None = None
    session_id = ""
    cost = 0.0
    try:
        report.write(f"live smoke starting: run={stamp}")
        report.write(
            f"  expected model: {expected.name} "
            f"({expected.model_id or expected.name}) via {expected.provider}"
        )
        report.write(f"  key env:        {key_env}")
        report.write(
            f"  spend cap:      ${args.max_cost:.2f} (expected ~${EXPECTED_SPEND_USD:.2f})"
        )
        report.write(f"  run dir:        {run_dir}")

        server = start_server(args.port, data_dir, run_dir, args.config_dir)
        try:
            if not wait_healthy(server, BOOT_TIMEOUT_S, report.write):
                return 1
            report.write(f"server healthy on 127.0.0.1:{args.port}")
            if not model_listed(args.port, expected.name):
                problems.append(f"expected model {expected.name} is not in GET /v1/models")
                return 1

            report.write(
                f"sending one real completion (max_tokens={SMOKE_MAX_TOKENS}, "
                f"marker={PROMPT_MARKER!r})..."
            )
            status, headers, body = post_completion_with_retry(
                args.port, REQUEST_TIMEOUT_S, report.write
            )
            if status is None:
                problems.append(f"could not reach shunt: {body}")
                return 1
            report.write(f"completion returned HTTP {status}")
            decision = decision_from_headers(headers)
            if decision is not None:
                served, decision_reason = decision
            session_id = header_value(headers, "X-Shunt-Session-Id") or ""
            problems.extend(verify_headers(status, headers, expected.name, STRATEGY))
            problems.extend(content_problems(body))

            row = read_session_row(data_dir, session_id, SESSION_WAIT_S) if session_id else None
            capture_problems, capture_warnings = verify_capture(
                row, expected.name, args.max_cost, PROMPT_MARKER
            )
            problems.extend(capture_problems)
            warnings.extend(capture_warnings)
            cost = float(row.get("cost") or 0.0) if row else 0.0
        finally:
            server.stop()

        for warning in warnings:
            report.write(f"WARNING: {warning}")
        result = {
            "run": stamp,
            "status": "PASS" if not problems else "FAIL",
            "expected_model": expected.name,
            "served_model": served,
            "decision_reason": decision_reason,
            "session_id": session_id,
            "recorded_cost_usd": cost,
            "spend_cap_usd": args.max_cost,
            "problems": problems,
            "warnings": warnings,
            "run_log": str(run_dir / "live-smoke.log"),
            "data_dir": str(data_dir),
        }
        (run_dir / "live-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        if problems:
            report.write("FAIL: " + "; ".join(problems))
            report.write(f"recorded in {run_dir} (keep the run log as the outcome record)")
            return 1
        report.write(
            f"PASS: one real session routed through Shunt and captured. "
            f"model={served} reason={decision_reason} cost=${cost:.6f} session={session_id}"
        )
        report.write(f"recorded in {run_dir} (keep the run log as the outcome record)")
        return 0
    finally:
        report.close()


if __name__ == "__main__":
    sys.exit(main())
