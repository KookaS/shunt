"""Read-only install diagnosis for `shunt doctor` — what is live, what is inert, and why."""

# STRICTLY READ-ONLY, and non-spending. Every check inspects configuration, the filesystem
# and the environment; none calls a provider, none writes a row, and none constructs the real
# embedding model. That last one is not incidental: loading it downloads ~600MB, and the
# command a user runs to find out why nothing works must never be the command that fills
# their disk. Credential checks therefore test PRESENCE of the env var, never its validity —
# validating a key costs a request, and an invalid key is a different failure with a
# different fix.
#
# THE GOVERNING RULE, learned the hard way: this module must DIAGNOSE a broken install, never
# crash on one. Every acquisition that can fail on bad user input is wrapped, because the
# states that make it fail — malformed router.yaml, a model name absent from the registry, a
# work_dir that does not exist — are precisely the states a user runs `doctor` to understand.
# An uncaught traceback here is a worse failure than the misconfiguration it is reporting.
#
# The split mirrors `inspection.py`: this module assembles plain dataclasses and the CLI
# renders them, so nothing here prints and the report is equally usable as JSON.

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from shunt.proxy.context_transfer import CONTEXT_TRANSFER_SUMMARY

if TYPE_CHECKING:
    from pathlib import Path

    from shunt.db.loop_health import LoopHealth
    from shunt.models import ModelPool
    from shunt.router.policy import RouterPolicy

# A partial download is not a usable cache: fastembed writes into the directory before it
# finishes, so "the directory is non-empty" reported a truncated model as healthy.
_MIN_MODEL_BYTES = 1024 * 1024

# How each degraded cache state reads when the active strategy never embeds. The kNN wording
# ("the first routed request downloads it") is a promise about behaviour that simply does not
# happen under a fixed strategy, so it needs its own phrasing rather than a shared one.
_CACHE_WORDING: Final[dict[str, str]] = {
    "absent": "absent — nothing has downloaded it, and nothing will",
    "unreadable": "present but NOT READABLE by this user",
    "incomplete": "present but INCOMPLETE (no model file of plausible size)",
}


# The report's SHAPE is a contract, not a consequence of which branch assembled it. Every name
# below appears in every report, in this order, whatever failed — a `--json` consumer keying on
# `credentials` used to KeyError the moment models.yaml was malformed, which is the branch it
# most needs to read. A check the branch could not compute is present with status "skipped".
CHECK_ORDER: tuple[str, ...] = (
    "registry",
    "credentials",
    "embedder",
    "port",
    "escalation",
    "loop",
    "config",
)

_SKIPPED_DETAIL = "not evaluated — an earlier failure above made this check meaningless"

# Shared with inspection.py's provenance vocabulary on purpose: `shunt escalate` and `shunt
# doctor` must use the same words for the same layer, or the two screens read as disagreeing.
_BUILTIN_DEFAULT = "built-in default"


@dataclass(frozen=True)
class Check:
    """One diagnosis line: its verdict, and the detail a user acts on."""

    name: str
    detail: str
    ok: bool = True
    warn: bool = False
    skipped: bool = False

    @property
    def status(self) -> str:
        """One of: ok | warn | fail | skipped — the machine-readable form of the verdict."""
        if self.skipped:
            return "skipped"
        if not self.ok:
            return "fail"
        return "warn" if self.warn else "ok"


@dataclass(frozen=True)
class DoctorReport:
    """The whole diagnosis, ordered as a reader reasons about it."""

    checks: tuple[Check, ...]

    @property
    def serviceable(self) -> bool:
        """Whether the router could serve a request at all — the exit code's only input."""
        return all(check.ok for check in self.checks)


def _report(*checks: Check) -> DoctorReport:
    """Pad to the full check set in CHECK_ORDER — the shape must not vary by branch."""
    # A skipped check is ok=True on purpose: the real failure is already carried by the check
    # that broke, and counting its fallout twice would say "three things are wrong" about one.
    found = {check.name: check for check in checks}
    return DoctorReport(
        checks=tuple(
            found.get(name, Check(name, _SKIPPED_DETAIL, skipped=True)) for name in CHECK_ORDER
        )
    )


def _first_routed_model(policy: RouterPolicy, pool: ModelPool) -> tuple[str | None, str]:
    """The model a fresh install routes its FIRST request to, and how that was derived."""
    # Asks the real objects rather than restating their rules: a fixed strategy's pick is
    # deterministic and needs no neighbours, and a neighbour-consulting strategy has no verified
    # outcomes on a fresh install, so `RouterEngine.decide` cold-starts. Reimplementing either
    # here would be a second copy of the routing decision, free to drift from the one that runs.
    from shunt.router.cold_start import ColdStartStrategy
    from shunt.router.selection import SelectionRule
    from shunt.router.strategies.registry import build_strategy

    try:
        if _consults_neighbors(policy.strategy):
            return ColdStartStrategy().select(pool), "cold start"
        chosen, _rule = build_strategy(policy.strategy, SelectionRule()).select([], pool)
    except Exception:  # noqa: BLE001 (undecidable pick must not invent a verdict — abstain)
        return None, "undetermined"
    return chosen, f"strategy {policy.strategy}"


def _unreachable_first_pick(pool: ModelPool, first_pick: str | None, source: str) -> str | None:
    """The detail line when the first-routed model's key is unset, else None."""
    if first_pick is None:
        return None
    model = pool.get_model(first_pick)
    if model is None or not model.api_key_env_var:
        return None
    if os.environ.get(model.api_key_env_var, "").strip():
        return None
    # A missing key is NOT survivable by falling back: ProxyRouter sends a placeholder key and
    # `_route_with_fallback` re-raises on 401/403 without trying another candidate. So the very
    # first request dies, however many other models are authenticated.
    return (
        f"\nThe first request routes to {first_pick} ({source}) and {model.api_key_env_var} is "
        f"MISSING, so it fails on auth. A missing key is not recoverable by fallback — the "
        f"router re-raises on 401/403 rather than trying another model."
    )


def _credentials_check(pool: ModelPool, first_pick: str | None, pick_source: str) -> Check:
    """Which provider key each live model needs, whether it is set, and can the FIRST pick run."""
    # Grouped by ENV VAR rather than by provider: the variable is what the user sets, and two
    # providers sharing one variable would otherwise read as two separate problems.
    needed: dict[str, list[str]] = {}
    keyless: list[str] = []
    for name in pool.model_names():
        model = pool.get_model(name)
        if model is None:
            continue
        if model.api_key_env_var:
            needed.setdefault(model.api_key_env_var, []).append(name)
        else:
            # A local/ollama-style provider declares no key. Counting it as MISSING made doctor
            # exit 1 on a working install and print a bare ": MISSING" with no variable name.
            keyless.append(name)

    lines = [f"(no key required): {', '.join(keyless)}"] if keyless else []
    unlocked = len(keyless)
    for var, models in sorted(needed.items()):
        # PRESENCE only — never the value, and never a length or prefix, both of which leak.
        is_set = bool(os.environ.get(var, "").strip())
        unlocked += len(models) if is_set else 0
        state = "set" if is_set else "MISSING"
        lines.append(f"{var}: {state} — unlocks {len(models)} model(s): {', '.join(models)}")

    if not lines:
        return Check("credentials", "no live models to authenticate", ok=False)
    detail = "\n".join(lines)
    total = len(keyless) + sum(len(m) for m in needed.values())
    if unlocked == 0:
        return Check(
            "credentials",
            f"{detail}\nNo provider key is set, so every model is unreachable and the router "
            f"cannot serve a request. Set one of the variables above.",
            ok=False,
        )
    # The join doctor previously did not make: it held the active strategy AND the unset vars,
    # and still reported "serviceable" on an install whose very first request dies on auth.
    blocked = _unreachable_first_pick(pool, first_pick, pick_source)
    if blocked is not None:
        return Check("credentials", detail + blocked, ok=False)
    return Check("credentials", detail, warn=unlocked < total)


def _registry_check(pool: ModelPool) -> Check:
    """The model registry parsed, and how many models the live allow-list leaves routable."""
    # Health via the pool's OWN `is_healthy`, the same predicate the fixed strategies scan with,
    # so doctor and the router cannot disagree about what is routable. Note the ceiling honestly:
    # a breaker lives in the ModelPool INSTANCE, so this reads a freshly-built pool, not the
    # breakers of a server running in another process. It still catches the state that matters
    # here — a registry that resolves models none of which this process would route to.
    names = pool.model_names()
    if not names:
        return Check("registry", "the registry resolved NO live models", ok=False)
    healthy = [name for name in names if pool.is_healthy(name)]
    detail = f"{len(healthy)} of {len(names)} live model(s) routable: {', '.join(names)}"
    if not healthy:
        return Check(
            "registry",
            f"{detail}\nEvery model's circuit breaker is open, so the router has nothing to "
            f"route to and cannot serve a request.",
            ok=False,
        )
    broken = [name for name in names if name not in set(healthy)]
    if broken:
        return Check("registry", f"{detail}\nCircuit-broken: {', '.join(broken)}", warn=True)
    return Check("registry", detail)


def _loop_check(pool: ModelPool) -> Check:
    """Label coverage and routing spread — whether the learning loop has anything to learn from."""
    # §2's second named reuse. Deliberately does NOT construct an OutcomeStore when the database
    # is absent: the constructor os.makedirs()es and migrates one into existence, and doctor must
    # not manufacture the state it is reporting on. When the file already exists, the migrations
    # run are exactly the ones `shunt start` runs against it, and are no-ops on a current schema.
    from shunt.db.store import OutcomeStore, default_db_path
    from shunt.router.inspection import loop_health_for

    path = default_db_path()
    if not os.path.exists(path):
        return Check(
            "loop",
            f"no outcome corpus at {path} yet — nothing has been routed, so there are no "
            f"verified outcomes to learn from and kNN routes cold.",
            warn=True,
        )
    try:
        store = OutcomeStore(db_path=path)
    except Exception as exc:  # noqa: BLE001 (an unopenable store is a finding, not a crash)
        return Check("loop", f"the outcome database at {path} could not be opened: {exc}", ok=False)
    try:
        health = loop_health_for(store, pool)
    except Exception as exc:  # noqa: BLE001
        return Check("loop", f"loop health could not be computed: {exc}", ok=False)
    finally:
        store.close()
    return _loop_detail(health)


def _loop_detail(health: LoopHealth) -> Check:
    """Render the loop-health object — WARN on an empty or collapsed loop, never a hard fail."""
    coverage, collapse = health.label_coverage, health.routing_collapse
    lines = [
        f"{coverage.verified_labeled} verified / {coverage.any_labeled} labeled of "
        f"{coverage.eligible_sessions} eligible session(s) ({coverage.total_sessions} routed)",
        f"last {collapse.window_size} decision(s) spread over {collapse.distinct_models} "
        f"model(s), frontier share {collapse.frontier_share:.2f}",
    ]
    if health.support_deficient_models:
        lines.append(f"under-explored: {', '.join(health.support_deficient_models)}")
    if collapse.alarm:
        lines.append("COLLAPSE ALARM — routing has concentrated onto the expensive tail.")
        return Check("loop", "\n".join(lines), warn=True)
    if coverage.any_labeled == 0:
        lines.append(
            "Nothing is labeled, so kNN has no signal — `shunt flag` or a verified escalation "
            "outcome is what supplies it."
        )
        return Check("loop", "\n".join(lines), warn=True)
    return Check("loop", "\n".join(lines))


def _cache_state(cache: str) -> str:
    """One of: absent | unreadable | incomplete | complete. Never downloads, never raises."""
    # Tri-state, not a (present, complete) pair. Collapsing "does not exist" with "exists but is
    # unreadable" made doctor tell a user with a perfectly good 600MB cache that it was NOT
    # cached and would download on first request — three false clauses in one line, on the very
    # install state they ran the command to understand.
    if not os.path.isdir(cache):
        return "absent"
    try:
        with os.scandir(cache) as entries:
            if not any(entries):
                return "absent"
    except OSError:
        return "unreadable"

    unreadable = False

    def _note(_exc: OSError) -> None:
        # os.walk swallows per-directory errors by default, so a good model file inside an
        # unreadable subdirectory read as "incomplete" — advice to delete a valid cache.
        nonlocal unreadable
        unreadable = True

    for root, _dirs, files in os.walk(cache, onerror=_note):
        for name in files:
            if not name.endswith(".onnx"):
                continue
            try:
                if os.path.getsize(os.path.join(root, name)) >= _MIN_MODEL_BYTES:
                    return "complete"
            except OSError:
                unreadable = True
    return "unreadable" if unreadable else "incomplete"


def _consults_neighbors(strategy: str) -> bool:
    """Whether the ACTIVE strategy embeds at all — the engine's own predicate, not a name list."""
    # `RouterEngine.decide` branches on exactly this: a strategy whose `consults_neighbors` is
    # False goes through `_decide_fixed`, which is "no cold-start, embedding, or query". Asking
    # the strategy object rather than hardcoding {always_cheap, always_frontier} means a fourth
    # live strategy inherits the right verdict instead of silently getting the wrong one.
    # SelectionRule() is a pure knob-holder — constructing one costs nothing and touches nothing.
    from shunt.router.selection import SelectionRule
    from shunt.router.strategies.registry import build_strategy

    try:
        return bool(build_strategy(strategy, SelectionRule()).consults_neighbors)
    except Exception:  # noqa: BLE001 (an unbuildable strategy must not decide "no embedder needed")
        return True


def _embedder_unused_note(strategy: str) -> str:
    return (
        f"The active strategy {strategy} routes from the model pool alone and never embeds "
        f"(see RouterEngine._decide_fixed), so no routed request reads this cache today."
    )


def _embedder_check(strategy: str, needs_neighbors: bool) -> Check:
    """The active embedding model, its weights on disk, and whether this strategy needs them."""
    # Deliberately does NOT construct `Embedder`: that triggers the download this check exists
    # to warn about. Reads the config and looks at the cache directory instead.
    #
    # STRATEGY-AWARE by necessity, not politeness. `DoctorReport.serviceable` is `all(ok)`, so an
    # ok=False here sets the exit code — and under a fixed strategy an unreadable cache cannot
    # stop a single request being served. Reporting it as a FAIL made doctor print "Router CANNOT
    # serve a request" and exit 1 on an install that works, inverting the exit-code contract on
    # the very command whose job is that verdict. Same contract the module already applies to
    # inert escalation, which is degraded-but-serving and therefore a WARN.
    from shunt.router.embedder import embedding_cache_dir
    from shunt.router.embedding_config import load_embedding_config

    try:
        config = load_embedding_config()
    except Exception as exc:  # noqa: BLE001 (a config defect must report, never crash doctor)
        return Check("embedder", f"could not read embedding.yaml: {exc}", ok=False)
    try:
        # resolve_active, not the `active` key directly: SHUNT_EMBEDDER_MODEL overrides it,
        # and reporting the file's value while the env runs a different model is the drift
        # this command is supposed to catch rather than reproduce. Its failure is reported
        # separately because blaming the FILE for a bad env var misdirects the fix.
        model_name = config.resolve_active(os.environ).repo
        cache = embedding_cache_dir(config.cache_dir)
    except Exception as exc:  # noqa: BLE001
        return Check("embedder", f"active embedding model could not be resolved: {exc}", ok=False)

    state = _cache_state(cache)
    if state == "complete":
        return Check("embedder", f"{model_name} — weights cached in {cache}")
    if not needs_neighbors:
        # Every remaining state is a degraded cache, and none of them can break a fixed
        # strategy — WARN so the operator still sees it before switching to knn_semantic_cascade,
        # ok so the
        # exit code keeps telling the truth about whether a request can be served.
        return Check(
            "embedder",
            f"{model_name} — the cache at {cache} is {_CACHE_WORDING[state]}. "
            f"{_embedder_unused_note(strategy)} Fix it before switching to a "
            f"neighbour-consulting strategy.",
            warn=True,
        )
    if state == "unreadable":
        return Check(
            "embedder",
            f"{model_name} — the cache at {cache} exists but is NOT READABLE by this user. "
            f"The first routed request will fail rather than re-download; fix the permissions.",
            ok=False,
        )
    if state == "incomplete":
        return Check(
            "embedder",
            f"{model_name} — cache at {cache} is present but INCOMPLETE (no model file of "
            f"plausible size). A partial download fails at the first routed request; delete "
            f"the directory and let it re-download.",
            warn=True,
        )
    return Check(
        "embedder",
        f"{model_name} — NOT cached in {cache}. The first routed request downloads it "
        f"(~600MB, CPU-only). This is expected on a fresh install, not an error.",
        warn=True,
    )


def _port_check() -> Check:
    """Whether the address `shunt start` binds is already taken."""
    from shunt.bind import resolve_bind

    try:
        host, port = resolve_bind()
    except ValueError as exc:
        return Check("port", str(exc), ok=False)

    if port == 0:
        # Port 0 asks the kernel for an ephemeral port at bind time, so there is no address to
        # probe and none to hand a client. "127.0.0.1:0 is free" was true of nothing.
        return Check(
            "port",
            f"{host}:0 — port 0 asks the kernel for an ephemeral port at bind time, so there "
            f"is no fixed address to check and none to point a tool at. Set a real "
            f"SHUNT_PORT before configuring a client.",
            warn=True,
        )

    # Resolve rather than assume AF_INET: an AF_INET-only probe reported an IPv6-occupied port
    # as free (after which `shunt start` fails to bind), and raised gaierror on `::1`.
    # An empty SHUNT_HOST binds every interface under uvicorn exactly as 0.0.0.0 does, so it must
    # probe the same way — otherwise doctor declares a working config unserviceable.
    probe_host = "127.0.0.1" if host in ("", "0.0.0.0") else host  # noqa: S104 (compare, not bind)
    try:
        infos = socket.getaddrinfo(probe_host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        # UnicodeError (a ValueError, NOT an OSError) is what getaddrinfo raises for a hostname
        # label over 63 bytes — a pasted or garbled SHUNT_HOST, which is a plausible broken state.
        return Check("port", f"{host!r} does not resolve: {exc}", ok=False)

    # Every resolved address, not just the first: a host can resolve to both A and AAAA, and
    # checking one of them is how an occupied port got reported as free.
    in_use = False
    for family, socktype, proto, _canon, sockaddr in infos:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(0.25)
                in_use = sock.connect_ex(sockaddr) == 0
        except OSError:
            continue
        if in_use:
            break

    if in_use:
        # Not a hard failure: the most likely cause is that shunt is already running, which
        # is the state a user checking on their install is often in.
        return Check(
            "port",
            f"{host}:{port} is already accepting connections — shunt may already be running, "
            f"or another service holds the port.",
            warn=True,
        )
    return Check("port", f"{host}:{port} is free")


def _escalation_check(policy: RouterPolicy, work_dir: str | None) -> Check:
    """The ladder's state, plus the context-transfer disclosure when one is armed."""
    check = _escalation_state_check(policy, work_dir)
    note = _context_transfer_note(policy)
    if note is None:
        return check
    # Appended rather than reported as its own check: CHECK_ORDER is a contract a `--json`
    # consumer keys on, and this is a property of the escalation the check already describes.
    return Check(check.name, f"{check.detail}\n{note}", ok=check.ok, warn=True)


def _context_transfer_note(policy: RouterPolicy) -> str | None:
    """The one line a user must see before trusting what the escalated model was told."""
    if policy.escalation.context_transfer != CONTEXT_TRANSFER_SUMMARY:
        return None
    writer = policy.escalation.context_transfer_model or "the outgoing pre-escalation model"
    return (
        f"context_transfer: SUMMARY — on the first turn after an escalation shunt replaces "
        f"the prior conversation with a note written by {writer} (max "
        f"{policy.escalation.context_transfer_max_tokens} tokens), then resends it unchanged. "
        f"THE MODEL DOES NOT SEE WHAT YOU SEE. Failures degrade to 'full' (nothing dropped)."
    )


def _escalation_state_check(policy: RouterPolicy, work_dir: str | None) -> Check:
    """ENABLED and ARMED are different states — the distinction this command exists for."""
    # A user learns escalation is inert only from a boot warning in the logs today, and the
    # two ways it goes inert are different: no repo resolved at all, or a repo that resolves
    # but declares no test framework. The second is invisible to the launch-dir validator,
    # because an explicit capture.work_dir / SHUNT_WORK_DIR is never framework-checked.
    from shunt.verifiers.tier2 import detect_framework

    if not policy.escalation.enabled:
        return Check("escalation", "disabled in router.yaml (escalation.enabled: false)")
    if work_dir is None:
        return Check(
            "escalation",
            "enabled but INERT — no work_dir resolved. Its only signal is re-running a "
            "repo's own tests off the wire, and there is no repo to run. Launch shunt "
            "inside a git repo, set capture.work_dir / SHUNT_WORK_DIR, or pass --work-dir.",
            warn=True,
        )
    if not os.path.isdir(work_dir):
        # detect_framework os.listdir()s the path, so a missing or non-directory work_dir
        # raised FileNotFoundError / NotADirectoryError straight out of the command.
        return Check(
            "escalation",
            f"enabled but INERT — {work_dir} is not a directory, so nothing can be verified "
            f"there. Check capture.work_dir / SHUNT_WORK_DIR / --work-dir.",
            warn=True,
        )
    try:
        framework = detect_framework(work_dir)
    except OSError as exc:
        return Check("escalation", f"enabled but INERT — cannot read {work_dir}: {exc}", warn=True)
    if framework is None:
        return Check(
            "escalation",
            f"enabled but INERT — {work_dir} resolved, but it declares no test framework "
            f"this verifier recognises, so no outcome is ever verified there.",
            warn=True,
        )
    return Check("escalation", f"ARMED on {work_dir} (detected {framework})")


def _file_sets_strategy(policy_path: Path) -> bool:
    """Whether router.yaml actually names `strategy` — presence, not equality-to-default."""
    # Without this the source column claimed the file set `strategy` whenever a file existed,
    # sending a reader to hunt for a key that was never there.
    from shunt.models.config import strict_yaml_load

    try:
        data = strict_yaml_load(policy_path.read_text())
    except (OSError, ValueError):
        return False
    # strict_yaml_load is TYPED -> dict but returns whatever YAML parsed: None for an empty or
    # comment-only file, False for `false`. parse_router_policy treats those as "use defaults",
    # so the policy loads fine and the crash lands here — past every guard, in a render helper.
    if not isinstance(data, dict):
        return False
    section = data.get("router", data)
    return isinstance(section, dict) and "strategy" in section


def _strategy_source(policy_path: Path | None) -> str:
    """Which layer supplied `strategy` — env wins over file, file over the built-in default."""
    if os.environ.get("SHUNT_ROUTER_STRATEGY"):
        return "SHUNT_ROUTER_STRATEGY (env overrides the file)"
    if policy_path is not None and _file_sets_strategy(policy_path):
        return str(policy_path)
    return _BUILTIN_DEFAULT


def _packaged_path() -> Path | None:
    """The router.yaml shipped INSIDE the wheel, or None if it cannot be located."""
    from shunt.router.policy import packaged_policy_path

    try:
        return packaged_policy_path()
    except Exception:  # noqa: BLE001 (a missing package resource must not crash the report)
        return None


def _source_label(source: str, packaged: Path | None) -> str:
    """Relabel the PACKAGED config as a built-in default — the user never wrote that file."""
    # `resolved_policy_path()` falls back to the packaged router.yaml, so on a stock install
    # every provenance cell named a path inside site-packages and "built-in default" never
    # appeared at all. §2 asks doctor which values are defaults versus overridden; attributing
    # all of them to a file the user has never opened answers "all overridden", which is false
    # and sends them editing a file inside the wheel. A USER file is still named verbatim.
    if packaged is not None and source == str(packaged):
        return f"{_BUILTIN_DEFAULT} (shipped router.yaml)"
    return source


def _config_check(policy: RouterPolicy, policy_path: Path | None) -> Check:
    """The effective routing/escalation knobs and which layer supplied each value."""
    # Provenance for the escalation block comes from inspection.py's shared builder, so `shunt
    # escalate` and `shunt doctor` cannot disagree about which file set a knob, and a knob
    # added there appears here automatically. `strategy` is handled separately because it is
    # the one value carrying an ENV override.
    from shunt.router.inspection import escalation_config_items

    packaged = _packaged_path()
    strategy_source = _source_label(_strategy_source(policy_path), packaged)
    lines = [f"{'strategy':<20} {policy.strategy!s:<16} ({strategy_source})"]
    lines += [
        f"{item.key:<20} {item.value!s:<16} ({_source_label(item.source, packaged)})"
        for item in escalation_config_items(policy, policy_path)
    ]
    return Check("config", "\n".join(lines))


def _load_policy() -> tuple[RouterPolicy | None, Path | None, Check | None]:
    """The effective policy (env overlay applied), or a Check explaining why it would not load."""
    from shunt.router.policy import apply_env_overrides, load_router_policy, resolved_policy_path

    try:
        # apply_env_overrides is what the SERVER applies; load_router_policy alone returns the
        # file's view, so doctor used to report a strategy the server would never run.
        policy = apply_env_overrides(load_router_policy())
        return policy, resolved_policy_path(), None
    except Exception as exc:  # noqa: BLE001 (report the config defect, never crash on it)
        return None, None, Check("config", f"router policy could not be loaded: {exc}", ok=False)


def _load_pool(policy: RouterPolicy) -> tuple[ModelPool | None, Check | None]:
    """The live model pool, or a Check carrying the registry error verbatim."""
    from shunt.models import ModelPool

    try:
        pool = ModelPool(config_path=os.environ.get("SHUNT_MODEL_CONFIG_PATH"))
        # restrict_to_live raises a well-worded error naming the unknown model; that message IS
        # the diagnosis, and it used to reach the user only as a traceback.
        pool.restrict_to_live(policy.models)
    except Exception as exc:  # noqa: BLE001
        return None, Check("registry", f"model registry could not be loaded: {exc}", ok=False)
    return pool, None


def doctor_report(work_dir: str | None, launch_dir: str) -> DoctorReport:
    """Assemble every check. Never spends, never downloads, never mutates, never raises."""
    from shunt.router.inspection import resolve_inspection_work_dir

    policy, policy_path, policy_error = _load_policy()
    if policy is None or policy_error is not None:
        # Nothing downstream is meaningful without a policy, but the embedder and port checks
        # are independent of it and still worth reporting beside the failure. With no policy
        # there is no strategy to be aware of, so the embedder is judged conservatively (as if
        # neighbours were needed) — the same direction the unbuildable-strategy guard takes.
        failure = policy_error or Check("config", "router policy unavailable", ok=False)
        return _report(failure, _embedder_check("unknown", needs_neighbors=True), _port_check())

    needs_neighbors = _consults_neighbors(policy.strategy)
    embedder = _embedder_check(policy.strategy, needs_neighbors)

    pool, pool_error = _load_pool(policy)
    if pool is None or pool_error is not None:
        failure = pool_error or Check("registry", "model registry unavailable", ok=False)
        return _report(failure, embedder, _port_check(), _config_check(policy, policy_path))

    try:
        resolved, _source = resolve_inspection_work_dir(policy, work_dir, launch_dir)
    except (OSError, ValueError):
        # The resolver realpath()s the input; a hostile work_dir must not take the whole report
        # down. ValueError as well as OSError — a NUL byte in the path raises the former.
        resolved = work_dir

    first_pick, pick_source = _first_routed_model(policy, pool)
    return _report(
        _registry_check(pool),
        _credentials_check(pool, first_pick, pick_source),
        embedder,
        _port_check(),
        _escalation_check(policy, resolved),
        _loop_check(pool),
        _config_check(policy, policy_path),
    )
