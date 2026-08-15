"""FastAPI proxy server — OpenAI + Anthropic compatible /v1/ endpoints."""

from __future__ import annotations

import functools
import logging
import os
import signal
import sys
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import FrameType
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from shunt.capture import CaptureCoordinator, CaptureWorker, RefitScheduler, WorkDirResolver
from shunt.capture.trajectory import TrajectoryRecorder
from shunt.capture.trajectory_store import LiveTrajectorySink, load_key, resolve_live_dir
from shunt.db.outcome_index import OutcomeIndexAdapter
from shunt.db.store import OutcomeStore, SessionProvenance
from shunt.log_config import configure_logging
from shunt.models import ModelPool
from shunt.proxy.redaction import header_safe, redact_secrets
from shunt.proxy.router import BudgetExceededError, ProxyRouter, UpstreamError
from shunt.router.cold_start import ColdStartStrategy
from shunt.router.embedder import Embedder, embedding_cache_dir
from shunt.router.engine import RouterEngine
from shunt.router.inspection import loop_health_for
from shunt.router.policy import (
    CapturePolicy,
    ExplorationPolicy,
    RouterPolicy,
    apply_env_overrides,
    load_router_policy,
)
from shunt.router.selection import SelectionRule
from shunt.router.strategies import EXPLORATORY_STRATEGIES, build_strategy
from shunt.session import Session, SessionManager
from shunt.verifiers import AutoDetectVerifier
from shunt.verifiers.rerun import RerunConfirmingVerifier

logger = logging.getLogger(__name__)

_INACTIVITY_TIMEOUT = int(os.environ.get("SHUNT_SESSION_INACTIVITY_TIMEOUT", "900"))
_GRACE_PERIOD = int(os.environ.get("SHUNT_SESSION_GRACE_PERIOD", "120"))
_RETRY_COUNT = int(os.environ.get("SHUNT_RETRY_COUNT", "3"))
_MODEL_CONFIG_PATH = os.environ.get("SHUNT_MODEL_CONFIG_PATH")


def _model_inventory(model_pool: ModelPool) -> str:
    """`rank:name` for every routable model, cheapest (lowest rank) first."""
    listed = [f"{rank}:{model.name}" for rank, model in enumerate(model_pool.ranked_models())]
    return ", ".join(listed) or "(none)"


def _log_config_disclosure(
    policy: RouterPolicy,
    model_pool: ModelPool,
    embedder: Embedder | None = None,
    fingerprint_trusted: bool | None = None,
) -> None:
    """Print the loaded configuration at startup, so what is in force is never a guess."""
    # Names and choices only. Never an api_key_env_var VALUE, never a resolved key —
    # this line goes to container logs, which are routinely pasted into issues.
    logger.info("Shunt config | strategy=%s", policy.strategy)
    logger.info(
        "Shunt config | knn: k=%d success_rate_threshold=%.2f min_samples=%d",
        policy.policy.k,
        policy.policy.success_rate_threshold,
        policy.policy.min_samples,
    )
    logger.info(
        "Shunt config | exploration: enabled=%s budget_frac=%.2f conservative_alpha=%.2f "
        "prior=Beta(%.2f,%.2f) propensity_mc_samples=%d",
        policy.exploration.enabled,
        policy.exploration.explore_budget_frac,
        policy.exploration.conservative_alpha,
        policy.exploration.prior_alpha,
        policy.exploration.prior_beta,
        policy.exploration.propensity_mc_samples,
    )
    logger.info("Shunt config | models: %s", _model_inventory(model_pool))
    logger.info(
        "Shunt config | session: inactivity_timeout=%ds grace_period=%ds retry_count=%d",
        _INACTIVITY_TIMEOUT,
        _GRACE_PERIOD,
        _RETRY_COUNT,
    )
    cap = (
        "unlimited"
        if policy.budget.max_spend_usd is None
        else f"${policy.budget.max_spend_usd:.6f}"
    )
    logger.info("Shunt config | budget: max_spend_usd=%s", cap)
    if embedder is not None:
        # The RESOLVED embedder (from embedding.yaml, env applied), not the raw env var —
        # plus whether its fingerprint matches the stored corpus space.
        status = (
            "unknown"
            if fingerprint_trusted is None
            else ("trusted" if fingerprint_trusted else "STALE→reindex")
        )
        logger.info(
            "Shunt config | embedder=%s max_chars=%d fingerprint=%s data_dir=%s",
            embedder.model_name,
            embedder.max_chars,
            status,
            os.environ.get("SHUNT_DATA_DIR", "(default)"),
        )
    else:
        logger.info(
            "Shunt config | embedder=%s max_chars=%s data_dir=%s",
            os.environ.get("SHUNT_EMBEDDER_MODEL", "(default)"),
            os.environ.get("SHUNT_EMBED_MAX_CHARS", "(default)"),
            os.environ.get("SHUNT_DATA_DIR", "(default)"),
        )


def _build_work_dir_resolver(policy: RouterPolicy, launch_dir: str) -> WorkDirResolver:
    """The one resolver shared by the engine's decide-side key and the capture side."""
    return WorkDirResolver.from_config(
        work_dir=policy.capture.work_dir,
        work_dirs=policy.capture.work_dirs,
        launch_dir=launch_dir,
        trust_launch_dir=policy.capture.trust_launch_dir,
    )


def _capture_auto_configured(resolver: WorkDirResolver) -> bool:
    """True when off-wire capture can auto-record outcomes (a work_dir is resolvable).

    Asks the RESOLVER, not the policy: the launch-directory layer is validated at runtime,
    so config alone cannot say whether capture armed, and a disclosure that guesses lies.
    """
    return resolver.armed_layer() is not None


def _log_exploration_disclosure(
    policy: RouterPolicy, resolver: WorkDirResolver, *, cold_start_active: bool
) -> None:
    """Loud one-line startup disclosure of the exploration state (least-surprise)."""
    # Must not promise spending that cannot happen. The gate is COLD-START, not "any
    # outcome exists": while cold-start is active the engine returns before it can
    # explore, so a rig with 1 of the 20 outcomes it needs is still completely inert.
    # Keying this on `verified_outcomes > 0` claimed a "~1.4x envelope" after the very
    # first flagged session — observed in the local container, and wrong.
    if _effective_exploration(policy) is None:
        logger.info("Shunt exploration is OFF: routing exploits the current best model only.")
    elif cold_start_active:
        logger.warning(
            "Shunt exploration is enabled but INERT: not enough verified outcomes yet, so "
            "the router cold-starts every session to the cheap default and will not "
            "explore. It costs nothing extra today. Record outcomes with `shunt flag`.",
        )
    else:
        logger.warning(
            "Shunt exploration is ON (~1.4x cost envelope, budget_frac=%.2f): the router "
            "will occasionally try cheaper/alternative models to learn from verified "
            "outcomes. Disable with `shunt start --no-explore` or SHUNT_EXPLORATION_ENABLED=0.",
            policy.exploration.explore_budget_frac,
        )
        # Say which HALF is running. The conservative gate only permits a downshift once
        # it has banked slack from verified downshift successes, and it banks that slack in
        # this process's memory. Whether it can open depends on the outcome-write path:
        # auto-capture (a configured work_dir) feeds verified downshift outcomes back
        # in-process at session close, so the gate CAN open; with manual-only `shunt flag`
        # (a separate CLI process writing SQLite) slack stays 0 and a downshift never fires.
        if _capture_auto_configured(resolver):
            logger.warning(
                "Shunt downshift exploration is ARMED (conservative_alpha=%.2f): the "
                "gate banks slack from auto-captured verified downshift outcomes at session "
                "close, so it can open within this run and the router may try cheaper models.",
                policy.exploration.conservative_alpha,
            )
        else:
            logger.warning(
                "Shunt will only explore UPWARD (conservative_alpha=%.2f): no work_dir is "
                "configured, so outcomes are recorded only via the separate `shunt flag` CLI "
                "and never feed the in-process gate — so it cannot open and trying a cheaper "
                "model is off, however the alpha is tuned. Set a work_dir to arm it.",
                policy.exploration.conservative_alpha,
            )


def _build_engine(  # noqa: PLR0913 (engine-composition wiring from the resolved policy)
    model_pool: ModelPool,
    session_manager: SessionManager,
    outcome_store: OutcomeStore,
    policy: RouterPolicy,
    embedder: Embedder | None = None,
    trust_neighbors: bool = True,
    resolver: WorkDirResolver | None = None,
) -> RouterEngine:
    """Compose the live RouterEngine from the resolved router policy."""
    # KnnPolicy is the single source of the knn knobs: threshold + min_samples feed the
    # SelectionRule (used by both the knn strategy and the exploration threshold); k feeds
    # the neighbor query. The registry maps router.strategy → the active strategy.
    # No resolver ⇒ an empty one (no work_dir): escalation stays inert, its precondition unmet.
    resolver = resolver or WorkDirResolver()
    selection_rule = SelectionRule(
        min_success_rate=policy.policy.success_rate_threshold,
        min_samples=policy.policy.min_samples,
    )
    strategy = build_strategy(policy.strategy, selection_rule)
    return RouterEngine(
        model_pool=model_pool,
        session_manager=session_manager,
        outcome_index=OutcomeIndexAdapter(outcome_store),
        embedder=embedder or Embedder(),
        selection_rule=selection_rule,
        strategy=strategy,
        neighbor_k=policy.policy.k,
        exploration=_effective_exploration(policy),
        trust_neighbors=trust_neighbors,
        escalation=policy.escalation.to_config(),
        # The escalation task key is the REPO (resolved work_dir), same seam the capture
        # side keys against — so decide() and capture agree on the task, never the client. The
        # cast bridges the engine's `object` session (SessionManagerProtocol.get_session) to
        # WorkDirResolver.resolve's Session — at runtime get_session always returns a Session.
        task_key_resolver=cast("Callable[[object], str | None]", resolver.resolve),
        # The routing-collapse guard, read live at escalation time. Only invoked when
        # escalation is enabled and a recurrence is otherwise due.
        loop_health_alarm=_build_collapse_alarm(outcome_store, model_pool),
    )


def _build_collapse_alarm(outcome_store: OutcomeStore, model_pool: ModelPool) -> Callable[[], bool]:
    """A live routing-collapse alarm probe: True when the choice distribution has collapsed."""

    def _alarm() -> bool:
        return loop_health_for(outcome_store, model_pool).routing_collapse.alarm

    return _alarm


def _resolve_embedding_trust(embedder: Embedder, outcome_store: OutcomeStore) -> bool:
    """Trust kNN only when the configured fingerprint matches the stored one, or the corpus
    is genuinely fresh (no fingerprint AND no embeddings)."""
    # A mismatch, or a legacy pre-fingerprint corpus that already holds embeddings of an
    # unknown space, refuses neighbours (cold-start) until `shunt reindex` re-stamps.
    configured = embedder.fingerprint()
    stored = outcome_store.load_embedding_fingerprint()
    if stored is None:
        # No fingerprint recorded. Adopt + trust ONLY for a truly empty corpus. A legacy DB
        # that already holds embeddings from an unrecorded (possibly custom/foreign) embedder
        # must NOT be trusted blindly — same dims but a different space silently mis-routes.
        # Refuse until `shunt reindex` re-embeds with the configured embedder and stamps a
        # fingerprint (the same remedy as a mismatch below).
        if outcome_store.get_labeled_embeddings():
            logger.warning(
                "Shunt corpus holds embeddings but no recorded fingerprint (a pre-fingerprint "
                "DB). Serving cold-start (no kNN) to avoid routing on possibly foreign-space "
                "neighbours. Run `shunt reindex` (server stopped) to re-embed the corpus with "
                "the configured embedder %s and stamp its fingerprint.",
                configured,
            )
            return False
        outcome_store.save_embedding_fingerprint(configured)
        return True
    if stored == configured:
        return True
    logger.warning(
        "Shunt embedding space MISMATCH: the stored corpus fingerprint %s differs from the "
        "configured one %s. Serving cold-start (no kNN) to avoid routing on foreign-space "
        "neighbours. Run `shunt reindex` (server stopped) to re-embed the corpus.",
        stored,
        configured,
    )
    return False


def _log_missing_credentials(model_pool: ModelPool) -> None:
    """Name the unset key variables at startup rather than at the first 401."""
    # Without this the only symptom is the provider's own "Incorrect API key" text,
    # which never names the variable the operator actually has to set.
    missing: dict[str, list[str]] = {}
    for name in model_pool.model_names():
        model = model_pool.get_model(name)
        if model is None:
            continue
        if not os.environ.get(model.api_key_env_var):
            missing.setdefault(model.api_key_env_var, []).append(name)
    for env_var, models in sorted(missing.items()):
        logger.warning(
            "Shunt config | %s is NOT set — these models cannot be routed to: %s",
            env_var,
            ", ".join(sorted(models)),
        )
    if not missing:
        logger.info("Shunt config | credentials present for every configured model")


def _warm_embedder_in_background(engine: RouterEngine) -> None:
    """Start loading the embedding model now rather than on the first request."""
    # In a thread on purpose: the first load downloads ~600MB, and blocking startup on
    # it would mean no network → the server never starts at all, instead of starting
    # and reporting a clear error. Health stays answerable throughout.
    if not engine.needs_embeddings:
        # Nothing to warm, and nothing to report as "ready": the ~600MB download, its disk
        # write and its resident memory are all skipped for a strategy that never embeds.
        logger.info(
            "Shunt config | embedding model NOT loaded: the active strategy does not consult "
            "neighbours, so no embedding is ever computed (~600MB download/load skipped)."
        )
        return

    def _warm() -> None:
        try:
            engine.warm()
        except Exception as exc:
            logger.warning("Embedding model not ready: %s", exc)
        else:
            logger.info("Embedding model ready (cache: %s)", embedding_cache_dir())

    threading.Thread(target=_warm, name="shunt-embedder-warm", daemon=True).start()


def _effective_exploration(policy: RouterPolicy) -> ExplorationPolicy | None:
    """Exploration only applies to exploratory (knn) strategies; fixed ones never explore."""
    if not policy.exploration.enabled:
        return None
    if policy.strategy not in EXPLORATORY_STRATEGIES:
        return None
    return policy.exploration


def _log_capture_disclosure(policy: RouterPolicy, resolver: WorkDirResolver) -> None:
    """Say whether automatic outcome capture can run, or is manual-only (least-surprise)."""
    # Off-wire capture needs a repo root: an operator-configured one, or Shunt's own
    # validated launch directory. With none, no session can auto-label — the loop is inert
    # and `shunt flag` is the only outcome-write path. Naming WHICH layer armed it matters:
    # "capture is ON" alone leaves an operator unable to tell a configured repo from the
    # working directory they happened to launch in — whose tests are about to be executed.
    layer = resolver.armed_layer()
    if layer is not None:
        logger.info(
            "Shunt capture is ON via %s: verified outcomes are recorded automatically at "
            "session close by re-running THAT repo's tests off the wire — which executes the "
            "tree's own test code (conftest.py, build.rs, npm scripts).",
            layer,
        )
    else:
        logger.warning(
            "Shunt capture is MANUAL-ONLY: no work_dir resolved (SHUNT_WORK_DIR / "
            "capture.work_dir / --work-dir, and the launch directory is not a git repo with "
            "a detectable test framework, or capture.trust_launch_dir is false), so no "
            "session is labelled automatically. Record outcomes with `shunt flag`, or set "
            "a work_dir."
        )
    # Escalation's ONLY signal is a verified failure, which needs the repo above. Escalation
    # ships ON, so enabled-without-a-work_dir is the common default state — it must not brick
    # the router, but it must NEVER be silent: without this warning an operator would think
    # escalation is working while it can fire on nothing.
    if policy.escalation.enabled and not _capture_auto_configured(resolver):
        logger.warning(
            "Auto-escalation is ENABLED but capture resolved no repo, so it will NOT fire: "
            "it re-runs the REPO's tests at session close and has no repo to verify. Launch "
            "shunt from inside your repo, or set SHUNT_WORK_DIR / capture.work_dir / "
            "--work-dir to arm it. (docs/escalation.md)"
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Sampled FIRST, before the embedder-warm thread or any worker exists: a chdir by any
    # of them would otherwise silently repoint the launch-directory layer at another tree.
    launch_dir = os.getcwd()
    session_manager = SessionManager(
        inactivity_timeout=_INACTIVITY_TIMEOUT,
        grace_period=_GRACE_PERIOD,
    )
    model_pool = ModelPool(config_path=_MODEL_CONFIG_PATH)
    outcome_store = OutcomeStore()
    policy = apply_env_overrides(load_router_policy())
    model_pool.restrict_to_live(policy.models)
    _index = OutcomeIndexAdapter(outcome_store)
    # Build the embedder once; its fingerprint gates whether the stored corpus (a
    # possibly-foreign embedding space) may be trusted for kNN. Decided here, where both
    # fingerprints are visible, and injected into the engine as a boolean.
    embedder = Embedder()
    trust_neighbors = _resolve_embedding_trust(embedder, outcome_store)
    _log_config_disclosure(policy, model_pool, embedder, fingerprint_trusted=trust_neighbors)
    _log_missing_credentials(model_pool)
    # One resolver, shared by the engine (decide-side task key) and the capture worker
    # (capture-side task key) so both key escalation on the SAME repo. Built BEFORE the
    # exploration disclosure, which must report whether capture actually armed.
    resolver = _build_work_dir_resolver(policy, launch_dir)
    _log_exploration_disclosure(
        policy,
        resolver,
        cold_start_active=ColdStartStrategy().is_active_effective(
            _index.effective_labeled(), _index.effective_tier2()
        ),
    )
    engine = _build_engine(
        model_pool,
        session_manager,
        outcome_store,
        policy,
        embedder=embedder,
        trust_neighbors=trust_neighbors,
        resolver=resolver,
    )
    # Reload the exploration budget cap + gate slack persisted by the prior run, so a
    # restart does not silently reset the cost cap and downshift evidence to zero. Same for
    # the escalation failure log + decision counters — a restart must not wipe them.
    engine.restore_exploration_state(outcome_store.load_router_state())
    engine.restore_escalation_state(outcome_store.load_escalation_state())
    _warm_embedder_in_background(engine)
    worker = _wire_capture(session_manager, outcome_store, policy, engine, resolver)
    router = ProxyRouter(
        model_pool=model_pool,
        session_manager=session_manager,
        retry_count=_RETRY_COUNT,
        engine=engine,
        max_spend_usd=policy.budget.max_spend_usd,
    )
    app.state.session_manager = session_manager
    app.state.model_pool = model_pool
    app.state.router = router
    app.state.outcome_store = outcome_store
    yield
    worker.stop()
    _persist_router_state(engine, outcome_store)  # exact state on a clean shutdown
    outcome_store.close()


def _persist_router_state(engine: RouterEngine, outcome_store: OutcomeStore) -> None:
    """Flush the router's mutable exploration AND escalation state to disk (both no-op if empty)."""
    exploration = engine.snapshot_exploration_state()
    if exploration:
        outcome_store.save_router_state(exploration)
    escalation = engine.snapshot_escalation_state()
    if escalation:
        outcome_store.save_escalation_state(escalation)


def _build_verifier(policy: RouterPolicy) -> AutoDetectVerifier | RerunConfirmingVerifier:
    """Off-wire verifier, rerun-confirmed when escalation is on so a flake never trips it.

    Rerun-confirm re-runs a failing suite on unchanged state (fail-then-pass = flake, abstained);
    default capture behaviour is unchanged by escalation's enabled state (it ships ON).
    """
    base = AutoDetectVerifier(timeout=policy.capture.verify_timeout_seconds)
    if policy.escalation.enabled:
        return RerunConfirmingVerifier(base, reruns=policy.capture.rerun_confirmations)
    return base


def _build_trajectory_recorder(capture: CapturePolicy) -> TrajectoryRecorder | None:
    """Build the opt-in full-content recorder only when enabled; None (inert) otherwise.

    Enabling it requires the encryption key + the 'capture' extra — resolved here, at boot,
    never on the wire. Off by default keeps every unconfigured deployment behaviour-only.
    """
    if not capture.full_content:
        return None
    live_dir = resolve_live_dir(capture.trajectory_dir)
    sink = LiveTrajectorySink(live_dir, load_key())
    return TrajectoryRecorder(sink, enabled=True)


def _wire_capture(
    session_manager: SessionManager,
    outcome_store: OutcomeStore,
    policy: RouterPolicy,
    engine: RouterEngine,
    resolver: WorkDirResolver,
) -> CaptureWorker:
    """Build the capture worker+coordinator and wire close→enqueue."""
    _log_capture_disclosure(policy, resolver)
    coordinator = CaptureCoordinator(
        resolver=resolver,
        verifier=_build_verifier(policy),
        store=outcome_store,
        # The live caller: a verified Tier-2 outcome escalates/downshifts the NEXT session
        # by moving the in-process ConservativeGate — never mid-session (cache-safety).
        record_outcome_callback=engine.record_outcome,
        # Batch-first learning: re-fit the kNN index from the log every N captured outcomes.
        refit_scheduler=RefitScheduler(outcome_store, policy.refit.every_n_outcomes),
        # Opt-in full-content capture — built only when explicitly enabled; inert otherwise.
        trajectory_recorder=_build_trajectory_recorder(policy.capture),
    )
    worker = CaptureWorker(
        coordinator=coordinator,
        session_manager=session_manager,
        # Crash-tolerant cadence: flush the exploration budget/gate slack AND the escalation
        # failure log on the existing periodic sweep (no extra timer thread). Both advance on
        # decisions/outcomes, so this bounds loss to one sweep interval; clean shutdown is exact.
        on_sweep=functools.partial(_persist_router_state, engine, outcome_store),
    )
    # Always wire: a per-session resolve returning None just means manual-only for that
    # session, and the sweeper must run so untrafficked sessions still close.
    session_manager.set_verifier_callback(worker.enqueue)
    worker.start()
    return worker


app = FastAPI(
    title="Shunt Router",
    version="0.0.0",
    lifespan=lifespan,
)


def _get_tool_identity(request: Request) -> str:
    source_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "") or ""
    return SessionManager.compute_tool_identity(source_ip, user_agent)


def _store_session_with_provenance(
    outcome_store: OutcomeStore,
    router: ProxyRouter,
    session: Session,
    model_name: str,
    reason: str,
) -> None:
    import time

    from shunt.router.provenance import build_provenance

    # Prefer the engine's rich provenance (candidate scores, neighbors, real reason);
    # only synthesize one on the engine-less/hard-code path where none was recorded.
    provenance = session.decision_provenance or build_provenance(
        model_chosen=model_name,
        selection_rule_used=session.metadata.get("model_source", reason),
        fallback_chain_triggered=False,
        router_propensity=1.0,
    )
    # FALLBACK HONESTY: when the engine locked model A but the retry/fallback loop
    # actually served model B (A failed upstream), the engine's decision-time
    # provenance still names A — the model that was CHOSEN, not the model that RAN.
    # The sessions.model_chosen COLUMN is already the served model (what kNN learns
    # from), but decision_provenance.model_chosen is what `shunt explain` prints, so
    # a fallback session would explain as "chose A, fallback: no" while serving B.
    # Correct the provenance to the served model and mark the fallback.
    served_model = provenance.get("model_chosen")
    if served_model is not None and served_model != model_name:
        provenance = {**provenance, "model_chosen": model_name, "fallback_chain_triggered": True}
    session.decision_provenance = provenance
    outcome_store.store_session(
        session_id=session.session_id,
        prompt_text=session.metadata.get("last_prompt", ""),
        # The engine already computed this at decision time — persist it so the session
        # is queryable by the kNN read-back once a verified outcome lands.
        embedding=router.cached_embedding(session.session_id),
        model_chosen=model_name,
        cost=session.total_cost,
        cache_stats={"cache_tax": session.cache_tax, "prompt_tokens": session.prompt_length_tokens},
        duration=time.time() - session.start_time.timestamp(),
        decision_provenance=provenance,
        # Cost is UNKNOWN (not 0.0) when the provider reported no usage.cost.
        # Selection propensity and resolved model-version fingerprint are decided-once,
        # first-class columns — they cannot be reconstructed later, so persist them now.
        provenance=SessionProvenance(
            cost_known=not session.metadata.get("cost_unreported", False),
            selection_propensity=provenance.get("router_propensity"),
            model_fingerprint=router.model_fingerprint(model_name),
        ),
    )


async def _persist_after_stream(
    inner: AsyncGenerator[bytes, None],
    persist: Callable[[], None],
) -> AsyncGenerator[bytes, None]:
    """Yield *inner* through, persisting the session once the stream ends."""
    # Usage (cost, cache tax) only arrives on the final streamed chunk, so persisting
    # before the stream drains would record a zero-cost row. `finally` keeps the row
    # written even when the client disconnects early.
    try:
        async for chunk in inner:
            yield chunk
    except Exception as exc:
        # The endpoint's own redaction ran before this generator started, so an error
        # raised mid-stream used to escape uncaught into uvicorn's traceback logger with
        # the upstream body — including a quoted API key — intact.
        safe = redact_secrets(str(exc))
        logger.error("Upstream stream failed: %s", safe)
        # `from None` so the raw text cannot ride along on __cause__/__context__ into
        # any handler that logs with exc_info.
        raise UpstreamError(safe) from None
    finally:
        persist()


async def _build_decision_headers(
    session: Session,
    model_name: str,
    reason: str,
) -> dict[str, str]:
    # Single choke point for all 9 call sites: everything that reaches this header
    # is redacted, ASCII-only and single-line, so upstream error text can neither
    # leak a key nor split the response.
    return {
        "X-Shunt-Decision": header_safe(f"{model_name}; reason={reason}"),
        "X-Shunt-Session-Id": session.session_id,
    }


def _error_response_headers(headers: dict[str, str], exc: UpstreamError) -> dict[str, str]:
    """Mark a permanent refusal so a retry-happy SDK stops: the budget cap is not transient."""
    if isinstance(exc, BudgetExceededError):
        # The OpenAI SDK obeys `x-should-retry: false` before its status-code check, so the
        # header is the belt-and-suspenders on top of 402 — even a client that normalizes
        # 402 into a retry path must honor it.
        return {**headers, "x-should-retry": "false"}
    return headers


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _loop_health_payload(outcome_store: OutcomeStore, model_pool: ModelPool) -> dict[str, Any]:
    """Aggregate loop-health metrics as a JSON-able dict — no prompt_text, no PII."""
    from dataclasses import asdict

    return asdict(loop_health_for(outcome_store, model_pool))


@app.get("/admin/loop-health")
async def loop_health(request: Request) -> dict[str, Any]:
    """Read-only loop-health telemetry. Aggregates only (no prompts); access control is the
    deployment's bind/port reach (localhost default; container publishes to loopback only)."""
    outcome_store: OutcomeStore = request.app.state.outcome_store
    model_pool: ModelPool = request.app.state.model_pool
    return _loop_health_payload(outcome_store, model_pool)


@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, object]:
    """OpenAI-shaped model list so clients that auto-discover models don't 404.

    A stub over the local registry — no auth, no upstream call. Anthropic clients
    read the same ``data[].id`` field, so one shape serves both wires.
    """
    pool: ModelPool = request.app.state.model_pool
    data = [
        {"id": name, "object": "model", "created": 0, "owned_by": "shunt"}
        for name in pool.model_names()
    ]
    return {"object": "list", "data": data}


async def _json_body(request: Request) -> dict[str, Any] | JSONResponse:
    """Parse the request body, answering 400 (not 500) when the client sent bad JSON."""
    # An unguarded `await request.json()` raises JSONDecodeError, which FastAPI turns into
    # a 500 with a traceback in the log — reporting a CLIENT mistake as a server fault, and
    # making a genuine server failure harder to spot in the same log.
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Malformed JSON in request body",
                    "type": "bad_request",
                }
            },
        )
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Request body must be a JSON object",
                    "type": "bad_request",
                }
            },
        )
    return body


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await _json_body(request)
    if isinstance(body, JSONResponse):
        return body
    mgr: SessionManager = request.app.state.session_manager
    router: ProxyRouter = request.app.state.router

    session = mgr.find_or_create(_get_tool_identity(request))
    mgr.cleanup_expired()

    stream = body.get("stream", False)

    try:
        response_data, model_name, reason = await router.route_chat_completion(body, session)
    except UpstreamError as exc:
        safe = redact_secrets(str(exc))
        logger.error("Routing failed: %s", safe)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": safe, "type": "proxy_error"}},
            headers=_error_response_headers(
                await _build_decision_headers(session, "error", safe), exc
            ),
        )
    except Exception as exc:
        safe = redact_secrets(str(exc))
        logger.error("Unexpected error: %s", safe)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Unexpected proxy error", "type": "proxy_error"}},
            headers=await _build_decision_headers(session, "error", safe),
        )

    reason = session.metadata.get("model_source", reason)
    persist = functools.partial(
        _store_session_with_provenance,
        request.app.state.outcome_store,
        router,
        session,
        model_name,
        reason,
    )

    if stream:
        gen: AsyncGenerator[bytes, None] = response_data  # type: ignore[assignment]
        decision_headers = await _build_decision_headers(session, model_name, reason)
        return StreamingResponse(
            _persist_after_stream(gen, persist),
            media_type="text/event-stream",
            headers=decision_headers,
        )

    persist()
    return JSONResponse(
        content=response_data,
        headers=await _build_decision_headers(session, model_name, reason),
    )


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    body = await _json_body(request)
    if isinstance(body, JSONResponse):
        return body
    mgr: SessionManager = request.app.state.session_manager
    router: ProxyRouter = request.app.state.router

    session = mgr.find_or_create(_get_tool_identity(request))
    mgr.cleanup_expired()

    stream = body.get("stream", False)

    try:
        response_data, model_name, reason = await router.route_messages(body, session)
    except UpstreamError as exc:
        safe = redact_secrets(str(exc))
        logger.error("Routing failed: %s", safe)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": safe, "type": "proxy_error"}},
            headers=_error_response_headers(
                await _build_decision_headers(session, "error", safe), exc
            ),
        )
    except Exception as exc:
        safe = redact_secrets(str(exc))
        logger.error("Unexpected error: %s", safe)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Unexpected proxy error", "type": "proxy_error"}},
            headers=await _build_decision_headers(session, "error", safe),
        )

    reason = session.metadata.get("model_source", reason)
    persist = functools.partial(
        _store_session_with_provenance,
        request.app.state.outcome_store,
        router,
        session,
        model_name,
        reason,
    )

    if stream:
        gen: AsyncGenerator[bytes, None] = response_data  # type: ignore[assignment]
        decision_headers = await _build_decision_headers(session, model_name, reason)
        return StreamingResponse(
            _persist_after_stream(gen, persist),
            media_type="text/event-stream",
            headers=decision_headers,
        )

    persist()
    return JSONResponse(
        content=response_data,
        headers=await _build_decision_headers(session, model_name, reason),
    )


def run() -> None:
    host = os.environ.get("SHUNT_HOST", "127.0.0.1")
    port = int(os.environ.get("SHUNT_PORT", "8080"))

    level = configure_logging()
    if level == "DEBUG":
        logger.debug(
            "Debug logging ON. Third-party HTTP libraries stay at INFO on purpose — "
            "their DEBUG output includes Authorization headers."
        )

    def _shutdown(sig: int, frame: FrameType | None) -> None:
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=level.lower(),
    )
