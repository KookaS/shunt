"""FastAPI proxy server — OpenAI + Anthropic compatible /v1/ endpoints."""

from __future__ import annotations

import logging
import os
import signal
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from types import FrameType

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from shunt.db.store import OutcomeStore
from shunt.models import ModelPool
from shunt.proxy.router import ProxyRouter, UpstreamError
from shunt.session import Session, SessionManager

logger = logging.getLogger(__name__)

_INACTIVITY_TIMEOUT = int(os.environ.get("SHUNT_SESSION_INACTIVITY_TIMEOUT", "900"))
_GRACE_PERIOD = int(os.environ.get("SHUNT_SESSION_GRACE_PERIOD", "120"))
_RETRY_COUNT = int(os.environ.get("SHUNT_RETRY_COUNT", "3"))
_MODEL_CONFIG_PATH = os.environ.get("SHUNT_MODEL_CONFIG_PATH")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    session_manager = SessionManager(
        inactivity_timeout=_INACTIVITY_TIMEOUT,
        grace_period=_GRACE_PERIOD,
    )
    model_pool = ModelPool(config_path=_MODEL_CONFIG_PATH)
    outcome_store = OutcomeStore()
    router = ProxyRouter(
        model_pool=model_pool,
        session_manager=session_manager,
        retry_count=_RETRY_COUNT,
    )
    app.state.session_manager = session_manager
    app.state.model_pool = model_pool
    app.state.router = router
    app.state.outcome_store = outcome_store
    yield
    outcome_store.close()


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
    session: Session,
    model_name: str,
    reason: str,
) -> None:
    import time

    from shunt.router.provenance import build_provenance

    provenance = build_provenance(
        model_chosen=model_name,
        selection_rule_used=reason,
        fallback_chain_triggered=False,
        router_propensity=1.0,
    )
    session.decision_provenance = provenance
    outcome_store.store_session(
        session_id=session.session_id,
        prompt_text=session.metadata.get("last_prompt", ""),
        embedding=None,
        model_chosen=model_name,
        cost=session.total_cost,
        cache_stats={"cache_tax": session.cache_tax, "prompt_tokens": session.prompt_length_tokens},
        duration=time.time() - session.start_time.timestamp(),
        decision_provenance=provenance,
    )


async def _build_decision_headers(
    session: Session,
    model_name: str,
    reason: str,
) -> dict[str, str]:
    return {
        "X-Shunt-Decision": f"{model_name}; reason={reason}",
        "X-Shunt-Session-Id": session.session_id,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    mgr: SessionManager = request.app.state.session_manager
    router: ProxyRouter = request.app.state.router

    session = mgr.find_or_create(_get_tool_identity(request))
    mgr.cleanup_expired()

    stream = body.get("stream", False)

    try:
        response_data, model_name, reason = await router.route_chat_completion(body, session)
    except UpstreamError as exc:
        logger.error("Routing failed: %s", exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc), "type": "proxy_error"}},
            headers=await _build_decision_headers(session, "error", str(exc)),
        )
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Unexpected proxy error", "type": "proxy_error"}},
            headers=await _build_decision_headers(session, "error", str(exc)),
        )

    _store_session_with_provenance(
        request.app.state.outcome_store,
        session,
        model_name,
        reason,
    )

    if stream:
        gen: AsyncGenerator[bytes, None] = response_data  # type: ignore[assignment]
        decision_headers = await _build_decision_headers(session, model_name, reason)

        async def _stream_with_headers(
            inner: AsyncGenerator[bytes, None],
            headers: dict[str, str],
        ) -> AsyncGenerator[bytes, None]:
            yield b""
            async for chunk in inner:
                yield chunk

        return StreamingResponse(
            _stream_with_headers(gen, decision_headers),
            media_type="text/event-stream",
            headers=decision_headers,
        )

    return JSONResponse(
        content=response_data,
        headers=await _build_decision_headers(session, model_name, reason),
    )


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    body = await request.json()
    mgr: SessionManager = request.app.state.session_manager
    router: ProxyRouter = request.app.state.router

    session = mgr.find_or_create(_get_tool_identity(request))
    mgr.cleanup_expired()

    stream = body.get("stream", False)

    try:
        response_data, model_name, reason = await router.route_messages(body, session)
    except UpstreamError as exc:
        logger.error("Routing failed: %s", exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc), "type": "proxy_error"}},
            headers=await _build_decision_headers(session, "error", str(exc)),
        )
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Unexpected proxy error", "type": "proxy_error"}},
            headers=await _build_decision_headers(session, "error", str(exc)),
        )

    _store_session_with_provenance(
        request.app.state.outcome_store,
        session,
        model_name,
        reason,
    )

    if stream:
        gen: AsyncGenerator[bytes, None] = response_data  # type: ignore[assignment]
        decision_headers = await _build_decision_headers(session, model_name, reason)

        return StreamingResponse(
            gen,
            media_type="text/event-stream",
            headers=decision_headers,
        )

    return JSONResponse(
        content=response_data,
        headers=await _build_decision_headers(session, model_name, reason),
    )


def run() -> None:
    host = os.environ.get("SHUNT_HOST", "127.0.0.1")
    port = int(os.environ.get("SHUNT_PORT", "8080"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    def _shutdown(sig: int, frame: FrameType | None) -> None:
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
