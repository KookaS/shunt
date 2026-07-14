"""FastAPI proxy server — OpenAI + Anthropic compatible /v1/ endpoints."""

from __future__ import annotations

import os
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Shunt Router",
    version="0.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await request.json()
    return JSONResponse(
        content={"id": "stub", "object": "chat.completion", "choices": []},
        headers={"X-Shunt-Decision": "stub"},
    )


@app.post("/v1/messages")
async def messages(request: Request):
    await request.json()
    return JSONResponse(
        content={"id": "stub", "type": "message", "content": []},
        headers={"X-Shunt-Decision": "stub"},
    )


def run():
    host = os.environ.get("SHUNT_HOST", "127.0.0.1")
    port = int(os.environ.get("SHUNT_PORT", "8080"))

    def _shutdown(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
