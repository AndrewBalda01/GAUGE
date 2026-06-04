"""
OpenAI-compatible FastAPI endpoint backed by the local router.

Endpoints:
  POST /v1/chat/completions   — OpenAI chat format
  POST /v1/completions        — OpenAI legacy format
  GET  /v1/models             — list loaded engines
  GET  /health                — liveness + engine status
  GET  /router/stats          — routing decision stats
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from router.classifier import Difficulty, HeuristicClassifier
from router.policy import RouterPolicy
from serving.config_loader import load_config
from serving.engine import BaseEngine, GenerateRequest, load_engine


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    engines: dict[str, BaseEngine] = {}
    classifier: HeuristicClassifier | None = None
    policy: RouterPolicy | None = None


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Engines are loaded via configure_engines() before app start, or lazily.
    yield
    for engine in _state.engines.values():
        await engine.unload()


app = FastAPI(
    title="Local Inference API",
    version="0.1.0",
    description="OpenAI-compatible local LLM inference with cost/quality router",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Setup helpers (called from CLI / tests)
# ---------------------------------------------------------------------------

async def configure_engines(
    small_config: str,
    large_config: str,
    uncertain_fallback: str = "large",
) -> None:
    """Load two engines and wire up the router."""
    small_cfg = load_config(small_config)
    large_cfg = load_config(large_config)

    small_engine = load_engine(small_cfg)
    large_engine = load_engine(large_cfg)

    await small_engine.load()
    await large_engine.load()

    _state.engines[small_engine.name] = small_engine
    _state.engines[large_engine.name] = large_engine
    _state.classifier = HeuristicClassifier()
    _state.policy = RouterPolicy(
        small_engine_name=small_engine.name,
        large_engine_name=large_engine.name,
        uncertain_fallback=uncertain_fallback,
    )


# ---------------------------------------------------------------------------
# OpenAI request/response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local-router"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    model: str = "local-router"
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | None = None


def _usage_dict(tokens_in: int, tokens_out: int) -> dict:
    return {
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out,
    }


def _build_chat_response(
    text: str,
    tokens_in: int,
    tokens_out: int,
    model: str,
    routing_info: dict | None = None,
) -> dict:
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_dict(tokens_in, tokens_out),
    }
    if routing_info:
        resp["x_routing"] = routing_info
    return resp


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def _get_engine(query: str) -> tuple[BaseEngine, dict]:
    """Select engine via router; returns (engine, routing_info_dict)."""
    if _state.classifier is None or _state.policy is None:
        # No router configured — use first available engine
        if not _state.engines:
            raise HTTPException(status_code=503, detail="No engines loaded")
        engine = next(iter(_state.engines.values()))
        return engine, {"method": "no-router", "engine": engine.name}

    clf = _state.classifier.classify(query)
    decision = _state.policy.decide(clf)
    engine = _state.engines.get(decision.engine_name)
    if engine is None:
        # Fallback to any available engine
        engine = next(iter(_state.engines.values()))
        decision.fallback_used = True
    return engine, {
        "engine": engine.name,
        "difficulty": decision.difficulty.value,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "fallback_used": decision.fallback_used,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Concatenate messages for routing classification
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    query = user_msgs[-1] if user_msgs else ""
    system = next((m.content for m in req.messages if m.role == "system"), "")

    engine, routing_info = _get_engine(query)

    gen_req = GenerateRequest(
        prompt=query,
        system=system,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop or [],
    )

    if req.stream:
        async def _stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            async for token in engine.generate_stream(gen_req):
                data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": routing_info["engine"],
                    "choices": [{"delta": {"content": token}, "index": 0}],
                }
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    resp = await engine.generate(gen_req)
    return _build_chat_response(
        resp.text, resp.tokens_prompt, resp.tokens_generated,
        model=routing_info["engine"],
        routing_info=routing_info,
    )


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    engine, routing_info = _get_engine(req.prompt)
    gen_req = GenerateRequest(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop or [],
    )
    resp = await engine.generate(gen_req)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": routing_info["engine"],
        "choices": [{"text": resp.text, "index": 0, "finish_reason": "stop"}],
        "usage": _usage_dict(resp.tokens_prompt, resp.tokens_generated),
        "x_routing": routing_info,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "loaded": engine.is_loaded,
                "vram_mb": engine.vram_mb,
            }
            for name, engine in _state.engines.items()
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engines_loaded": len(_state.engines),
        "router_active": _state.policy is not None,
        "engines": {
            name: {"loaded": e.is_loaded, "vram_mb": e.vram_mb}
            for name, e in _state.engines.items()
        },
    }


@app.get("/router/stats")
async def router_stats():
    if _state.policy is None:
        return {"router": "not configured"}
    return _state.policy.report()
