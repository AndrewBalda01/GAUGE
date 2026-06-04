"""
@trace_llm decorator and context manager.

Usage — decorator:
    @trace_llm(app="eval-harness", route="small-model")
    async def call_model(prompt: str) -> LLMResult:
        ...

Usage — context manager:
    async with llm_span(model="claude-haiku-4-5", app="eval-harness") as span:
        result = await call_api(prompt)
        span.set_tokens(tokens_in=100, tokens_out=50)
        span.set_cost(0.0012)

Both paths:
  1. Create an OTel span with GenAI attributes.
  2. Emit the completed LLMSpanData to the collector (async, fire-and-forget).
"""

from __future__ import annotations

import asyncio
import functools
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

from otel.attributes import SPAN_CHAT, build_span_attributes
from collector.ingest import emit_span

_tracer_name = "llm-observability"


# ---------------------------------------------------------------------------
# LLMSpanData — the mutable context passed to user code in context manager
# ---------------------------------------------------------------------------

@dataclass
class LLMSpanData:
    """Mutable span context; user code fills in tokens/cost/status/content."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    model: str = ""
    system: str = ""
    operation: str = "chat"
    app: str = ""
    route: str = ""
    run_id: str = ""

    # Filled during / after the call
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    ttft_ms: float | None = None
    latency_ms: float = 0.0
    status: str = "ok"
    error: str = ""
    prompt: str = ""
    completion: str = ""

    # Timing
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc), repr=False)

    def set_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    def set_cost(self, cost_usd: float) -> None:
        self.cost_usd = cost_usd

    def set_ttft(self, ttft_ms: float) -> None:
        self.ttft_ms = ttft_ms

    def set_content(self, prompt: str = "", completion: str = "") -> None:
        self.prompt = prompt
        self.completion = completion

    def finish(self, status: str = "ok", error: str = "") -> None:
        self.latency_ms = (time.perf_counter() - self._t0) * 1000
        self.status = status
        self.error = error


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def llm_span(
    model: str,
    system: str = "",
    operation: str = "chat",
    app: str = "",
    route: str = "",
    run_id: str = "",
    include_content: bool = False,
) -> AsyncIterator[LLMSpanData]:
    span_data = LLMSpanData(
        model=model,
        system=system,
        operation=operation,
        app=app,
        route=route,
        run_id=run_id,
    )

    otel_span = None
    if _OTEL_AVAILABLE:
        tracer = trace.get_tracer(_tracer_name)
        otel_span = tracer.start_span(SPAN_CHAT)

    try:
        yield span_data
        span_data.finish(status="ok")
    except Exception as exc:
        span_data.finish(status="error", error=str(exc))
        if otel_span:
            otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    finally:
        if otel_span:
            attrs = build_span_attributes(
                system=span_data.system or "unknown",
                model=span_data.model,
                operation=span_data.operation,
                tokens_in=span_data.tokens_in,
                tokens_out=span_data.tokens_out,
                cost_usd=span_data.cost_usd,
                ttft_ms=span_data.ttft_ms,
                status=span_data.status,
                app=span_data.app,
                route=span_data.route,
                run_id=span_data.run_id,
                include_content=include_content,
                prompt=span_data.prompt,
                completion=span_data.completion,
            )
            for k, v in attrs.items():
                otel_span.set_attribute(k, v)
            otel_span.end()

        # Emit to our collector (non-blocking)
        asyncio.create_task(emit_span(span_data))


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def trace_llm(
    model: str = "",
    system: str = "",
    app: str = "",
    route: str = "",
    run_id: str = "",
    include_content: bool = False,
    model_kwarg: str | None = None,
):
    """
    Decorator that wraps an async function with an LLM span.

    The wrapped function receives a `span: LLMSpanData` kwarg it can use
    to record tokens, cost, etc.  If the function signature doesn't accept
    `span`, the kwarg is not injected.

    Example:
        @trace_llm(app="eval-harness")
        async def call(prompt: str, span: LLMSpanData | None = None) -> str:
            resp = await api_call(prompt)
            if span:
                span.set_tokens(resp.tokens_in, resp.tokens_out)
            return resp.text
    """
    def decorator(fn: Callable) -> Callable:
        import inspect
        accepts_span = "span" in inspect.signature(fn).parameters

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _model = model
            if model_kwarg and model_kwarg in kwargs:
                _model = str(kwargs[model_kwarg])

            async with llm_span(
                model=_model,
                system=system,
                app=app,
                route=route,
                run_id=run_id,
                include_content=include_content,
            ) as span:
                if accepts_span:
                    kwargs["span"] = span
                return await fn(*args, **kwargs)

        return wrapper
    return decorator
