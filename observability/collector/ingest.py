"""
Span ingestion: receives LLMSpanData from the instrumentation layer,
normalises it, estimates cost if missing, and persists to the store.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from collector.cost import estimate_cost

# Import lazily to avoid circular deps at module load time
_db_path: Path | None = None


def configure(db_path: Path) -> None:
    """Call once at startup to set the database path."""
    global _db_path
    _db_path = db_path


async def emit_span(span_data) -> None:
    """
    Called by instrumentation (fire-and-forget via asyncio.create_task).
    Converts LLMSpanData → LLMTrace and persists it.
    """
    from store.schema import LLMTrace
    from store.db import save_trace, init_db

    global _db_path
    if _db_path is None:
        _db_path = Path(__file__).parent.parent / "traces.db"

    await init_db(_db_path)

    # Auto-compute cost if not set
    cost = span_data.cost_usd
    if cost == 0.0 and span_data.tokens_in + span_data.tokens_out > 0:
        cost = estimate_cost(span_data.model, span_data.tokens_in, span_data.tokens_out)

    trace = LLMTrace(
        trace_id=span_data.trace_id,
        model=span_data.model,
        system=span_data.system,
        operation=span_data.operation,
        app=span_data.app,
        route=span_data.route,
        run_id=span_data.run_id,
        tokens_in=span_data.tokens_in,
        tokens_out=span_data.tokens_out,
        cost_usd=cost,
        latency_ms=span_data.latency_ms,
        ttft_ms=span_data.ttft_ms,
        status=span_data.status,
        error=span_data.error,
        prompt=span_data.prompt,
        completion=span_data.completion,
        timestamp=span_data.timestamp,
    )
    await save_trace(trace, _db_path)
