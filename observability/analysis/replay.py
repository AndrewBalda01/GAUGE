"""
Replay: fetch a stored LLM call and re-execute it.

Use cases:
  - Debug a problematic call by re-running it (possibly with a different model)
  - Regression check: does the same prompt now produce a different response?
  - Cost comparison: how much would this prompt cost on model X vs Y?
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from store.db import load_trace, _DEFAULT_DB
from store.schema import LLMTrace


@dataclass
class ReplayResult:
    original: LLMTrace
    replayed_model: str
    replayed_completion: str
    replayed_tokens_in: int
    replayed_tokens_out: int
    replayed_latency_ms: float
    replayed_cost_usd: float
    changed: bool                    # True if completion differs from original


async def replay_trace(
    trace_id: str,
    call_fn,        # async callable: (prompt, model, system) -> (text, tokens_in, tokens_out, latency_ms)
    target_model: str | None = None,
    db_path: Path = _DEFAULT_DB,
) -> ReplayResult:
    """
    Load a stored trace and re-execute it.

    `call_fn` is injected by the caller (keeps this module model-agnostic).
    Signature: async def call_fn(prompt, model, system) -> (text, tokens_in, tokens_out, latency_ms)
    """
    trace = await load_trace(trace_id, db_path)
    if trace is None:
        raise ValueError(f"Trace {trace_id!r} not found in {db_path}")

    model = target_model or trace.model
    text, tokens_in, tokens_out, latency_ms = await call_fn(
        prompt=trace.prompt,
        model=model,
        system=trace.system,
    )

    from collector.cost import estimate_cost
    cost = estimate_cost(model, tokens_in, tokens_out)

    return ReplayResult(
        original=trace,
        replayed_model=model,
        replayed_completion=text,
        replayed_tokens_in=tokens_in,
        replayed_tokens_out=tokens_out,
        replayed_latency_ms=latency_ms,
        replayed_cost_usd=cost,
        changed=(text.strip() != trace.completion.strip()),
    )
