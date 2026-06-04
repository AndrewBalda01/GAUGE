"""Pydantic models for the observability store."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class LLMTrace(BaseModel):
    """One captured LLM call — mirrors the OTel span data model."""
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    model: str
    system: str = ""           # e.g. "anthropic", "local.llama_cpp"
    operation: str = "chat"
    app: str = ""              # calling application identifier
    route: str = ""            # router decision (e.g. "small-model")
    run_id: str = ""           # eval-harness run ID

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: float | None = None

    status: str = "ok"         # "ok" | "error" | "timeout"
    error: str = ""

    prompt: str = ""           # stored only when include_content=True
    completion: str = ""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Alert(BaseModel):
    """A fired anomaly alert."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    kind: str                   # "cost_spike" | "latency_spike" | "error_rate" | "drift"
    message: str
    severity: str = "warning"   # "info" | "warning" | "critical"
    metric_value: float | None = None
    threshold: float | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    app: str = ""
    fired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False


class AggregateWindow(BaseModel):
    """Pre-computed aggregate for a time window."""
    window_start: datetime
    window_end: datetime
    app: str = ""
    model: str = ""
    n_calls: int = 0
    n_errors: int = 0
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    latency_mean_ms: float | None = None
