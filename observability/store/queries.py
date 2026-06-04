"""
Aggregate queries: cost/period, latency percentiles, error rate.
These power the dashboard and drift detection.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from store.db import query_traces, _DEFAULT_DB
from store.schema import AggregateWindow, LLMTrace


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


async def aggregate_window(
    since: datetime,
    until: datetime,
    app: str | None = None,
    model: str | None = None,
    db_path: Path = _DEFAULT_DB,
) -> AggregateWindow:
    traces = await query_traces(app=app, model=model, since=since, until=until, db_path=db_path)
    ok = [t for t in traces if t.status == "ok"]
    errors = [t for t in traces if t.status != "ok"]

    latencies = [t.latency_ms for t in ok]
    return AggregateWindow(
        window_start=since,
        window_end=until,
        app=app or "",
        model=model or "",
        n_calls=len(traces),
        n_errors=len(errors),
        total_cost_usd=sum(t.cost_usd for t in traces),
        total_tokens_in=sum(t.tokens_in for t in traces),
        total_tokens_out=sum(t.tokens_out for t in traces),
        latency_p50_ms=_pct(latencies, 0.50),
        latency_p95_ms=_pct(latencies, 0.95),
        latency_p99_ms=_pct(latencies, 0.99),
        latency_mean_ms=statistics.mean(latencies) if latencies else None,
    )


async def cost_per_day(
    days: int = 7,
    app: str | None = None,
    db_path: Path = _DEFAULT_DB,
) -> list[dict]:
    """Return list of {date, cost_usd} for the last N days."""
    now = datetime.now(timezone.utc)
    result = []
    for i in range(days - 1, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = day_start + timedelta(days=1)
        traces = await query_traces(app=app, since=day_start, until=day_end, db_path=db_path)
        result.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "cost_usd": round(sum(t.cost_usd for t in traces), 6),
            "n_calls": len(traces),
        })
    return result


async def latency_percentiles(
    hours: int = 24,
    app: str | None = None,
    model: str | None = None,
    db_path: Path = _DEFAULT_DB,
) -> dict[str, float | None]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    traces = await query_traces(app=app, model=model, since=since, status="ok", db_path=db_path)
    lats = [t.latency_ms for t in traces]
    return {
        "p50": _pct(lats, 0.50),
        "p95": _pct(lats, 0.95),
        "p99": _pct(lats, 0.99),
        "mean": statistics.mean(lats) if lats else None,
        "n": len(lats),
    }


async def error_rate(
    hours: int = 24,
    app: str | None = None,
    db_path: Path = _DEFAULT_DB,
) -> dict:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    traces = await query_traces(app=app, since=since, db_path=db_path)
    total = len(traces)
    errors = sum(1 for t in traces if t.status != "ok")
    return {
        "total": total,
        "errors": errors,
        "rate": errors / total if total > 0 else 0.0,
    }
