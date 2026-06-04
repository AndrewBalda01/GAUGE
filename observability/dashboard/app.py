"""
Observability dashboard — FastAPI REST API.

Endpoints:
  GET  /traces            — recent traces (filterable)
  GET  /traces/{id}       — single trace
  POST /traces/{id}/replay — re-execute a stored call
  GET  /metrics/summary   — aggregate metrics for a window
  GET  /metrics/cost      — cost per day
  GET  /metrics/latency   — latency percentiles
  GET  /metrics/errors    — error rate
  GET  /alerts            — recent alerts
  POST /alerts/check      — trigger anomaly check now
  GET  /drift             — run drift detection on-demand
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from store.db import init_db, query_traces, query_alerts, load_trace, _DEFAULT_DB
from store.queries import aggregate_window, cost_per_day, latency_percentiles, error_rate
from analysis.drift import detect_drift
from analysis.anomaly import check_anomalies, check_drift_and_alert

_DB_PATH = _DEFAULT_DB


def set_db_path(path: Path) -> None:
    global _DB_PATH
    _DB_PATH = path


from contextlib import asynccontextmanager
from typing import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db(_DB_PATH)
    yield


app = FastAPI(
    title="LLM Observability Dashboard",
    version="0.1.0",
    description="Production-ready observability for LLM applications",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

@app.get("/traces")
async def list_traces(
    app_name: str | None = Query(None, alias="app"),
    model: str | None = None,
    status: str | None = None,
    hours: int = 24,
    limit: int = 100,
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    traces = await query_traces(
        app=app_name, model=model, since=since, status=status,
        limit=limit, db_path=_DB_PATH,
    )
    return {
        "count": len(traces),
        "traces": [t.model_dump() for t in traces],
    }


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    trace = await load_trace(trace_id, _DB_PATH)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump()


class ReplayRequest(BaseModel):
    target_model: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"


@app.post("/traces/{trace_id}/replay")
async def replay(trace_id: str, req: ReplayRequest):
    import os
    import httpx

    trace = await load_trace(trace_id, _DB_PATH)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    if not trace.prompt:
        raise HTTPException(status_code=400, detail="Trace has no stored prompt (include_content was False)")

    model = req.target_model or trace.model

    async def _call(prompt: str, model: str, system: str):
        import time
        api_key = os.environ.get(req.api_key_env, "")
        payload = {
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            resp.raise_for_status()
        latency = (time.perf_counter() - t0) * 1000
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0), latency

    from analysis.replay import replay_trace
    result = await replay_trace(trace_id, _call, target_model=model, db_path=_DB_PATH)
    return {
        "original_completion": result.original.completion[:500],
        "replayed_completion": result.replayed_completion[:500],
        "replayed_model": result.replayed_model,
        "replayed_latency_ms": result.replayed_latency_ms,
        "replayed_cost_usd": result.replayed_cost_usd,
        "changed": result.changed,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@app.get("/metrics/summary")
async def metrics_summary(
    app_name: str | None = Query(None, alias="app"),
    model: str | None = None,
    hours: int = 24,
):
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)
    agg = await aggregate_window(since, now, app=app_name, model=model, db_path=_DB_PATH)
    return agg.model_dump()


@app.get("/metrics/cost")
async def metrics_cost(
    app_name: str | None = Query(None, alias="app"),
    days: int = 7,
):
    rows = await cost_per_day(days=days, app=app_name, db_path=_DB_PATH)
    return {"days": rows}


@app.get("/metrics/latency")
async def metrics_latency(
    app_name: str | None = Query(None, alias="app"),
    model: str | None = None,
    hours: int = 24,
):
    return await latency_percentiles(hours=hours, app=app_name, model=model, db_path=_DB_PATH)


@app.get("/metrics/errors")
async def metrics_errors(
    app_name: str | None = Query(None, alias="app"),
    hours: int = 24,
):
    return await error_rate(hours=hours, app=app_name, db_path=_DB_PATH)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@app.get("/alerts")
async def list_alerts(
    acknowledged: bool | None = None,
    limit: int = 50,
):
    alerts = await query_alerts(acknowledged=acknowledged, limit=limit, db_path=_DB_PATH)
    return {"count": len(alerts), "alerts": [a.model_dump() for a in alerts]}


@app.post("/alerts/check")
async def trigger_alert_check(
    app_name: str | None = Query(None, alias="app"),
    window_hours: int = 1,
):
    anomaly_alerts = await check_anomalies(app=app_name, window_hours=window_hours, db_path=_DB_PATH)
    drift_alerts   = await check_drift_and_alert(app=app_name, db_path=_DB_PATH)
    all_alerts = anomaly_alerts + drift_alerts
    return {
        "fired": len(all_alerts),
        "alerts": [a.model_dump() for a in all_alerts],
    }


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

@app.get("/drift")
async def get_drift(
    app_name: str | None = Query(None, alias="app"),
    baseline_hours: int = 24,
    current_hours: int = 1,
):
    now = datetime.now(timezone.utc)
    current_start  = now - timedelta(hours=current_hours)
    baseline_end   = current_start
    baseline_start = baseline_end - timedelta(hours=baseline_hours)

    report = await detect_drift(
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        current_start=current_start,
        current_end=now,
        app=app_name,
        db_path=_DB_PATH,
    )
    return {
        "any_drift": report.any_drift,
        "summary": report.summary(),
        "metrics": [
            {
                "name": m.name,
                "baseline": m.baseline_value,
                "current": m.current_value,
                "delta_pct": m.delta_pct,
                "drifted": m.drifted,
                "threshold_pct": m.threshold_pct,
            }
            for m in report.metrics
        ],
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "db": str(_DB_PATH)}
