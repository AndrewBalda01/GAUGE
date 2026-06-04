"""
Anomaly detection and alert firing.

Rules:
  - cost_spike     : mean cost in last window > baseline * (1 + threshold)
  - latency_spike  : p95 latency in last window > baseline * (1 + threshold)
  - error_rate     : error rate in last window > absolute threshold
  - drift          : any metric in DriftReport exceeded its threshold

All fired alerts are persisted to the DB.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from analysis.drift import DriftReport, detect_drift
from store.db import save_alert, query_traces, _DEFAULT_DB
from store.schema import Alert
from store.queries import aggregate_window


async def check_anomalies(
    app: str | None = None,
    window_hours: int = 1,
    baseline_hours: int = 24,
    cost_spike_pct: float = 40.0,
    latency_spike_pct: float = 40.0,
    error_rate_threshold: float = 0.10,
    db_path: Path = _DEFAULT_DB,
) -> list[Alert]:
    """
    Compare the last `window_hours` against the prior `baseline_hours` window.
    Returns list of newly fired alerts (also persisted to DB).
    """
    now = datetime.now(timezone.utc)
    current_start  = now - timedelta(hours=window_hours)
    baseline_start = now - timedelta(hours=baseline_hours + window_hours)
    baseline_end   = current_start

    current  = await aggregate_window(current_start, now, app=app, db_path=db_path)
    baseline = await aggregate_window(baseline_start, baseline_end, app=app, db_path=db_path)

    alerts: list[Alert] = []

    # --- Cost spike ---
    if baseline.n_calls > 0 and current.n_calls > 0:
        b_cost_mean = baseline.total_cost_usd / baseline.n_calls
        c_cost_mean = current.total_cost_usd  / current.n_calls
        if b_cost_mean > 0 and c_cost_mean > b_cost_mean * (1 + cost_spike_pct / 100):
            pct = (c_cost_mean - b_cost_mean) / b_cost_mean * 100
            a = Alert(
                kind="cost_spike",
                message=f"Cost per call rose {pct:.1f}% vs baseline "
                        f"(${c_cost_mean:.5f} vs ${b_cost_mean:.5f})",
                severity="warning" if pct < 80 else "critical",
                metric_value=c_cost_mean,
                threshold=b_cost_mean * (1 + cost_spike_pct / 100),
                window_start=current_start,
                window_end=now,
                app=app or "",
            )
            alerts.append(a)

    # --- Latency spike ---
    b_lat = baseline.latency_p95_ms
    c_lat = current.latency_p95_ms
    if b_lat and c_lat and c_lat > b_lat * (1 + latency_spike_pct / 100):
        pct = (c_lat - b_lat) / b_lat * 100
        a = Alert(
            kind="latency_spike",
            message=f"p95 latency rose {pct:.1f}% vs baseline ({c_lat:.0f}ms vs {b_lat:.0f}ms)",
            severity="warning" if pct < 80 else "critical",
            metric_value=c_lat,
            threshold=b_lat * (1 + latency_spike_pct / 100),
            window_start=current_start,
            window_end=now,
            app=app or "",
        )
        alerts.append(a)

    # --- Error rate ---
    if current.n_calls > 0:
        err_rate = current.n_errors / current.n_calls
        if err_rate > error_rate_threshold:
            a = Alert(
                kind="error_rate",
                message=f"Error rate {err_rate:.1%} exceeds threshold {error_rate_threshold:.0%} "
                        f"({current.n_errors}/{current.n_calls} calls failed)",
                severity="critical" if err_rate > 0.25 else "warning",
                metric_value=err_rate,
                threshold=error_rate_threshold,
                window_start=current_start,
                window_end=now,
                app=app or "",
            )
            alerts.append(a)

    # Persist all
    for a in alerts:
        await save_alert(a, db_path)

    return alerts


async def check_drift_and_alert(
    app: str | None = None,
    baseline_hours: int = 24,
    current_hours: int = 1,
    db_path: Path = _DEFAULT_DB,
) -> list[Alert]:
    now = datetime.now(timezone.utc)
    current_start  = now - timedelta(hours=current_hours)
    baseline_end   = current_start
    baseline_start = baseline_end - timedelta(hours=baseline_hours)

    report = await detect_drift(
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        current_start=current_start,
        current_end=now,
        app=app,
        db_path=db_path,
    )

    alerts: list[Alert] = []
    if report.any_drift:
        a = Alert(
            kind="drift",
            message=report.summary(),
            severity="warning",
            window_start=current_start,
            window_end=now,
            app=app or "",
        )
        await save_alert(a, db_path)
        alerts.append(a)

    return alerts
