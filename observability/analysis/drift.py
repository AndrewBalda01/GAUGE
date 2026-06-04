"""
Drift detection: compare metric distributions between two time windows.

Detects:
  - Cost drift      (mean cost per call changed significantly)
  - Latency drift   (p95 latency shifted)
  - Token drift     (mean tokens_out changed — prompt/model behaviour shift)
  - Error rate drift
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from store.db import query_traces, _DEFAULT_DB
from store.schema import LLMTrace


@dataclass
class DriftMetric:
    name: str
    baseline_value: float | None
    current_value: float | None
    delta_pct: float | None   # (current - baseline) / baseline * 100
    drifted: bool             # exceeded threshold
    threshold_pct: float


@dataclass
class DriftReport:
    baseline_start: datetime
    baseline_end: datetime
    current_start: datetime
    current_end: datetime
    app: str = ""
    metrics: list[DriftMetric] = field(default_factory=list)

    @property
    def any_drift(self) -> bool:
        return any(m.drifted for m in self.metrics)

    def summary(self) -> str:
        if not self.any_drift:
            return "No drift detected."
        drifted = [m for m in self.metrics if m.drifted]
        parts = [f"{m.name}: {m.delta_pct:+.1f}%" for m in drifted]
        return "Drift detected — " + ", ".join(parts)


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _pct95(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = int(0.95 * (len(s) - 1))
    return s[idx]


def _delta_pct(baseline: float | None, current: float | None) -> float | None:
    if baseline is None or current is None or baseline == 0:
        return None
    return (current - baseline) / baseline * 100


async def detect_drift(
    baseline_start: datetime,
    baseline_end: datetime,
    current_start: datetime,
    current_end: datetime,
    app: str | None = None,
    cost_threshold_pct: float = 30.0,
    latency_threshold_pct: float = 30.0,
    token_threshold_pct: float = 20.0,
    error_rate_threshold_pct: float = 5.0,  # absolute percentage points
    db_path: Path = _DEFAULT_DB,
) -> DriftReport:
    baseline_traces = await query_traces(app=app, since=baseline_start, until=baseline_end, db_path=db_path)
    current_traces  = await query_traces(app=app, since=current_start,  until=current_end,  db_path=db_path)

    def _ok(traces: list[LLMTrace]) -> list[LLMTrace]:
        return [t for t in traces if t.status == "ok"]

    b_ok = _ok(baseline_traces)
    c_ok = _ok(current_traces)

    metrics: list[DriftMetric] = []

    # --- Cost per call ---
    b_cost = _mean_or_none([t.cost_usd for t in b_ok])
    c_cost = _mean_or_none([t.cost_usd for t in c_ok])
    delta = _delta_pct(b_cost, c_cost)
    metrics.append(DriftMetric(
        name="cost_per_call",
        baseline_value=b_cost,
        current_value=c_cost,
        delta_pct=delta,
        drifted=abs(delta) > cost_threshold_pct if delta is not None else False,
        threshold_pct=cost_threshold_pct,
    ))

    # --- Latency p95 ---
    b_lat = _pct95([t.latency_ms for t in b_ok])
    c_lat = _pct95([t.latency_ms for t in c_ok])
    delta = _delta_pct(b_lat, c_lat)
    metrics.append(DriftMetric(
        name="latency_p95",
        baseline_value=b_lat,
        current_value=c_lat,
        delta_pct=delta,
        drifted=abs(delta) > latency_threshold_pct if delta is not None else False,
        threshold_pct=latency_threshold_pct,
    ))

    # --- Mean tokens out (output length drift = behaviour shift) ---
    b_tok = _mean_or_none([t.tokens_out for t in b_ok])
    c_tok = _mean_or_none([t.tokens_out for t in c_ok])
    delta = _delta_pct(b_tok, c_tok)
    metrics.append(DriftMetric(
        name="mean_tokens_out",
        baseline_value=b_tok,
        current_value=c_tok,
        delta_pct=delta,
        drifted=abs(delta) > token_threshold_pct if delta is not None else False,
        threshold_pct=token_threshold_pct,
    ))

    # --- Error rate (absolute difference in %) ---
    b_err_rate = (len(baseline_traces) - len(b_ok)) / max(len(baseline_traces), 1) * 100
    c_err_rate = (len(current_traces)  - len(c_ok))  / max(len(current_traces), 1)  * 100
    delta_abs = c_err_rate - b_err_rate
    metrics.append(DriftMetric(
        name="error_rate",
        baseline_value=b_err_rate,
        current_value=c_err_rate,
        delta_pct=delta_abs,  # reuse field as absolute delta for error rate
        drifted=abs(delta_abs) > error_rate_threshold_pct,
        threshold_pct=error_rate_threshold_pct,
    ))

    return DriftReport(
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        current_start=current_start,
        current_end=current_end,
        app=app or "",
        metrics=metrics,
    )
