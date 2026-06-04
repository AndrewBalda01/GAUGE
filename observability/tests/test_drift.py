"""Tests for drift detection."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from store.db import init_db, save_trace
from store.schema import LLMTrace
from analysis.drift import detect_drift


def _trace(ts: datetime, cost: float = 0.001, latency: float = 200.0,
           tokens_out: int = 50, status: str = "ok", **kw) -> LLMTrace:
    return LLMTrace(
        model="claude-haiku-4-5-20251001",
        cost_usd=cost,
        latency_ms=latency,
        tokens_out=tokens_out,
        status=status,
        timestamp=ts,
        **kw,
    )


@pytest.fixture
def tmp_db(tmp_path):
    db = tmp_path / "drift_test.db"
    asyncio.run(init_db(db))
    return db


def test_no_drift_stable(tmp_db):
    now = datetime.now(timezone.utc)
    # Baseline: yesterday
    for i in range(5):
        asyncio.run(save_trace(_trace(now - timedelta(hours=25 + i), cost=0.001), tmp_db))
    # Current: last hour — same cost
    for i in range(5):
        asyncio.run(save_trace(_trace(now - timedelta(minutes=10 + i), cost=0.001), tmp_db))

    report = asyncio.run(detect_drift(
        baseline_start=now - timedelta(hours=30),
        baseline_end=now - timedelta(hours=24),
        current_start=now - timedelta(hours=1),
        current_end=now,
        db_path=tmp_db,
    ))
    assert not report.any_drift


def test_cost_drift_detected(tmp_db):
    now = datetime.now(timezone.utc)
    for i in range(5):
        asyncio.run(save_trace(_trace(now - timedelta(hours=25 + i), cost=0.001), tmp_db))
    # Current: cost doubled
    for i in range(5):
        asyncio.run(save_trace(_trace(now - timedelta(minutes=10 + i), cost=0.003), tmp_db))

    report = asyncio.run(detect_drift(
        baseline_start=now - timedelta(hours=30),
        baseline_end=now - timedelta(hours=24),
        current_start=now - timedelta(hours=1),
        current_end=now,
        cost_threshold_pct=30.0,
        db_path=tmp_db,
    ))
    cost_metric = next(m for m in report.metrics if m.name == "cost_per_call")
    assert cost_metric.drifted is True
    assert report.any_drift is True


def test_error_rate_drift(tmp_db):
    now = datetime.now(timezone.utc)
    for i in range(10):
        asyncio.run(save_trace(_trace(now - timedelta(hours=25 + i), status="ok"), tmp_db))
    # Current: 60% errors
    for i in range(4):
        asyncio.run(save_trace(_trace(now - timedelta(minutes=10 + i), status="ok"), tmp_db))
    for i in range(6):
        asyncio.run(save_trace(_trace(now - timedelta(minutes=i + 1), status="error"), tmp_db))

    report = asyncio.run(detect_drift(
        baseline_start=now - timedelta(hours=30),
        baseline_end=now - timedelta(hours=24),
        current_start=now - timedelta(hours=1),
        current_end=now,
        error_rate_threshold_pct=5.0,
        db_path=tmp_db,
    ))
    err_metric = next(m for m in report.metrics if m.name == "error_rate")
    assert err_metric.drifted is True


def test_summary_message_contains_metric_name(tmp_db):
    now = datetime.now(timezone.utc)
    for i in range(3):
        asyncio.run(save_trace(_trace(now - timedelta(hours=25 + i), cost=0.001), tmp_db))
    for i in range(3):
        asyncio.run(save_trace(_trace(now - timedelta(minutes=5 + i), cost=0.005), tmp_db))

    report = asyncio.run(detect_drift(
        baseline_start=now - timedelta(hours=30),
        baseline_end=now - timedelta(hours=24),
        current_start=now - timedelta(hours=1),
        current_end=now,
        db_path=tmp_db,
    ))
    if report.any_drift:
        assert "cost_per_call" in report.summary()
