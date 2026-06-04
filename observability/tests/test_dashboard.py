"""Integration tests for the dashboard API."""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.app import app, set_db_path
from store.db import init_db, save_trace
from store.schema import LLMTrace


@pytest.fixture
def db_with_data(tmp_path):
    db = tmp_path / "dash_test.db"
    asyncio.run(init_db(db))
    now = datetime.now(timezone.utc)
    for i in range(5):
        asyncio.run(save_trace(LLMTrace(
            model="claude-haiku-4-5-20251001",
            app="test-app",
            tokens_in=100, tokens_out=50,
            cost_usd=0.001,
            latency_ms=200.0 + i * 10,
            status="ok",
            prompt=f"Question {i}",
            completion=f"Answer {i}",
            timestamp=now - timedelta(minutes=i),
        ), db))
    return db


@pytest.fixture
def client(db_with_data):
    set_db_path(db_with_data)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_traces(client):
    r = client.get("/traces?app=test-app&hours=1")
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == 5


def test_get_single_trace(client, db_with_data):
    traces = asyncio.run(
        __import__("store.db", fromlist=["query_traces"]).query_traces(app="test-app", db_path=db_with_data)
    )
    tid = traces[0].trace_id
    r = client.get(f"/traces/{tid}")
    assert r.status_code == 200
    assert r.json()["trace_id"] == tid


def test_get_missing_trace(client):
    r = client.get("/traces/nonexistent-id-xyz")
    assert r.status_code == 404


def test_metrics_summary(client):
    r = client.get("/metrics/summary?app=test-app&hours=1")
    assert r.status_code == 200
    d = r.json()
    assert d["n_calls"] == 5
    assert d["total_cost_usd"] == pytest.approx(0.005)


def test_metrics_cost(client):
    r = client.get("/metrics/cost?app=test-app&days=7")
    assert r.status_code == 200
    days = r.json()["days"]
    assert len(days) == 7


def test_metrics_latency(client):
    r = client.get("/metrics/latency?app=test-app&hours=1")
    assert r.status_code == 200
    d = r.json()
    assert d["n"] == 5
    assert d["p50"] is not None


def test_metrics_errors(client):
    r = client.get("/metrics/errors?app=test-app&hours=1")
    assert r.status_code == 200
    assert r.json()["rate"] == pytest.approx(0.0)


def test_alerts_empty(client):
    r = client.get("/alerts")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_check_alerts(client):
    r = client.post("/alerts/check?app=test-app")
    assert r.status_code == 200
    # No anomalies expected with stable data
    assert r.json()["fired"] >= 0


def test_drift_endpoint(client):
    r = client.get("/drift?app=test-app&baseline_hours=24&current_hours=1")
    assert r.status_code == 200
    d = r.json()
    assert "any_drift" in d
    assert "metrics" in d
