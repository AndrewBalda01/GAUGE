"""Tests for store persistence (in-memory/temp DB)."""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from store.db import init_db, save_trace, load_trace, query_traces, save_alert, query_alerts
from store.schema import Alert, LLMTrace


def _trace(**kwargs) -> LLMTrace:
    defaults = dict(model="claude-haiku-4-5-20251001", tokens_in=100, tokens_out=50, latency_ms=200.0)
    return LLMTrace(**{**defaults, **kwargs})


@pytest.fixture
def tmp_db(tmp_path):
    db = tmp_path / "test.db"
    asyncio.run(init_db(db))
    return db


def test_save_and_load_trace(tmp_db):
    t = _trace(trace_id="abc123", app="eval-harness", prompt="What is tort?", completion="Tort is...")
    asyncio.run(save_trace(t, tmp_db))
    loaded = asyncio.run(load_trace("abc123", tmp_db))
    assert loaded is not None
    assert loaded.trace_id == "abc123"
    assert loaded.app == "eval-harness"
    assert loaded.completion == "Tort is..."


def test_load_nonexistent_returns_none(tmp_db):
    result = asyncio.run(load_trace("nonexistent-id", tmp_db))
    assert result is None


def test_query_traces_by_app(tmp_db):
    asyncio.run(save_trace(_trace(app="eval-harness"), tmp_db))
    asyncio.run(save_trace(_trace(app="local-inference"), tmp_db))
    asyncio.run(save_trace(_trace(app="eval-harness"), tmp_db))
    results = asyncio.run(query_traces(app="eval-harness", db_path=tmp_db))
    assert len(results) == 2
    assert all(t.app == "eval-harness" for t in results)


def test_query_traces_by_status(tmp_db):
    asyncio.run(save_trace(_trace(status="ok"), tmp_db))
    asyncio.run(save_trace(_trace(status="error", error="timeout"), tmp_db))
    ok_results = asyncio.run(query_traces(status="ok", db_path=tmp_db))
    err_results = asyncio.run(query_traces(status="error", db_path=tmp_db))
    assert len(ok_results) == 1
    assert len(err_results) == 1


def test_save_and_query_alert(tmp_db):
    a = Alert(kind="cost_spike", message="Cost rose 50%", severity="warning",
              metric_value=0.005, threshold=0.003, app="eval-harness")
    asyncio.run(save_alert(a, tmp_db))
    alerts = asyncio.run(query_alerts(db_path=tmp_db))
    assert len(alerts) == 1
    assert alerts[0].kind == "cost_spike"


def test_query_unacknowledged_alerts(tmp_db):
    asyncio.run(save_alert(Alert(kind="latency_spike", message="Slow", acknowledged=False), tmp_db))
    asyncio.run(save_alert(Alert(kind="cost_spike", message="Costly", acknowledged=True), tmp_db))
    unack = asyncio.run(query_alerts(acknowledged=False, db_path=tmp_db))
    assert len(unack) == 1
    assert unack[0].kind == "latency_spike"
