"""Unit tests for the perf benchmark module."""

import asyncio
import math
import pytest
from benchmark.perf import benchmark_engine, _percentile
from serving.engine import MockEngine


def test_percentile_basic():
    assert _percentile([1, 2, 3, 4, 5], 0.5) == pytest.approx(3.0)


def test_percentile_empty():
    assert math.isnan(_percentile([], 0.5))


def test_benchmark_engine_mock():
    async def _run():
        engine = MockEngine(name="bench-mock", tokens_per_second=100.0, ttft_ms=40.0)
        await engine.load()
        return await benchmark_engine(engine, n_runs=5, max_tokens=16)

    result = asyncio.run(_run())
    assert result.n_runs == 5
    assert result.latency_ms_mean > 0
    assert result.tokens_per_second_mean > 0
    assert result.ttft_ms_mean == pytest.approx(40.0)
    assert len(result.latencies_ms) == 5


def test_benchmark_to_dict_has_all_keys():
    async def _run():
        engine = MockEngine()
        await engine.load()
        return await benchmark_engine(engine, n_runs=3, max_tokens=8)

    result = asyncio.run(_run())
    d = result.to_dict()
    for key in ("latency_p50_ms", "latency_p95_ms", "tok_per_sec_mean", "vram_peak_mb"):
        assert key in d


def test_benchmark_percentiles_ordering():
    async def _run():
        engine = MockEngine()
        await engine.load()
        return await benchmark_engine(engine, n_runs=10, max_tokens=16)

    result = asyncio.run(_run())
    assert result.latency_ms_p50 <= result.latency_ms_p95 <= result.latency_ms_p99
