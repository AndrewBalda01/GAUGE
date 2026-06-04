"""Unit tests for the engine abstraction (MockEngine only — no hardware needed)."""

import asyncio
import pytest
from serving.engine import GenerateRequest, MockEngine


@pytest.fixture
def engine():
    return MockEngine(name="test-mock", tokens_per_second=100.0, ttft_ms=50.0)


def test_load_unload(engine):
    assert not engine.is_loaded
    asyncio.run(engine.load())
    assert engine.is_loaded
    asyncio.run(engine.unload())
    assert not engine.is_loaded


def test_generate_returns_response(engine):
    asyncio.run(engine.load())
    req = GenerateRequest(prompt="What is a contract?", max_tokens=32)
    resp = asyncio.run(engine.generate(req))
    assert resp.text
    assert resp.tokens_generated > 0
    assert resp.total_ms > 0
    assert resp.ttft_ms == pytest.approx(50.0)


def test_generate_stream_yields_tokens(engine):
    asyncio.run(engine.load())
    req = GenerateRequest(prompt="Define tort law.", max_tokens=16)

    async def _collect():
        tokens = []
        async for tok in engine.generate_stream(req):
            tokens.append(tok)
        return tokens

    tokens = asyncio.run(_collect())
    assert len(tokens) > 0
    assert "".join(tokens).strip() != ""


def test_vram_is_zero_for_mock(engine):
    asyncio.run(engine.load())
    assert engine.vram_mb == 0.0


def test_tokens_per_second_matches_config(engine):
    asyncio.run(engine.load())
    req = GenerateRequest(prompt="Hello", max_tokens=32)
    resp = asyncio.run(engine.generate(req))
    assert resp.tokens_per_second == pytest.approx(100.0)
