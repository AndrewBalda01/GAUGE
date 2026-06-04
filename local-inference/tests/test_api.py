"""Integration tests for the FastAPI endpoints using TestClient + mock engines."""

import asyncio
import pytest
from fastapi.testclient import TestClient

from api.main import app, _state, configure_engines
from serving.engine import MockEngine


@pytest.fixture(autouse=True)
def reset_state():
    _state.engines.clear()
    _state.classifier = None
    _state.policy = None
    yield
    _state.engines.clear()
    _state.classifier = None
    _state.policy = None


@pytest.fixture
def client_with_engines():
    asyncio.run(configure_engines("mock-small", "mock-large"))
    return TestClient(app)


@pytest.fixture
def client_no_router():
    engine = MockEngine(name="solo-mock")
    asyncio.run(engine.load())
    _state.engines["solo-mock"] = engine
    return TestClient(app)


# --- Health ---

def test_health_no_engines():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["engines_loaded"] == 0


def test_health_with_engines(client_with_engines):
    r = client_with_engines.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["engines_loaded"] == 2
    assert d["router_active"] is True


# --- Models ---

def test_list_models_empty():
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"] == []


def test_list_models_with_engines(client_with_engines):
    r = client_with_engines.get("/v1/models")
    data = r.json()["data"]
    assert len(data) == 2
    names = {m["id"] for m in data}
    assert "mock-small" in names
    assert "mock-large" in names


# --- Chat completions ---

def test_chat_completion_no_engine():
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is a contract?"}]
    })
    assert r.status_code == 503


def test_chat_completion_simple_query(client_with_engines):
    r = client_with_engines.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is tort law?"}],
        "max_tokens": 32,
    })
    assert r.status_code == 200
    d = r.json()
    assert d["choices"][0]["message"]["role"] == "assistant"
    assert d["choices"][0]["message"]["content"]
    assert "x_routing" in d
    assert d["x_routing"]["engine"] in ("mock-small", "mock-large")


def test_chat_completion_with_system(client_with_engines):
    r = client_with_engines.post("/v1/chat/completions", json={
        "messages": [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": "Define negligence."},
        ],
        "max_tokens": 32,
    })
    assert r.status_code == 200


# --- Completions ---

def test_completion(client_with_engines):
    r = client_with_engines.post("/v1/completions", json={
        "prompt": "Define estoppel.",
        "max_tokens": 32,
    })
    assert r.status_code == 200
    d = r.json()
    assert d["choices"][0]["text"]


# --- Router stats ---

def test_router_stats_not_configured():
    client = TestClient(app)
    r = client.get("/router/stats")
    assert r.status_code == 200
    assert "not configured" in r.json().get("router", "")


def test_router_stats_after_requests(client_with_engines):
    for _ in range(3):
        client_with_engines.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is a contract?"}],
            "max_tokens": 16,
        })
    r = client_with_engines.get("/router/stats")
    d = r.json()
    assert d["total_queries"] == 3
