"""Unit tests for aggregate metrics."""

import pytest
from evals.metrics.aggregate import pass_at_k, percentiles, summarise_run
from store.schema import (
    CaseResult, DeterministicScores, EvalRun, ModelConfig, RunStatus
)


def _make_run(n_ok: int = 5, n_fail: int = 0) -> EvalRun:
    config = ModelConfig(model_id="test-model")
    run = EvalRun(
        dataset_name="legal_qa",
        dataset_version="v1",
        dataset_sha256="abc",
        config=config,
        status=RunStatus.COMPLETED,
    )
    for i in range(n_ok):
        run.results.append(CaseResult(
            case_id=f"c-{i}",
            run_id=run.id,
            model_output="ok",
            latency_ms=100.0 + i * 10,
            tokens_in=50,
            tokens_out=20,
            cost_usd=0.001,
            deterministic=DeterministicScores(exact_match=True),
        ))
    for i in range(n_fail):
        run.results.append(CaseResult(
            case_id=f"f-{i}",
            run_id=run.id,
            model_output="",
            latency_ms=0.0,
            error="timeout",
        ))
    return run


def test_pass_at_k_all_correct():
    assert pass_at_k(10, 10, 1) == 1.0


def test_pass_at_k_none_correct():
    assert pass_at_k(10, 0, 1) == 0.0


def test_pass_at_k_partial():
    v = pass_at_k(10, 5, 1)
    assert 0.0 < v < 1.0


def test_percentiles_basic():
    p = percentiles([100.0, 200.0, 300.0, 400.0, 500.0])
    assert p["p50"] == pytest.approx(300.0)
    assert p["p95"] >= 400.0


def test_percentiles_empty():
    p = percentiles([])
    import math
    assert math.isnan(p["p50"])


def test_summarise_run_pass_rate():
    run = _make_run(n_ok=4, n_fail=1)
    s = summarise_run(run)
    assert s["pass_rate"] == pytest.approx(1.0)   # all ok cases have exact_match=True
    assert s["ok"] == 4
    assert s["failed"] == 1


def test_summarise_run_cost():
    run = _make_run(n_ok=3)
    s = summarise_run(run)
    assert s["total_cost_usd"] == pytest.approx(0.003)
