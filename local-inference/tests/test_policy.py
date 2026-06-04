"""Unit tests for the routing policy."""

import pytest
from router.classifier import ClassificationResult, Difficulty
from router.policy import RouterPolicy


def _clf_result(diff: Difficulty, conf: float = 0.8) -> ClassificationResult:
    return ClassificationResult(
        difficulty=diff,
        confidence=conf,
        reason="test",
        method="heuristic",
        feature_scores={},
    )


@pytest.fixture
def policy():
    return RouterPolicy(
        small_engine_name="small",
        large_engine_name="large",
        uncertain_fallback="large",
        confidence_threshold=0.55,
    )


def test_simple_routes_to_small(policy):
    d = policy.decide(_clf_result(Difficulty.SIMPLE, 0.9))
    assert d.engine_name == "small"
    assert d.savings_eligible is True


def test_complex_routes_to_large(policy):
    d = policy.decide(_clf_result(Difficulty.COMPLEX, 0.85))
    assert d.engine_name == "large"
    assert d.savings_eligible is False


def test_uncertain_routes_to_large_by_default(policy):
    d = policy.decide(_clf_result(Difficulty.UNCERTAIN, 0.4))
    assert d.engine_name == "large"
    assert d.fallback_used is True


def test_uncertain_can_fallback_to_small():
    p = RouterPolicy("small", "large", uncertain_fallback="small")
    d = p.decide(_clf_result(Difficulty.UNCERTAIN, 0.3))
    assert d.engine_name == "small"
    assert d.savings_eligible is True


def test_low_confidence_simple_becomes_uncertain(policy):
    # confidence below threshold → treated as uncertain
    d = policy.decide(_clf_result(Difficulty.SIMPLE, conf=0.3))
    assert d.fallback_used is True


def test_stats_accumulate(policy):
    policy.decide(_clf_result(Difficulty.SIMPLE, 0.9))
    policy.decide(_clf_result(Difficulty.SIMPLE, 0.9))
    policy.decide(_clf_result(Difficulty.COMPLEX, 0.9))
    r = policy.report()
    assert r["total_queries"] == 3
    assert r["routed_to_small"] == 2
    assert r["routed_to_large"] == 1


def test_small_fraction(policy):
    for _ in range(4):
        policy.decide(_clf_result(Difficulty.SIMPLE, 0.9))
    for _ in range(1):
        policy.decide(_clf_result(Difficulty.COMPLEX, 0.9))
    r = policy.report()
    assert r["small_fraction"] == pytest.approx(0.8)
