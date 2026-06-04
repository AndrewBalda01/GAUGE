"""Unit tests for cost estimation."""

import pytest
from collector.cost import estimate_cost


def test_known_model():
    cost = estimate_cost("claude-haiku-4-5-20251001", tokens_in=1_000_000, tokens_out=0)
    assert cost == pytest.approx(0.80)


def test_output_cost():
    cost = estimate_cost("claude-sonnet-4-6", tokens_in=0, tokens_out=1_000_000)
    assert cost == pytest.approx(15.0)


def test_combined_cost():
    cost = estimate_cost("claude-haiku-4-5-20251001", tokens_in=1000, tokens_out=500)
    expected = (1000 * 0.80 + 500 * 4.0) / 1_000_000
    assert cost == pytest.approx(expected)


def test_unknown_model_returns_zero():
    assert estimate_cost("unknown-model-xyz", 1000, 500) == 0.0


def test_local_model_returns_zero():
    assert estimate_cost("qwen3-0.6b-q4", 5000, 2000) == 0.0


def test_partial_model_id_match():
    # "claude-opus-4-8-extra" still matches "claude-opus-4-8"
    cost = estimate_cost("claude-opus-4-8-extra-variant", tokens_in=1_000_000, tokens_out=0)
    assert cost == pytest.approx(15.0)
