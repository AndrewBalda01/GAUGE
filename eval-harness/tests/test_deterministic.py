"""Unit tests for deterministic metrics."""

import pytest
from store.schema import MetricKind, TestCase as EvalTestCase
from evals.metrics.deterministic import score_deterministic


def _case(**kwargs) -> EvalTestCase:
    defaults = {"id": "test-1", "input": "Q", "metrics": [MetricKind.EXACT_MATCH]}
    return EvalTestCase(**{**defaults, **kwargs})


def test_exact_match_true():
    c = _case(expected="A void contract has no legal effect.")
    s = score_deterministic(c, "A void contract has no legal effect.")
    assert s.exact_match is True


def test_exact_match_normalises_whitespace():
    c = _case(expected="a void contract has no legal effect .")
    s = score_deterministic(c, "  A void contract has   no legal effect .  ")
    assert s.exact_match is True


def test_exact_match_false():
    c = _case(expected="Correct answer.")
    s = score_deterministic(c, "Different answer.")
    assert s.exact_match is False


def test_no_expected_no_exact():
    c = _case(expected=None)
    s = score_deterministic(c, "anything")
    assert s.exact_match is None


def test_regex_match():
    c = _case(
        expected=None,
        metadata={"expected_pattern": r"consent.*contract"},
        metrics=[MetricKind.REGEX],
    )
    s = score_deterministic(c, "The lawful bases include consent and contract.")
    assert s.regex_match is True


def test_regex_no_match():
    c = _case(
        expected=None,
        metadata={"expected_pattern": r"specific performance"},
        metrics=[MetricKind.REGEX],
    )
    s = score_deterministic(c, "Only damages are available.")
    assert s.regex_match is False


def test_json_schema_valid():
    c = _case(
        expected=None,
        expected_schema={"type": "object", "required": ["score", "reasoning"]},
        metrics=[MetricKind.JSON_SCHEMA],
    )
    s = score_deterministic(c, '{"score": 4, "reasoning": "good"}')
    assert s.json_schema_valid is True


def test_json_schema_invalid_missing_key():
    c = _case(
        expected=None,
        expected_schema={"type": "object", "required": ["score", "reasoning"]},
        metrics=[MetricKind.JSON_SCHEMA],
    )
    s = score_deterministic(c, '{"score": 4}')
    assert s.json_schema_valid is False


def test_json_schema_not_json():
    c = _case(
        expected=None,
        expected_schema={"type": "object"},
        metrics=[MetricKind.JSON_SCHEMA],
    )
    s = score_deterministic(c, "plain text")
    assert s.json_schema_valid is False
