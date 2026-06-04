"""Tests for heuristic filtering (no API calls)."""

import pytest
from spec.schema import DifficultyLevel, SyntheticExample
from quality.filter import heuristic_filter, filter_heuristic, _repetition_score


def _ex(q: str, a: str) -> SyntheticExample:
    return SyntheticExample(id="t-1", topic="contract", input=q, expected=a)


_GOOD_Q = "What is consideration in contract law?"
_GOOD_A = (
    "Consideration is something of value exchanged between parties "
    "that makes a contract legally binding. It can be an act, "
    "forbearance, or a promise, and must be sufficient but need not be adequate."
)


def test_good_example_passes():
    r = heuristic_filter(_ex(_GOOD_Q, _GOOD_A))
    assert r.passed


def test_boilerplate_fails():
    r = heuristic_filter(_ex(_GOOD_Q, "As an AI, I cannot provide legal advice. Please consult a lawyer."))
    assert not r.passed
    assert "boilerplate" in r.reason


def test_high_repetition_fails():
    repeated = ("the contract the contract the contract " * 10).strip() + "."
    r = heuristic_filter(_ex(_GOOD_Q, repeated))
    assert not r.passed
    assert "repetition" in r.reason


def test_too_many_questions_fails():
    q_answer = "What? Where? When? How? Why? Something?" * 2
    r = heuristic_filter(_ex(_GOOD_Q, q_answer))
    assert not r.passed


def test_repetition_score_low_for_varied_text():
    score = _repetition_score(_GOOD_A)
    assert score < 0.35


def test_repetition_score_high_for_repetitive():
    text = "the contract the contract the contract the contract the contract."
    score = _repetition_score(text)
    assert score > 0.35


def test_filter_heuristic_batch():
    good = _ex(_GOOD_Q, _GOOD_A)
    bad  = _ex(_GOOD_Q, "As an AI I cannot provide this. Please see a lawyer for advice.")
    result = filter_heuristic([good, bad])
    assert len(result.kept) == 1
    assert len(result.rejected_heuristic) == 1


def test_filter_stamps_stage():
    result = filter_heuristic([_ex(_GOOD_Q, _GOOD_A)])
    assert "filter" in result.kept[0].stage_passed


def test_filter_result_summary():
    result = filter_heuristic([_ex(_GOOD_Q, _GOOD_A)])
    s = result.summary()
    assert "kept" in s or "Filter" in s
