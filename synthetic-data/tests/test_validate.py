"""Tests for structural validation."""

import pytest
from spec.schema import DifficultyLevel, SyntheticExample
from quality.validate import validate, validate_batch


def _ex(**kwargs) -> SyntheticExample:
    defaults = dict(
        id="t-1", topic="contract",
        input="What is consideration in contract law?",
        expected=(
            "Consideration is something of value exchanged between parties "
            "that makes a contract legally binding. It can be an act, "
            "forbearance, or a promise, and must be sufficient."
        ),
    )
    return SyntheticExample(**{**defaults, **kwargs})


def test_valid_example_passes():
    assert validate(_ex()).passed


def test_short_question_fails():
    r = validate(_ex(input="What?"))
    assert not r.passed
    assert "short" in r.reason


def test_short_answer_fails():
    r = validate(_ex(expected="It is a rule."))
    assert not r.passed
    assert "short" in r.reason


def test_answer_repeating_question_fails():
    q = "What is consideration in contract law"
    r = validate(_ex(input=q + "?", expected=q + " refers to value exchanged."))
    assert not r.passed


def test_answer_without_terminal_punctuation_fails():
    r = validate(_ex(expected=(
        "Consideration is something of value exchanged between parties "
        "in a contract making it legally binding without punctuation"
    )))
    assert not r.passed


def test_validate_batch_separates():
    good = _ex()
    bad  = _ex(id="t-bad", input="Why?")
    passed, failed = validate_batch([good, bad])
    assert len(passed) == 1
    assert len(failed) == 1
    assert passed[0].id == "t-1"


def test_validate_stamps_stage(clear=None):
    ex = _ex()
    passed, _ = validate_batch([ex])
    assert "validate" in passed[0].stage_passed
