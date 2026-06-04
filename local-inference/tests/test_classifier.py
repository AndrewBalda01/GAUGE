"""Unit tests for the heuristic query classifier."""

import pytest
from router.classifier import Difficulty, HeuristicClassifier


@pytest.fixture
def clf():
    return HeuristicClassifier()


# --- Simple queries ---

def test_simple_definition_query(clf):
    r = clf.classify("What is a contract?")
    assert r.difficulty == Difficulty.SIMPLE
    assert r.confidence > 0.5


def test_simple_define_query(clf):
    r = clf.classify("Define consideration in contract law.")
    assert r.difficulty in (Difficulty.SIMPLE, Difficulty.UNCERTAIN)


# --- Complex queries ---

def test_complex_analysis_query(clf):
    r = clf.classify(
        "Compare and contrast the ratio decidendi and obiter dicta, "
        "analysing their role in the doctrine of stare decisis."
    )
    assert r.difficulty == Difficulty.COMPLEX
    assert r.confidence > 0.5


def test_complex_code_query(clf):
    r = clf.classify(
        "Write a Python function that validates a contract schema using Pydantic. "
        "```python\ndef validate(data): ...\n```"
    )
    assert r.difficulty == Difficulty.COMPLEX


def test_complex_multi_step(clf):
    r = clf.classify(
        "Analyse the pros and cons of arbitration versus litigation, "
        "discussing costs, enforceability, and confidentiality in detail."
    )
    assert r.difficulty == Difficulty.COMPLEX


# --- Confidence ---

def test_confidence_is_between_0_and_1(clf):
    for q in [
        "What is tort?",
        "Define negligence and explain its elements in depth.",
        "x",
    ]:
        r = clf.classify(q)
        assert 0.0 <= r.confidence <= 1.0


def test_classification_result_has_reason(clf):
    r = clf.classify("Explain fiduciary duty.")
    assert r.reason
    assert r.method == "heuristic"


def test_feature_scores_are_floats(clf):
    r = clf.classify("What is promissory estoppel?")
    for k, v in r.feature_scores.items():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
