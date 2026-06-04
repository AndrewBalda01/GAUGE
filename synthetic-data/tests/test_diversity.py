"""Tests for diversity analysis."""

import pytest
from spec.schema import DifficultyLevel, SyntheticExample
from quality.diversity import analyse_diversity, _tfidf, _build_vocabulary, _tokenize


def _ex(i: int, topic: str, text: str) -> SyntheticExample:
    return SyntheticExample(
        id=f"div-{i}", topic=topic,
        input=text,
        expected=(
            f"This covers the legal principles of {topic} law, including "
            f"key concepts and their application in practice."
        ),
    )


def test_empty_returns_zero_coverage():
    r = analyse_diversity([])
    assert r.n_examples == 0
    assert r.coverage_score == 0.0


def test_single_example():
    ex = _ex(0, "contract", "What is consideration in contract law?")
    r = analyse_diversity([ex], n_clusters=3)
    assert r.n_examples == 1
    assert 0.0 <= r.coverage_score <= 1.0


def test_cluster_count_capped_at_n_examples():
    examples = [_ex(i, "tort", f"Question {i} about negligence") for i in range(3)]
    r = analyse_diversity(examples, n_clusters=10)
    assert r.n_clusters <= 3


def test_diverse_examples_have_reasonable_balance():
    topics = ["contract", "tort", "criminal", "ip", "privacy", "property"]
    examples = [
        _ex(i, t, f"What is the key principle in {t} law regarding {t} doctrine?")
        for i, t in enumerate(topics * 2)
    ]
    r = analyse_diversity(examples, n_clusters=6)
    assert r.balance_score >= 0.0
    assert r.coverage_score >= 0.5


def test_tokenize_removes_stopwords():
    tokens = _tokenize("What is a contract and why is it important")
    assert "a" not in tokens
    assert "is" not in tokens
    assert "contract" in tokens


def test_tfidf_vectors_normalised():
    import math
    docs = [["contract", "consideration", "offer"], ["tort", "negligence", "duty"]]
    vocab = _build_vocabulary(docs, min_df=1)
    vecs = _tfidf(docs, vocab)
    for v in vecs:
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6 or norm < 1e-9   # normalised or zero
