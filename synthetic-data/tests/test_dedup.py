"""Tests for exact and semantic deduplication."""

import pytest
from spec.schema import SyntheticExample, DifficultyLevel
from quality.dedup import (
    exact_dedup, semantic_dedup, full_dedup,
    _shingles, _minhash_signature, _jaccard_estimate,
)


def _ex(i: int, q: str, a: str = None) -> SyntheticExample:
    return SyntheticExample(
        id=f"ex-{i}",
        topic="contract",
        input=q,
        expected=a or (
            "Consideration is something of value exchanged in a contract "
            "making it legally binding between the parties involved."
        ),
    )


# --- Exact dedup ---

def test_exact_dedup_keeps_unique():
    examples = [_ex(1, "What is consideration?"), _ex(2, "What is an offer?")]
    kept, dupes = exact_dedup(examples)
    assert len(kept) == 2
    assert len(dupes) == 0


def test_exact_dedup_removes_identical():
    q = "What is consideration?"
    examples = [_ex(1, q), _ex(2, q)]
    kept, dupes = exact_dedup(examples)
    assert len(kept) == 1
    assert len(dupes) == 1


def test_exact_dedup_normalises_case():
    examples = [_ex(1, "What is consideration?"), _ex(2, "WHAT IS CONSIDERATION?")]
    kept, dupes = exact_dedup(examples)
    assert len(kept) == 1


# --- MinHash internals ---

def test_shingles_nonempty():
    s = _shingles("consideration is value exchanged")
    assert len(s) > 0


def test_shingles_short_text():
    s = _shingles("hi")
    assert isinstance(s, set)


def test_minhash_signature_length():
    sig = _minhash_signature("what is consideration in contract law")
    assert len(sig) == 128


def test_jaccard_identical():
    sig = _minhash_signature("what is consideration in contract law")
    assert _jaccard_estimate(sig, sig) == pytest.approx(1.0)


def test_jaccard_different():
    s1 = _minhash_signature("consideration is value in a contract forming the basis")
    s2 = _minhash_signature("negligence requires duty of care breach causation damages")
    j = _jaccard_estimate(s1, s2)
    assert j < 0.5


# --- Semantic dedup ---

def test_semantic_dedup_removes_near_duplicates():
    # Use almost-identical sentences to guarantee they exceed the threshold
    q  = "consideration contract law parties binding offer acceptance formation"
    q2 = "consideration contract law parties binding offer acceptance formation agreement"
    examples = [
        _ex(1, q),
        _ex(2, q2),
        _ex(3, "negligence tort duty care breach causation damages personal injury"),
    ]
    kept, near = semantic_dedup(examples, threshold=0.7)
    # q and q2 share many trigrams; one should be flagged as near-dup
    assert len(kept) <= 2
    assert len(kept) + len(near) == 3


# --- Full dedup ---

def test_full_dedup_stamps_stage():
    examples = [
        _ex(1, "What is consideration in contract law?"),
        _ex(2, "Define negligence in tort law and its elements."),
    ]
    result = full_dedup(examples)
    assert all("dedup" in ex.stage_passed for ex in result.kept)


def test_full_dedup_summary_contains_counts():
    examples = [_ex(i, f"Question about topic {i} in legal context.") for i in range(5)]
    result = full_dedup(examples)
    s = result.summary()
    assert "in" in s and "kept" in s
