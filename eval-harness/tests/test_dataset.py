"""Integration test: load and verify the bundled dataset."""

import pytest
from evals.cases import load_dataset, filter_cases


def test_load_legal_qa():
    cases, manifest = load_dataset("legal_qa", "v1")
    assert len(cases) == 60
    assert manifest.case_count == 60
    assert manifest.sha256  # non-empty


def test_hash_integrity():
    """load_dataset raises on hash mismatch — here we just verify it loads clean."""
    cases, manifest = load_dataset("legal_qa", "v1")
    import hashlib
    from pathlib import Path
    path = Path(__file__).parent.parent / "datasets" / "legal_qa.v1.jsonl"
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h == manifest.sha256


def test_filter_by_tag():
    cases, _ = load_dataset("legal_qa", "v1")
    contract = filter_cases(cases, tags=["contract"])
    assert len(contract) > 0
    assert all("contract" in c.tags for c in contract)


def test_filter_by_id():
    cases, _ = load_dataset("legal_qa", "v1")
    subset = filter_cases(cases, ids=["lq-001", "lq-002"])
    assert len(subset) == 2
    assert {c.id for c in subset} == {"lq-001", "lq-002"}


def test_all_cases_have_id_and_input():
    cases, _ = load_dataset("legal_qa", "v1")
    for c in cases:
        assert c.id
        assert c.input
