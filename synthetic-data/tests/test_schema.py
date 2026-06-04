"""Tests for the SyntheticExample schema and eval-harness compatibility."""

import pytest
from pydantic import ValidationError
from spec.schema import DifficultyLevel, SyntheticExample


def _valid() -> dict:
    return dict(
        id="s-001",
        topic="contract",
        input="What is consideration in contract law?",
        expected=(
            "Consideration is something of value exchanged between parties "
            "that makes a contract legally binding."
        ),
    )


def test_valid_example_constructs():
    ex = SyntheticExample(**_valid())
    assert ex.id == "s-001"
    assert ex.difficulty == DifficultyLevel.BASIC


def test_empty_input_raises():
    with pytest.raises(ValidationError):
        SyntheticExample(**{**_valid(), "input": "   "})


def test_empty_expected_raises():
    with pytest.raises(ValidationError):
        SyntheticExample(**{**_valid(), "expected": ""})


def test_strips_whitespace():
    ex = SyntheticExample(**{**_valid(), "input": "  What is consideration?  "})
    assert ex.input == "What is consideration?"


def test_to_eval_harness_dict_keys():
    ex = SyntheticExample(**_valid())
    d = ex.to_eval_harness_dict()
    assert set(d.keys()) == {"id", "input", "expected", "metrics", "tags"}


def test_to_eval_harness_dict_tags_include_topic():
    ex = SyntheticExample(**{**_valid(), "subtopic": "formation"})
    d = ex.to_eval_harness_dict()
    assert "contract" in d["tags"]
    assert "formation" in d["tags"]


def test_to_eval_harness_dict_no_pipeline_fields():
    ex = SyntheticExample(**_valid())
    d = ex.to_eval_harness_dict()
    assert "stage_passed" not in d
    assert "generation_model" not in d


def test_stage_passed_accumulates():
    ex = SyntheticExample(**_valid())
    ex.stage_passed.append("validate")
    ex.stage_passed.append("dedup")
    assert len(ex.stage_passed) == 2


def test_seed_metadata_stored():
    ex = SyntheticExample(**{
        **_valid(),
        "seed_persona": "a law student",
        "seed_scenario": "exam revision",
    })
    assert ex.seed_persona == "a law student"
