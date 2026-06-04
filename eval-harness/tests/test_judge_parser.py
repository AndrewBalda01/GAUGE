"""Unit tests for judge response parsing (no API calls)."""

import pytest
from evals.metrics.judge import _parse_judge_response, cohens_kappa


_VALID_JSON = """{
  "correctness": {"score": 4, "reasoning": "Mostly correct."},
  "completeness": {"score": 3, "reasoning": "Misses one point."},
  "format_adherence": {"score": 5, "reasoning": "Well structured."},
  "hallucination": {"detected": false, "evidence": "none"}
}"""


def test_parse_valid_response():
    s = _parse_judge_response(_VALID_JSON)
    assert s.correctness is not None
    assert s.correctness.score == 4
    assert s.completeness.score == 3
    assert s.format_adherence.score == 5
    assert s.hallucination_detected is False


def test_parse_fenced_markdown():
    fenced = f"```json\n{_VALID_JSON}\n```"
    s = _parse_judge_response(fenced)
    assert s.correctness.score == 4


def test_parse_invalid_json_returns_empty():
    s = _parse_judge_response("not json at all")
    assert s.correctness is None
    assert s.raw_response == "not json at all"


def test_cohens_kappa_perfect():
    labels = [4, 3, 5, 2, 4]
    assert cohens_kappa(labels, labels) == 1.0


def test_cohens_kappa_negative():
    # perfect systematic disagreement → kappa = -1
    j = [1, 5, 1, 5]  # binarised [0, 1, 0, 1]
    h = [5, 1, 5, 1]  # binarised [1, 0, 1, 0]
    k = cohens_kappa(j, h)
    assert k == pytest.approx(-1.0)


def test_cohens_kappa_empty():
    import math
    assert math.isnan(cohens_kappa([], []))
