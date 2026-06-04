"""Deterministic metrics: exact match, regex, JSON-schema validity."""

from __future__ import annotations

import json
import re

from store.schema import DeterministicScores, TestCase


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def score_deterministic(case: TestCase, model_output: str) -> DeterministicScores:
    scores = DeterministicScores()

    # --- exact match ---
    if case.expected is not None:
        scores.exact_match = _normalise(model_output) == _normalise(case.expected)

    # --- regex match (uses case.metadata["expected_pattern"] if present) ---
    pattern: str | None = case.metadata.get("expected_pattern")
    # also pick up extra fields stored in Pydantic model_extra (from JSONL)
    extra = getattr(case, "model_extra", None) or {}
    raw_pattern = extra.get("expected_pattern")
    if raw_pattern:
        pattern = raw_pattern
    if pattern:
        scores.regex_match = bool(re.search(pattern, model_output, re.IGNORECASE | re.DOTALL))

    # --- JSON schema validity ---
    if case.expected_schema is not None:
        try:
            parsed = json.loads(model_output)
            scores.json_schema_valid = _validate_schema(parsed, case.expected_schema)
        except json.JSONDecodeError:
            scores.json_schema_valid = False

    return scores


def _validate_schema(data: object, schema: dict) -> bool:
    """Minimal structural schema check (type + required keys). No jsonschema dep."""
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(data, dict):
            return False
        for key in schema.get("required", []):
            if key not in data:
                return False
    elif expected_type == "array":
        if not isinstance(data, list):
            return False
    elif expected_type == "string":
        if not isinstance(data, str):
            return False
    return True
