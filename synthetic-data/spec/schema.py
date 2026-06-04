"""
Target schema for synthetic examples.

A SyntheticExample is the unit produced by the pipeline and consumed by:
  - eval-harness (Project 01): fed directly as TestCase
  - fine-tuning pipelines (future)

The schema is intentionally compatible with eval-harness TestCase so that
produced examples can be appended to legal_qa.v1.jsonl without transformation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DifficultyLevel(str, Enum):
    BASIC      = "basic"       # single-concept factual question
    APPLIED    = "applied"     # requires applying a rule to a scenario
    ANALYTICAL = "analytical"  # compare / critique / argue


class SyntheticExample(BaseModel):
    """One generated Q&A pair, validated and ready for eval-harness."""
    id: str
    input: str                              # the question / prompt
    expected: str                           # reference answer
    topic: str                              # e.g. "contract", "tort", "ip"
    subtopic: str = ""                      # e.g. "formation", "breach"
    difficulty: DifficultyLevel = DifficultyLevel.BASIC
    tags: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=lambda: ["judge"])
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Pipeline provenance (not written to eval-harness JSONL)
    generation_model: str = ""
    seed_persona: str = ""
    seed_scenario: str = ""
    stage_passed: list[str] = Field(default_factory=list)  # validate, dedup, filter

    @field_validator("input", "expected")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be empty")
        return v.strip()

    def to_eval_harness_dict(self) -> dict:
        """Produce a dict compatible with eval-harness JSONL format."""
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "metrics": self.metrics,
            "tags": [self.topic] + ([self.subtopic] if self.subtopic else []) + self.tags,
        }
