"""Pydantic models — source of truth for DB schema and serialization."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MetricKind(str, Enum):
    EXACT_MATCH = "exact_match"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    JUDGE = "judge"
    PASS_AT_K = "pass_at_k"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Dataset / test case
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    id: str
    input: str
    expected: str | None = None
    expected_schema: dict[str, Any] | None = None
    metrics: list[MetricKind] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    name: str
    version: str
    sha256: str
    case_count: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""


# ---------------------------------------------------------------------------
# Model / prompt configuration
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    model_id: str
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str = ""
    extra_params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-case result
# ---------------------------------------------------------------------------

class DeterministicScores(BaseModel):
    exact_match: bool | None = None
    regex_match: bool | None = None
    json_schema_valid: bool | None = None


class JudgeDimension(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


class JudgeScores(BaseModel):
    correctness: JudgeDimension | None = None
    completeness: JudgeDimension | None = None
    format_adherence: JudgeDimension | None = None
    hallucination_detected: bool | None = None
    hallucination_evidence: str | None = None
    raw_response: str = ""


class CaseResult(BaseModel):
    case_id: str
    run_id: str
    model_output: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    deterministic: DeterministicScores = Field(default_factory=DeterministicScores)
    judge: JudgeScores | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def judge_avg_score(self) -> float | None:
        if self.judge is None:
            return None
        scores = [
            d.score
            for d in [
                self.judge.correctness,
                self.judge.completeness,
                self.judge.format_adherence,
            ]
            if d is not None
        ]
        return sum(scores) / len(scores) if scores else None


# ---------------------------------------------------------------------------
# Full eval run
# ---------------------------------------------------------------------------

class EvalRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    config: ModelConfig
    status: RunStatus = RunStatus.PENDING
    results: list[CaseResult] = Field(default_factory=list)
    git_ref: str = ""
    branch: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    notes: str = ""

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def mean_latency_ms(self) -> float | None:
        lats = [r.latency_ms for r in self.results if r.error is None]
        return sum(lats) / len(lats) if lats else None

    @property
    def pass_rate(self) -> float | None:
        ok = [r for r in self.results if r.error is None]
        if not ok:
            return None
        passing = [r for r in ok if r.deterministic.exact_match is True]
        return len(passing) / len(ok)


# ---------------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------------

class RegressionReport(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    delta_judge_avg: float | None = None
    delta_pass_rate: float | None = None
    delta_cost_usd: float | None = None
    delta_latency_ms: float | None = None
    regression_detected: bool = False
    threshold_judge: float = 0.3
    threshold_pass_rate: float = 0.05
    details: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
