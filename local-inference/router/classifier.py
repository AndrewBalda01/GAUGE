"""
Query difficulty classifier.

Two tiers of classification:
  1. Heuristic  — O(1), no model needed, ~95% of cases.
  2. LLM-based  — optional, called when heuristic is uncertain.

Difficulty levels:
  SIMPLE  → route to small fast model
  COMPLEX → route to large accurate model
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Difficulty(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    UNCERTAIN = "uncertain"   # heuristic not confident; escalate to LLM classifier


@dataclass
class ClassificationResult:
    difficulty: Difficulty
    confidence: float           # 0–1
    reason: str
    method: str                 # "heuristic" | "llm"
    feature_scores: dict[str, float]


# ---------------------------------------------------------------------------
# Heuristic features
# ---------------------------------------------------------------------------

_CODE_PATTERN = re.compile(
    r"```|\bdef\b|\bclass\b|\bimport\b|\bfunction\b|\breturn\b|"
    r"\b(SELECT|INSERT|UPDATE|DELETE)\b",
    re.IGNORECASE,
)

_MATH_PATTERN = re.compile(
    r"\b(integral|derivative|matrix|eigenvalue|probability|calculus|"
    r"gradient|theorem|proof|equation)\b",
    re.IGNORECASE,
)

_MULTI_STEP_PATTERN = re.compile(
    r"\b(compare|contrast|analyse|analyze|evaluate|discuss|explain .{0,20} and .{0,20}|"
    r"pros and cons|advantages and disadvantages|step.by.step|in detail)\b",
    re.IGNORECASE,
)

_LEGAL_COMPLEX_PATTERN = re.compile(
    r"\b(argue|distinguish|apply|analyse|ratio decidendi|obiter|stare decisis|"
    r"estoppel|constructive|vicarious|fiduciary|ultra vires)\b",
    re.IGNORECASE,
)

_SIMPLE_PATTERN = re.compile(
    r"^(what is|define|what does|what are|who is|when was|where is)\b",
    re.IGNORECASE,
)


def _score_features(query: str) -> dict[str, float]:
    q = query.strip()
    words = q.split()
    n = len(words)

    return {
        "length": min(n / 50, 1.0),                        # 0→short, 1→50+ words
        "has_code": float(bool(_CODE_PATTERN.search(q))),
        "has_math": float(bool(_MATH_PATTERN.search(q))),
        "multi_step": float(bool(_MULTI_STEP_PATTERN.search(q))),
        "legal_complex": float(bool(_LEGAL_COMPLEX_PATTERN.search(q))),
        "is_simple_question": float(bool(_SIMPLE_PATTERN.match(q))),
        "has_question_mark": float("?" in q),
        "multiple_sentences": float(q.count(".") >= 2 or q.count(";") >= 1),
    }


_COMPLEX_WEIGHTS = {
    "length": 0.20,
    "has_code": 0.45,
    "has_math": 0.40,
    "multi_step": 0.40,
    "legal_complex": 0.35,
    "multiple_sentences": 0.15,
}

_SIMPLE_WEIGHTS = {
    "is_simple_question": 0.40,
    "has_question_mark": 0.10,
}

_UNCERTAIN_BAND = (0.30, 0.55)   # raw score in this range → uncertain


class HeuristicClassifier:
    """
    Fast, stateless query difficulty classifier based on hand-crafted features.
    No model inference required.
    """

    def classify(self, query: str) -> ClassificationResult:
        features = _score_features(query)

        complex_score = sum(
            features.get(k, 0) * w for k, w in _COMPLEX_WEIGHTS.items()
        )
        simple_boost = sum(
            features.get(k, 0) * w for k, w in _SIMPLE_WEIGHTS.items()
        )
        raw = complex_score - simple_boost
        # Clamp to [0, 1]
        raw = max(0.0, min(1.0, raw))

        lo, hi = _UNCERTAIN_BAND
        if raw < lo:
            difficulty = Difficulty.SIMPLE
            confidence = 1.0 - (raw / lo) * 0.4  # 0.6→1.0
        elif raw > hi:
            difficulty = Difficulty.COMPLEX
            confidence = 0.6 + ((raw - hi) / (1.0 - hi)) * 0.4
        else:
            difficulty = Difficulty.UNCERTAIN
            # confidence represents distance from the centre of the band
            mid = (lo + hi) / 2
            confidence = abs(raw - mid) / ((hi - lo) / 2) * 0.5

        top_features = sorted(features.items(), key=lambda x: -x[1])
        reason = ", ".join(f"{k}={v:.2f}" for k, v in top_features[:3] if v > 0)

        return ClassificationResult(
            difficulty=difficulty,
            confidence=round(confidence, 3),
            reason=reason or "no strong signal",
            method="heuristic",
            feature_scores=features,
        )
