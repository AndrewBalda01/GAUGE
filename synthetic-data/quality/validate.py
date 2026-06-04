"""
Structural validation stage.

Checks that each SyntheticExample passes basic quality gates
before entering the dedup and filter stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from spec.schema import SyntheticExample


@dataclass
class ValidationResult:
    passed: bool
    reason: str = ""


_MIN_INPUT_WORDS  = 5
_MIN_ANSWER_WORDS = 15
_MAX_INPUT_CHARS  = 800
_MAX_ANSWER_CHARS = 2000


def validate(ex: SyntheticExample) -> ValidationResult:
    """Run all structural checks; returns first failure or pass."""
    input_words  = len(ex.input.split())
    answer_words = len(ex.expected.split())

    if input_words < _MIN_INPUT_WORDS:
        return ValidationResult(False, f"question too short ({input_words} words < {_MIN_INPUT_WORDS})")

    if answer_words < _MIN_ANSWER_WORDS:
        return ValidationResult(False, f"answer too short ({answer_words} words < {_MIN_ANSWER_WORDS})")

    if len(ex.input) > _MAX_INPUT_CHARS:
        return ValidationResult(False, f"question too long ({len(ex.input)} chars)")

    if len(ex.expected) > _MAX_ANSWER_CHARS:
        return ValidationResult(False, f"answer too long ({len(ex.expected)} chars)")

    # Answer should not simply repeat the question
    q_lower = ex.input.lower()
    a_lower = ex.expected.lower()
    if a_lower.startswith(q_lower[:30]):
        return ValidationResult(False, "answer appears to repeat the question")

    # Answer must end with sentence-terminal punctuation
    if ex.expected.strip()[-1] not in ".!?)":
        return ValidationResult(False, "answer does not end with punctuation")

    return ValidationResult(True)


def validate_batch(
    examples: list[SyntheticExample],
) -> tuple[list[SyntheticExample], list[tuple[SyntheticExample, str]]]:
    """
    Returns (passed, failed) where failed items include the failure reason.
    Also stamps stage_passed on survivors.
    """
    passed, failed = [], []
    for ex in examples:
        r = validate(ex)
        if r.passed:
            ex.stage_passed.append("validate")
            passed.append(ex)
        else:
            failed.append((ex, r.reason))
    return passed, failed
