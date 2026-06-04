"""LLM-as-judge: structured rubric evaluation returning typed JSON."""

from __future__ import annotations

import json
import os
import time

import httpx

from store.schema import JudgeDimension, JudgeScores, ModelConfig

_JUDGE_SYSTEM = """\
You are a strict, impartial evaluator for AI-generated answers to legal questions.
Score each dimension independently. Return ONLY a JSON object — no prose before or after it.
"""

_JUDGE_TEMPLATE = """\
Question:
{question}

Reference answer:
{reference}

Model answer:
{answer}

Evaluate the model answer on the following dimensions.
Return a JSON object with exactly this structure:

{{
  "correctness": {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "completeness": {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "format_adherence": {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "hallucination": {{"detected": <true|false>, "evidence": "<quote or 'none'>"}}
}}

Scoring guide:
- correctness 5: fully accurate, no errors | 3: mostly correct, minor gap | 1: wrong
- completeness 5: covers all key points | 3: covers main but misses some | 1: superficial
- format_adherence 5: clear, well-structured | 3: readable but disorganised | 1: incoherent
- hallucination detected=true if the model states a falsehood or fabricates citations/cases.
"""


def _parse_judge_response(raw: str) -> JudgeScores:
    """Extract and validate the JSON block from the judge response."""
    # strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return JudgeScores(raw_response=raw)

    scores = JudgeScores(raw_response=raw)

    for field in ("correctness", "completeness", "format_adherence"):
        block = data.get(field)
        if isinstance(block, dict) and "score" in block:
            try:
                setattr(scores, field, JudgeDimension(
                    score=int(block["score"]),
                    reasoning=str(block.get("reasoning", "")),
                ))
            except Exception:
                pass

    hall = data.get("hallucination")
    if isinstance(hall, dict):
        scores.hallucination_detected = bool(hall.get("detected", False))
        scores.hallucination_evidence = str(hall.get("evidence", ""))

    return scores


async def judge_case(
    question: str,
    reference: str,
    model_output: str,
    judge_config: ModelConfig,
) -> JudgeScores:
    """Call the judge model and return structured scores."""
    prompt = _JUDGE_TEMPLATE.format(
        question=question,
        reference=reference,
        answer=model_output,
    )
    api_key = os.environ.get(judge_config.api_key_env, "")
    payload = {
        "model": judge_config.model_id,
        "max_tokens": 512,
        "temperature": 0.0,          # deterministic judge
        "system": _JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{judge_config.base_url}/messages",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()

    data = resp.json()
    raw = "".join(
        b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
    )
    return _parse_judge_response(raw)


# ---------------------------------------------------------------------------
# Calibration helper (Cohen's kappa vs human labels)
# ---------------------------------------------------------------------------

def cohens_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Simple Cohen's kappa for ordinal agreement (1-5 scale binned to pass/fail)."""
    assert len(judge_labels) == len(human_labels), "label lists must match"
    n = len(judge_labels)
    if n == 0:
        return float("nan")

    # binarise at threshold >= 3
    j = [1 if s >= 3 else 0 for s in judge_labels]
    h = [1 if s >= 3 else 0 for s in human_labels]

    po = sum(a == b for a, b in zip(j, h)) / n
    pj1 = sum(j) / n
    ph1 = sum(h) / n
    pe = pj1 * ph1 + (1 - pj1) * (1 - ph1)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
