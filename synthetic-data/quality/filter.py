"""
Quality filtering: heuristic rules + optional LLM-as-judge.

Two-stage approach:
  1. Fast heuristics — no API call: catch obvious failures.
  2. LLM judge       — reuses the judge rubric from eval-harness (Project 01).
     Only called when heuristics pass (saves cost).

The judge threshold is configurable; default is mean score ≥ 3.0/5.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

from spec.schema import SyntheticExample

# ---------------------------------------------------------------------------
# Heuristic rules
# ---------------------------------------------------------------------------

_BOILERPLATE = re.compile(
    r"(as an ai|i cannot|i'm unable|i don't have access|"
    r"please consult|see a lawyer|this is not legal advice|"
    r"note that|it's important to note)",
    re.IGNORECASE,
)

_REPETITION_THRESHOLD = 0.35   # fraction of repeated 3-grams → poor answer


def _repetition_score(text: str) -> float:
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    trigrams = [tuple(words[i: i + 3]) for i in range(len(words) - 2)]
    return 1 - len(set(trigrams)) / len(trigrams)


@dataclass
class HeuristicResult:
    passed: bool
    reason: str = ""


def heuristic_filter(ex: SyntheticExample) -> HeuristicResult:
    if _BOILERPLATE.search(ex.expected):
        return HeuristicResult(False, "answer contains boilerplate/disclaimer")
    if _repetition_score(ex.expected) > _REPETITION_THRESHOLD:
        return HeuristicResult(False, "answer has high repetition")
    if ex.expected.count("?") > 3:
        return HeuristicResult(False, "answer contains too many questions")
    if len(ex.expected.split()) > len(ex.input.split()) * 10:
        return HeuristicResult(False, "answer disproportionately long vs question")
    return HeuristicResult(True)


# ---------------------------------------------------------------------------
# LLM judge (reuses eval-harness rubric)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a strict legal education quality assessor.
Score the Q&A pair on correctness and educational value.
Return ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}
5 = excellent, 3 = acceptable, 1 = wrong or useless.
"""

_JUDGE_PROMPT = """\
Question: {question}
Answer:   {answer}
Score this Q&A pair (1-5) for correctness and educational value.
"""


async def judge_quality(
    ex: SyntheticExample,
    judge_model: str = "claude-haiku-4-5-20251001",
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> tuple[int, str]:
    """Returns (score 1-5, reason). Falls back to (3, 'parse error') on failure."""
    api_key = os.environ.get(api_key_env, "")
    payload = {
        "model": judge_model,
        "max_tokens": 128,
        "temperature": 0.0,
        "system": _JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": _JUDGE_PROMPT.format(
            question=ex.input, answer=ex.expected
        )}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        raw = "".join(
            b.get("text", "") for b in resp.json().get("content", [])
            if b.get("type") == "text"
        ).strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        import json
        data = json.loads(raw)
        return int(data["score"]), str(data.get("reason", ""))
    except Exception:
        return 3, "parse error"


# ---------------------------------------------------------------------------
# Combined filter pass
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    kept: list[SyntheticExample]
    rejected_heuristic: list[tuple[SyntheticExample, str]]
    rejected_judge: list[tuple[SyntheticExample, int, str]]

    def summary(self) -> str:
        total = len(self.kept) + len(self.rejected_heuristic) + len(self.rejected_judge)
        return (
            f"Filter: {total} in → {len(self.kept)} kept  "
            f"(heuristic={len(self.rejected_heuristic)}, judge={len(self.rejected_judge)})"
        )


def filter_heuristic(examples: list[SyntheticExample]) -> FilterResult:
    """Heuristic-only pass (no API calls)."""
    kept, rejected = [], []
    for ex in examples:
        r = heuristic_filter(ex)
        if r.passed:
            if "filter" not in ex.stage_passed:
                ex.stage_passed.append("filter")
            kept.append(ex)
        else:
            rejected.append((ex, r.reason))
    return FilterResult(kept=kept, rejected_heuristic=rejected, rejected_judge=[])


async def filter_with_judge(
    examples: list[SyntheticExample],
    min_score: int = 3,
    judge_model: str = "claude-haiku-4-5-20251001",
    concurrency: int = 5,
) -> FilterResult:
    """Heuristic + LLM judge pass."""
    import asyncio

    # Stage 1: heuristics
    heuristic = filter_heuristic(examples)

    # Stage 2: judge survivors
    sem = asyncio.Semaphore(concurrency)

    async def _judge(ex: SyntheticExample) -> tuple[SyntheticExample, int, str]:
        async with sem:
            score, reason = await judge_quality(ex, judge_model=judge_model)
            return ex, score, reason

    results = await asyncio.gather(*[_judge(ex) for ex in heuristic.kept])

    final_kept: list[SyntheticExample] = []
    rejected_judge: list[tuple[SyntheticExample, int, str]] = []
    for ex, score, reason in results:
        if score >= min_score:
            ex.stage_passed.append("filter")
            final_kept.append(ex)
        else:
            rejected_judge.append((ex, score, reason))

    return FilterResult(
        kept=final_kept,
        rejected_heuristic=heuristic.rejected_heuristic,
        rejected_judge=rejected_judge,
    )
