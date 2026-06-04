"""
Structured generation: turns a TopicSlot + Seed into a SyntheticExample.

Design decisions:
  - Temperature 0.8 for variety, but JSON output forced for structure.
  - One example per call (avoids batched JSON that LLMs hallucinate more).
  - System prompt establishes the rubric; user prompt carries the slot+seed.
  - Retries on parse failure (up to 3).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import AsyncIterator

import httpx

from spec.plan import TopicSlot
from spec.schema import DifficultyLevel, SyntheticExample
from generate.seed import Seed

_SYSTEM = """\
You are an expert legal educator creating high-quality Q&A examples for an AI evaluation dataset.

Your output must be a single valid JSON object with exactly these keys:
  "input"    : string — the question (clear, self-contained, no jargon unexplained)
  "expected" : string — the reference answer (accurate, concise, 2-5 sentences)

Rules:
- The question must be genuinely answerable from general legal knowledge.
- The answer must be factually correct and jurisdictionally neutral unless stated.
- Do not start the answer with "The answer is" or repeat the question.
- Return ONLY the JSON object — no prose before or after.
"""

_DIFFICULTY_GUIDE = {
    DifficultyLevel.BASIC:
        "single-concept factual question (e.g. 'What is X?' or 'Define Y.')",
    DifficultyLevel.APPLIED:
        "applies a legal rule to a concrete scenario (e.g. 'In this situation, what would…?')",
    DifficultyLevel.ANALYTICAL:
        "requires comparison, critique or argument (e.g. 'Compare X and Y', 'Analyse the tension between…')",
}

_USER_TEMPLATE = """\
Topic:     {topic}
Subtopic:  {subtopic}
Difficulty: {difficulty} — {difficulty_guide}

Persona (who is asking / the intended audience):
  {persona}

Scenario (the context in which this question arises):
  {scenario}

Question style:
  {framing}

{edge_case_line}
{extra_line}

Generate ONE Q&A pair as a JSON object.
"""


def _build_prompt(slot: TopicSlot, seed: Seed) -> str:
    edge_line = (
        f"Edge case to incorporate: {slot.edge_cases[0]}"
        if slot.edge_cases else ""
    )
    extra_line = (
        f"Additional instruction: {slot.extra_instructions}"
        if slot.extra_instructions else ""
    )
    return _USER_TEMPLATE.format(
        topic=slot.topic,
        subtopic=slot.subtopic or "(general)",
        difficulty=slot.difficulty.value,
        difficulty_guide=_DIFFICULTY_GUIDE[slot.difficulty],
        persona=seed.persona,
        scenario=seed.scenario,
        framing=seed.framing,
        edge_case_line=edge_line,
        extra_line=extra_line,
    )


def _parse_response(raw: str) -> tuple[str, str] | None:
    """Extract (input, expected) from raw LLM output; returns None on failure."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        q = str(data.get("input", "")).strip()
        a = str(data.get("expected", "")).strip()
        if q and a:
            return q, a
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: regex search for keys
    m_q = re.search(r'"input"\s*:\s*"([^"]+)"', text, re.DOTALL)
    m_a = re.search(r'"expected"\s*:\s*"([^"]+)"', text, re.DOTALL)
    if m_q and m_a:
        return m_q.group(1).strip(), m_a.group(1).strip()
    return None


async def generate_one(
    slot: TopicSlot,
    seed: Seed,
    model: str = "claude-haiku-4-5-20251001",
    api_key_env: str = "ANTHROPIC_API_KEY",
    temperature: float = 0.8,
    max_retries: int = 3,
) -> SyntheticExample | None:
    """
    Generate a single synthetic example for the given slot + seed.
    Returns None if all retries fail.
    """
    api_key = os.environ.get(api_key_env, "")
    prompt = _build_prompt(slot, seed)
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(max_retries):
            payload = {
                "model": model,
                "max_tokens": 512,
                "temperature": temperature,
                "system": _SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            }
            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                raw = "".join(
                    b.get("text", "")
                    for b in resp.json().get("content", [])
                    if b.get("type") == "text"
                )
                parsed = _parse_response(raw)
                if parsed:
                    question, answer = parsed
                    ex_id = f"syn-{slot.topic[:3]}-{uuid.uuid4().hex[:6]}"
                    return SyntheticExample(
                        id=ex_id,
                        input=question,
                        expected=answer,
                        topic=slot.topic,
                        subtopic=slot.subtopic,
                        difficulty=slot.difficulty,
                        tags=[slot.difficulty.value],
                        generation_model=model,
                        seed_persona=seed.persona,
                        seed_scenario=seed.scenario,
                    )
            except Exception:
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1.0 * (attempt + 1))

    return None


async def generate_slot(
    slot: TopicSlot,
    seeds: list[Seed],
    model: str = "claude-haiku-4-5-20251001",
    concurrency: int = 3,
) -> list[SyntheticExample]:
    """Generate all examples for one slot (uses different seeds for diversity)."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(seed: Seed) -> SyntheticExample | None:
        async with sem:
            return await generate_one(slot, seed, model=model)

    results = await asyncio.gather(*[_one(s) for s in seeds[: slot.count]])
    return [r for r in results if r is not None]
