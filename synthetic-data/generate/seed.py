"""
Seed diversifiers: personas, scenarios, and parameter axes.

Seeding forces the LLM to generate from different starting points,
which is the primary weapon against mode collapse (the LLM otherwise
repeats the same ~10 patterns at slightly different surface forms).

Each seed contributes to the generation prompt as an instructional frame,
not as content — the LLM still produces the Q&A, it just does so
from a different perspective.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Seed:
    persona: str        # who is asking / who is being tested
    scenario: str       # context in which the question arises
    framing: str        # HOW the question should be phrased


# ---------------------------------------------------------------------------
# Persona bank
# ---------------------------------------------------------------------------

PERSONAS: list[str] = [
    "a first-year law student revising for an exam",
    "a paralegal drafting a client advice memo",
    "a small business owner with no legal background",
    "a compliance officer at a fintech company",
    "a journalist investigating a corporate scandal",
    "a non-profit director concerned about data protection",
    "a startup founder negotiating their first commercial contract",
    "an HR manager handling an employment dispute",
    "a property developer facing a planning dispute",
    "a software engineer who accidentally infringed a patent",
    "a PhD student studying comparative law",
    "a senior in-house counsel preparing for litigation",
]

# ---------------------------------------------------------------------------
# Scenario bank (context / triggering event)
# ---------------------------------------------------------------------------

SCENARIOS: list[str] = [
    "during a transaction that has gone wrong",
    "after receiving a legal letter before action",
    "when advising a client in writing",
    "during a job interview for a legal role",
    "while preparing a court submission",
    "in a university tutorial setting",
    "when onboarding a new jurisdiction",
    "in the context of cross-border operations",
    "following a regulatory investigation",
    "during due diligence for an acquisition",
    "in response to a customer complaint",
    "while drafting a policy document",
]

# ---------------------------------------------------------------------------
# Framing / question-style axes
# ---------------------------------------------------------------------------

FRAMINGS: list[str] = [
    "ask a direct definitional question",
    "ask a question that requires distinguishing two related concepts",
    "pose a scenario and ask what the legal outcome would be",
    "ask for a list of key elements or requirements",
    "ask a 'why does this matter in practice?' question",
    "ask about an edge case or exception to a general rule",
    "ask to compare two legal mechanisms",
    "ask about the procedural steps required",
    "ask about the consequences of failing to comply",
    "ask about a recent or historically significant case illustration",
]


def sample_seeds(n: int, rng: random.Random | None = None) -> list[Seed]:
    """Return n unique seed combinations."""
    r = rng or random.Random()
    seeds: list[Seed] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(seeds) < n and attempts < n * 10:
        attempts += 1
        p = r.choice(PERSONAS)
        s = r.choice(SCENARIOS)
        f = r.choice(FRAMINGS)
        key = (p, s, f)
        if key not in seen:
            seen.add(key)
            seeds.append(Seed(persona=p, scenario=s, framing=f))
    return seeds
