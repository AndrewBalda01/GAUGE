"""
Generation plan: defines desired distribution (topics, difficulties, edge cases).

The plan is the 'intelligent' piece that prevents mode collapse —
without it, the LLM would gravitate to the same easy patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spec.schema import DifficultyLevel


@dataclass
class TopicSlot:
    """A quota for one (topic, subtopic, difficulty) combination."""
    topic: str
    subtopic: str
    difficulty: DifficultyLevel
    count: int                       # desired number of examples
    edge_cases: list[str] = field(default_factory=list)
    extra_instructions: str = ""


@dataclass
class GenerationPlan:
    name: str
    description: str
    slots: list[TopicSlot] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(s.count for s in self.slots)

    def summary(self) -> str:
        lines = [f"Plan: {self.name}  ({self.total} total examples)"]
        for s in self.slots:
            lines.append(
                f"  {s.topic}/{s.subtopic or '*'} [{s.difficulty.value}] × {s.count}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default plan: legal Q&A expansion (feeds eval-harness legal_qa dataset)
# ---------------------------------------------------------------------------

LEGAL_QA_EXPANSION_PLAN = GenerationPlan(
    name="legal_qa_expansion",
    description="Expand the legal Q&A eval dataset with varied, high-quality examples",
    slots=[
        # --- Contract law ---
        TopicSlot("contract", "formation",     DifficultyLevel.BASIC,      count=4),
        TopicSlot("contract", "formation",     DifficultyLevel.APPLIED,    count=3,
                  edge_cases=["email as offer", "silence as acceptance"]),
        TopicSlot("contract", "breach",        DifficultyLevel.BASIC,      count=3),
        TopicSlot("contract", "breach",        DifficultyLevel.ANALYTICAL, count=2,
                  edge_cases=["partial performance", "anticipatory repudiation"]),
        TopicSlot("contract", "remedies",      DifficultyLevel.APPLIED,    count=3),
        TopicSlot("contract", "terms",         DifficultyLevel.APPLIED,    count=3,
                  edge_cases=["exclusion clause in standard form contract"]),

        # --- Tort law ---
        TopicSlot("tort", "negligence",        DifficultyLevel.BASIC,      count=4),
        TopicSlot("tort", "negligence",        DifficultyLevel.APPLIED,    count=3,
                  edge_cases=["nervous shock", "pure economic loss"]),
        TopicSlot("tort", "liability",         DifficultyLevel.ANALYTICAL, count=2),
        TopicSlot("tort", "defamation",        DifficultyLevel.BASIC,      count=2),

        # --- Criminal law ---
        TopicSlot("criminal", "mens_rea",      DifficultyLevel.BASIC,      count=3),
        TopicSlot("criminal", "defences",      DifficultyLevel.APPLIED,    count=3,
                  edge_cases=["self-defence", "insanity plea", "duress"]),
        TopicSlot("criminal", "property",      DifficultyLevel.APPLIED,    count=2),

        # --- IP ---
        TopicSlot("ip", "copyright",           DifficultyLevel.BASIC,      count=3),
        TopicSlot("ip", "patents",             DifficultyLevel.APPLIED,    count=2,
                  edge_cases=["software patents", "patent exhaustion"]),
        TopicSlot("ip", "trademarks",          DifficultyLevel.APPLIED,    count=2),

        # --- Privacy / GDPR ---
        TopicSlot("privacy", "gdpr",           DifficultyLevel.APPLIED,    count=4,
                  edge_cases=["cross-border transfers", "legitimate interest balancing"]),
        TopicSlot("privacy", "rights",         DifficultyLevel.BASIC,      count=2),

        # --- Company law ---
        TopicSlot("company", "directors",      DifficultyLevel.ANALYTICAL, count=3,
                  edge_cases=["shadow director", "wrongful trading"]),
        TopicSlot("company", "insolvency",     DifficultyLevel.APPLIED,    count=2),

        # --- Property ---
        TopicSlot("property", "easements",     DifficultyLevel.APPLIED,    count=2),
        TopicSlot("property", "adverse",       DifficultyLevel.BASIC,      count=2),

        # --- Procedure / ADR ---
        TopicSlot("procedure", "arbitration",  DifficultyLevel.ANALYTICAL, count=2,
                  edge_cases=["enforcement of foreign award"]),
        TopicSlot("procedure", "adr",          DifficultyLevel.APPLIED,    count=2),
    ],
)
