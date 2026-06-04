"""
Routing policy: translates a ClassificationResult into an engine selection.

Principles:
  - Simple queries  → small engine (fast, cheap)
  - Complex queries → large engine (accurate)
  - Uncertain       → configurable fallback (default: large to be safe)
  - If chosen engine unavailable → fallback to other engine
"""

from __future__ import annotations

from dataclasses import dataclass

from router.classifier import ClassificationResult, Difficulty


@dataclass
class RoutingDecision:
    engine_name: str          # name of the chosen engine config
    difficulty: Difficulty
    confidence: float
    reason: str               # human-readable explanation
    fallback_used: bool = False
    savings_eligible: bool = False  # True when small engine was chosen


@dataclass
class RouterStats:
    total: int = 0
    routed_small: int = 0
    routed_large: int = 0
    fallback_used: int = 0

    @property
    def small_fraction(self) -> float:
        return self.routed_small / self.total if self.total > 0 else 0.0

    @property
    def savings_estimate(self) -> str:
        if self.total == 0:
            return "—"
        pct = self.small_fraction * 100
        return f"{pct:.1f}% queries served by small model"


class RouterPolicy:
    """
    Stateful policy that records routing decisions for reporting.
    Thread-safe for asyncio (single-thread event loop).
    """

    def __init__(
        self,
        small_engine_name: str,
        large_engine_name: str,
        uncertain_fallback: str = "large",   # "small" | "large"
        confidence_threshold: float = 0.55,  # below this → treat as uncertain
    ) -> None:
        self.small_engine_name = small_engine_name
        self.large_engine_name = large_engine_name
        self.uncertain_fallback = uncertain_fallback
        self.confidence_threshold = confidence_threshold
        self.stats = RouterStats()

    def decide(self, classification: ClassificationResult) -> RoutingDecision:
        self.stats.total += 1
        diff = classification.difficulty
        conf = classification.confidence

        # Low-confidence heuristic result → treat as uncertain
        if conf < self.confidence_threshold and diff != Difficulty.UNCERTAIN:
            diff = Difficulty.UNCERTAIN

        if diff == Difficulty.SIMPLE:
            engine = self.small_engine_name
            self.stats.routed_small += 1
            return RoutingDecision(
                engine_name=engine,
                difficulty=diff,
                confidence=conf,
                reason=f"Simple query (confidence={conf:.2f}) → small model",
                savings_eligible=True,
            )

        if diff == Difficulty.COMPLEX:
            engine = self.large_engine_name
            self.stats.routed_large += 1
            return RoutingDecision(
                engine_name=engine,
                difficulty=diff,
                confidence=conf,
                reason=f"Complex query (confidence={conf:.2f}) → large model",
            )

        # Uncertain
        if self.uncertain_fallback == "small":
            engine = self.small_engine_name
            self.stats.routed_small += 1
            eligible = True
        else:
            engine = self.large_engine_name
            self.stats.routed_large += 1
            eligible = False

        self.stats.fallback_used += 1
        return RoutingDecision(
            engine_name=engine,
            difficulty=diff,
            confidence=conf,
            reason=f"Uncertain query → fallback to {self.uncertain_fallback} model",
            fallback_used=True,
            savings_eligible=eligible,
        )

    def report(self) -> dict:
        s = self.stats
        return {
            "total_queries": s.total,
            "routed_to_small": s.routed_small,
            "routed_to_large": s.routed_large,
            "fallback_used": s.fallback_used,
            "small_fraction": round(s.small_fraction, 3),
            "savings_estimate": s.savings_estimate,
        }
