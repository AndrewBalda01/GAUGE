"""
Cost estimation: USD per 1M tokens for known models.

Usage:
    cost = estimate_cost("claude-haiku-4-5-20251001", tokens_in=1000, tokens_out=500)
"""

from __future__ import annotations

# (input_per_1M_usd, output_per_1M_usd)
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-8":              (15.00,  75.00),
    "claude-sonnet-4-6":            (3.00,   15.00),
    "claude-haiku-4-5-20251001":    (0.80,    4.00),
    "claude-haiku-4-5":             (0.80,    4.00),
    # OpenAI
    "gpt-4o":                       (5.00,   15.00),
    "gpt-4o-mini":                  (0.15,    0.60),
    "gpt-3.5-turbo":                (0.50,    1.50),
    # Open models (local = $0 — included for completeness)
    "qwen3":                        (0.0,     0.0),
    "llama":                        (0.0,     0.0),
    "mistral":                      (0.0,     0.0),
}


def estimate_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated USD cost; returns 0.0 for unknown / local models."""
    key = next(
        (k for k in _PRICE_TABLE if model_id.lower().startswith(k.lower())),
        None,
    )
    if key is None:
        return 0.0
    price_in, price_out = _PRICE_TABLE[key]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def all_models() -> list[str]:
    return list(_PRICE_TABLE.keys())
