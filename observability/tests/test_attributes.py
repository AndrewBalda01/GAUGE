"""Unit tests for OTel attribute builder."""

from otel.attributes import build_span_attributes, GEN_AI_SYSTEM, LLM_COST_USD, LLM_TTFT_MS


def test_basic_attributes():
    attrs = build_span_attributes(
        system="anthropic",
        model="claude-haiku-4-5-20251001",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.0012,
    )
    assert attrs[GEN_AI_SYSTEM] == "anthropic"
    assert attrs[LLM_COST_USD] == 0.0012


def test_ttft_only_when_provided():
    attrs_no_ttft = build_span_attributes(system="x", model="m")
    assert LLM_TTFT_MS not in attrs_no_ttft

    attrs_with_ttft = build_span_attributes(system="x", model="m", ttft_ms=80.0)
    assert attrs_with_ttft[LLM_TTFT_MS] == 80.0


def test_content_not_included_by_default():
    from otel.attributes import LLM_PROMPT, LLM_COMPLETION
    attrs = build_span_attributes(
        system="x", model="m",
        include_content=False,
        prompt="secret prompt",
        completion="secret output",
    )
    assert LLM_PROMPT not in attrs
    assert LLM_COMPLETION not in attrs


def test_content_included_when_requested():
    from otel.attributes import LLM_PROMPT, LLM_COMPLETION
    attrs = build_span_attributes(
        system="x", model="m",
        include_content=True,
        prompt="hello",
        completion="world",
    )
    assert attrs[LLM_PROMPT] == "hello"
    assert attrs[LLM_COMPLETION] == "world"


def test_long_content_truncated():
    from otel.attributes import LLM_PROMPT
    long_prompt = "x" * 5000
    attrs = build_span_attributes(
        system="x", model="m",
        include_content=True,
        prompt=long_prompt,
    )
    assert len(attrs[LLM_PROMPT]) <= 2000
