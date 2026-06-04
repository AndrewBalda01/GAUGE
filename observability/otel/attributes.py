"""
OpenTelemetry GenAI semantic conventions (draft spec + extensions).
Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/

We define constants for every attribute we capture, keeping the code
aligned with the emerging standard that production observability stacks use.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GenAI semantic convention attributes
# ---------------------------------------------------------------------------

# System / model
GEN_AI_SYSTEM          = "gen_ai.system"          # e.g. "anthropic", "local.llama_cpp"
GEN_AI_REQUEST_MODEL   = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL  = "gen_ai.response.model"

# Tokens
GEN_AI_USAGE_INPUT_TOKENS   = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS  = "gen_ai.usage.output_tokens"

# Request parameters
GEN_AI_REQUEST_MAX_TOKENS   = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE  = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P        = "gen_ai.request.top_p"

# Operation
GEN_AI_OPERATION_NAME       = "gen_ai.operation.name"   # "chat", "completion", "embed"

# ---------------------------------------------------------------------------
# Extended attributes (project-specific)
# ---------------------------------------------------------------------------

# Cost
LLM_COST_USD            = "llm.cost_usd"
LLM_COST_INPUT_USD      = "llm.cost.input_usd"
LLM_COST_OUTPUT_USD     = "llm.cost.output_usd"

# Timing (beyond standard OTel span duration)
LLM_TTFT_MS             = "llm.ttft_ms"           # time-to-first-token

# Status
LLM_STATUS              = "llm.status"             # "ok" | "error" | "timeout"
LLM_ERROR_TYPE          = "llm.error_type"

# Application context (set by the calling app via metadata)
LLM_APP                 = "llm.app"                # e.g. "eval-harness", "local-inference"
LLM_ROUTE               = "llm.route"              # router decision, e.g. "small-model"
LLM_RUN_ID              = "llm.run_id"             # eval run ID (Project 01 link)

# Prompt / completion (optional, can be disabled for privacy)
LLM_PROMPT              = "llm.prompt"
LLM_COMPLETION          = "llm.completion"

# ---------------------------------------------------------------------------
# Span name conventions
# ---------------------------------------------------------------------------

SPAN_CHAT      = "gen_ai.chat"
SPAN_COMPLETE  = "gen_ai.complete"
SPAN_EMBED     = "gen_ai.embed"


# ---------------------------------------------------------------------------
# Helper: build attribute dict from common fields
# ---------------------------------------------------------------------------

def build_span_attributes(
    *,
    system: str,
    model: str,
    operation: str = "chat",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    ttft_ms: float | None = None,
    status: str = "ok",
    app: str = "",
    route: str = "",
    run_id: str = "",
    include_content: bool = False,
    prompt: str = "",
    completion: str = "",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, object]:
    attrs: dict[str, object] = {
        GEN_AI_SYSTEM: system,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: model,
        GEN_AI_OPERATION_NAME: operation,
        GEN_AI_USAGE_INPUT_TOKENS: tokens_in,
        GEN_AI_USAGE_OUTPUT_TOKENS: tokens_out,
        LLM_COST_USD: cost_usd,
        LLM_STATUS: status,
    }
    if ttft_ms is not None:
        attrs[LLM_TTFT_MS] = ttft_ms
    if app:
        attrs[LLM_APP] = app
    if route:
        attrs[LLM_ROUTE] = route
    if run_id:
        attrs[LLM_RUN_ID] = run_id
    if max_tokens is not None:
        attrs[GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
    if temperature is not None:
        attrs[GEN_AI_REQUEST_TEMPERATURE] = temperature
    if include_content:
        attrs[LLM_PROMPT] = prompt[:2000]       # truncate for safety
        attrs[LLM_COMPLETION] = completion[:2000]
    return attrs
