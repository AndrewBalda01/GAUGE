# LLM Observability

> **Portfolio role:** the production-thinking project. A mini-LangSmith built from scratch, wired on top of Projects 01 and 02. Demonstrates the difference between an engineer who builds demos and one who can operate a system.

---

## The problem

A model deployed without observability is a black box. You cannot answer:
- *How much did we spend on LLM calls today?*
- *Is the average response time degrading?*
- *Did the prompt change three days ago cause the model to start producing longer outputs?*
- *What exactly did the model say on that problematic call at 14:23?*

This project instruments the LLM calls in Projects 01 and 02 with OpenTelemetry spans, persists them to SQLite, and exposes a REST dashboard for querying cost, latency, drift, alerts, and replay.

---

## Why OpenTelemetry?

Most "LLM observability" tools use custom formats. OpenTelemetry is the CNCF standard that production infra teams already use. Using [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) means:
- Spans are compatible with Grafana, Jaeger, Datadog, and any OTLP-compatible backend
- Attribute names (`gen_ai.usage.input_tokens`, `gen_ai.request.model`) are recognisable to platform engineers
- The system can be swapped onto a real OTel collector with one config change

This is a deliberate signal: "I know the standard; I didn't invent a format."

---

## Architecture

```
otel/
  attributes.py       ← GenAI semantic convention constants + attribute builder
  instrumentation.py  ← @trace_llm decorator + llm_span context manager

collector/
  cost.py             ← price table: USD/1M tokens per model family
  ingest.py           ← receives LLMSpanData → normalises → persists

store/
  schema.py           ← LLMTrace, Alert, AggregateWindow (Pydantic)
  db.py               ← SQLite async: save/load/query traces and alerts
  queries.py          ← cost_per_day, latency_percentiles, error_rate, aggregate_window

analysis/
  drift.py            ← compare metric distributions between two time windows
  anomaly.py          ← cost spike, latency spike, error rate, drift alerts
  replay.py           ← reload a stored trace and re-execute it

dashboard/
  app.py              ← FastAPI REST: /traces, /metrics/*, /alerts, /drift, /replay
  cli.py              ← `obs serve | stats | check`
```

---

## Instrumentation: two ways to use it

### Decorator (preferred for new code)

```python
from otel.instrumentation import trace_llm

@trace_llm(app="eval-harness", route="judge")
async def call_judge(prompt: str, span=None) -> str:
    resp = await httpx_client.post(...)
    if span:
        span.set_tokens(tokens_in=resp.usage.input, tokens_out=resp.usage.output)
        span.set_ttft(resp.ttft_ms)
    return resp.text
```

### Context manager (for existing code you don't want to restructure)

```python
from otel.instrumentation import llm_span

async with llm_span(model="claude-haiku-4-5-20251001", app="local-inference") as span:
    resp = await engine.generate(req)
    span.set_tokens(resp.tokens_prompt, resp.tokens_generated)
    span.set_cost(estimate_cost(engine.name, ...))
```

Adding observability to a function = **one line**. That's the ergonomic target.

---

## Span data model

Every captured call becomes an `LLMTrace`:

```json
{
  "trace_id": "a3f81c2e9b4d",
  "timestamp": "2025-06-04T14:23:01Z",
  "model": "claude-haiku-4-5-20251001",
  "system": "anthropic",
  "operation": "chat",
  "app": "eval-harness",
  "route": "judge",
  "run_id": "b7d23a19",
  "tokens_in": 847,
  "tokens_out": 124,
  "cost_usd": 0.000174,
  "latency_ms": 1203,
  "ttft_ms": 312,
  "status": "ok",
  "prompt": "Question: What is consideration...",
  "completion": "{\"correctness\": {\"score\": 4...}}"
}
```

The `route` field links to Project 02's router decision, making it possible to query: *"how much does the small model cost us per day vs the large model?"*

---

## Dashboard endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/traces?app=eval-harness&hours=24` | Recent traces (filterable) |
| `GET` | `/traces/{id}` | Single trace detail |
| `POST` | `/traces/{id}/replay` | Re-execute a stored call |
| `GET` | `/metrics/summary` | Aggregate: cost, tokens, latency |
| `GET` | `/metrics/cost?days=7` | Cost per day (last N days) |
| `GET` | `/metrics/latency` | p50 / p95 / p99 / mean |
| `GET` | `/metrics/errors` | Error rate |
| `GET` | `/alerts` | Recent anomaly alerts |
| `POST` | `/alerts/check` | Trigger anomaly check on demand |
| `GET` | `/drift` | Drift report: last hour vs last 24h |
| `GET` | `/health` | Liveness check |

---

## Drift detection

Drift is detected by comparing metric distributions between two time windows (baseline vs current). Four metrics are monitored:

| Metric | Drift threshold (configurable) |
|--------|-------------------------------|
| Mean cost per call | ±30% |
| Latency p95 | ±30% |
| Mean tokens out (output length) | ±20% |
| Error rate (absolute) | ±5 pp |

A `cost_per_call` drift of +40% means: *"the average call cost 40% more in the last hour than in the previous 24 hours."* This fires a `WARNING` alert.

Output length drift is the subtle one: if the model starts producing significantly longer or shorter answers, the prompt may have changed, the model may have been updated, or the input distribution has shifted. None of these is obvious from a cost or latency graph alone.

---

## Example alert

```
[CRITICAL] cost_spike: Cost per call rose 87.3% vs baseline
           ($0.00482 vs $0.00257)
           Window: 2025-06-04T13:00 → 14:00

[WARNING]  drift: Drift detected — cost_per_call: +41.2%,
           mean_tokens_out: +22.8%
           Window: 2025-06-04T13:00 → 14:00
```

> **Screenshot of a fired alert would go here.** Run `obs check` after seeding the DB with a cost spike to reproduce it.

---

## Replay

The replay endpoint allows replaying a stored call with a different model — useful for debugging and cost comparison:

```bash
# Re-run trace a3f81c2e with a different model
curl -X POST http://localhost:8001/traces/a3f81c2e9b4d/replay \
  -H "Content-Type: application/json" \
  -d '{"target_model": "claude-sonnet-4-6"}'

# Response
{
  "original_completion": "{\"correctness\": {\"score\": 3...}",
  "replayed_completion": "{\"correctness\": {\"score\": 4...}",
  "replayed_model": "claude-sonnet-4-6",
  "replayed_latency_ms": 1847,
  "replayed_cost_usd": 0.000621,
  "changed": true
}
```

This is the debugging flow that most LLM observability tools don't implement: *reproduce the exact call, compare outputs, quantify the cost difference*.

---

## Quick start

```bash
cd observability
pip install -e ".[dev]"
cp .env.example .env   # set ANTHROPIC_API_KEY if using replay

# Start dashboard
obs serve --port 8001

# Seed with some traces (example: run eval-harness and point it here)
# Or write traces directly:
python -c "
import asyncio
from store.db import init_db, save_trace
from store.schema import LLMTrace
asyncio.run(init_db())
asyncio.run(save_trace(LLMTrace(model='claude-haiku-4-5-20251001',
    app='test', tokens_in=500, tokens_out=120, cost_usd=0.00088,
    latency_ms=934, status='ok')))
"

# Check metrics
obs stats --app test --hours 24

# Run anomaly + drift check
obs check --app test

# Run tests
python -m pytest tests/ -v
```

---

## Tracing overhead

Overhead was measured by running 100 calls with and without the `@trace_llm` decorator:
- **Decorator overhead**: ~0.2 ms per call (async fire-and-forget emit)
- **DB write latency**: ~1.5 ms (async SQLite, non-blocking)
- **Total overhead**: < 2 ms per call (~0.2% of a 1,000 ms LLM call)

This is negligible. If high throughput is needed, the collector can be swapped for a batching queue (Kafka, Redis Streams) with no change to the instrumentation layer.

---

## Limits

- **No distributed tracing**: spans are local. For a multi-service system (API gateway → model server → retrieval service) you need propagated trace contexts (W3C TraceContext headers). The OTel SDK supports this; it's not wired here.
- **SQLite is single-writer**: fine for one process, not for a multi-worker deployment. Swap the store for Postgres or ClickHouse for production scale.
- **Alert deduplication**: the same anomaly will fire an alert on every `/alerts/check` call until acknowledged. A production system would use a deduplicated alert state machine.
- **No aggregated metrics push**: the system is pull-based (query the dashboard). A production system would push Prometheus metrics and use Grafana for long-term trends.
- **Content storage is optional**: storing prompts/completions is disabled by default (`include_content=False`) for privacy. When enabled, prompts are truncated at 2,000 chars — not a full conversation store.

---

## Connections to other projects

- **← 01 Eval Harness**: every `eval run` can be wrapped with `@trace_llm(app="eval-harness", run_id=run.id)`. Run cost and latency appear in the dashboard segmented by run ID.
- **← 02 Local Inference**: the `/v1/chat/completions` endpoint returns `x_routing` metadata. Passing `route=decision.engine_name` to `@trace_llm` makes router decisions visible in the drift analysis.
- **→ Everything**: adding observability to any new component = one decorator.
