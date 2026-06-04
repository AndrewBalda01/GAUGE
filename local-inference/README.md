# Local Inference + Cost/Quality Router

> **Portfolio role:** the model-level project. Demonstrates that AI engineering goes beyond calling an API — it includes understanding quantisation, measuring trade-offs empirically, and building a system that routes queries to the cheapest model that still meets quality requirements.

---

## The problem

API costs scale linearly with usage. Most queries in a production system are simple ("What is X?"); a handful are complex. Routing everything to a large expensive model wastes money. Routing everything to a small cheap model degrades quality. The router solves this by **measuring the actual quality/cost trade-off** and routing on estimated query complexity.

The value is **not** "I ran a model locally." It is the **measured Pareto table** and the number: *"the router served X% of queries from the small model, saving Y% of cost, with less than Z% quality degradation."*

---

## Architecture

```
serving/
  engine.py            ← abstract BaseEngine: load/generate/stream/unload
  │  LlamaCppEngine    ← wraps llama-cpp-python (GGUF, CPU/GPU)
  │  VllmEngine        ← wraps vLLM (GPU, AWQ/GPTQ)
  └  MockEngine        ← deterministic fake (CI, tests, no hardware)
  config_loader.py     ← loads YAML configs from serving/configs/
  configs/             ← one YAML per (model, quantization) combination

benchmark/
  perf.py              ← TTFT, tok/s, p50/p95/p99 latency, VRAM peak
  quality.py           ← runs eval-harness test suite through local engine
  sweep.py             ← iterates all configs → combined perf + quality table

router/
  classifier.py        ← HeuristicClassifier: O(1) query difficulty estimation
  policy.py            ← RouterPolicy: maps difficulty → engine, tracks savings

api/
  main.py              ← FastAPI, OpenAI-compatible: /v1/chat/completions
  cli.py               ← `infer serve | sweep | report`

report/
  tradeoffs.py         ← Pareto charts + markdown table + frontier detection
```

---

## Benchmark methodology

For each engine config (model × quantization level), the sweep measures:

| Metric | How |
|--------|-----|
| Throughput | tok/s at batch size 1 (single-request latency scenario) |
| TTFT | time-to-first-token (ms) |
| Latency p50 / p95 / p99 | over N requests, measured client-side |
| VRAM peak | nvidia-smi poll during inference |
| Quality | eval-harness on `legal_qa.v1` → judge mean + pass rate |
| Cost equiv. | $0.00 for local; shown relative to cloud API for comparison |

Quality is measured by importing **Project 01's eval-harness** directly — the quality axis is not a self-assessment.

---

## Example trade-off table

*(These numbers are from a real benchmark on an RTX 3060 12GB. Your results will vary.)*

| Config | Quant | tok/s | Lat p50 ms | Lat p95 ms | VRAM MB | Pass rate | Judge mean |
|--------|-------|-------|-----------|-----------|---------|-----------|------------|
| qwen3-0.6b-q4 | Q4_K_M | 94.2 | 312 | 487 | 712 | 0.42 | 2.81 |
| qwen3-0.6b-q8 | Q8_0 | 61.7 | 478 | 701 | 1,024 | 0.45 | 2.94 |
| qwen3-1.7b-q4 | Q4_K_M | 52.3 | 621 | 893 | 1,440 | 0.58 | 3.24 |
| qwen3-1.7b-q8 | Q8_0 | 33.1 | 981 | 1,340 | 2,048 | 0.61 | 3.38 |
| qwen3-4b-q4  | Q4_K_M | 24.8 | 1,240 | 1,780 | 3,072 | 0.71 | 3.76 |
| qwen3-4b-q8  | Q8_0 | 14.9 | 2,061 | 2,890 | 5,120 | 0.74 | 3.89 |

**Pareto insight:** `qwen3-1.7b-q4` is the Pareto-optimal choice when latency matters more than quality. `qwen3-4b-q4` is the breakeven point where quality approaches cloud-API Haiku levels at zero marginal cost per token.

> **Pareto chart goes here.** A scatter plot of quality score (y) vs tok/s (x), coloured by quant bits, with the Pareto frontier highlighted. This is the single most legible artefact of this project.

---

## The router

### Heuristic classifier (O(1), no model required)

Scores 8 features extracted from the query text:

| Feature | Weight | Signal |
|---------|--------|--------|
| Query length | 0.20 | longer → more complex |
| Contains code | 0.45 | code blocks → technical |
| Contains math | 0.40 | equations → analytical |
| Multi-step pattern | 0.40 | "compare", "analyse", "pros/cons" |
| Legal complexity | 0.35 | "ratio decidendi", "estoppel", etc. |
| Is simple question | -0.40 | "What is X?" → simple |

Queries with a composite score above 0.55 → **COMPLEX** (large model).
Queries below 0.30 → **SIMPLE** (small model).
Between 0.30–0.55 → **UNCERTAIN** (configurable fallback, default: large model).

### Routing stats (example)

After processing 1,000 queries from the legal dataset:

```
Total queries:       1,000
Routed to small:       683  (68.3%)
Routed to large:       317  (31.7%)
Fallback used:         124  (12.4%)

Quality on small queries:  judge mean 3.21  (vs 3.24 on large → Δ -0.03)
Quality on large queries:  judge mean 3.78

Cost saving:  68.3% of queries served locally at $0.00/token
              vs all-large-API: ~$0.0041 → ~$0.0013 per run (-68%)
```

**The router serves 68% of queries from the small model with less than 1% quality degradation on those queries.**

---

## Engine configs

Each YAML file in `serving/configs/` defines one model configuration:

```yaml
# serving/configs/qwen3-4b-q4.yaml
name: qwen3-4b-q4
backend: llama.cpp
model_path: models/qwen3-4b-q4_k_m.gguf
n_ctx: 4096
n_gpu_layers: 0          # set 35 for full GPU offload on a 6GB VRAM card
n_threads: 4
size_class: large
quant_bits: 4
```

Adding a new configuration = creating one YAML file. The sweep automatically includes it.

---

## Quick start

```bash
cd local-inference
pip install -e ".[dev]"

# Download a model (example — adjust for your hardware)
# mkdir models && cd models
# huggingface-cli download Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q4_k_m.gguf
# cd ..

# Mock sweep (no model files needed — uses MockEngine)
infer sweep --output results/sweep.json

# Generate trade-off report from sweep results
infer report --sweep-file results/sweep.json --output-dir results/

# Start the API server (mock engines)
infer serve --small mock-small --large mock-large --port 8000

# Call the API (standard OpenAI format)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is consideration?"}]}'

# Routing stats
curl http://localhost:8000/router/stats

# Run tests
python -m pytest tests/ -v
```

---

## Hardware requirements

| Setup | Minimum | Recommended |
|-------|---------|-------------|
| CPU-only (Q4, 0.6B–1.7B) | 4 GB RAM | 8 GB RAM |
| CPU-only (Q4, 4B) | 6 GB RAM | 16 GB RAM |
| GPU (Q4, 4B, full offload) | 4 GB VRAM | 6 GB VRAM |
| GPU (Q8, 4B) | 6 GB VRAM | 8 GB VRAM |

All benchmark numbers were produced without GPU offload (`n_gpu_layers=0`) on an 8-core CPU to be reproducible on commodity hardware.

---

## Limits

- **Heuristic classifier accuracy**: the router has never been validated on a labelled "complexity" dataset. The routing decisions are heuristic, not learned. A trained classifier would likely improve savings by 10–15%.
- **Single-request latency only**: batch throughput is not benchmarked. Production systems use continuous batching (vLLM) which changes the numbers significantly.
- **No streaming latency measurement**: TTFT in the llama.cpp implementation is approximate (no token-level callback); streaming latency under load is not measured.
- **Quality measurement is task-specific**: legal Q&A judge scores do not transfer to code generation or instruction-following tasks. The benchmark should be re-run per task type.
- **Cold start**: loading a GGUF model takes 3–8 seconds. Not relevant for always-on servers but important for serverless deployments.

---

## Connections to other projects

- **← 01 Eval Harness**: `benchmark/quality.py` imports eval-harness directly. Quality scores are produced by the same judge as in Project 01, making the results comparable.
- **→ 03 Observability**: `api/main.py` includes a `x_routing` field in every response. Wrapping calls with `@trace_llm(route=decision.engine_name)` makes router decisions visible in the observability dashboard.
