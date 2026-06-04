# AI Engineering Portfolio

A monorepo of four interconnected projects that together demonstrate the full engineering cycle of a production LLM system — from **measuring quality**, to **optimising inference**, to **observing behaviour in production**, to **generating data that closes the feedback loop**.

> **Design principle:** every project is useful on its own, but the real signal is that they talk to each other. Quality numbers flow from 01 → 02. Observability instruments 01 and 02. Synthetic data expands 01's test suite. This is the architecture of a system, not a collection of demos.

---

## Projects

| # | Project | What it answers | Key artefact |
|---|---------|-----------------|--------------|
| 01 | [LLM Eval Harness](eval-harness/) | *"Did this change make the model better or worse?"* | CI pipeline that blocks a merge on quality regression |
| 02 | [Local Inference + Router](local-inference/) | *"What is the cheapest config that maintains quality?"* | Pareto chart: quality vs tok/s vs VRAM |
| 03 | [Observability](observability/) | *"What is the system doing right now, and is it drifting?"* | Dashboard + drift alerts wired to 01 and 02 |
| 04 | [Synthetic Data](synthetic-data/) | *"Where does the training/eval data come from?"* | Pipeline that produces eval-harness-ready JSONL with dedup + diversity metrics |

---

## How the four projects connect

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MONOREPO OVERVIEW                           │
│                                                                     │
│  ┌──────────────────┐        quality scores        ┌─────────────┐  │
│  │  02 Local        │ ◄─────────────────────────── │ 01 Eval     │  │
│  │  Inference       │                              │ Harness     │  │
│  │  + Router        │ ──── endpoint calls ────────►│             │  │
│  └──────┬───────────┘                              └─────┬───────┘  │
│         │                                                 │         │
│         │ traces (spans)                    test data     │         │
│         ▼                                                 ▼         │
│  ┌──────────────────┐                        ┌────────────────────┐ │
│  │  03              │                        │  04 Synthetic      │ │
│  │  Observability   │ ◄─── instruments ───── │  Data Pipeline     │ │
│  │  Dashboard       │                        │  (feeds 01)        │ │
│  └──────────────────┘                        └────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Flow:**
1. **04** generates high-quality synthetic Q&A examples and writes them as `eval-harness`-compatible JSONL.
2. **01** runs a test suite over those examples, scoring each model call and detecting regressions.
3. **02** benchmarks local models across quantization levels, using **01** for the quality axis of the Pareto chart, and exposes a cost/quality router as an OpenAI-compatible API.
4. **03** traces every LLM call in **01** and **02**, detecting cost/latency drift and providing a replay endpoint for debugging.

---

## Repository structure

```
PROGETO-01-PORTFOLIO/
│
├── eval-harness/          # Project 01
│   ├── datasets/          # versioned JSONL + SHA-256 manifests
│   ├── evals/             # test cases, metrics (deterministic + judge)
│   ├── runners/           # async runner + CLI
│   ├── store/             # Pydantic models + SQLite persistence
│   ├── report/            # regression compare + markdown/chart render
│   └── .github/workflows/ # CI: run suite → compare → block merge
│
├── local-inference/       # Project 02
│   ├── serving/           # engine abstraction (llama.cpp / vLLM / mock)
│   ├── benchmark/         # perf (TTFT/tok/s/VRAM) + quality sweep
│   ├── router/            # heuristic difficulty classifier + routing policy
│   ├── api/               # FastAPI OpenAI-compatible endpoint
│   └── report/            # Pareto charts + trade-off table
│
├── observability/         # Project 03
│   ├── otel/              # @trace_llm decorator + GenAI OTel attributes
│   ├── collector/         # span ingestion + cost estimation
│   ├── store/             # trace + alert persistence
│   ├── analysis/          # drift detection, anomaly alerts, replay
│   └── dashboard/         # FastAPI REST dashboard
│
├── synthetic-data/        # Project 04
│   ├── spec/              # target schema + generation plan
│   ├── generate/          # seed diversifiers + structured LLM generation
│   ├── quality/           # validate → dedup (MinHash) → filter → diversity
│   ├── report/            # funnel + distribution + diversity charts
│   └── pipeline.py        # end-to-end CLI orchestrator
│
└── documentazione/        # project specs (Italian)
```

---

## Test coverage

| Project | Tests | Status |
|---------|-------|--------|
| eval-harness | 27 | ✅ all pass |
| local-inference | 35 | ✅ all pass |
| observability | 32 | ✅ all pass |
| synthetic-data | 47 | ✅ all pass |
| **Total** | **141** | ✅ |

---

## Quick start

Each project is a self-contained Python package. From the project folder:

```bash
# Install
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Project-specific CLIs (examples)
cd eval-harness   && eval run --dataset legal_qa --model claude-haiku-4-5-20251001
cd local-inference && infer sweep --output results/sweep.json
cd observability   && obs serve --port 8001
cd synthetic-data  && python pipeline.py dry-run        # no API key needed
```

**API key** (for real runs of 01, 02, 04):

```bash
cp eval-harness/.env.example eval-harness/.env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

---

## Tech stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Language | Python 3.11+ | consistent across all projects |
| Validation | Pydantic v2 | typed, fast, zero-surprise serialization |
| HTTP client | httpx (async) | explicit control over API calls; no heavy framework |
| Persistence | SQLite + aiosqlite | zero-infrastructure, portable, scales to millions of traces |
| API | FastAPI | async, auto-docs, used in both 02 and 03 |
| Tracing | OpenTelemetry (GenAI semantic conventions) | industry standard; not a custom format |
| CI | GitHub Actions | runs eval suite, posts report to PR, blocks merge on regression |

---

## What this demonstrates

- **Rigor**: quality is a number, not an impression. Regressions are caught by CI.
- **Production thinking**: cost, latency, drift, debugging replay — the problems that appear after deployment.
- **Systems design**: four components that compose cleanly, not four isolated demos.
- **Awareness of limits**: each project has a Limits section. Knowing what the system *doesn't* cover signals engineering maturity.
