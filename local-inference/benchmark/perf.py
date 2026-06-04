"""
Performance benchmark: TTFT, tok/s, p50/p95/p99 latency, VRAM peak.

Usage (programmatic):
    results = await benchmark_engine(engine, prompts=PROMPTS, n_runs=20)

Usage (CLI):
    python -m benchmark.perf --config qwen3-0.6b-q4 --n-runs 20
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field

import psutil

from serving.engine import BaseEngine, GenerateRequest, GenerateResponse


# ---------------------------------------------------------------------------
# Default prompt set — short / medium / long to stress different scenarios
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS: list[str] = [
    "What is a contract?",
    "Define consideration in contract law.",
    "What are the elements of negligence in tort law?",
    "Explain the difference between void and voidable contracts.",
    "What is promissory estoppel and when does it apply?",
    "Describe the doctrine of frustration of contract.",
    "What remedies are available for breach of contract? List and briefly explain each.",
    "Explain the key principles of the rule of law and why it matters in a democratic society.",
    "What is the difference between a fiduciary duty and a general duty of care? Provide examples.",
    "Describe the main categories of intellectual property, their duration, and what they protect.",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SingleRunResult:
    prompt: str
    response: GenerateResponse
    vram_mb_before: float
    vram_mb_after: float


@dataclass
class PerfResult:
    config_name: str
    n_runs: int
    prompts_used: int

    # Latency (total generation time)
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_ms_mean: float

    # TTFT
    ttft_ms_p50: float
    ttft_ms_p95: float
    ttft_ms_mean: float

    # Throughput
    tokens_per_second_mean: float
    tokens_per_second_p50: float

    # Tokens
    tokens_generated_mean: float
    tokens_prompt_mean: float

    # Memory
    vram_mb_peak: float
    ram_mb_delta: float

    # Raw series (for plotting)
    latencies_ms: list[float] = field(default_factory=list)
    ttfts_ms: list[float] = field(default_factory=list)
    tps_series: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "n_runs": self.n_runs,
            "latency_p50_ms": round(self.latency_ms_p50, 1),
            "latency_p95_ms": round(self.latency_ms_p95, 1),
            "latency_p99_ms": round(self.latency_ms_p99, 1),
            "latency_mean_ms": round(self.latency_ms_mean, 1),
            "ttft_p50_ms": round(self.ttft_ms_p50, 1),
            "ttft_p95_ms": round(self.ttft_ms_p95, 1),
            "ttft_mean_ms": round(self.ttft_ms_mean, 1),
            "tok_per_sec_mean": round(self.tokens_per_second_mean, 1),
            "tok_per_sec_p50": round(self.tokens_per_second_p50, 1),
            "vram_peak_mb": round(self.vram_mb_peak, 1),
            "ram_delta_mb": round(self.ram_mb_delta, 1),
        }


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


async def benchmark_engine(
    engine: BaseEngine,
    prompts: list[str] | None = None,
    n_runs: int = 20,
    max_tokens: int = 128,
    temperature: float = 0.0,
    concurrency: int = 1,
) -> PerfResult:
    """
    Run `n_runs` inference calls over the given prompts (cycling if needed)
    and return a PerfResult with latency/throughput/memory stats.
    """
    if not engine.is_loaded:
        await engine.load()

    prompts = prompts or BENCHMARK_PROMPTS
    ram = psutil.Process()

    ram_before = ram.memory_info().rss / 1024 / 1024
    vram_before = engine.vram_mb

    latencies: list[float] = []
    ttfts: list[float] = []
    tps_list: list[float] = []
    tokens_generated_list: list[float] = []
    tokens_prompt_list: list[float] = []
    vram_peak = vram_before

    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int) -> GenerateResponse:
        prompt = prompts[idx % len(prompts)]
        req = GenerateRequest(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        async with sem:
            return await engine.generate(req)

    tasks = [_one(i) for i in range(n_runs)]
    responses = await asyncio.gather(*tasks)

    for resp in responses:
        latencies.append(resp.total_ms)
        ttfts.append(resp.ttft_ms)
        tps_list.append(resp.tokens_per_second)
        tokens_generated_list.append(resp.tokens_generated)
        tokens_prompt_list.append(resp.tokens_prompt)
        v = engine.vram_mb
        if v > vram_peak:
            vram_peak = v

    ram_after = ram.memory_info().rss / 1024 / 1024

    return PerfResult(
        config_name=engine.name,
        n_runs=n_runs,
        prompts_used=len(prompts),
        latency_ms_p50=_percentile(latencies, 0.50),
        latency_ms_p95=_percentile(latencies, 0.95),
        latency_ms_p99=_percentile(latencies, 0.99),
        latency_ms_mean=statistics.mean(latencies),
        ttft_ms_p50=_percentile(ttfts, 0.50),
        ttft_ms_p95=_percentile(ttfts, 0.95),
        ttft_ms_mean=statistics.mean(ttfts),
        tokens_per_second_mean=statistics.mean(tps_list),
        tokens_per_second_p50=_percentile(tps_list, 0.50),
        tokens_generated_mean=statistics.mean(tokens_generated_list),
        tokens_prompt_mean=statistics.mean(tokens_prompt_list),
        vram_mb_peak=vram_peak,
        ram_mb_delta=ram_after - ram_before,
        latencies_ms=latencies,
        ttfts_ms=ttfts,
        tps_series=tps_list,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from serving.config_loader import load_config
    from serving.engine import load_engine

    parser = argparse.ArgumentParser(description="Performance benchmark")
    parser.add_argument("--config", required=True, help="Config name (e.g. qwen3-0.6b-q4)")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = load_engine(cfg)

    async def _main():
        await engine.load()
        result = await benchmark_engine(
            engine,
            n_runs=args.n_runs,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
        await engine.unload()
        d = result.to_dict()
        print(json.dumps(d, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(d, f, indent=2)

    asyncio.run(_main())
