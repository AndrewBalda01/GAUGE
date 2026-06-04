"""
Sweep: run perf + quality benchmarks across all engine configs.
Produces the trade-off table used for Pareto analysis.

Usage:
    python -m benchmark.sweep --output results/sweep.json
    python -m benchmark.sweep --configs qwen3-0.6b-q4,qwen3-1.7b-q4 --output results/sweep.json
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from benchmark.perf import PerfResult, benchmark_engine
from benchmark.quality import _HARNESS_AVAILABLE, run_quality_eval
from serving.config_loader import load_all_configs, load_config
from serving.engine import load_engine

console = Console()


@dataclass
class ConfigSweepResult:
    config_name: str
    backend: str
    size_class: str
    quant_bits: int
    model_path: str = ""

    # Perf
    perf: dict = field(default_factory=dict)

    # Quality (from eval-harness)
    quality_pass_rate: float | None = None
    quality_judge_mean: float | None = None
    quality_total_cases: int = 0

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "backend": self.backend,
            "size_class": self.size_class,
            "quant_bits": self.quant_bits,
            "model_path": self.model_path,
            **self.perf,
            "quality_pass_rate": self.quality_pass_rate,
            "quality_judge_mean": self.quality_judge_mean,
            "quality_total_cases": self.quality_total_cases,
        }


async def sweep_config(
    cfg: dict,
    *,
    n_perf_runs: int = 20,
    max_tokens: int = 128,
    run_quality: bool = True,
    quality_tags: list[str] | None = None,
) -> ConfigSweepResult:
    engine = load_engine(cfg)
    result = ConfigSweepResult(
        config_name=cfg["name"],
        backend=cfg.get("backend", "unknown"),
        size_class=cfg.get("size_class", "unknown"),
        quant_bits=cfg.get("quant_bits", 0),
        model_path=cfg.get("model_path", cfg.get("model_id", "")),
    )

    try:
        await engine.load()
        console.print(f"  [green]loaded[/green] {cfg['name']}")

        # --- perf benchmark ---
        perf = await benchmark_engine(engine, n_runs=n_perf_runs, max_tokens=max_tokens)
        result.perf = perf.to_dict()
        console.print(
            f"  perf  p50={perf.latency_ms_p50:.0f}ms  "
            f"tok/s={perf.tokens_per_second_mean:.1f}  "
            f"VRAM={perf.vram_mb_peak:.0f}MB"
        )

        # --- quality benchmark ---
        if run_quality and _HARNESS_AVAILABLE:
            eval_run = await run_quality_eval(
                engine,
                tags=quality_tags,
                max_tokens=max_tokens,
                concurrency=2,
            )
            result.quality_pass_rate = eval_run.pass_rate
            result.quality_total_cases = len(eval_run.results)
            console.print(
                f"  quality  pass_rate={result.quality_pass_rate or 0:.2f}  "
                f"cases={result.quality_total_cases}"
            )
        elif run_quality and not _HARNESS_AVAILABLE:
            console.print("  [yellow]quality skipped[/yellow] — eval-harness not found")

    except Exception as exc:
        console.print(f"  [red]ERROR[/red] {cfg['name']}: {exc}")
    finally:
        await engine.unload()

    return result


async def run_sweep(
    config_names: list[str] | None = None,
    exclude_mock: bool = False,
    n_perf_runs: int = 20,
    max_tokens: int = 128,
    run_quality: bool = True,
    output_path: str = "results/sweep.json",
) -> list[ConfigSweepResult]:
    if config_names:
        cfgs = [load_config(n) for n in config_names]
    else:
        cfgs = load_all_configs(exclude_mock=exclude_mock)

    console.print(f"[bold]Sweep[/bold] — {len(cfgs)} configs, {n_perf_runs} perf runs each")
    results: list[ConfigSweepResult] = []

    for cfg in cfgs:
        console.print(f"\n[bold]{cfg['name']}[/bold]")
        r = await sweep_config(
            cfg,
            n_perf_runs=n_perf_runs,
            max_tokens=max_tokens,
            run_quality=run_quality,
        )
        results.append(r)

    # Save results
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in results]
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    console.print(f"\n[green]Sweep complete[/green] → {out}")

    # Print summary table
    _print_table(results)
    return results


def _print_table(results: list[ConfigSweepResult]) -> None:
    t = Table(title="Sweep Results")
    cols = [
        "config", "quant", "tok/s (mean)", "lat p50 ms",
        "lat p95 ms", "VRAM MB", "pass rate", "judge mean"
    ]
    for c in cols:
        t.add_column(c)
    for r in results:
        t.add_row(
            r.config_name,
            str(r.quant_bits),
            f"{r.perf.get('tok_per_sec_mean', 0):.1f}",
            f"{r.perf.get('latency_p50_ms', 0):.0f}",
            f"{r.perf.get('latency_p95_ms', 0):.0f}",
            f"{r.perf.get('vram_peak_mb', 0):.0f}",
            f"{r.quality_pass_rate:.2f}" if r.quality_pass_rate is not None else "—",
            f"{r.quality_judge_mean:.2f}" if r.quality_judge_mean is not None else "—",
        )
    console.print(t)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sweep all engine configs")
    parser.add_argument("--configs", default=None, help="Comma-separated config names")
    parser.add_argument("--n-perf-runs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--exclude-mock", action="store_true")
    parser.add_argument("--output", default="results/sweep.json")
    args = parser.parse_args()

    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else None
    asyncio.run(run_sweep(
        config_names=config_names,
        exclude_mock=args.exclude_mock,
        n_perf_runs=args.n_perf_runs,
        max_tokens=args.max_tokens,
        run_quality=not args.no_quality,
        output_path=args.output,
    ))
