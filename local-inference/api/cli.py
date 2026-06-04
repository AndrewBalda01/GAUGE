"""CLI to start the API server and run sweeps."""

from __future__ import annotations

import asyncio
import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
def main() -> None:
    """Local Inference CLI."""


@main.command()
@click.option("--small", default="mock-small", show_default=True, help="Small engine config name")
@click.option("--large", default="mock-large", show_default=True, help="Large engine config name")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--uncertain-fallback", default="large", type=click.Choice(["small", "large"]))
@click.option("--reload", is_flag=True, help="Enable hot-reload (dev only)")
def serve(small, large, host, port, uncertain_fallback, reload) -> None:
    """Start the OpenAI-compatible API server."""
    import uvicorn
    from api.main import app, configure_engines

    async def _startup():
        await configure_engines(small, large, uncertain_fallback)

    asyncio.run(_startup())
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@main.command()
@click.option("--configs", default=None, help="Comma-separated config names (default: all)")
@click.option("--n-perf-runs", default=20, type=int, show_default=True)
@click.option("--max-tokens", default=128, type=int, show_default=True)
@click.option("--no-quality", is_flag=True)
@click.option("--exclude-mock", is_flag=True)
@click.option("--output", default="results/sweep.json", show_default=True)
def sweep(configs, n_perf_runs, max_tokens, no_quality, exclude_mock, output) -> None:
    """Sweep all engine configs and produce trade-off table."""
    from benchmark.sweep import run_sweep
    config_names = [c.strip() for c in configs.split(",")] if configs else None
    asyncio.run(run_sweep(
        config_names=config_names,
        exclude_mock=exclude_mock,
        n_perf_runs=n_perf_runs,
        max_tokens=max_tokens,
        run_quality=not no_quality,
        output_path=output,
    ))


@main.command()
@click.option("--sweep-file", default="results/sweep.json", show_default=True)
@click.option("--output-dir", default="results/", show_default=True)
def report(sweep_file, output_dir) -> None:
    """Generate Pareto charts from a sweep results file."""
    from report.tradeoffs import render_tradeoff_report
    render_tradeoff_report(sweep_file, output_dir)


if __name__ == "__main__":
    main()
