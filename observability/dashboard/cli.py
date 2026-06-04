"""CLI for the observability stack."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
def main() -> None:
    """LLM Observability CLI."""


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8001, type=int, show_default=True)
@click.option("--db", default="traces.db", show_default=True)
@click.option("--reload", is_flag=True)
def serve(host, port, db, reload) -> None:
    """Start the observability dashboard API."""
    import uvicorn
    from dashboard.app import set_db_path
    set_db_path(Path(db))
    uvicorn.run("dashboard.app:app", host=host, port=port, reload=reload)


@main.command()
@click.option("--app", "app_name", default=None)
@click.option("--hours", default=24, type=int, show_default=True)
@click.option("--db", default="traces.db")
def stats(app_name, hours, db) -> None:
    """Print metrics summary to terminal."""
    from store.db import init_db
    from store.queries import aggregate_window, latency_percentiles, error_rate, cost_per_day
    from datetime import datetime, timedelta, timezone
    from rich.console import Console
    from rich.table import Table

    console = Console()
    db_path = Path(db)

    async def _run():
        await init_db(db_path)
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=hours)
        agg = await aggregate_window(since, now, app=app_name, db_path=db_path)
        lat = await latency_percentiles(hours=hours, app=app_name, db_path=db_path)
        err = await error_rate(hours=hours, app=app_name, db_path=db_path)

        t = Table(title=f"Metrics — last {hours}h" + (f" [{app_name}]" if app_name else ""))
        t.add_column("Metric"); t.add_column("Value")
        t.add_row("Calls", str(agg.n_calls))
        t.add_row("Errors", str(agg.n_errors))
        t.add_row("Error rate", f"{err['rate']:.1%}")
        t.add_row("Total cost", f"${agg.total_cost_usd:.5f}")
        t.add_row("Latency p50", f"{lat['p50'] or 0:.0f} ms")
        t.add_row("Latency p95", f"{lat['p95'] or 0:.0f} ms")
        t.add_row("Latency p99", f"{lat['p99'] or 0:.0f} ms")
        t.add_row("Tokens in", f"{agg.total_tokens_in:,}")
        t.add_row("Tokens out", f"{agg.total_tokens_out:,}")
        console.print(t)

    asyncio.run(_run())


@main.command()
@click.option("--app", "app_name", default=None)
@click.option("--db", default="traces.db")
def check(app_name, db) -> None:
    """Run anomaly + drift check and print any alerts."""
    from store.db import init_db
    from analysis.anomaly import check_anomalies, check_drift_and_alert
    from rich.console import Console

    console = Console()
    db_path = Path(db)

    async def _run():
        await init_db(db_path)
        anomaly = await check_anomalies(app=app_name, db_path=db_path)
        drift   = await check_drift_and_alert(app=app_name, db_path=db_path)
        all_a = anomaly + drift
        if not all_a:
            console.print("[green]No anomalies or drift detected.[/green]")
        for a in all_a:
            color = "red" if a.severity == "critical" else "yellow"
            console.print(f"[{color}][{a.severity.upper()}] {a.kind}:[/{color}] {a.message}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
