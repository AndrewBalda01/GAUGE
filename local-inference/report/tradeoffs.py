"""
Trade-off report: Pareto charts and markdown table for sweep results.

Reads the JSON produced by benchmark/sweep.py and generates:
  - Markdown table: config × tok/s × latency × VRAM × quality × quant
  - Pareto chart: quality score vs tok/s (scatter, colored by quant bits)
  - Pareto chart: quality score vs VRAM (scatter)
  - Latency bar chart across configs
"""

from __future__ import annotations

import json
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def render_markdown_table(rows: list[dict]) -> str:
    header = (
        "| Config | Quant bits | tok/s | Lat p50 ms | Lat p95 ms "
        "| VRAM MB | Pass rate | Judge mean |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r.get('config', '—')} "
            f"| {r.get('quant_bits', '—')} "
            f"| {_fmt(r.get('tok_per_sec_mean'))} "
            f"| {_fmt(r.get('latency_p50_ms', r.get('latency_ms_p50')))} "
            f"| {_fmt(r.get('latency_p95_ms', r.get('latency_ms_p95')))} "
            f"| {_fmt(r.get('vram_peak_mb'))} "
            f"| {_fmt(r.get('quality_pass_rate'), 3)} "
            f"| {_fmt(r.get('quality_judge_mean'), 3)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _plot_pareto_quality_vs_tps(rows: list[dict], out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        return None

    # Filter to rows that have quality data
    data = [r for r in rows if r.get("quality_judge_mean") is not None]
    if not data:
        # If no quality data, use pass_rate if available, else skip
        data = [r for r in rows if r.get("quality_pass_rate") is not None]
        y_key, y_label = "quality_pass_rate", "Pass Rate"
    else:
        y_key, y_label = "quality_judge_mean", "Judge Mean Score (1–5)"

    if not data:
        return None

    quant_vals = sorted({r.get("quant_bits", 0) for r in data})
    colors = {q: cm.viridis(i / max(len(quant_vals) - 1, 1)) for i, q in enumerate(quant_vals)}

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in data:
        x = r.get("tok_per_sec_mean", 0)
        y = r.get(y_key, 0)
        q = r.get("quant_bits", 0)
        ax.scatter(x, y, c=[colors[q]], s=100, zorder=3)
        ax.annotate(r.get("config", ""), (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    # Legend for quant bits
    for q in quant_vals:
        ax.scatter([], [], c=[colors[q]], label=f"Q{q}", s=80)
    ax.legend(title="Quant bits", fontsize=9)

    ax.set_xlabel("Throughput (tok/s)")
    ax.set_ylabel(y_label)
    ax.set_title("Pareto: Quality vs Throughput")
    ax.grid(True, alpha=0.3)

    p = out_dir / "pareto_quality_vs_tps.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return p


def _plot_pareto_quality_vs_vram(rows: list[dict], out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    data = [r for r in rows if r.get("vram_peak_mb", 0) > 0]
    if not data:
        return None

    y_key = "quality_judge_mean" if any(r.get("quality_judge_mean") for r in data) else "quality_pass_rate"
    y_label = "Judge Mean Score" if y_key == "quality_judge_mean" else "Pass Rate"

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in data:
        ax.scatter(r.get("vram_peak_mb", 0), r.get(y_key, 0), s=100, color="#4C72B0", zorder=3)
        ax.annotate(r.get("config", ""), (r.get("vram_peak_mb", 0), r.get(y_key, 0)),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("VRAM Peak (MB)")
    ax.set_ylabel(y_label)
    ax.set_title("Pareto: Quality vs VRAM")
    ax.grid(True, alpha=0.3)

    p = out_dir / "pareto_quality_vs_vram.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return p


def _plot_latency_bar(rows: list[dict], out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    if not rows:
        return None

    labels = [r.get("config", f"cfg-{i}") for i, r in enumerate(rows)]
    p50 = [r.get("latency_p50_ms", r.get("latency_ms_p50", 0)) or 0 for r in rows]
    p95 = [r.get("latency_p95_ms", r.get("latency_ms_p95", 0)) or 0 for r in rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, p50, w, label="p50", color="#4C72B0")
    ax.bar(x + w / 2, p95, w, label="p95", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency p50 / p95 by Config")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    p = out_dir / "latency_bar.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Pareto frontier helper
# ---------------------------------------------------------------------------

def pareto_frontier(rows: list[dict], x_key: str, y_key: str) -> list[dict]:
    """Return the subset of rows on the Pareto frontier (maximise both x and y)."""
    valid = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    valid.sort(key=lambda r: r[x_key], reverse=True)
    frontier = []
    best_y = float("-inf")
    for r in valid:
        if r[y_key] > best_y:
            frontier.append(r)
            best_y = r[y_key]
    return frontier


# ---------------------------------------------------------------------------
# Top-level render function
# ---------------------------------------------------------------------------

def render_tradeoff_report(sweep_json: str, output_dir: str) -> None:
    from rich.console import Console
    console = Console()

    path = Path(sweep_json)
    if not path.exists():
        console.print(f"[red]Sweep file not found:[/red] {path}")
        return

    rows: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Markdown table
    md = render_markdown_table(rows)
    table_path = out / "tradeoffs.md"
    table_path.write_text(
        f"# Trade-off Table\n\n{md}\n\n"
        f"Generated from: `{path.name}`\n",
        encoding="utf-8",
    )
    console.print(f"[green]Table[/green] → {table_path}")

    # Pareto frontier annotation
    frontier = pareto_frontier(rows, "tok_per_sec_mean", "quality_judge_mean")
    if frontier:
        console.print(f"[bold]Pareto frontier[/bold] ({len(frontier)} configs):")
        for r in frontier:
            console.print(
                f"  {r.get('config')}: "
                f"tok/s={r.get('tok_per_sec_mean', 0):.1f}  "
                f"judge={r.get('quality_judge_mean', '—')}"
            )

    # Charts
    saved = []
    for fn in [_plot_pareto_quality_vs_tps, _plot_pareto_quality_vs_vram, _plot_latency_bar]:
        p = fn(rows, out)
        if p:
            saved.append(p)
            console.print(f"[green]Chart[/green] → {p}")

    if not saved:
        console.print("[yellow]matplotlib not available — charts skipped[/yellow]")
