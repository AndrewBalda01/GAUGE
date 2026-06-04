"""
Pipeline statistics report: distribution, discard rates, diversity charts.

Produces:
  - Markdown report (pipeline_report.md)
  - Distribution bar chart (topic × difficulty)
  - Diversity scatter/bar (before vs after clustering)
  - Discard funnel chart (validate → dedup → filter → final)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from spec.schema import SyntheticExample


# ---------------------------------------------------------------------------
# Discard funnel
# ---------------------------------------------------------------------------

def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.1f}"


def render_markdown_report(
    raw_count: int,
    after_validate: int,
    after_dedup: int,
    after_filter: int,
    final: list[SyntheticExample],
    diversity_before=None,
    diversity_after=None,
) -> str:
    def _rate(a: int, b: int) -> str:
        if b == 0:
            return "—"
        return f"{(1 - a / b) * 100:.1f}%"

    lines = [
        "# Synthetic Data Pipeline Report",
        "",
        "## Funnel",
        "",
        "| Stage | Count | Discarded |",
        "|---|---|---|",
        f"| Generated (raw) | {raw_count} | — |",
        f"| After validate | {after_validate} | {_rate(after_validate, raw_count)} |",
        f"| After dedup | {after_dedup} | {_rate(after_dedup, after_validate)} |",
        f"| After filter | {after_filter} | {_rate(after_filter, after_dedup)} |",
        "",
        f"**Final dataset: {len(final)} examples**  "
        f"(overall discard rate: {_rate(len(final), raw_count)})",
        "",
        "## Topic distribution",
        "",
        "| Topic | Count |",
        "|---|---|",
    ]
    topic_counts = Counter(ex.topic for ex in final)
    for topic, cnt in sorted(topic_counts.items()):
        lines.append(f"| {topic} | {cnt} |")

    lines += [
        "",
        "## Difficulty distribution",
        "",
        "| Difficulty | Count |",
        "|---|---|",
    ]
    diff_counts = Counter(ex.difficulty.value for ex in final)
    for diff, cnt in sorted(diff_counts.items()):
        lines.append(f"| {diff} | {cnt} |")

    if diversity_before and diversity_after:
        lines += [
            "",
            "## Diversity (before vs after dedup + filter)",
            "",
            "| Metric | Before | After |",
            "|---|---|---|",
            f"| Examples | {diversity_before.n_examples} | {diversity_after.n_examples} |",
            f"| Clusters | {diversity_before.n_clusters} | {diversity_after.n_clusters} |",
            f"| Coverage score | {diversity_before.coverage_score:.3f} | {diversity_after.coverage_score:.3f} |",
            f"| Balance score | {diversity_before.balance_score:.3f} | {diversity_after.balance_score:.3f} |",
        ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_distribution(final: list[SyntheticExample], out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    topic_diff: dict[str, dict[str, int]] = {}
    for ex in final:
        td = topic_diff.setdefault(ex.topic, {})
        td[ex.difficulty.value] = td.get(ex.difficulty.value, 0) + 1

    topics = sorted(topic_diff)
    diffs  = ["basic", "applied", "analytical"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    x = np.arange(len(topics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (diff, color) in enumerate(zip(diffs, colors)):
        counts = [topic_diff[t].get(diff, 0) for t in topics]
        ax.bar(x + (i - 1) * w, counts, w, label=diff, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Topic × Difficulty distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    p = out_dir / "distribution.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return p


def plot_diversity_comparison(diversity_before, diversity_after, out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    labels = [f"C{i}" for i in range(max(diversity_before.n_clusters, diversity_after.n_clusters))]

    def _pad(sizes, n):
        return sizes + [0] * (n - len(sizes))

    n = len(labels)
    b_sizes = _pad(diversity_before.cluster_sizes, n)
    a_sizes = _pad(diversity_after.cluster_sizes, n)

    x = np.arange(n)
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - w / 2, b_sizes, w, label="Before filter", color="#C44E52", alpha=0.7)
    ax.bar(x + w / 2, a_sizes, w, label="After filter",  color="#55A868", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Examples per cluster")
    ax.set_title("Diversity: cluster sizes before vs after quality pipeline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    p = out_dir / "diversity_comparison.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return p


def plot_funnel(
    raw: int, after_validate: int, after_dedup: int, after_filter: int, out_dir: Path
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    stages = ["Generated", "Post-validate", "Post-dedup", "Post-filter"]
    counts = [raw, after_validate, after_dedup, after_filter]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(stages[::-1], counts[::-1], color=colors[::-1])
    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=10)
    ax.set_xlabel("Examples")
    ax.set_title("Pipeline funnel — discard by stage")
    ax.set_xlim(0, raw * 1.1)
    ax.grid(axis="x", alpha=0.3)

    p = out_dir / "funnel.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return p
