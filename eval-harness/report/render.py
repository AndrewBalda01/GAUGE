"""Generate Markdown reports and matplotlib charts for eval runs."""

from __future__ import annotations

import math
from pathlib import Path

from evals.metrics.aggregate import summarise_run
from store.schema import EvalRun, RegressionReport


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def render_run_report(run: EvalRun) -> str:
    s = summarise_run(run)
    lat = s["latency_ms"]
    lines = [
        f"# Eval Run Report",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| Run ID | `{run.id}` |",
        f"| Model | `{run.config.model_id}` |",
        f"| Dataset | `{s['dataset']}` |",
        f"| Branch | `{run.branch or '—'}` |",
        f"| Git ref | `{run.git_ref or '—'}` |",
        f"| Created | {run.created_at.strftime('%Y-%m-%d %H:%M UTC')} |",
        f"",
        f"## Metrics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total cases | {s['total_cases']} |",
        f"| Successful | {s['ok']} |",
        f"| Failed (API error) | {s['failed']} |",
        f"| Pass rate (exact match) | {_fmt(s['pass_rate'], 2)} |",
        f"| Judge mean score (1-5) | {_fmt(s['judge_mean'], 3)} |",
        f"| Judge stdev | {_fmt(s['judge_stdev'], 3)} |",
        f"| Latency p50 | {_fmt(lat.get('p50'), 1)} ms |",
        f"| Latency p95 | {_fmt(lat.get('p95'), 1)} ms |",
        f"| Latency p99 | {_fmt(lat.get('p99'), 1)} ms |",
        f"| Total cost | ${_fmt(s['total_cost_usd'], 4)} |",
        f"| Tokens in | {s['tokens_in_total']:,} |",
        f"| Tokens out | {s['tokens_out_total']:,} |",
        f"",
    ]

    if run.results:
        lines += [
            "## Per-case results",
            "",
            "| Case ID | Judge avg | Exact match | Latency ms | Cost USD | Error |",
            "|---|---|---|---|---|---|",
        ]
        for r in run.results:
            lines.append(
                f"| {r.case_id} "
                f"| {_fmt(r.judge_avg_score, 2)} "
                f"| {'✓' if r.deterministic.exact_match else ('✗' if r.deterministic.exact_match is False else '—')} "
                f"| {r.latency_ms:.0f} "
                f"| {r.cost_usd:.5f} "
                f"| {r.error or ''} |"
            )
        lines.append("")

    return "\n".join(lines)


def render_regression_report(reg: RegressionReport) -> str:
    icon = "🔴 REGRESSION DETECTED" if reg.regression_detected else "✅ No regression"
    lines = [
        f"# Regression Report — {icon}",
        f"",
        f"| | Value |",
        f"|---|---|",
        f"| Baseline run | `{reg.baseline_run_id}` |",
        f"| Candidate run | `{reg.candidate_run_id}` |",
        f"| Δ Judge avg | {_fmt(reg.delta_judge_avg, 3)} (threshold: ±{reg.threshold_judge}) |",
        f"| Δ Pass rate | {_fmt(reg.delta_pass_rate, 3)} (threshold: ±{reg.threshold_pass_rate}) |",
        f"| Δ Cost USD | {_fmt(reg.delta_cost_usd, 5)} |",
        f"| Δ Latency p50 | {_fmt(reg.delta_latency_ms, 1)} ms |",
        f"",
    ]
    if reg.details:
        lines += [
            "## Regressed cases",
            "",
            "| Case ID | Baseline | Candidate | Δ |",
            "|---|---|---|---|",
        ]
        for d in reg.details:
            lines.append(
                f"| {d['case_id']} | {d['baseline_score']} | {d['candidate_score']} | {d['delta']} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts (matplotlib, optional)
# ---------------------------------------------------------------------------

def plot_run_summary(run: EvalRun, out_dir: Path) -> list[Path]:
    """Generate cost/latency/score charts. Returns list of saved paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    ok = [r for r in run.results if r.error is None]
    if not ok:
        return []

    # --- latency histogram ---
    latencies = [r.latency_ms for r in ok]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(latencies, bins=20, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cases")
    ax.set_title(f"Latency distribution — {run.config.model_id}")
    p = out_dir / "latency_hist.png"
    fig.savefig(p, bbox_inches="tight", dpi=120)
    plt.close(fig)
    saved.append(p)

    # --- judge score distribution ---
    j_scores = [r.judge_avg_score for r in ok if r.judge_avg_score is not None]
    if j_scores:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(j_scores, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color="#55A868", edgecolor="white")
        ax.set_xlabel("Judge average score (1–5)")
        ax.set_ylabel("Cases")
        ax.set_title(f"Judge score distribution — {run.config.model_id}")
        p = out_dir / "judge_hist.png"
        fig.savefig(p, bbox_inches="tight", dpi=120)
        plt.close(fig)
        saved.append(p)

    return saved


def plot_comparison(runs: list[EvalRun], out_dir: Path) -> list[Path]:
    """Bar chart comparing multiple runs on judge score + cost."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    summaries = [summarise_run(r) for r in runs]
    labels = [s["model"] for s in summaries]
    judge_means = [s["judge_mean"] or 0 for s in summaries]
    costs = [s["total_cost_usd"] for s in summaries]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, judge_means, color="#4C72B0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Judge mean score (1–5)")
    ax.set_title("Model comparison — judge score")
    p = out_dir / "compare_judge.png"
    fig.savefig(p, bbox_inches="tight", dpi=120)
    plt.close(fig)
    saved.append(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, costs, color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Total cost (USD)")
    ax.set_title("Model comparison — total cost")
    p = out_dir / "compare_cost.png"
    fig.savefig(p, bbox_inches="tight", dpi=120)
    plt.close(fig)
    saved.append(p)

    return saved
