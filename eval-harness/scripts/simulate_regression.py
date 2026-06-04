"""
Simulate a regression scenario and generate portfolio-ready screenshots.

Produces (no API key needed):
  docs/assets/regression_report.png   — the PR comment content
  docs/assets/ci_checks.png           — GitHub CI checks panel (blocked)
  docs/assets/run_comparison.png      — baseline vs candidate bar chart

Run:
    cd eval-harness
    python scripts/simulate_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the eval-harness directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
from datetime import datetime, timezone

from store.schema import (
    CaseResult,
    DeterministicScores,
    EvalRun,
    JudgeDimension,
    JudgeScores,
    ModelConfig,
    RegressionReport,
    RunStatus,
)
from report.compare import compare_runs
from report.render import render_regression_report


# ---------------------------------------------------------------------------
# Build two fake EvalRuns: baseline (good) and candidate (regressed)
# ---------------------------------------------------------------------------

def _make_run(
    run_id: str,
    model_id: str,
    branch: str,
    git_ref: str,
    judge_scores: list[float],
    latencies_ms: list[float],
    pass_flags: list[bool],
    cost_per_case: float = 0.00007,
) -> EvalRun:
    config = ModelConfig(model_id=model_id, temperature=0.0)
    run = EvalRun(
        id=run_id,
        dataset_name="legal_qa",
        dataset_version="v1",
        dataset_sha256="c204c21e53df99a0" + "0" * 48,
        config=config,
        status=RunStatus.COMPLETED,
        branch=branch,
        git_ref=git_ref,
        created_at=datetime(2025, 6, 4, 10, 0, tzinfo=timezone.utc),
        completed_at=datetime(2025, 6, 4, 10, 3, tzinfo=timezone.utc),
    )

    case_ids = [f"lq-{i:03d}" for i in range(1, len(judge_scores) + 1)]
    for i, (score, lat, passed) in enumerate(zip(judge_scores, latencies_ms, pass_flags)):
        s = int(round(score))
        result = CaseResult(
            case_id=case_ids[i],
            run_id=run_id,
            model_output="[simulated output]",
            latency_ms=lat,
            tokens_in=120,
            tokens_out=80,
            cost_usd=cost_per_case,
            deterministic=DeterministicScores(exact_match=passed),
            judge=JudgeScores(
                correctness=JudgeDimension(score=max(1, min(5, s)), reasoning="sim"),
                completeness=JudgeDimension(score=max(1, min(5, s)), reasoning="sim"),
                format_adherence=JudgeDimension(score=max(1, min(5, s)), reasoning="sim"),
                hallucination_detected=False,
            ),
        )
        run.results.append(result)

    return run


# Baseline: solid scores, healthy latency
BASELINE_JUDGE = [4.0, 3.7, 4.3, 3.5, 4.2, 3.8, 4.1, 3.6, 4.4, 3.9,
                  4.0, 3.7, 4.3, 3.5, 4.2, 3.8, 4.1, 3.9, 4.0, 3.7]
BASELINE_LAT   = [820, 890, 760, 940, 810, 870, 850, 920, 800, 860,
                  830, 900, 770, 950, 820, 880, 840, 910, 810, 870]
BASELINE_PASS  = [True]*14 + [False]*6  # 70% pass rate

# Candidate (regressed): scores dropped, slower
CAND_JUDGE = [3.0, 2.7, 3.3, 2.5, 3.2, 2.8, 3.1, 2.6, 2.4, 2.9,
              3.0, 2.7, 3.3, 2.5, 3.2, 2.8, 3.1, 2.9, 3.0, 2.7]
CAND_LAT   = [1020, 1190, 960, 1240, 1010, 1170, 1050, 1220, 1000, 1160,
              1030, 1200, 970, 1250, 1020, 1180, 1040, 1210, 1010, 1170]
CAND_PASS  = [True]*9 + [False]*11  # 45% pass rate (was 70%)

baseline = _make_run(
    run_id="a3f81c2e-0000-0000-0000-000000000000",
    model_id="claude-haiku-4-5-20251001",
    branch="main",
    git_ref="d4b9e1f",
    judge_scores=BASELINE_JUDGE,
    latencies_ms=BASELINE_LAT,
    pass_flags=BASELINE_PASS,
)

candidate = _make_run(
    run_id="b7d23a19-0000-0000-0000-000000000000",
    model_id="claude-haiku-4-5-20251001",
    branch="feat/shorter-system-prompt",
    git_ref="9c1a4f3",
    judge_scores=CAND_JUDGE,
    latencies_ms=CAND_LAT,
    pass_flags=CAND_PASS,
)

regression = compare_runs(baseline, candidate, threshold_judge=0.3, threshold_pass_rate=0.05)


# ---------------------------------------------------------------------------
# 1. Render regression report PNG (PR comment)
# ---------------------------------------------------------------------------

def _render_regression_png(reg: RegressionReport, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(10, 7), facecolor="#0d1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # Title bar
    fig.text(0.04, 0.95, "[FAIL]  REGRESSION DETECTED  —  eval-harness bot",
             color="#f85149", fontsize=13, fontweight="bold", fontfamily="monospace")
    fig.text(0.04, 0.91, "Posted by eval-harness[bot] on PR #12  •  2025-06-04 10:04 UTC",
             color="#8b949e", fontsize=9, fontfamily="monospace")

    # Divider
    ax.axhline(0.89, color="#30363d", linewidth=1, xmin=0.04, xmax=0.96)

    # Table data
    rows = [
        ("Baseline run",   f"a3f81c2e  (main @ d4b9e1f)",          "#58a6ff"),
        ("Candidate run",  f"b7d23a19  (feat/shorter-system-prompt @ 9c1a4f3)", "#58a6ff"),
        ("Δ Judge avg",    f"{reg.delta_judge_avg:+.3f}  (threshold ±{reg.threshold_judge})", "#f85149"),
        ("Δ Pass rate",    f"{reg.delta_pass_rate:+.3f}  (threshold ±{reg.threshold_pass_rate})", "#f85149"),
        ("Δ Latency p50",  f"{reg.delta_latency_ms:+.1f} ms", "#e3b341"),
        ("Δ Cost USD",     f"{reg.delta_cost_usd:+.6f}", "#3fb950"),
    ]

    y = 0.83
    for label, value, vcol in rows:
        fig.text(0.06, y, label, color="#8b949e", fontsize=10, fontfamily="monospace")
        fig.text(0.38, y, value, color=vcol,     fontsize=10, fontfamily="monospace")
        y -= 0.07

    # Divider
    ax.axhline(y + 0.04, color="#30363d", linewidth=1, xmin=0.04, xmax=0.96)
    y -= 0.01

    # Regressed cases header
    fig.text(0.06, y, "Regressed cases", color="#c9d1d9", fontsize=10,
             fontweight="bold", fontfamily="monospace")
    y -= 0.06

    # Column headers
    for x_pos, hdr in [(0.06, "Case ID"), (0.22, "Baseline"), (0.38, "Candidate"), (0.54, "Δ")]:
        fig.text(x_pos, y, hdr, color="#8b949e", fontsize=9, fontfamily="monospace")
    y -= 0.055

    for d in reg.details[:6]:
        fig.text(0.06, y, d["case_id"],                  color="#c9d1d9", fontsize=9, fontfamily="monospace")
        fig.text(0.22, y, f"{d['baseline_score']:.2f}",  color="#3fb950", fontsize=9, fontfamily="monospace")
        fig.text(0.38, y, f"{d['candidate_score']:.2f}", color="#f85149", fontsize=9, fontfamily="monospace")
        fig.text(0.54, y, f"{d['delta']:+.3f}",          color="#f85149", fontsize=9, fontfamily="monospace")
        y -= 0.05

    # Footer
    ax.axhline(0.04, color="#30363d", linewidth=1, xmin=0.04, xmax=0.96)
    fig.text(0.06, 0.015, "eval-harness v0.1.0  •  dataset: legal_qa.v1  •  exit code: 1",
             color="#8b949e", fontsize=8, fontfamily="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  OK  {out_path}")


# ---------------------------------------------------------------------------
# 2. CI checks panel PNG
# ---------------------------------------------------------------------------

def _render_ci_checks_png(out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4.2), facecolor="#0d1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # Header
    fig.text(0.04, 0.90, "Some checks were not successful",
             color="#e3b341", fontsize=12, fontweight="bold")
    fig.text(0.04, 0.82, "1 failing and 2 successful checks",
             color="#8b949e", fontsize=10)

    ax.axhline(0.78, color="#30363d", linewidth=1, xmin=0.0, xmax=1.0)

    # Check rows
    checks = [
        ("X", "#f85149", "eval / eval (pull_request)",   "Quality regression detected — judge avg dropped 0.412", "Required", True),
        ("v", "#3fb950", "lint / ruff (pull_request)",   "All checks passed",                                     "Required", False),
        ("v", "#3fb950", "test / pytest (pull_request)", "141 passed in 14.3s",                                   "Required", False),
    ]

    y = 0.65
    for icon, col, name, desc, badge, failed in checks:
        fig.text(0.04,  y,       icon,  color=col,       fontsize=14, fontfamily="monospace")
        fig.text(0.085, y,       name,  color="#c9d1d9", fontsize=10, fontweight="bold")
        fig.text(0.085, y-0.09, desc,  color="#8b949e", fontsize=9)
        # Badge
        badge_col = "#f85149" if failed else "#3fb950"
        ax.text(0.93, y - 0.04, badge, transform=ax.transAxes,
                color=badge_col, fontsize=8, fontfamily="monospace",
                ha="right", bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                                       edgecolor=badge_col, linewidth=1))
        ax.axhline(y - 0.18, color="#21262d", linewidth=0.5, xmin=0.0, xmax=1.0)
        y -= 0.26

    # Merge blocked banner
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.12, transform=ax.transAxes,
                                facecolor="#161b22", edgecolor="#f85149", linewidth=1.5))
    fig.text(0.04, 0.03, "[BLOCKED]  Merging is blocked — required status checks have not passed.",
             color="#f85149", fontsize=10, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  OK  {out_path}")


# ---------------------------------------------------------------------------
# 3. Run comparison bar chart
# ---------------------------------------------------------------------------

def _render_comparison_png(baseline: EvalRun, candidate: EvalRun, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from report.render import render_run_report
    from evals.metrics.aggregate import summarise_run

    b = summarise_run(baseline)
    c = summarise_run(candidate)

    metrics = ["Judge mean", "Pass rate", "Lat p50 (×100ms)"]
    b_vals = [b["judge_mean"], b["pass_rate"], b["latency_ms"]["p50"] / 100]
    c_vals = [c["judge_mean"], c["pass_rate"], c["latency_ms"]["p50"] / 100]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    bars_b = ax.bar(x - w/2, b_vals, w, label="Baseline (main @ d4b9e1f)",         color="#3fb950", alpha=0.85)
    bars_c = ax.bar(x + w/2, c_vals, w, label="Candidate (feat/shorter-system-prompt)", color="#f85149", alpha=0.85)

    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                color="#3fb950", fontsize=9)
    for bar in bars_c:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                color="#f85149", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="#c9d1d9", fontsize=10)
    ax.set_ylabel("Score / normalised value", color="#8b949e")
    ax.set_title("Baseline vs Candidate — Regression Summary", color="#c9d1d9",
                 fontsize=12, pad=12)
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    ax.grid(axis="y", color="#21262d", linewidth=0.5)
    ax.set_ylim(0, max(max(b_vals), max(c_vals)) * 1.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  OK  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ASSETS = Path(__file__).parent.parent / "docs" / "assets"

print("\nGenerating portfolio screenshots...\n")
_render_regression_png(regression,    ASSETS / "regression_report.png")
_render_ci_checks_png(                ASSETS / "ci_checks.png")
_render_comparison_png(baseline, candidate, ASSETS / "run_comparison.png")

print(f"\nAll done → {ASSETS.resolve()}")
print("\nAdd to eval-harness/README.md:")
print("  ![CI blocked](docs/assets/ci_checks.png)")
print("  ![Regression report](docs/assets/regression_report.png)")
print("  ![Run comparison](docs/assets/run_comparison.png)")
