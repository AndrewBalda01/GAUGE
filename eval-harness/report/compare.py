"""Regression detection: compare two EvalRun objects."""

from __future__ import annotations

from store.schema import EvalRun, RegressionReport
from evals.metrics.aggregate import summarise_run


def compare_runs(
    baseline: EvalRun,
    candidate: EvalRun,
    threshold_judge: float = 0.3,
    threshold_pass_rate: float = 0.05,
) -> RegressionReport:
    """Return a RegressionReport; regression_detected=True if any threshold is exceeded."""
    b = summarise_run(baseline)
    c = summarise_run(candidate)

    delta_judge = None
    if b["judge_mean"] is not None and c["judge_mean"] is not None:
        delta_judge = c["judge_mean"] - b["judge_mean"]

    delta_pass = None
    if b["pass_rate"] is not None and c["pass_rate"] is not None:
        delta_pass = c["pass_rate"] - b["pass_rate"]

    delta_cost = (
        c["total_cost_usd"] - b["total_cost_usd"]
        if b["total_cost_usd"] is not None and c["total_cost_usd"] is not None
        else None
    )

    delta_lat = None
    b_lat = b["latency_ms"].get("p50")
    c_lat = c["latency_ms"].get("p50")
    if b_lat is not None and c_lat is not None:
        delta_lat = c_lat - b_lat

    regression = False
    if delta_judge is not None and delta_judge < -threshold_judge:
        regression = True
    if delta_pass is not None and delta_pass < -threshold_pass_rate:
        regression = True

    # Per-case details for cases that regressed
    details = []
    b_map = {r.case_id: r for r in baseline.results}
    c_map = {r.case_id: r for r in candidate.results}
    for case_id, c_res in c_map.items():
        b_res = b_map.get(case_id)
        if b_res is None:
            continue
        b_score = b_res.judge_avg_score
        c_score = c_res.judge_avg_score
        if b_score is not None and c_score is not None and c_score < b_score - threshold_judge:
            details.append({
                "case_id": case_id,
                "baseline_score": round(b_score, 3),
                "candidate_score": round(c_score, 3),
                "delta": round(c_score - b_score, 3),
            })

    return RegressionReport(
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        delta_judge_avg=round(delta_judge, 4) if delta_judge is not None else None,
        delta_pass_rate=round(delta_pass, 4) if delta_pass is not None else None,
        delta_cost_usd=round(delta_cost, 6) if delta_cost is not None else None,
        delta_latency_ms=round(delta_lat, 2) if delta_lat is not None else None,
        regression_detected=regression,
        threshold_judge=threshold_judge,
        threshold_pass_rate=threshold_pass_rate,
        details=details,
    )
