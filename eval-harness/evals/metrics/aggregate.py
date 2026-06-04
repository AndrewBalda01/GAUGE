"""Aggregate metrics: pass@k, percentiles, cost/latency summaries."""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from store.schema import CaseResult, EvalRun


# ---------------------------------------------------------------------------
# pass@k  (unbiased estimator from Chen et al. 2021)
# ---------------------------------------------------------------------------

def _comb(n: int, k: int) -> float:
    if k > n:
        return 0.0
    return math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: n samples, c correct, estimate pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------

def percentiles(values: Sequence[float], qs: Sequence[float] = (50, 95, 99)) -> dict[str, float]:
    if not values:
        return {f"p{int(q)}": float("nan") for q in qs}
    sorted_v = sorted(values)
    result = {}
    for q in qs:
        idx = (q / 100) * (len(sorted_v) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(sorted_v) - 1)
        result[f"p{int(q)}"] = sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo])
    return result


# ---------------------------------------------------------------------------
# Run-level summary
# ---------------------------------------------------------------------------

def summarise_run(run: EvalRun) -> dict:
    ok = [r for r in run.results if r.error is None]
    failed = [r for r in run.results if r.error is not None]

    latencies = [r.latency_ms for r in ok]
    costs = [r.cost_usd for r in ok]
    judge_scores = [s for r in ok if (s := r.judge_avg_score) is not None]

    exact_matches = [r for r in ok if r.deterministic.exact_match is True]

    summary = {
        "run_id": run.id,
        "model": run.config.model_id,
        "dataset": f"{run.dataset_name}.{run.dataset_version}",
        "total_cases": len(run.results),
        "ok": len(ok),
        "failed": len(failed),
        "pass_rate": len(exact_matches) / len(ok) if ok else None,
        "judge_mean": statistics.mean(judge_scores) if judge_scores else None,
        "judge_stdev": statistics.stdev(judge_scores) if len(judge_scores) > 1 else None,
        "latency_ms": percentiles(latencies),
        "total_cost_usd": sum(costs),
        "mean_cost_usd": statistics.mean(costs) if costs else None,
        "tokens_in_total": sum(r.tokens_in for r in ok),
        "tokens_out_total": sum(r.tokens_out for r in ok),
    }

    if ok:
        n = len(ok)
        c = len(exact_matches)
        summary["pass_at_1"] = pass_at_k(n, c, 1)

    return summary
