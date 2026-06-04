"""
Quality benchmark: run eval-harness test cases through a local engine
and return scores compatible with the eval-harness schema.

This is the bridge between Project 02 and Project 01.
The local engine acts as a drop-in "model" for the eval runner.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Make eval-harness importable when running from the monorepo root
_HARNESS_PATH = Path(__file__).parent.parent.parent / "eval-harness"
if str(_HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(_HARNESS_PATH))

from serving.engine import BaseEngine, GenerateRequest

try:
    from evals.cases import load_dataset, filter_cases
    from evals.metrics.deterministic import score_deterministic
    from store.schema import (
        CaseResult,
        EvalRun,
        ModelConfig,
        RunStatus,
    )
    _HARNESS_AVAILABLE = True
except ImportError:
    _HARNESS_AVAILABLE = False


async def run_quality_eval(
    engine: BaseEngine,
    dataset_name: str = "legal_qa",
    dataset_version: str = "v1",
    tags: list[str] | None = None,
    case_ids: list[str] | None = None,
    max_tokens: int = 512,
    concurrency: int = 3,
    system_prompt: str = "",
) -> "EvalRun":
    """
    Run the eval-harness test suite through a local engine.
    Returns an EvalRun (identical schema to Project 01 runs).
    """
    if not _HARNESS_AVAILABLE:
        raise ImportError(
            f"eval-harness not found at {_HARNESS_PATH}. "
            "Make sure eval-harness is installed or in the Python path."
        )

    if not engine.is_loaded:
        await engine.load()

    cases, manifest = load_dataset(dataset_name, dataset_version)
    cases = filter_cases(cases, tags=tags, ids=case_ids)

    # Fake ModelConfig for schema compatibility
    config = ModelConfig(
        model_id=engine.name,
        base_url="local",
        temperature=0.0,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )

    run = EvalRun(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_sha256=manifest.sha256,
        config=config,
        status=RunStatus.RUNNING,
        notes=f"local-inference engine={engine.name}",
    )

    sem = asyncio.Semaphore(concurrency)

    async def _eval_case(case) -> CaseResult:
        async with sem:
            req = GenerateRequest(
                prompt=case.input,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            try:
                resp = await engine.generate(req)
                det = score_deterministic(case, resp.text)
                return CaseResult(
                    case_id=case.id,
                    run_id=run.id,
                    model_output=resp.text,
                    latency_ms=resp.total_ms,
                    tokens_in=resp.tokens_prompt,
                    tokens_out=resp.tokens_generated,
                    cost_usd=0.0,  # local = no API cost
                    deterministic=det,
                )
            except Exception as exc:
                return CaseResult(
                    case_id=case.id,
                    run_id=run.id,
                    model_output="",
                    latency_ms=0.0,
                    error=str(exc),
                )

    results = await asyncio.gather(*[_eval_case(c) for c in cases])
    run.results = list(results)
    run.status = RunStatus.COMPLETED
    run.completed_at = datetime.now(timezone.utc)
    return run
