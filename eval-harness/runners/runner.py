"""Async runner: executes a test suite against a ModelConfig."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone

import httpx

from evals.cases import load_dataset, filter_cases
from evals.metrics.deterministic import score_deterministic
from store.schema import (
    CaseResult,
    DeterministicScores,
    EvalRun,
    ModelConfig,
    RunStatus,
    TestCase,
)

# Cost table USD per 1M tokens (input, output) — extend as needed
_COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4-8":    (15.0, 75.0),
    "claude-sonnet-4-6":  (3.0,  15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    # local / unknown models default to 0
}


def _estimate_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    key = next((k for k in _COST_TABLE if model_id.startswith(k)), None)
    if key is None:
        return 0.0
    price_in, price_out = _COST_TABLE[key]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


async def _call_model(
    client: httpx.AsyncClient,
    config: ModelConfig,
    case: TestCase,
) -> CaseResult:
    """Single async call to the model (Anthropic Messages API format)."""
    api_key = os.environ.get(config.api_key_env, "")
    messages = [{"role": "user", "content": case.input}]
    payload: dict = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": messages,
    }
    if config.system_prompt:
        payload["system"] = config.system_prompt
    payload.update(config.extra_params)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    run_id = str(uuid.uuid4())  # placeholder, overwritten by caller
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{config.base_url}/messages",
            json=payload,
            headers=headers,
            timeout=120.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        content_blocks = data.get("content", [])
        model_output = "".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        )
        usage = data.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        cost = _estimate_cost(config.model_id, tokens_in, tokens_out)

        det = score_deterministic(case, model_output)
        return CaseResult(
            case_id=case.id,
            run_id=run_id,
            model_output=model_output,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            deterministic=det,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return CaseResult(
            case_id=case.id,
            run_id=run_id,
            model_output="",
            latency_ms=latency_ms,
            error=str(exc),
        )


async def run_suite(
    config: ModelConfig,
    dataset_name: str = "legal_qa",
    dataset_version: str = "v1",
    tags: list[str] | None = None,
    case_ids: list[str] | None = None,
    concurrency: int = 5,
    git_ref: str = "",
    branch: str = "",
    notes: str = "",
) -> EvalRun:
    cases, manifest = load_dataset(dataset_name, dataset_version)
    cases = filter_cases(cases, tags=tags, ids=case_ids)

    run = EvalRun(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_sha256=manifest.sha256,
        config=config,
        status=RunStatus.RUNNING,
        git_ref=git_ref,
        branch=branch,
        notes=notes,
    )

    sem = asyncio.Semaphore(concurrency)

    async def bounded_call(case: TestCase) -> CaseResult:
        async with sem:
            result = await _call_model(client, config, case)
            result.run_id = run.id
            return result

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[bounded_call(c) for c in cases])

    run.results = list(results)
    run.status = RunStatus.COMPLETED
    run.completed_at = datetime.now(timezone.utc)
    return run
