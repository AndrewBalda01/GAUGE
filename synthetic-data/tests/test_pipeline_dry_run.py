"""Integration test: full pipeline in dry-run mode (no API calls)."""

import asyncio
import json
from pathlib import Path

import pytest
from pipeline import run_pipeline
from spec.plan import LEGAL_QA_EXPANSION_PLAN


def test_dry_run_produces_output(tmp_path):
    output = tmp_path / "synth.jsonl"
    report_dir = str(tmp_path)

    result = asyncio.run(run_pipeline(
        plan=LEGAL_QA_EXPANSION_PLAN,
        dry_run=True,
        output_path=str(output),
        report_dir=report_dir,
    ))

    assert len(result) > 0
    assert output.exists()

    # Every line must be valid JSON with required keys
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result)
    for line in lines:
        d = json.loads(line)
        assert "id" in d
        assert "input" in d
        assert "expected" in d
        assert "metrics" in d


def test_dry_run_report_written(tmp_path):
    asyncio.run(run_pipeline(
        plan=LEGAL_QA_EXPANSION_PLAN,
        dry_run=True,
        output_path=str(tmp_path / "out.jsonl"),
        report_dir=str(tmp_path),
    ))
    report = tmp_path / "pipeline_report.md"
    assert report.exists()
    content = report.read_text(encoding="utf-8")
    assert "Funnel" in content
    assert "Topic distribution" in content


def test_dry_run_all_stages_pass(tmp_path):
    result = asyncio.run(run_pipeline(
        plan=LEGAL_QA_EXPANSION_PLAN,
        dry_run=True,
        output_path=str(tmp_path / "out.jsonl"),
        report_dir=str(tmp_path),
    ))
    for ex in result:
        # Dry-run examples pass validate, dedup, filter
        assert "validate" in ex.stage_passed
        assert "filter" in ex.stage_passed


def test_dry_run_covers_all_topics(tmp_path):
    result = asyncio.run(run_pipeline(
        plan=LEGAL_QA_EXPANSION_PLAN,
        dry_run=True,
        output_path=str(tmp_path / "out.jsonl"),
        report_dir=str(tmp_path),
    ))
    expected_topics = {s.topic for s in LEGAL_QA_EXPANSION_PLAN.slots}
    produced_topics = {ex.topic for ex in result}
    assert produced_topics == expected_topics


def test_dry_run_output_compatible_with_plan_total(tmp_path):
    result = asyncio.run(run_pipeline(
        plan=LEGAL_QA_EXPANSION_PLAN,
        dry_run=True,
        output_path=str(tmp_path / "out.jsonl"),
        report_dir=str(tmp_path),
    ))
    # Dry-run generates plan.total examples; dedup may remove exact duplicates.
    # All survivors must pass the heuristic filter.
    assert len(result) > 0
    assert len(result) <= LEGAL_QA_EXPANSION_PLAN.total
