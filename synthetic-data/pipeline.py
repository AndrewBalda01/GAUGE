"""
End-to-end pipeline orchestration.

Flow:
  1. Load plan (slots + quotas)
  2. Sample seeds (diversifiers)
  3. Generate raw examples (LLM calls, one per slot × seed)
  4. Validate (structural checks)
  5. Dedup (exact + semantic MinHash)
  6. Filter (heuristic + optional LLM judge)
  7. Diversity analysis
  8. Write output JSONL (eval-harness compatible)
  9. Generate report + charts

CLI: python pipeline.py run --output output/legal_qa_synth.v1.jsonl
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from generate.generator import generate_slot
from generate.seed import sample_seeds
from quality.dedup import full_dedup
from quality.diversity import analyse_diversity
from quality.filter import filter_heuristic, filter_with_judge
from quality.validate import validate_batch
from report.stats import (
    plot_distribution,
    plot_diversity_comparison,
    plot_funnel,
    render_markdown_report,
)
from spec.plan import LEGAL_QA_EXPANSION_PLAN, GenerationPlan
from spec.schema import SyntheticExample

load_dotenv()
console = Console()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    plan: GenerationPlan,
    model: str = "claude-haiku-4-5-20251001",
    concurrency: int = 5,
    use_judge: bool = False,
    judge_min_score: int = 3,
    near_dedup_threshold: float = 0.75,
    output_path: str = "output/legal_qa_synth.v1.jsonl",
    report_dir: str = "output/",
    dry_run: bool = False,
) -> list[SyntheticExample]:

    console.print(f"\n[bold]Synthetic Data Pipeline[/bold] — {plan.name}")
    console.print(plan.summary())
    console.print()

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 — Generate
    console.print("[bold]Step 1/5[/bold] Generating…")
    all_raw: list[SyntheticExample] = []
    seeds = sample_seeds(max(s.count for s in plan.slots) + 5)

    if dry_run:
        # Produce mock examples without API calls
        for slot in plan.slots:
            for i in range(slot.count):
                s = seeds[i % len(seeds)]
                idx = len(all_raw)
                label = f"{slot.topic}_{slot.subtopic or 'general'}_{slot.difficulty.value}_{i}"
                all_raw.append(SyntheticExample(
                    id=f"dry-{slot.topic[:3]}-{idx:04d}",
                    input=(
                        f"[DRY RUN #{idx}] In {slot.topic} law, what is the significance "
                        f"of {slot.subtopic or slot.topic} under the {label} doctrine?"
                    ),
                    expected=(
                        f"[DRY RUN] The {slot.subtopic or slot.topic} doctrine in {slot.topic} law "
                        f"(example {idx}) refers to a fundamental legal principle governing the "
                        f"rights and obligations of parties. It is applied by courts through "
                        f"examination of the facts and circumstances of each case. "
                        f"Understanding {label} is essential for legal practitioners."
                    ),
                    topic=slot.topic,
                    subtopic=slot.subtopic,
                    difficulty=slot.difficulty,
                    tags=[slot.difficulty.value],
                    generation_model="dry-run",
                    seed_persona=s.persona,
                ))
    else:
        for slot in plan.slots:
            slot_seeds = seeds[: slot.count + 2]
            examples = await generate_slot(slot, slot_seeds, model=model, concurrency=concurrency)
            all_raw.extend(examples)
            console.print(f"  {slot.topic}/{slot.subtopic} → {len(examples)}/{slot.count}")

    console.print(f"  [green]Generated {len(all_raw)} raw examples[/green]")

    # 2 — Validate
    console.print("[bold]Step 2/5[/bold] Validating…")
    after_validate, v_failed = validate_batch(all_raw)
    console.print(
        f"  passed={len(after_validate)}  "
        f"rejected={len(v_failed)}"
        + (f"  (e.g. '{v_failed[0][1]}')" if v_failed else "")
    )

    # 3 — Dedup
    console.print("[bold]Step 3/5[/bold] Deduplicating…")
    dedup_result = full_dedup(after_validate, near_threshold=near_dedup_threshold)
    console.print(f"  {dedup_result.summary()}")

    # Diversity BEFORE filter (for the before/after chart)
    div_before = analyse_diversity(dedup_result.kept)

    # 4 — Filter
    console.print("[bold]Step 4/5[/bold] Filtering…")
    if use_judge and not dry_run:
        filter_result = await filter_with_judge(
            dedup_result.kept, min_score=judge_min_score, judge_model=model
        )
    else:
        filter_result = filter_heuristic(dedup_result.kept)
        for ex in filter_result.kept:
            ex.stage_passed.append("filter")

    console.print(f"  {filter_result.summary()}")
    final = filter_result.kept

    # Diversity AFTER filter
    div_after = analyse_diversity(final)

    # 5 — Write output
    console.print("[bold]Step 5/5[/bold] Writing output…")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in final:
            f.write(json.dumps(ex.to_eval_harness_dict()) + "\n")
    console.print(f"  [green]Written {len(final)} examples → {out_path}[/green]")

    # Report
    report_md = render_markdown_report(
        raw_count=len(all_raw),
        after_validate=len(after_validate),
        after_dedup=len(dedup_result.kept),
        after_filter=len(final),
        final=final,
        diversity_before=div_before,
        diversity_after=div_after,
    )
    rp = out_dir / "pipeline_report.md"
    rp.write_text(report_md, encoding="utf-8")
    console.print(f"  report → {rp}")

    for fn in [plot_distribution, ]:
        p = fn(final, out_dir)
        if p:
            console.print(f"  chart  → {p}")

    p = plot_diversity_comparison(div_before, div_after, out_dir)
    if p:
        console.print(f"  chart  → {p}")

    p = plot_funnel(len(all_raw), len(after_validate), len(dedup_result.kept), len(final), out_dir)
    if p:
        console.print(f"  chart  → {p}")

    console.print(f"\n[bold green]Done.[/bold green]  {len(final)} examples ready for eval-harness.")
    console.print(f"  Diversity: {div_after.summary()}")
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Synthetic data generation pipeline."""


@cli.command()
@click.option("--model", default="claude-haiku-4-5-20251001", show_default=True)
@click.option("--concurrency", default=5, type=int, show_default=True)
@click.option("--use-judge/--no-judge", default=False, show_default=True)
@click.option("--judge-min-score", default=3, type=int, show_default=True)
@click.option("--near-dedup-threshold", default=0.75, type=float, show_default=True)
@click.option("--output", default="output/legal_qa_synth.v1.jsonl", show_default=True)
@click.option("--report-dir", default="output/", show_default=True)
@click.option("--dry-run", is_flag=True, help="Mock generation (no API calls)")
def run(model, concurrency, use_judge, judge_min_score,
        near_dedup_threshold, output, report_dir, dry_run) -> None:
    """Run the full synthetic data generation pipeline."""
    asyncio.run(run_pipeline(
        plan=LEGAL_QA_EXPANSION_PLAN,
        model=model,
        concurrency=concurrency,
        use_judge=use_judge,
        judge_min_score=judge_min_score,
        near_dedup_threshold=near_dedup_threshold,
        output_path=output,
        report_dir=report_dir,
        dry_run=dry_run,
    ))


@cli.command()
@click.option("--output", default="output/legal_qa_synth.v1.jsonl", show_default=True)
@click.option("--report-dir", default="output/", show_default=True)
def dry_run(output, report_dir) -> None:
    """Mock run (no API calls) — for testing the pipeline locally."""
    asyncio.run(run_pipeline(
        plan=LEGAL_QA_EXPANSION_PLAN,
        dry_run=True,
        output_path=output,
        report_dir=report_dir,
    ))


if __name__ == "__main__":
    cli()
