"""CLI entry point: eval run | eval compare | eval list"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_config(model: str, system_prompt: str, temperature: float, max_tokens: int):
    from store.schema import ModelConfig
    return ModelConfig(
        model_id=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """LLM Evaluation Harness CLI."""


@main.command()
@click.option("--dataset", default="legal_qa", show_default=True)
@click.option("--version", "ds_version", default="v1", show_default=True)
@click.option("--model", default="claude-haiku-4-5-20251001", show_default=True)
@click.option("--system-prompt", default="", help="System prompt for the model under test.")
@click.option("--temperature", default=0.0, type=float, show_default=True)
@click.option("--max-tokens", default=1024, type=int, show_default=True)
@click.option("--concurrency", default=5, type=int, show_default=True)
@click.option("--branch", default="", help="Git branch name (for regression tracking).")
@click.option("--git-ref", default="", help="Git commit SHA.")
@click.option("--tags", default=None, help="Comma-separated tag filter.")
@click.option("--use-judge/--no-judge", default=False, show_default=True)
@click.option("--judge-model", default="claude-sonnet-4-6", show_default=True)
@click.option("--output-dir", default="reports", show_default=True)
@click.option("--notes", default="")
def run(
    dataset, ds_version, model, system_prompt, temperature, max_tokens,
    concurrency, branch, git_ref, tags, use_judge, judge_model, output_dir, notes,
) -> None:
    """Run the eval suite and persist results."""
    from runners.runner import run_suite
    from evals.metrics.judge import judge_case
    from store.db import init_db, save_run
    from report.render import render_run_report, plot_run_summary
    from store.schema import ModelConfig

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    config = _get_config(model, system_prompt, temperature, max_tokens)

    async def _run():
        await init_db()
        console.print(f"[bold]Running suite[/bold] dataset={dataset}.{ds_version} model={model}")
        eval_run = await run_suite(
            config=config,
            dataset_name=dataset,
            dataset_version=ds_version,
            tags=tag_list,
            concurrency=concurrency,
            git_ref=git_ref,
            branch=branch,
            notes=notes,
        )

        # optional judge pass
        if use_judge:
            from evals.cases import load_dataset
            cases, _ = load_dataset(dataset, ds_version)
            case_map = {c.id: c for c in cases}
            judge_cfg = ModelConfig(model_id=judge_model, temperature=0.0)
            for result in eval_run.results:
                if result.error:
                    continue
                case = case_map.get(result.case_id)
                if case is None or case.expected is None:
                    continue
                result.judge = await judge_case(
                    question=case.input,
                    reference=case.expected,
                    model_output=result.model_output,
                    judge_config=judge_cfg,
                )

        await save_run(eval_run)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_md = render_run_report(eval_run)
        report_path = out / f"run_{eval_run.id[:8]}.md"
        report_path.write_text(report_md, encoding="utf-8")
        plot_run_summary(eval_run, out)

        console.print(f"[green]Run complete[/green] id={eval_run.id[:8]}")
        console.print(f"  cost=${eval_run.total_cost_usd:.4f}  "
                      f"ok={len([r for r in eval_run.results if not r.error])}/{len(eval_run.results)}")
        console.print(f"  report → {report_path}")
        return eval_run

    asyncio.run(_run())


@main.command()
@click.option("--baseline-branch", default="main", show_default=True)
@click.option("--candidate-branch", required=True)
@click.option("--dataset", default="legal_qa", show_default=True)
@click.option("--version", "ds_version", default="v1", show_default=True)
@click.option("--report-path", default="reports/regression.md", show_default=True)
@click.option("--threshold-judge", default=0.3, type=float, show_default=True)
@click.option("--threshold-pass-rate", default=0.05, type=float, show_default=True)
def compare(baseline_branch, candidate_branch, dataset, ds_version,
            report_path, threshold_judge, threshold_pass_rate) -> None:
    """Compare latest runs on two branches; exit 1 on regression."""
    from store.db import init_db, latest_run_on_branch, save_regression
    from report.compare import compare_runs
    from report.render import render_regression_report

    async def _compare():
        await init_db()
        baseline = await latest_run_on_branch(baseline_branch, dataset, ds_version)
        candidate = await latest_run_on_branch(candidate_branch, dataset, ds_version)

        if baseline is None:
            console.print(f"[yellow]No baseline run found on branch '{baseline_branch}' — skipping comparison.[/yellow]")
            return False
        if candidate is None:
            console.print(f"[red]No candidate run found on branch '{candidate_branch}'.[/red]")
            return False

        reg = compare_runs(baseline, candidate, threshold_judge, threshold_pass_rate)
        await save_regression(reg)

        report_md = render_regression_report(reg)
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(report_md, encoding="utf-8")

        if reg.regression_detected:
            console.print("[bold red]REGRESSION DETECTED[/bold red]")
            for d in reg.details:
                console.print(f"  {d['case_id']}: {d['baseline_score']} → {d['candidate_score']}")
        else:
            console.print("[bold green]No regression detected[/bold green]")

        console.print(f"  report → {report_path}")
        return reg.regression_detected

    regressed = asyncio.run(_compare())
    if regressed:
        sys.exit(1)


@main.command(name="list")
@click.option("--dataset", default=None)
@click.option("--branch", default=None)
@click.option("--limit", default=20, type=int, show_default=True)
def list_runs(dataset, branch, limit) -> None:
    """List recent eval runs."""
    from store.db import init_db, list_runs as _list

    async def _list_cmd():
        await init_db()
        rows = await _list(dataset=dataset, branch=branch, limit=limit)
        t = Table(title="Eval Runs")
        for col in ("id", "dataset", "model_id", "status", "branch", "created_at"):
            t.add_column(col)
        for r in rows:
            t.add_row(r["id"][:8], r["dataset"], r["model_id"],
                      r["status"], r.get("branch") or "—", r["created_at"][:16])
        console.print(t)

    asyncio.run(_list_cmd())


if __name__ == "__main__":
    main()
