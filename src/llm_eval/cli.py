"""CLI entry point for the eval pipeline."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

import click
from rich.console import Console
from rich.table import Table

from llm_eval.config.loader import load_settings
from llm_eval.dataset.validator import validate_dataset_file
from llm_eval.models.types import EvalRun, TriggerType
from llm_eval.service import EvalService

console = Console()


@click.group()
@click.version_option(package_name="llm-eval")
def main() -> None:
    """LLM Eval CI/CD — automated evaluation with merge gates."""


@main.command()
@click.option("--scope", default="full", type=click.Choice(["full", "retrieval", "smoke"]))
@click.option("--trigger", default="manual", type=click.Choice(["pr", "scheduled", "manual"]))
@click.option("--sha", default=None, help="Git SHA for reproducibility")
@click.option("--branch", default=None, help="Git branch name")
@click.option("--output", default=None, type=click.Path(), help="Write run JSON to file")
@click.option("--no-gate", is_flag=True, help="Skip GitHub status posting")
def run(scope: str, trigger: str, sha: str | None, branch: str | None, output: str | None, no_gate: bool) -> None:
    """Execute the eval pipeline."""
    service = EvalService()
    trigger_type = TriggerType(trigger)
    if trigger_type == TriggerType.PR and scope == "full":
        scope = service.eval_config.eval.scope_on_pr

    console.print(f"[bold]Running eval[/bold] scope={scope} trigger={trigger}")

    eval_run = asyncio.run(
        service.run_eval(scope=scope, trigger_type=trigger_type, git_sha=sha, git_branch=branch)
    )

    _print_summary(eval_run)

    if output:
        Path(output).write_text(eval_run.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Run saved to {output}")

    if not no_gate:
        posted = service.post_gate(eval_run, sha=sha)
        if posted:
            console.print("[green]GitHub status posted[/green]")

    if eval_run.status.value == "failed":
        sys.exit(1)


@main.command()
@click.option("--run-id", required=True, help="Eval run UUID")
@click.option("--sha", required=True, help="Commit SHA to attach status to")
def gate(run_id: str, sha: str) -> None:
    """Post merge gate status for an existing run."""
    service = EvalService()
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        console.print(f"[red]Invalid run UUID: {run_id}[/red]")
        sys.exit(1)

    run_data = service.local_storage.load_run(run_uuid)
    if not run_data:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)

    posted = service.post_gate(run_data, sha=sha)
    if posted:
        console.print("[green]Gate status posted successfully[/green]")
    else:
        console.print("[yellow]Gate status not posted (check GITHUB_TOKEN/REPO)[/yellow]")


@main.command("validate-dataset")
@click.option("--path", default="data/golden_dataset/questions.yaml")
def validate_dataset(path: str) -> None:
    """Validate golden dataset schema and governance rules."""
    from llm_eval.config.loader import _project_root

    full_path = _project_root() / path
    try:
        validate_dataset_file(full_path)
        console.print(f"[green]Dataset valid:[/green] {full_path}")
    except Exception as exc:
        console.print(f"[red]Validation failed:[/red] {exc}")
        sys.exit(1)


@main.command()
@click.option("--scope", default="full")
@click.option("--pr", default=None, type=int, help="PR number for agent comment")
def analyze(scope: str, pr: int | None) -> None:
    """Run eval with CrewAI failure analysis on gate failures."""
    service = EvalService()
    eval_run = asyncio.run(service.run_with_agents(scope=scope, pr_number=pr))
    _print_summary(eval_run)
    if eval_run.status.value == "failed":
        sys.exit(1)


def _print_summary(eval_run: EvalRun) -> None:
    table = Table(title=f"Eval Run {eval_run.run_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    if eval_run.metrics:
        m = eval_run.metrics
        table.add_row("Hallucination rate", f"{m.hallucination_rate:.2%}")
        table.add_row("Answer relevancy", f"{m.answer_relevancy:.3f}")
        table.add_row("Faithfulness", f"{m.faithfulness:.3f}")
        table.add_row("Context recall", f"{m.context_recall:.3f}")
        table.add_row("p50 latency", f"{m.p50_latency_ms:.0f} ms")
        table.add_row("p95 latency", f"{m.p95_latency_ms:.0f} ms")
        table.add_row("Cost per query", f"${m.cost_per_query_usd:.6f}")
        table.add_row("Errors", str(m.error_count))

    table.add_row("Status", eval_run.status.value.upper())
    console.print(table)

    if eval_run.gate_results:
        console.print("\n[bold]Gate Results[/bold]")
        for g in eval_run.gate_results:
            color = {"pass": "green", "warn": "yellow", "block": "red"}.get(g.status.value, "white")
            console.print(f"  [{color}]{g.status.value.upper()}[/{color}] {g.message}")


if __name__ == "__main__":
    main()
