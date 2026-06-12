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
@click.option("--report/--no-report", default=True, help="Generate detailed Excel/PDF reports")
def run(
    scope: str,
    trigger: str,
    sha: str | None,
    branch: str | None,
    output: str | None,
    no_gate: bool,
    report: bool,
) -> None:
    """Execute the eval pipeline."""
    service = EvalService()
    trigger_type = TriggerType(trigger)
    if trigger_type == TriggerType.PR and scope == "full":
        scope = service.eval_config.eval.scope_on_pr

    console.print(f"[bold]Running eval[/bold] scope={scope} trigger={trigger}")

    service.eval_config.reporting.auto_export = report
    eval_run = asyncio.run(
        service.run_eval(scope=scope, trigger_type=trigger_type, git_sha=sha, git_branch=branch)
    )

    _print_summary(eval_run, service)

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
@click.option("--run-id", required=True, help="Eval run UUID")
@click.option(
    "--format",
    "fmt",
    default="both",
    type=click.Choice(["excel", "pdf", "both", "json"]),
    help="Report output format",
)
@click.option("--output-dir", default=None, type=click.Path(), help="Custom output directory")
def report(run_id: str, fmt: str, output_dir: str | None) -> None:
    """Generate detailed evaluation report for an existing run."""
    from pathlib import Path
    from uuid import UUID

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

    formats = {"excel": ["excel"], "pdf": ["pdf"], "both": ["excel", "pdf"], "json": []}[fmt]
    try:
        paths = service.generate_report(
            run_data,
            formats=formats or None,
        )
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("Install reporting deps: pip install -e '.[reporting]'")
        sys.exit(1)

    if output_dir and paths:
        import shutil

        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for key, src in paths.items():
            shutil.copy2(src, dest / Path(src).name)
        console.print(f"[green]Reports copied to {dest}[/green]")

    console.print("[bold]Report generated:[/bold]")
    for key, path in paths.items():
        console.print(f"  {key}: {path}")


@main.command()
@click.option("--run-id", required=True, help="Current run UUID to compare")
@click.option("--baseline", default="auto", help="Baseline run UUID or 'auto' for rolling avg")
@click.option("--days", default=7, help="Baseline window in days when using auto")
def compare(run_id: str, baseline: str, days: int) -> None:
    """Compare a run against historical baseline metrics."""
    from uuid import UUID

    from llm_eval.reporting.comparison import compare_metrics, get_baseline

    service = EvalService()
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        console.print(f"[red]Invalid run UUID: {run_id}[/red]")
        sys.exit(1)

    run_data = service.local_storage.load_run(run_uuid)
    if not run_data or not run_data.metrics:
        console.print(f"[red]Run not found or has no metrics: {run_id}[/red]")
        sys.exit(1)

    if baseline == "auto":
        baseline_metrics = get_baseline(service.local_storage, days)
        baseline_label = f"{days}-day rolling average"
    else:
        try:
            baseline_run = service.local_storage.load_run(UUID(baseline))
        except ValueError:
            console.print(f"[red]Invalid baseline UUID: {baseline}[/red]")
            sys.exit(1)
        if not baseline_run or not baseline_run.metrics:
            console.print(f"[red]Baseline run not found: {baseline}[/red]")
            sys.exit(1)
        baseline_metrics = baseline_run.metrics.model_dump()
        baseline_label = str(baseline)

    comparisons = compare_metrics(run_data.metrics, baseline_metrics)
    if not comparisons:
        console.print("[yellow]No baseline data available for comparison.[/yellow]")
        sys.exit(0)

    table = Table(title=f"Comparison vs {baseline_label}")
    table.add_column("Metric", style="cyan")
    table.add_column("Current", style="green")
    table.add_column("Baseline")
    table.add_column("Delta")
    table.add_column("Trend")

    for c in comparisons:
        trend_color = {"improved": "green", "regressed": "red", "stable": "yellow"}.get(
            c.trend, "white"
        )
        table.add_row(
            c.metric,
            f"{c.current:.4f}",
            f"{c.baseline:.4f}",
            f"{c.delta:+.4f} ({c.delta_pct:+.1f}%)",
            f"[{trend_color}]{c.trend.upper()}[/{trend_color}]",
        )
    console.print(table)


@main.command()
@click.option("--scope", default="full")
@click.option("--pr", default=None, type=int, help="PR number for agent comment")
def analyze(scope: str, pr: int | None) -> None:
    """Run eval with CrewAI failure analysis on gate failures."""
    service = EvalService()
    eval_run = asyncio.run(service.run_with_agents(scope=scope, pr_number=pr))
    _print_summary(eval_run, service)
    if eval_run.status.value == "failed":
        sys.exit(1)


def _print_summary(eval_run: EvalRun, service: EvalService | None = None) -> None:
    table = Table(title=f"Eval Run {eval_run.run_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Gate", style="dim")

    gate_map = {g.metric: g.status.value.upper() for g in eval_run.gate_results}

    if eval_run.metrics:
        m = eval_run.metrics
        rows = [
            ("Hallucination rate", f"{m.hallucination_rate:.2%}", "hallucination_rate"),
            ("Answer relevancy", f"{m.answer_relevancy:.3f}", "answer_relevancy"),
            ("Faithfulness", f"{m.faithfulness:.3f}", "faithfulness"),
            ("Context recall", f"{m.context_recall:.3f}", None),
            ("Accuracy", f"{m.accuracy:.3f}", "accuracy"),
            ("Precision", f"{m.precision:.3f}", None),
            ("Recall", f"{m.recall:.3f}", None),
            ("F1 score", f"{m.f1_score:.3f}", None),
            ("Injection resistance", f"{m.prompt_injection_resistance:.1%}", "prompt_injection_resistance"),
            ("Jailbreak resistance", f"{m.jailbreak_resistance:.1%}", "jailbreak_resistance"),
            ("p50 latency", f"{m.p50_latency_ms:.0f} ms", None),
            ("p95 latency", f"{m.p95_latency_ms:.0f} ms", "p95_latency_ms"),
            ("Cost per query", f"${m.cost_per_query_usd:.6f}", None),
            ("Errors", str(m.error_count), None),
        ]
        for label, value, gate_key in rows:
            gate_status = gate_map.get(gate_key, "") if gate_key else ""
            table.add_row(label, value, gate_status)

    table.add_row("Status", eval_run.status.value.upper(), "")
    console.print(table)

    if eval_run.gate_results:
        console.print("\n[bold]Gate Results[/bold]")
        for g in eval_run.gate_results:
            color = {"pass": "green", "warn": "yellow", "block": "red"}.get(g.status.value, "white")
            console.print(f"  [{color}]{g.status.value.upper()}[/{color}] {g.message}")

    if service:
        report_dir = service.local_storage.reports_path() / str(eval_run.run_id)
        excel = report_dir / "evaluation_report.xlsx"
        pdf = report_dir / "evaluation_report.pdf"
        if excel.exists() or pdf.exists():
            console.print("\n[bold]Detailed Reports[/bold]")
            if excel.exists():
                console.print(f"  Excel: {excel}")
            if pdf.exists():
                console.print(f"  PDF: {pdf}")
            console.print("  Generate manually: llm-eval report --run-id " + str(eval_run.run_id))


if __name__ == "__main__":
    main()
