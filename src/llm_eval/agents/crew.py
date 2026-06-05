"""CrewAI failure analysis — runs after eval failures."""

from __future__ import annotations

import json
from pathlib import Path


async def run_failure_analysis(
    report_path: str,
    diff_path: str | None = None,
    model: str = "claude-sonnet-4-5",
) -> str:
    """Analyze eval failures and return markdown for PR comment."""
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError:
        return _fallback_analysis(report_path, diff_path)

    report = Path(report_path).read_text(encoding="utf-8")

    eval_analyst = Agent(
        role="LLM Evaluation Analyst",
        goal="Identify which questions failed and classify failure modes",
        backstory="Expert at reading eval metric reports and spotting patterns",
        llm=model,
        verbose=False,
    )

    root_cause_agent = Agent(
        role="Root Cause Analyst",
        goal="Map each failure to a specific change in the PR diff",
        backstory="Expert at prompt engineering and LLM behavior analysis",
        llm=model,
        verbose=False,
    )

    fix_agent = Agent(
        role="Prompt Engineer",
        goal="Generate concrete, testable fixes for each identified root cause",
        backstory="Senior prompt engineer with deep knowledge of RAG systems",
        llm=model,
        verbose=False,
    )

    analyze_task = Task(
        description=f"""
        Read this eval report JSON:
        {report[:8000]}

        List all questions with hallucination_verdict=hallucinated or failed gates.
        Group them by failure pattern.
        """,
        expected_output="Structured list of failures with pattern classification",
        agent=eval_analyst,
    )

    diff_content = ""
    if diff_path and Path(diff_path).exists():
        diff_content = Path(diff_path).read_text(encoding="utf-8")[:4000]

    root_cause_task = Task(
        description=f"""
        Given the failure analysis and this PR diff:
        {diff_content or '(diff not available)'}

        Identify which specific change most likely caused each failure group.
        """,
        expected_output="Root cause map: failure group -> PR change",
        agent=root_cause_agent,
        context=[analyze_task],
    )

    fix_task = Task(
        description="""
        For each root cause, suggest a specific, testable fix.
        Format: (a) problem, (b) proposed change, (c) which questions it should fix.
        """,
        expected_output="Fix suggestions ready to post as PR comment",
        agent=fix_agent,
        context=[root_cause_task],
    )

    crew = Crew(
        agents=[eval_analyst, root_cause_agent, fix_agent],
        tasks=[analyze_task, root_cause_task, fix_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return f"## EvalBot Analysis\n\n{result}"


def _fallback_analysis(report_path: str, diff_path: str | None) -> str:
    """Rule-based analysis when CrewAI is not installed."""
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    gates = data.get("gate_results", [])
    failed = [g for g in gates if g.get("status") in ("block", "warn")]

    lines = ["## EvalBot Analysis (fallback mode)", ""]
    if failed:
        lines.append(f"**{len(failed)} gate(s) triggered:**")
        for g in failed:
            lines.append(f"- **{g['metric']}**: {g['message']}")
    else:
        lines.append("No gate failures detected in report.")

    lines.append("")
    lines.append(
        "_Install `llm-eval[agents]` for full CrewAI-powered root cause analysis._"
    )
    return "\n".join(lines)
