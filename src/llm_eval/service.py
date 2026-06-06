"""Main eval pipeline orchestration — ties all layers together."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from llm_eval.config.loader import (
    AppSettings,
    EvalConfig,
    PipelineConfig,
    compute_config_hash,
    load_eval_config,
    load_pipeline_config,
    load_settings,
)
from llm_eval.execution.collector import ResponseCollector
from llm_eval.execution.runner import EvalRunner
from llm_eval.gates.evaluator import GateEvaluator
from llm_eval.gates.github import GitHubStatusPoster
from llm_eval.models.types import (
    EvalRun,
    GateResult,
    GateStatus,
    GoldenQuestion,
    QuestionResult,
    RunStatus,
    TriggerType,
)
from llm_eval.scorers.orchestrator import ScorerOrchestrator
from llm_eval.storage.local import LocalStorage
from llm_eval.storage.postgres import PostgresStorage
from llm_eval.storage.s3 import create_output_store

logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "local"


def _get_git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "local"


def _resolve_scope(scope: str, trigger_type: TriggerType, eval_config: EvalConfig) -> str:
    if trigger_type == TriggerType.PR and scope == "full":
        return eval_config.eval.scope_on_pr
    return scope


def _check_per_question_gates(
    scored_results: list[QuestionResult],
    questions: list[GoldenQuestion],
) -> list[GateResult]:
    q_map = {q.id: q for q in questions}
    sla_violations = 0
    sla_checked = 0
    relevancy_violations = 0
    relevancy_checked = 0

    for result in scored_results:
        question = q_map.get(result.question_id)
        if not question:
            continue
        if question.sla_latency_ms is not None:
            sla_checked += 1
            if result.latency_ms > question.sla_latency_ms:
                sla_violations += 1
        if question.min_relevancy_score is not None and not result.error:
            relevancy_checked += 1
            if result.scores.get("answer_relevancy", 0) < question.min_relevancy_score:
                relevancy_violations += 1

    extra: list[GateResult] = []
    if sla_checked and sla_violations:
        extra.append(
            GateResult(
                metric="sla_latency_violations",
                status=GateStatus.WARN,
                value=float(sla_violations),
                threshold=0.0,
                message=(
                    f"{sla_violations}/{sla_checked} questions exceeded per-question SLA latency"
                ),
            )
        )
    if relevancy_checked and relevancy_violations:
        extra.append(
            GateResult(
                metric="min_relevancy_violations",
                status=GateStatus.WARN,
                value=float(relevancy_violations),
                threshold=0.0,
                message=(
                    f"{relevancy_violations}/{relevancy_checked} questions "
                    "below min_relevancy_score"
                ),
            )
        )
    return extra


class EvalService:
    def __init__(
        self,
        settings: AppSettings | None = None,
        eval_config: EvalConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
        root: Path | None = None,
    ):
        from llm_eval.config.loader import _project_root

        if settings is None or eval_config is None or pipeline_config is None:
            s, e, p = load_settings(root)
            settings = settings or s
            eval_config = eval_config or e
            pipeline_config = pipeline_config or p

        self.settings = settings
        self.eval_config = eval_config
        self.pipeline_config = pipeline_config
        self.root = root or _project_root()
        self.local_storage = LocalStorage(settings.local_storage_path)
        self.output_store = create_output_store(settings, self.local_storage)
        self.postgres: PostgresStorage | None = None
        if settings.database_url:
            self.postgres = PostgresStorage(settings.database_url)

    def _should_run_agents(self, run: EvalRun) -> bool:
        if not self.eval_config.agents.enabled:
            return False

        has_block = any(g.status == GateStatus.BLOCK for g in run.gate_results)
        has_warn = any(g.status == GateStatus.WARN for g in run.gate_results)
        run_on = self.eval_config.agents.run_on

        if run_on == "always":
            return True
        if run_on == "warn":
            return has_block or has_warn
        if run_on == "failure":
            return run.status == RunStatus.FAILED or has_block
        return run.status == RunStatus.FAILED

    async def run_eval(
        self,
        scope: str = "full",
        trigger_type: TriggerType = TriggerType.MANUAL,
        git_sha: str | None = None,
        git_branch: str | None = None,
    ) -> EvalRun:
        scope = _resolve_scope(scope, trigger_type, self.eval_config)
        config_hash = compute_config_hash(
            self.eval_config.model_dump(),
            self.pipeline_config.model_dump(),
        )

        run = EvalRun(
            git_sha=git_sha or _get_git_sha(),
            git_branch=git_branch or _get_git_branch(),
            trigger_type=trigger_type,
            model_version=self.pipeline_config.model.name,
            config_hash=config_hash,
            scope=scope,
        )

        runner = EvalRunner(self.settings, self.eval_config, self.pipeline_config, self.root)
        collector = ResponseCollector(self.pipeline_config)
        scorers = ScorerOrchestrator(self.settings, self.eval_config)
        questions = runner.load_questions(scope)

        raw_results = await runner.run(scope)
        enriched = collector.enrich_with_costs(raw_results)
        scored_results, scorer_metrics = await scorers.run(enriched)
        base_metrics = collector.aggregate(scored_results)

        run.metrics = base_metrics.model_copy(
            update={
                "hallucination_rate": scorer_metrics.hallucination_rate,
                "answer_relevancy": scorer_metrics.answer_relevancy,
                "faithfulness": scorer_metrics.faithfulness,
                "context_recall": scorer_metrics.context_recall,
            }
        )

        cost_baseline = None
        if self.postgres:
            cost_baseline = self.postgres.get_cost_baseline()
        if cost_baseline is None:
            cost_baseline = self.local_storage.get_cost_baseline()

        gate_evaluator = GateEvaluator(self.eval_config, cost_baseline)
        run.gate_results = gate_evaluator.evaluate(run.metrics)
        run.gate_results.extend(_check_per_question_gates(scored_results, questions))
        state, _ = gate_evaluator.overall_state(run.gate_results)
        run.status = RunStatus.PASSED if state == "success" else RunStatus.FAILED
        run.finished_at = datetime.now(timezone.utc)

        for result in scored_results:
            self.output_store.save_raw_output(run.run_id, result)

        self.local_storage.save_run(run)
        if self.postgres:
            try:
                s3_keys = {
                    r.question_id: f"{run.run_id}/{r.question_id}/response.json"
                    for r in scored_results
                }
                self.postgres.save_run(run, scored_results, s3_keys)
            except Exception as exc:
                logger.warning("Failed to save run to Postgres: %s", exc)

        return run

    def post_gate(self, run: EvalRun, sha: str | None = None) -> bool:
        poster = GitHubStatusPoster(
            token=self.settings.github_token,
            repository=self.settings.github_repository,
            dashboard_url=self.settings.dashboard_url,
        )
        evaluator = GateEvaluator(self.eval_config)
        state, description = evaluator.overall_state(run.gate_results)
        return poster.post_status(sha or run.git_sha, state, description, run.run_id)

    async def run_with_agents(
        self,
        scope: str = "full",
        pr_number: int | None = None,
        diff_path: str | None = None,
    ) -> EvalRun:
        run = await self.run_eval(scope=scope, trigger_type=TriggerType.PR)

        if self._should_run_agents(run):
            try:
                from llm_eval.agents.crew import run_failure_analysis

                report_path = str(self.local_storage.runs_path / f"{run.run_id}.json")
                comment = await run_failure_analysis(
                    report_path=report_path,
                    diff_path=diff_path,
                    model=self.eval_config.agents.model,
                )
                if pr_number and self.settings.github_token:
                    poster = GitHubStatusPoster(
                        self.settings.github_token,
                        self.settings.github_repository,
                    )
                    poster.post_pr_comment(pr_number, comment)
            except ImportError:
                logger.info("CrewAI not installed; skipping agent analysis")
            except Exception as exc:
                logger.warning("Agent failure analysis failed: %s", exc)

        self.post_gate(run)
        return run
