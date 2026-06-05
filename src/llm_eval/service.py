"""Main eval pipeline orchestration — ties all layers together."""

from __future__ import annotations

import subprocess
from datetime import datetime
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
from llm_eval.models.types import EvalRun, RunStatus, TriggerType
from llm_eval.scorers.orchestrator import ScorerOrchestrator
from llm_eval.storage.local import LocalStorage
from llm_eval.storage.postgres import PostgresStorage
from llm_eval.storage.s3 import create_output_store


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
        self.output_store = create_output_store(settings)
        self.local_storage = LocalStorage(settings.local_storage_path)
        self.postgres: PostgresStorage | None = None
        if settings.database_url:
            self.postgres = PostgresStorage(settings.database_url)

    async def run_eval(
        self,
        scope: str = "full",
        trigger_type: TriggerType = TriggerType.MANUAL,
        git_sha: str | None = None,
        git_branch: str | None = None,
    ) -> EvalRun:
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
        state, _ = gate_evaluator.overall_state(run.gate_results)
        run.status = RunStatus.PASSED if state == "success" else RunStatus.FAILED
        run.finished_at = datetime.utcnow()

        s3_keys: dict[str, str] = {}
        for result in scored_results:
            s3_keys[result.question_id] = self.output_store.save_raw_output(run.run_id, result)

        self.local_storage.save_run(run)
        if self.postgres:
            try:
                self.postgres.save_run(run, scored_results, s3_keys)
            except Exception:
                pass

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
        has_failure = run.status == RunStatus.FAILED or any(
            g.status.value in ("block", "warn") for g in run.gate_results
        )

        if has_failure and self.eval_config.agents.enabled:
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
                pass

        self.post_gate(run)
        return run
