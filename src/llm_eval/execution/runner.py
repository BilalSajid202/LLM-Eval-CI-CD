"""Parallel eval execution — local asyncio or Modal workers."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from llm_eval.config.loader import AppSettings, EvalConfig, PipelineConfig
from llm_eval.dataset.loader import filter_by_scope, load_golden_dataset
from llm_eval.models.types import GoldenQuestion, QuestionResult
from llm_eval.pipeline.rag_pipeline import create_pipeline


class EvalRunner:
    def __init__(
        self,
        settings: AppSettings,
        eval_config: EvalConfig,
        pipeline_config: PipelineConfig,
        root: Path | None = None,
    ):
        from llm_eval.config.loader import _project_root

        self.settings = settings
        self.eval_config = eval_config
        self.pipeline_config = pipeline_config
        self.root = root or _project_root()
        self.pipeline = create_pipeline(pipeline_config, api_key=settings.anthropic_api_key)

    def load_questions(self, scope: str) -> list[GoldenQuestion]:
        dataset_path = self.root / self.eval_config.eval.dataset_path
        questions = load_golden_dataset(dataset_path)
        return filter_by_scope(questions, scope)

    async def run_question(self, question: GoldenQuestion) -> QuestionResult:
        timeout = self.eval_config.eval.timeout_per_question_s
        start = time.perf_counter()
        try:
            output = await asyncio.wait_for(
                self.pipeline.run(question.question),
                timeout=timeout,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return QuestionResult(
                question_id=question.id,
                question=question.question,
                category=question.category.value,
                expected_answer=question.expected_answer,
                answer=output.answer,
                sources=output.sources,
                contexts=output.contexts,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return QuestionResult(
                question_id=question.id,
                question=question.question,
                category=question.category.value,
                expected_answer=question.expected_answer,
                answer="",
                latency_ms=latency_ms,
                error=str(exc),
            )

    async def run_local(self, scope: str = "full") -> list[QuestionResult]:
        questions = self.load_questions(scope)
        workers = self.eval_config.eval.parallel_workers
        semaphore = asyncio.Semaphore(workers)

        async def run_with_limit(question: GoldenQuestion) -> QuestionResult:
            async with semaphore:
                return await self.run_question(question)

        return list(await asyncio.gather(*[run_with_limit(q) for q in questions]))

    async def run(self, scope: str = "full") -> list[QuestionResult]:
        if self.settings.eval_mode == "modal":
            return await self._run_modal(scope)
        return await self.run_local(scope)

    async def _run_modal(self, scope: str) -> list[QuestionResult]:
        try:
            from llm_eval.execution.modal_app import run_modal_eval
        except ImportError as exc:
            raise RuntimeError(
                "Modal mode requires: pip install 'llm-eval[modal]'"
            ) from exc
        return await run_modal_eval(scope, self.eval_config, self.pipeline_config)
