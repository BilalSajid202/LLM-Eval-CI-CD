"""Modal serverless workers for parallel eval execution."""

from __future__ import annotations

import modal

from llm_eval.config.loader import EvalConfig, PipelineConfig
from llm_eval.models.types import QuestionResult

app = modal.App("llm-eval")


@app.function(
    secrets=[modal.Secret.from_name("llm-secrets")],
    timeout=300,
    max_containers=20,
)
async def eval_worker(question_batch: list[dict]) -> list[dict]:
    import asyncio

    from llm_eval.models.types import GoldenQuestion
    from llm_eval.pipeline.rag_pipeline import create_pipeline

    pipeline_config = PipelineConfig()
    pipeline = create_pipeline(pipeline_config)
    results = []

    for raw in question_batch:
        q = GoldenQuestion.model_validate(raw)
        start = __import__("time").perf_counter()
        try:
            output = await pipeline.run(q.question)
            latency_ms = (__import__("time").perf_counter() - start) * 1000
            results.append(
                QuestionResult(
                    question_id=q.id,
                    question=q.question,
                    category=q.category.value,
                    expected_answer=q.expected_answer,
                    answer=output.answer,
                    sources=output.sources,
                    contexts=output.contexts,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    latency_ms=latency_ms,
                ).model_dump()
            )
        except Exception as exc:
            latency_ms = (__import__("time").perf_counter() - start) * 1000
            results.append(
                QuestionResult(
                    question_id=q.id,
                    question=q.question,
                    category=q.category.value,
                    expected_answer=q.expected_answer,
                    answer="",
                    latency_ms=latency_ms,
                    error=str(exc),
                ).model_dump()
            )

    return results


async def run_modal_eval(
    scope: str,
    eval_config: EvalConfig,
    pipeline_config: PipelineConfig,
) -> list[QuestionResult]:
    from llm_eval.config.loader import _project_root
    from llm_eval.dataset.loader import filter_by_scope, load_golden_dataset

    root = _project_root()
    questions = filter_by_scope(
        load_golden_dataset(root / eval_config.eval.dataset_path),
        scope,
    )
    batch_size = eval_config.eval.batch_size
    batches = [
        [q.model_dump() for q in questions[i : i + batch_size]]
        for i in range(0, len(questions), batch_size)
    ]

    with app.run():
        all_results: list[dict] = []
        for batch in batches:
            batch_results = eval_worker.remote(batch)
            all_results.extend(batch_results)

    return [QuestionResult.model_validate(r) for r in all_results]
