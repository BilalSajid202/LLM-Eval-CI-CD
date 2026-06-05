"""Runs all scorers in parallel and merges results."""

from __future__ import annotations

import asyncio

from llm_eval.config.loader import AppSettings, EvalConfig
from llm_eval.models.types import QuestionResult, RunMetrics
from llm_eval.scorers.cost import CostScorer
from llm_eval.scorers.hallucination import HallucinationScorer
from llm_eval.scorers.latency import LatencyScorer
from llm_eval.scorers.ragas_scorer import RagasScorer
from llm_eval.scorers.relevancy import RelevancyScorer


class ScorerOrchestrator:
    def __init__(self, settings: AppSettings, eval_config: EvalConfig):
        self.settings = settings
        self.eval_config = eval_config
        self.scorers = self._build_scorers()

    def _build_scorers(self) -> list:
        cfg = self.eval_config.scorers
        scorers = []
        if cfg.hallucination.enabled:
            scorers.append(
                HallucinationScorer(
                    model=cfg.hallucination.model or "claude-haiku-4-5",
                    api_key=self.settings.anthropic_api_key,
                )
            )
        if cfg.relevancy.enabled:
            scorers.append(
                RelevancyScorer(model_name=cfg.relevancy.model_name or "all-MiniLM-L6-v2")
            )
        if cfg.ragas.enabled:
            scorers.append(
                RagasScorer(judge_model=cfg.ragas.judge_model or "claude-haiku-4-5")
            )
        scorers.extend([LatencyScorer(), CostScorer()])
        return scorers

    async def run(self, results: list[QuestionResult]) -> tuple[list[QuestionResult], RunMetrics]:
        per_question_scorers = [
            s for s in self.scorers if hasattr(s, "score_per_question")
        ]

        current = results
        for scorer in per_question_scorers:
            if scorer.name in ("latency", "cost"):
                continue
            current = await scorer.score_per_question(current)

        aggregate_tasks = [s.score(current) for s in self.scorers]
        aggregate_results = await asyncio.gather(*aggregate_tasks)

        merged: dict[str, float] = {}
        for agg in aggregate_results:
            merged.update(agg)

        metrics = RunMetrics(
            hallucination_rate=merged.get("hallucination_rate", 0.0),
            answer_relevancy=merged.get("answer_relevancy", 0.0),
            faithfulness=merged.get("faithfulness", 0.0),
            context_recall=merged.get("context_recall", 0.0),
            p50_latency_ms=merged.get("p50_latency_ms", 0.0),
            p95_latency_ms=merged.get("p95_latency_ms", 0.0),
            cost_per_query_usd=merged.get("cost_per_query_usd", 0.0),
            total_cost_usd=merged.get("total_cost_usd", 0.0),
            error_count=sum(1 for r in current if r.error),
        )
        return current, metrics
