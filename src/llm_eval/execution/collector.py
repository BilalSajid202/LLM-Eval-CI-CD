"""Aggregates execution results and computes run-level statistics."""

from __future__ import annotations

import numpy as np

from llm_eval.config.loader import PipelineConfig
from llm_eval.models.types import QuestionResult, RunMetrics


class ResponseCollector:
    def __init__(self, pipeline_config: PipelineConfig):
        self.pricing = pipeline_config.pricing

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1_000_000) * self.pricing.input_per_million
        output_cost = (completion_tokens / 1_000_000) * self.pricing.output_per_million
        return input_cost + output_cost

    def enrich_with_costs(self, results: list[QuestionResult]) -> list[QuestionResult]:
        enriched = []
        for r in results:
            cost = self.compute_cost(r.prompt_tokens, r.completion_tokens)
            enriched.append(r.model_copy(update={"cost_usd": cost}))
        return enriched

    def aggregate(self, results: list[QuestionResult]) -> RunMetrics:
        valid = [r for r in results if not r.error]
        latencies = [r.latency_ms for r in valid]
        costs = [r.cost_usd for r in results]

        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        total_cost = sum(costs)
        cost_per_query = total_cost / len(results) if results else 0.0

        return RunMetrics(
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            cost_per_query_usd=cost_per_query,
            total_cost_usd=total_cost,
            error_count=sum(1 for r in results if r.error),
        )
