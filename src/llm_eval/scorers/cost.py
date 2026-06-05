"""Cost per query scorer."""

from __future__ import annotations

from llm_eval.models.types import QuestionResult
from llm_eval.scorers.base import BaseScorer


class CostScorer(BaseScorer):
    name = "cost"

    async def score_per_question(self, results: list[QuestionResult]) -> list[QuestionResult]:
        return results

    async def score(self, results: list[QuestionResult]) -> dict[str, float]:
        costs = [r.cost_usd for r in results]
        total = sum(costs)
        per_query = total / len(results) if results else 0.0
        return {"cost_per_query_usd": per_query, "total_cost_usd": total}
