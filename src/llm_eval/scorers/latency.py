"""Latency percentile scorer."""

from __future__ import annotations

import numpy as np

from llm_eval.models.types import QuestionResult
from llm_eval.scorers.base import BaseScorer


class LatencyScorer(BaseScorer):
    name = "latency"

    async def score_per_question(self, results: list[QuestionResult]) -> list[QuestionResult]:
        return results

    async def score(self, results: list[QuestionResult]) -> dict[str, float]:
        latencies = [r.latency_ms for r in results if not r.error]
        if not latencies:
            return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0}
        return {
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }
