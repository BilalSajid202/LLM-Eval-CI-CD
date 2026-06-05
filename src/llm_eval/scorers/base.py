"""Base scorer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_eval.models.types import QuestionResult


class BaseScorer(ABC):
    name: str

    @abstractmethod
    async def score(self, results: list[QuestionResult]) -> dict[str, float]:
        """Return aggregate metric name -> value."""

    @abstractmethod
    async def score_per_question(
        self, results: list[QuestionResult]
    ) -> list[QuestionResult]:
        """Annotate each result with per-question scores."""
