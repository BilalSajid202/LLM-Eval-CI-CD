"""Embedding-based answer relevancy scorer."""

from __future__ import annotations

from llm_eval.models.types import QuestionResult
from llm_eval.scorers.base import BaseScorer


class RelevancyScorer(BaseScorer):
    name = "answer_relevancy"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                self._model = False
        return self._model

    def _cosine_similarity(self, a: str, b: str) -> float:
        model = self._get_model()
        if model is False:
            return self._token_overlap(a, b)
        from sentence_transformers import util

        emb_a = model.encode(a, convert_to_tensor=True)
        emb_b = model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b))

    def _token_overlap(self, question: str, answer: str) -> float:
        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())
        if not q_tokens or not a_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / len(q_tokens | a_tokens)

    async def score_per_question(self, results: list[QuestionResult]) -> list[QuestionResult]:
        scored = []
        for r in results:
            if r.error or not r.answer:
                scored.append(r.model_copy(update={"scores": {**r.scores, self.name: 0.0}}))
                continue
            sim = self._cosine_similarity(r.question, r.answer)
            scored.append(r.model_copy(update={"scores": {**r.scores, self.name: sim}}))
        return scored

    async def score(self, results: list[QuestionResult]) -> dict[str, float]:
        if not any(self.name in r.scores for r in results):
            results = await self.score_per_question(results)
        valid = [r.scores[self.name] for r in results if not r.error and r.answer and self.name in r.scores]
        mean = sum(valid) / len(valid) if valid else 0.0
        return {self.name: mean}
