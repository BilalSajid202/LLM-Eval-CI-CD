"""LLM-as-judge hallucination scorer."""

from __future__ import annotations

import json
import os

from llm_eval.models.types import QuestionResult
from llm_eval.scorers.base import BaseScorer

JUDGE_PROMPT = """You are a hallucination evaluator. Given a question, an answer, and
retrieved context, determine if the answer makes any claims NOT supported by the context.

Return JSON only: {{"verdict": "faithful" | "hallucinated", "reason": "..."}}

Question: {question}
Answer: {answer}
Context: {context}
"""


class HallucinationScorer(BaseScorer):
    name = "hallucination_rate"

    def __init__(self, model: str = "claude-haiku-4-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def _judge(self, question: str, answer: str, contexts: list[str]) -> str:
        if not answer.strip():
            return "hallucinated"

        context = "\n".join(contexts) if contexts else "(no context retrieved)"
        prompt = JUDGE_PROMPT.format(question=question, answer=answer, context=context)

        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else "{}"
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return data.get("verdict", "hallucinated")
        except Exception:
            pass

        return self._heuristic_judge(answer, contexts)

    def _heuristic_judge(self, answer: str, contexts: list[str]) -> str:
        """Fallback when API unavailable — keyword overlap heuristic."""
        if "don't have enough information" in answer.lower():
            return "faithful"
        if not contexts:
            return "hallucinated" if len(answer.split()) > 5 else "faithful"
        context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        return "faithful" if overlap > 0.3 else "hallucinated"

    async def score_per_question(self, results: list[QuestionResult]) -> list[QuestionResult]:
        import asyncio

        async def score_one(r: QuestionResult) -> QuestionResult:
            if r.error:
                return r.model_copy(
                    update={"scores": {**r.scores, "hallucination_verdict": 1.0}}
                )
            verdict = await self._judge(r.question, r.answer, r.contexts)
            value = 1.0 if verdict == "hallucinated" else 0.0
            return r.model_copy(
                update={
                    "scores": {**r.scores, "hallucination_verdict": value},
                    "metadata": {**r.metadata, "hallucination_verdict": verdict},
                }
            )

        return list(await asyncio.gather(*[score_one(r) for r in results]))

    async def score(self, results: list[QuestionResult]) -> dict[str, float]:
        scored = await self.score_per_question(results)
        valid = [r for r in scored if not r.error]
        if not valid:
            return {self.name: 0.0}
        rate = sum(r.scores.get("hallucination_verdict", 0) for r in valid) / len(valid)
        return {self.name: rate}
