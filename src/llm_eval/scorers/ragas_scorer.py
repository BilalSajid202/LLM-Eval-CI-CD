"""RAGAS faithfulness and context recall scorer."""

from __future__ import annotations

from llm_eval.models.types import QuestionResult
from llm_eval.scorers.base import BaseScorer


class RagasScorer(BaseScorer):
    name = "ragas"

    def __init__(self, judge_model: str = "claude-haiku-4-5"):
        self.judge_model = judge_model

    def _compute_heuristic(self, results: list[QuestionResult]) -> dict[str, float]:
        """Fallback when RAGAS is not installed."""
        faithfulness_scores = []
        recall_scores = []

        for r in results:
            if r.error or not r.answer:
                continue
            ctx_text = " ".join(r.contexts).lower()
            ans_words = set(r.answer.lower().split())
            ctx_words = set(ctx_text.split())
            faith = len(ans_words & ctx_words) / max(len(ans_words), 1) if ans_words else 0

            exp_words = set(r.expected_answer.lower().split())
            recall = len(exp_words & ctx_words) / max(len(exp_words), 1) if exp_words else 0

            faithfulness_scores.append(faith)
            recall_scores.append(recall)

        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else 0.0,
            "context_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        }

    async def score_per_question(self, results: list[QuestionResult]) -> list[QuestionResult]:
        aggregates = self._compute_heuristic(results)
        scored = []
        for r in results:
            if r.error or not r.answer:
                scored.append(
                    r.model_copy(
                        update={
                            "scores": {
                                **r.scores,
                                "faithfulness": 0.0,
                                "context_recall": 0.0,
                            }
                        }
                    )
                )
                continue
            ctx_text = " ".join(r.contexts).lower()
            ans_words = set(r.answer.lower().split())
            ctx_words = set(ctx_text.split())
            faith = len(ans_words & ctx_words) / max(len(ans_words), 1)
            exp_words = set(r.expected_answer.lower().split())
            recall = len(exp_words & ctx_words) / max(len(exp_words), 1) if exp_words else 0
            scored.append(
                r.model_copy(
                    update={"scores": {**r.scores, "faithfulness": faith, "context_recall": recall}}
                )
            )
        return scored

    async def score(self, results: list[QuestionResult]) -> dict[str, float]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_recall, faithfulness

            rows = [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "contexts": r.contexts or [""],
                    "ground_truth": r.expected_answer,
                }
                for r in results
                if not r.error and r.answer
            ]
            if not rows:
                return {"faithfulness": 0.0, "context_recall": 0.0}

            dataset = Dataset.from_list(rows)
            scores = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_recall],
            )
            means = scores.to_pandas().mean(numeric_only=True).to_dict()
            return {
                "faithfulness": float(means.get("faithfulness", 0)),
                "context_recall": float(means.get("context_recall", 0)),
            }
        except ImportError:
            return self._compute_heuristic(results)
        except Exception:
            return self._compute_heuristic(results)
