"""Accuracy, precision, recall, and F1 scoring against expected answers."""

from __future__ import annotations

from pathlib import PurePosixPath

from llm_eval.models.types import QuestionResult
from llm_eval.scorers.base import BaseScorer

ACCURACY_PASS_THRESHOLD = 0.7


def _token_set(text: str) -> set[str]:
    return {w.strip(".,!?;:\"'()[]") for w in text.lower().split() if w.strip(".,!?;:\"'()[]")}


def _normalize_source(path: str) -> str:
    return str(PurePosixPath(path.replace("\\", "/")))


class AccuracyScorer(BaseScorer):
    name = "accuracy"

    def _answer_recall(self, expected: str, answer: str) -> float:
        exp = _token_set(expected)
        ans = _token_set(answer)
        if not exp:
            return 1.0 if ans else 0.0
        return len(exp & ans) / len(exp)

    def _answer_precision(self, expected: str, answer: str) -> float:
        exp = _token_set(expected)
        ans = _token_set(answer)
        if not ans:
            return 0.0
        return len(exp & ans) / len(ans)

    def _source_precision(self, expected_sources: list[str], sources: list[str]) -> float | None:
        if not expected_sources:
            return None
        if not sources:
            return 0.0
        exp = {_normalize_source(s) for s in expected_sources}
        got = {_normalize_source(s) for s in sources}
        return len(exp & got) / len(got)

    def _source_recall(self, expected_sources: list[str], sources: list[str]) -> float | None:
        if not expected_sources:
            return None
        if not sources:
            return 0.0
        exp = {_normalize_source(s) for s in expected_sources}
        got = {_normalize_source(s) for s in sources}
        return len(exp & got) / len(exp)

    def _f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _score_one(self, result: QuestionResult) -> QuestionResult:
        if result.error or not result.answer:
            return result.model_copy(
                update={
                    "scores": {
                        **result.scores,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                    },
                    "metadata": {
                        **result.metadata,
                        "accuracy_pass": False,
                    },
                }
            )

        ans_recall = self._answer_recall(result.expected_answer, result.answer)
        ans_precision = self._answer_precision(result.expected_answer, result.answer)
        src_precision = self._source_precision(result.metadata.get("expected_sources", []), result.sources)
        src_recall = self._source_recall(result.metadata.get("expected_sources", []), result.sources)

        precision = src_precision if src_precision is not None else ans_precision
        recall = src_recall if src_recall is not None else ans_recall
        accuracy = (ans_recall + ans_precision) / 2
        f1 = self._f1(precision, recall)

        return result.model_copy(
            update={
                "scores": {
                    **result.scores,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                },
                "metadata": {
                    **result.metadata,
                    "accuracy_pass": accuracy >= ACCURACY_PASS_THRESHOLD,
                },
            }
        )

    async def score_per_question(self, results: list[QuestionResult]) -> list[QuestionResult]:
        return [self._score_one(r) for r in results]

    async def score(self, results: list[QuestionResult]) -> dict[str, float]:
        if not any("accuracy" in r.scores for r in results):
            results = await self.score_per_question(results)

        valid = [r for r in results if not r.error and r.answer and "accuracy" in r.scores]
        if not valid:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        return {
            "accuracy": sum(r.scores["accuracy"] for r in valid) / len(valid),
            "precision": sum(r.scores["precision"] for r in valid) / len(valid),
            "recall": sum(r.scores["recall"] for r in valid) / len(valid),
            "f1_score": sum(r.scores["f1_score"] for r in valid) / len(valid),
        }
