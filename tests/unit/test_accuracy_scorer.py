import asyncio

from llm_eval.models.types import QuestionResult
from llm_eval.scorers.accuracy_scorer import AccuracyScorer


def _make_result(**kwargs) -> QuestionResult:
    defaults = {
        "question_id": "q_001",
        "question": "What is the refund policy?",
        "category": "factual",
        "expected_answer": "Enterprise subscriptions can be refunded within 30 days.",
        "answer": "Enterprise subscriptions can be refunded within 30 days of purchase.",
        "sources": ["data/knowledge_base/docs/billing.md"],
        "metadata": {"expected_sources": ["data/knowledge_base/docs/billing.md"]},
    }
    defaults.update(kwargs)
    return QuestionResult(**defaults)


def test_high_accuracy_match():
    scorer = AccuracyScorer()
    scored = asyncio.run(scorer.score_per_question([_make_result()]))
    assert scored[0].scores["accuracy"] > 0.7
    assert scored[0].metadata["accuracy_pass"] is True


def test_low_accuracy_mismatch():
    scorer = AccuracyScorer()
    result = _make_result(answer="The weather is sunny today.")
    scored = asyncio.run(scorer.score_per_question([result]))
    assert scored[0].scores["accuracy"] < 0.5
    assert scored[0].metadata["accuracy_pass"] is False


def test_source_precision_recall():
    scorer = AccuracyScorer()
    result = _make_result(
        sources=["data/knowledge_base/docs/billing.md", "data/knowledge_base/docs/other.md"],
        metadata={"expected_sources": ["data/knowledge_base/docs/billing.md"]},
    )
    scored = asyncio.run(scorer.score_per_question([result]))
    assert scored[0].scores["precision"] == 0.5
    assert scored[0].scores["recall"] == 1.0


def test_f1_computed():
    scorer = AccuracyScorer()
    scored = asyncio.run(scorer.score_per_question([_make_result()]))
    assert 0 < scored[0].scores["f1_score"] <= 1.0
