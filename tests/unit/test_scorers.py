import asyncio

import pytest

from llm_eval.models.types import QuestionResult
from llm_eval.scorers.cost import CostScorer
from llm_eval.scorers.hallucination import HallucinationScorer
from llm_eval.scorers.latency import LatencyScorer
from llm_eval.scorers.relevancy import RelevancyScorer


def _make_result(**kwargs) -> QuestionResult:
    defaults = {
        "question_id": "q_001",
        "question": "What is the refund policy?",
        "category": "factual",
        "expected_answer": "30 days",
        "answer": "Enterprise subscriptions can be refunded within 30 days.",
        "contexts": ["Enterprise subscriptions can be refunded within 30 days of purchase."],
        "latency_ms": 1200,
        "cost_usd": 0.001,
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }
    defaults.update(kwargs)
    return QuestionResult(**defaults)


def test_latency_scorer():
    results = [_make_result(latency_ms=100), _make_result(latency_ms=200)]
    scores = asyncio.run(LatencyScorer().score(results))
    assert scores["p50_latency_ms"] == 150
    assert scores["p95_latency_ms"] >= 190


def test_cost_scorer():
    results = [_make_result(cost_usd=0.002), _make_result(cost_usd=0.004)]
    scores = asyncio.run(CostScorer().score(results))
    assert scores["cost_per_query_usd"] == pytest.approx(0.003)
    assert scores["total_cost_usd"] == pytest.approx(0.006)


def test_relevancy_scorer_fallback():
    scorer = RelevancyScorer()
    results = [_make_result()]
    scored = asyncio.run(scorer.score_per_question(results))
    assert 0 <= scored[0].scores["answer_relevancy"] <= 1


def test_hallucination_heuristic_faithful():
    scorer = HallucinationScorer(api_key="")
    result = _make_result(
        answer="Enterprise subscriptions can be refunded within 30 days of purchase.",
        contexts=["Enterprise subscriptions can be refunded within 30 days of purchase."],
    )
    scored = asyncio.run(scorer.score_per_question([result]))
    assert scored[0].metadata.get("hallucination_verdict") == "faithful"


def test_hallucination_error_excluded_from_rate():
    scorer = HallucinationScorer(api_key="")
    errored = _make_result(error="timeout", answer="")
    scored = asyncio.run(scorer.score_per_question([errored]))
    assert "hallucination_verdict" not in scored[0].scores
    rate = asyncio.run(scorer.score(scored))
    assert rate["hallucination_rate"] == 0.0


def test_hallucination_refusal():
    scorer = HallucinationScorer(api_key="")
    result = _make_result(
        answer="I don't have enough information to answer that.",
        contexts=[],
    )
    scored = asyncio.run(scorer.score_per_question([result]))
    assert scored[0].metadata.get("hallucination_verdict") == "faithful"
