from llm_eval.config.loader import load_eval_config
from llm_eval.gates.evaluator import GateEvaluator
from llm_eval.models.types import RunMetrics


def test_gates_pass(project_root):
    config = load_eval_config(project_root)
    metrics = RunMetrics(
        hallucination_rate=0.02,
        answer_relevancy=0.85,
        faithfulness=0.80,
        accuracy=0.80,
        prompt_injection_resistance=1.0,
        jailbreak_resistance=1.0,
        p95_latency_ms=2500,
        cost_per_query_usd=0.001,
    )
    results = GateEvaluator(config, cost_baseline=0.001).evaluate(metrics)
    assert all(r.status.value != "block" for r in results)


def test_gates_block_hallucination(project_root):
    config = load_eval_config(project_root)
    metrics = RunMetrics(hallucination_rate=0.10, p95_latency_ms=1000)
    results = GateEvaluator(config).evaluate(metrics)
    hall = next(r for r in results if r.metric == "hallucination_rate")
    assert hall.status.value == "block"


def test_gates_warn_latency(project_root):
    config = load_eval_config(project_root)
    metrics = RunMetrics(
        hallucination_rate=0.01,
        answer_relevancy=0.85,
        faithfulness=0.80,
        accuracy=0.80,
        prompt_injection_resistance=1.0,
        jailbreak_resistance=1.0,
        p95_latency_ms=3500,
    )
    results = GateEvaluator(config).evaluate(metrics)
    lat = next(r for r in results if r.metric == "p95_latency_ms")
    assert lat.status.value == "warn"
    assert all(r.status.value != "block" for r in results)


def test_cost_regression_block(project_root):
    config = load_eval_config(project_root)
    metrics = RunMetrics(
        hallucination_rate=0.01,
        p95_latency_ms=1000,
        faithfulness=0.9,
        answer_relevancy=0.9,
        accuracy=0.85,
        prompt_injection_resistance=1.0,
        jailbreak_resistance=1.0,
        cost_per_query_usd=0.003,
    )
    results = GateEvaluator(config, cost_baseline=0.001).evaluate(metrics)
    cost = next((r for r in results if r.metric == "cost_regression_pct"), None)
    assert cost is not None
    assert cost.status.value == "block"
