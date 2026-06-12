"""Metric comparison and run ranking against historical baselines."""

from __future__ import annotations

from llm_eval.models.types import EvalRun, MetricComparison, RunMetrics
from llm_eval.storage.local import LocalStorage

METRIC_DIRECTIONS: dict[str, str] = {
    "hallucination_rate": "lower_is_better",
    "answer_relevancy": "higher_is_better",
    "faithfulness": "higher_is_better",
    "context_recall": "higher_is_better",
    "accuracy": "higher_is_better",
    "precision": "higher_is_better",
    "recall": "higher_is_better",
    "f1_score": "higher_is_better",
    "prompt_injection_resistance": "higher_is_better",
    "jailbreak_resistance": "higher_is_better",
    "p50_latency_ms": "lower_is_better",
    "p95_latency_ms": "lower_is_better",
    "cost_per_query_usd": "lower_is_better",
}


def _trend(direction: str, delta: float, stable_threshold: float = 0.01) -> str:
    if abs(delta) <= stable_threshold:
        return "stable"
    if direction == "higher_is_better":
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def compare_metrics(current: RunMetrics, baseline: dict[str, float]) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        current_val = getattr(current, metric, 0.0)
        baseline_val = baseline.get(metric)
        if baseline_val is None:
            continue
        delta = current_val - baseline_val
        delta_pct = (delta / baseline_val * 100) if baseline_val else 0.0
        comparisons.append(
            MetricComparison(
                metric=metric,
                current=current_val,
                baseline=baseline_val,
                delta=delta,
                delta_pct=delta_pct,
                direction=direction,
                trend=_trend(direction, delta),
            )
        )
    return comparisons


def rank_run(run: EvalRun, historical: list[EvalRun]) -> tuple[int | None, int]:
    if not run.metrics:
        return None, 0

    candidates = [r for r in historical if r.metrics and r.run_id != run.run_id]
    if not candidates:
        return None, 0

    def composite_score(metrics: RunMetrics) -> float:
        return (
            metrics.accuracy * 0.2
            + metrics.answer_relevancy * 0.15
            + metrics.faithfulness * 0.15
            + metrics.f1_score * 0.15
            + metrics.prompt_injection_resistance * 0.1
            + metrics.jailbreak_resistance * 0.1
            + (1 - metrics.hallucination_rate) * 0.15
        )

    all_runs = candidates + [run]
    ranked = sorted(all_runs, key=lambda r: composite_score(r.metrics), reverse=True)
    rank = next(i + 1 for i, r in enumerate(ranked) if r.run_id == run.run_id)
    return rank, len(all_runs)


def get_baseline(storage: LocalStorage, days: int = 7) -> dict[str, float]:
    return storage.get_metrics_baseline(days)
