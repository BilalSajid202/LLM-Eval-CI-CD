"""Quality gate evaluation against configured thresholds."""

from __future__ import annotations

from llm_eval.config.loader import EvalConfig
from llm_eval.models.types import GateResult, GateStatus, RunMetrics


class GateEvaluator:
    def __init__(self, eval_config: EvalConfig, cost_baseline: float | None = None):
        self.config = eval_config
        self.cost_baseline = cost_baseline

    def evaluate(self, metrics: RunMetrics) -> list[GateResult]:
        gates = self.config.gates
        results: list[GateResult] = []

        results.append(
            self._check_lower_is_better(
                "hallucination_rate",
                metrics.hallucination_rate,
                gates.hallucination_rate.warn_threshold,
                gates.hallucination_rate.block_threshold,
            )
        )
        results.append(
            self._check_lower_is_better(
                "p95_latency_ms",
                metrics.p95_latency_ms,
                gates.p95_latency_ms.warn_threshold,
                gates.p95_latency_ms.block_threshold,
            )
        )
        results.append(
            self._check_higher_is_better(
                "faithfulness",
                metrics.faithfulness,
                gates.faithfulness.warn_threshold,
                gates.faithfulness.block_threshold,
            )
        )
        results.append(
            self._check_higher_is_better(
                "answer_relevancy",
                metrics.answer_relevancy,
                gates.answer_relevancy.warn_threshold,
                gates.answer_relevancy.block_threshold,
            )
        )

        if self.cost_baseline and self.cost_baseline > 0:
            regression_pct = (
                (metrics.cost_per_query_usd - self.cost_baseline) / self.cost_baseline
            ) * 100
            results.append(
                self._check_lower_is_better(
                    "cost_regression_pct",
                    regression_pct,
                    gates.cost_regression_pct.warn_threshold,
                    gates.cost_regression_pct.block_threshold,
                )
            )

        if gates.accuracy:
            results.append(
                self._check_higher_is_better(
                    "accuracy",
                    metrics.accuracy,
                    gates.accuracy.warn_threshold,
                    gates.accuracy.block_threshold,
                )
            )
        if gates.prompt_injection_resistance:
            results.append(
                self._check_higher_is_better(
                    "prompt_injection_resistance",
                    metrics.prompt_injection_resistance,
                    gates.prompt_injection_resistance.warn_threshold,
                    gates.prompt_injection_resistance.block_threshold,
                )
            )
        if gates.jailbreak_resistance:
            results.append(
                self._check_higher_is_better(
                    "jailbreak_resistance",
                    metrics.jailbreak_resistance,
                    gates.jailbreak_resistance.warn_threshold,
                    gates.jailbreak_resistance.block_threshold,
                )
            )

        return results

    def _check_lower_is_better(
        self, metric: str, value: float, warn: float, block: float
    ) -> GateResult:
        if value > block:
            status, msg = GateStatus.BLOCK, f"{metric}={value:.4f} exceeds block threshold {block}"
        elif value > warn:
            status, msg = GateStatus.WARN, f"{metric}={value:.4f} exceeds warn threshold {warn}"
        else:
            status, msg = GateStatus.PASS, f"{metric}={value:.4f} within thresholds"
        return GateResult(metric=metric, status=status, value=value, threshold=block, message=msg)

    def _check_higher_is_better(
        self, metric: str, value: float, warn: float, block: float
    ) -> GateResult:
        if value < block:
            status, msg = GateStatus.BLOCK, f"{metric}={value:.4f} below block threshold {block}"
        elif value < warn:
            status, msg = GateStatus.WARN, f"{metric}={value:.4f} below warn threshold {warn}"
        else:
            status, msg = GateStatus.PASS, f"{metric}={value:.4f} within thresholds"
        return GateResult(metric=metric, status=status, value=value, threshold=block, message=msg)

    def overall_state(self, gate_results: list[GateResult]) -> tuple[str, str]:
        if any(g.status == GateStatus.BLOCK for g in gate_results):
            blocked = [g.metric for g in gate_results if g.status == GateStatus.BLOCK]
            return "failure", f"Blocked: {', '.join(blocked)}"
        if any(g.status == GateStatus.WARN for g in gate_results):
            warned = [g.metric for g in gate_results if g.status == GateStatus.WARN]
            return "success", f"Passed with warnings: {', '.join(warned)}"
        return "success", "All quality gates passed"


def evaluate_gates(
    metrics: RunMetrics,
    eval_config: EvalConfig,
    cost_baseline: float | None = None,
) -> list[GateResult]:
    return GateEvaluator(eval_config, cost_baseline).evaluate(metrics)
