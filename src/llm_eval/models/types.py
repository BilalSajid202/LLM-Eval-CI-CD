"""Shared domain types for the eval pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TriggerType(str, Enum):
    PR = "pr"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    WEBHOOK = "webhook"


class RunStatus(str, Enum):
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class GateStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class QuestionCategory(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    EDGE_CASE = "edge_case"
    ADVERSARIAL = "adversarial"


class GoldenQuestion(BaseModel):
    id: str
    category: QuestionCategory
    question: str
    expected_answer: str
    expected_sources: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    sla_latency_ms: int | None = None
    min_relevancy_score: float | None = None


class PipelineOutput(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class QuestionResult(BaseModel):
    question_id: str
    question: str
    category: str
    expected_answer: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    name: str
    value: float
    p50: float | None = None
    p95: float | None = None


class RunMetrics(BaseModel):
    hallucination_rate: float = 0.0
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    context_recall: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    prompt_injection_resistance: float = 1.0
    jailbreak_resistance: float = 1.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    cost_per_query_usd: float = 0.0
    total_cost_usd: float = 0.0
    error_count: int = 0

    def to_metric_list(self) -> list[MetricResult]:
        return [
            MetricResult(name="hallucination_rate", value=self.hallucination_rate),
            MetricResult(name="answer_relevancy", value=self.answer_relevancy),
            MetricResult(name="faithfulness", value=self.faithfulness),
            MetricResult(name="context_recall", value=self.context_recall),
            MetricResult(name="accuracy", value=self.accuracy),
            MetricResult(name="precision", value=self.precision),
            MetricResult(name="recall", value=self.recall),
            MetricResult(name="f1_score", value=self.f1_score),
            MetricResult(
                name="prompt_injection_resistance", value=self.prompt_injection_resistance
            ),
            MetricResult(name="jailbreak_resistance", value=self.jailbreak_resistance),
            MetricResult(
                name="p50_latency_ms", value=self.p50_latency_ms, p50=self.p50_latency_ms
            ),
            MetricResult(
                name="p95_latency_ms", value=self.p95_latency_ms, p95=self.p95_latency_ms
            ),
            MetricResult(name="cost_per_query_usd", value=self.cost_per_query_usd),
            MetricResult(name="total_cost_usd", value=self.total_cost_usd),
        ]


class GateResult(BaseModel):
    metric: str
    status: GateStatus
    value: float
    threshold: float
    message: str


class EvalRun(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    git_sha: str = "local"
    git_branch: str = "local"
    trigger_type: TriggerType = TriggerType.MANUAL
    model_version: str = ""
    config_hash: str = ""
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.RUNNING
    gate_results: list[GateResult] = Field(default_factory=list)
    metrics: RunMetrics | None = None
    scope: str = "full"


class MetricComparison(BaseModel):
    metric: str
    current: float
    baseline: float
    delta: float
    delta_pct: float
    direction: str  # "higher_is_better" | "lower_is_better"
    trend: str  # "improved" | "regressed" | "stable"


class CategoryBreakdown(BaseModel):
    category: str
    question_count: int
    accuracy: float
    answer_relevancy: float
    hallucination_rate: float
    pass_rate: float


class EvaluationReport(BaseModel):
    run_id: UUID
    generated_at: datetime = Field(default_factory=_utc_now)
    run: EvalRun
    question_count: int = 0
    comparisons: list[MetricComparison] = Field(default_factory=list)
    category_breakdown: list[CategoryBreakdown] = Field(default_factory=list)
    overall_rank: int | None = None
    total_runs_compared: int = 0
    insights: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    report_paths: dict[str, str] = Field(default_factory=dict)
