import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from llm_eval.config.loader import load_eval_config
from llm_eval.models.types import EvalRun, GateResult, GateStatus, QuestionResult, RunMetrics, RunStatus
from llm_eval.reporting.comparison import compare_metrics
from llm_eval.reporting.report_builder import ReportBuilder
from llm_eval.storage.local import LocalStorage


def _make_result(**kwargs) -> QuestionResult:
    defaults = {
        "question_id": "q_001",
        "question": "What is the refund policy?",
        "category": "factual",
        "expected_answer": "30 days",
        "answer": "Refunds within 30 days.",
        "scores": {
            "accuracy": 0.85,
            "answer_relevancy": 0.9,
            "precision": 0.8,
            "recall": 0.75,
            "f1_score": 0.77,
            "faithfulness": 0.8,
            "context_recall": 0.7,
            "hallucination_verdict": 0.0,
        },
        "metadata": {"accuracy_pass": True, "tags": []},
    }
    defaults.update(kwargs)
    return QuestionResult(**defaults)


def _make_run() -> EvalRun:
    return EvalRun(
        run_id=uuid4(),
        status=RunStatus.PASSED,
        finished_at=datetime.now(timezone.utc),
        metrics=RunMetrics(
            hallucination_rate=0.02,
            answer_relevancy=0.85,
            faithfulness=0.80,
            accuracy=0.82,
            precision=0.78,
            recall=0.75,
            f1_score=0.76,
            prompt_injection_resistance=1.0,
            jailbreak_resistance=1.0,
            p95_latency_ms=2000,
            cost_per_query_usd=0.001,
        ),
        gate_results=[
            GateResult(
                metric="accuracy",
                status=GateStatus.PASS,
                value=0.82,
                threshold=0.55,
                message="ok",
            )
        ],
    )


def test_compare_metrics_trend(project_root):
    current = RunMetrics(accuracy=0.85, hallucination_rate=0.02, answer_relevancy=0.9)
    baseline = {"accuracy": 0.75, "hallucination_rate": 0.04, "answer_relevancy": 0.85}
    comparisons = compare_metrics(current, baseline)
    accuracy_cmp = next(c for c in comparisons if c.metric == "accuracy")
    hall_cmp = next(c for c in comparisons if c.metric == "hallucination_rate")
    assert accuracy_cmp.trend == "improved"
    assert hall_cmp.trend == "improved"


def test_report_builder_insights(project_root, tmp_path):
    config = load_eval_config(project_root)
    storage = LocalStorage(tmp_path)
    builder = ReportBuilder(storage, config)
    run = _make_run()
    results = [_make_result(), _make_result(question_id="q_002", category="adversarial")]
    report = builder.build(run, results)
    assert report.question_count == 2
    assert len(report.insights) > 0
    assert len(report.recommendations) > 0
    assert report.category_breakdown


def test_excel_export(project_root, tmp_path):
    pytest.importorskip("openpyxl")
    from llm_eval.reporting.excel_exporter import export_excel

    config = load_eval_config(project_root)
    storage = LocalStorage(tmp_path)
    builder = ReportBuilder(storage, config)
    run = _make_run()
    results = [_make_result()]
    report = builder.build(run, results)
    path = tmp_path / "report.xlsx"
    export_excel(report, results, path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_pdf_export(project_root, tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    from llm_eval.reporting.pdf_exporter import export_pdf

    config = load_eval_config(project_root)
    storage = LocalStorage(tmp_path)
    builder = ReportBuilder(storage, config)
    run = _make_run()
    results = [_make_result()]
    report = builder.build(run, results)
    path = tmp_path / "report.pdf"
    export_pdf(report, results, path)
    assert path.exists()
    assert path.stat().st_size > 0
