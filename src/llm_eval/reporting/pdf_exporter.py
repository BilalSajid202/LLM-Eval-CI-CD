"""Export evaluation reports to PDF with tables and embedded charts."""

from __future__ import annotations

import io
from pathlib import Path

from llm_eval.models.types import EvaluationReport, QuestionResult


def _require_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        return (
            colors,
            letter,
            ParagraphStyle,
            getSampleStyleSheet,
            inch,
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ImportError(
            "PDF export requires reportlab and matplotlib. "
            "Install with: pip install 'llm-eval[reporting]'"
        ) from exc


def _make_chart_image(report: EvaluationReport) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not report.run.metrics:
        return None

    m = report.run.metrics
    labels = [
        "Accuracy", "Relevancy", "Faithfulness", "F1",
        "Precision", "Recall", "Inj. Res.", "JB Res.",
    ]
    values = [
        m.accuracy, m.answer_relevancy, m.faithfulness, m.f1_score,
        m.precision, m.recall, m.prompt_injection_resistance, m.jailbreak_resistance,
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics Overview")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def export_pdf(report: EvaluationReport, results: list[QuestionResult], path: Path) -> None:
    (
        colors,
        letter,
        ParagraphStyle,
        getSampleStyleSheet,
        inch,
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    ) = _require_reportlab()

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            rightMargin=0.75 * inch, leftMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=18, spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=13, spaceAfter=8)
    body_style = styles["Normal"]
    story: list = []

    run = report.run
    story.append(Paragraph("LLM Evaluation Report", title_style))
    story.append(Paragraph(
        f"Run {run.run_id} | Status: <b>{run.status.value.upper()}</b> | "
        f"Scope: {run.scope} | Model: {run.model_version}",
        body_style,
    ))
    story.append(Spacer(1, 0.2 * inch))

    if run.metrics:
        m = run.metrics
        metric_data = [
            ["Metric", "Value"],
            ["Hallucination Rate", f"{m.hallucination_rate:.2%}"],
            ["Answer Relevancy", f"{m.answer_relevancy:.3f}"],
            ["Faithfulness", f"{m.faithfulness:.3f}"],
            ["Context Recall", f"{m.context_recall:.3f}"],
            ["Accuracy", f"{m.accuracy:.3f}"],
            ["Precision", f"{m.precision:.3f}"],
            ["Recall", f"{m.recall:.3f}"],
            ["F1 Score", f"{m.f1_score:.3f}"],
            ["Injection Resistance", f"{m.prompt_injection_resistance:.1%}"],
            ["Jailbreak Resistance", f"{m.jailbreak_resistance:.1%}"],
            ["p95 Latency", f"{m.p95_latency_ms:.0f} ms"],
            ["Cost/Query", f"${m.cost_per_query_usd:.6f}"],
        ]
        t = Table(metric_data, colWidths=[2.5 * inch, 2 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ]))
        story.append(Paragraph("Aggregate Metrics", heading_style))
        story.append(t)
        story.append(Spacer(1, 0.2 * inch))

    chart_bytes = _make_chart_image(report)
    if chart_bytes:
        story.append(Paragraph("Metrics Chart", heading_style))
        story.append(Image(io.BytesIO(chart_bytes), width=6 * inch, height=3 * inch))
        story.append(Spacer(1, 0.2 * inch))

    if report.comparisons:
        story.append(Paragraph("Baseline Comparison", heading_style))
        comp_data = [["Metric", "Current", "Baseline", "Delta %", "Trend"]]
        for c in report.comparisons:
            comp_data.append([
                c.metric, f"{c.current:.4f}", f"{c.baseline:.4f}",
                f"{c.delta_pct:.1f}%", c.trend.upper(),
            ])
        ct = Table(comp_data, colWidths=[1.8 * inch, 1 * inch, 1 * inch, 0.8 * inch, 0.8 * inch])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(ct)
        story.append(Spacer(1, 0.2 * inch))

    if report.category_breakdown:
        story.append(Paragraph("Category Breakdown", heading_style))
        cat_data = [["Category", "Count", "Accuracy", "Relevancy", "Pass Rate"]]
        for cat in report.category_breakdown:
            cat_data.append([
                cat.category, str(cat.question_count),
                f"{cat.accuracy:.3f}", f"{cat.answer_relevancy:.3f}", f"{cat.pass_rate:.1%}",
            ])
        cat_t = Table(cat_data)
        cat_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(cat_t)
        story.append(PageBreak())

    story.append(Paragraph("Gate Results", heading_style))
    gate_data = [["Metric", "Status", "Message"]]
    for g in run.gate_results:
        gate_data.append([g.metric, g.status.value.upper(), g.message[:80]])
    gt = Table(gate_data, colWidths=[1.5 * inch, 0.8 * inch, 3.5 * inch])
    gt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(gt)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Key Insights", heading_style))
    for insight in report.insights:
        story.append(Paragraph(f"• {insight}", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Recommendations", heading_style))
    for rec in report.recommendations:
        story.append(Paragraph(f"• {rec}", body_style))
    story.append(PageBreak())

    story.append(Paragraph("Per-Question Results (Top 25 by lowest accuracy)", heading_style))
    sorted_results = sorted(results, key=lambda r: r.scores.get("accuracy", 0))
    raw_data = [["ID", "Category", "Accuracy", "Relevancy", "Halluc.", "Pass"]]
    for r in sorted_results[:25]:
        hall = "Yes" if r.scores.get("hallucination_verdict", 0) >= 0.5 else "No"
        passed = "PASS" if r.metadata.get("accuracy_pass") else "FAIL"
        raw_data.append([
            r.question_id, r.category,
            f"{r.scores.get('accuracy', 0):.3f}",
            f"{r.scores.get('answer_relevancy', 0):.3f}",
            hall, passed,
        ])
    rt = Table(raw_data)
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(rt)

    if report.overall_rank:
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(
            f"Overall Performance Rank: <b>{report.overall_rank}</b> of "
            f"<b>{report.total_runs_compared}</b> recent runs",
            body_style,
        ))

    doc.build(story)
