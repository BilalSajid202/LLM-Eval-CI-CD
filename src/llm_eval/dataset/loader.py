"""Golden dataset loading and scope filtering."""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_eval.models.types import GoldenQuestion, QuestionCategory


def load_golden_dataset(path: Path | str) -> list[GoldenQuestion]:
    dataset_path = Path(path)
    with dataset_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []

    return [GoldenQuestion.model_validate(item) for item in raw]


def filter_by_scope(
    questions: list[GoldenQuestion],
    scope: str,
) -> list[GoldenQuestion]:
    if scope == "full":
        return questions
    if scope == "smoke":
        return questions[:3]
    if scope == "retrieval":
        return [q for q in questions if q.expected_sources]
    return questions


def group_by_category(questions: list[GoldenQuestion]) -> dict[str, list[GoldenQuestion]]:
    grouped: dict[str, list[GoldenQuestion]] = {}
    for q in questions:
        grouped.setdefault(q.category.value, []).append(q)
    return grouped


CATEGORY_ORDER = [c.value for c in QuestionCategory]
