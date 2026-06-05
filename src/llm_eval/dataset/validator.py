"""Golden dataset validation rules."""

from __future__ import annotations

from pathlib import Path

from llm_eval.dataset.loader import load_golden_dataset
from llm_eval.models.types import GoldenQuestion


class DatasetValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Dataset validation failed with {len(errors)} error(s)")


def validate_dataset(questions: list[GoldenQuestion]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for q in questions:
        if q.id in seen_ids:
            errors.append(f"Duplicate question id: {q.id}")
        seen_ids.add(q.id)

        if not q.question.strip():
            errors.append(f"{q.id}: question is empty")

        if not q.expected_answer.strip():
            errors.append(f"{q.id}: expected_answer is empty")

        if q.category.value in ("factual", "reasoning") and not q.expected_sources:
            errors.append(f"{q.id}: factual/reasoning questions require expected_sources")

        if q.category.value in ("edge_case", "adversarial") and not q.tags:
            errors.append(f"{q.id}: edge_case/adversarial questions should have tags")

    return errors


def validate_dataset_file(path: Path | str) -> None:
    questions = load_golden_dataset(path)
    errors = validate_dataset(questions)
    if errors:
        raise DatasetValidationError(errors)
