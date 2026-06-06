"""Local filesystem storage fallback for development."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from llm_eval.models.types import EvalRun, QuestionResult

logger = logging.getLogger(__name__)


class LocalStorage:
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.base_path / "runs"
        self.outputs_path = self.base_path / "outputs"
        self.runs_path.mkdir(exist_ok=True)
        self.outputs_path.mkdir(exist_ok=True)

    def save_run(self, run: EvalRun) -> None:
        path = self.runs_path / f"{run.run_id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    def save_raw_output(self, run_id: UUID, result: QuestionResult) -> str:
        key = f"{run_id}/{result.question_id}/response.json"
        path = self.outputs_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return str(key)

    def load_run(self, run_id: UUID) -> EvalRun | None:
        path = self.runs_path / f"{run_id}.json"
        if not path.exists():
            return None
        return EvalRun.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[EvalRun]:
        runs = []
        for path in self.runs_path.glob("*.json"):
            try:
                runs.append(EvalRun.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Skipping corrupt run file %s: %s", path, exc)
        return sorted(runs, key=lambda r: r.started_at, reverse=True)

    def get_cost_baseline(self, days: int = 7) -> float | None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        costs = []
        for run in self.list_runs():
            if not run.metrics:
                continue
            started = run.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if started >= cutoff:
                costs.append(run.metrics.cost_per_query_usd)
        if len(costs) < 2:
            return None
        return sum(costs) / len(costs)
