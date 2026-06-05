"""PostgreSQL storage for eval runs and metrics."""

from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

from llm_eval.models.types import EvalRun, GateResult, QuestionResult, RunMetrics


class PostgresStorage:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool = None

    def _get_pool(self):
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(self.database_url, min_size=1, max_size=5)
        return self._pool

    def save_run(
        self,
        run: EvalRun,
        question_results: list[QuestionResult] | None = None,
        s3_keys: dict[str, str] | None = None,
    ) -> None:
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval_runs (
                        run_id, git_sha, git_branch, trigger_type, model_version,
                        config_hash, started_at, finished_at, status, gate_result, scope
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        finished_at = EXCLUDED.finished_at,
                        status = EXCLUDED.status,
                        gate_result = EXCLUDED.gate_result
                    """,
                    (
                        str(run.run_id),
                        run.git_sha,
                        run.git_branch,
                        run.trigger_type.value,
                        run.model_version,
                        run.config_hash,
                        run.started_at,
                        run.finished_at,
                        run.status.value,
                        json.dumps([g.model_dump() for g in run.gate_results]),
                        run.scope,
                    ),
                )

                if run.metrics:
                    for metric in run.metrics.to_metric_list():
                        cur.execute(
                            """
                            INSERT INTO run_metrics (run_id, metric_name, value, p50, p95)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (run_id, metric_name) DO UPDATE SET
                                value = EXCLUDED.value,
                                p50 = EXCLUDED.p50,
                                p95 = EXCLUDED.p95
                            """,
                            (
                                str(run.run_id),
                                metric.name,
                                metric.value,
                                metric.p50,
                                metric.p95,
                            ),
                        )

                if question_results:
                    for qr in question_results:
                        s3_key = (s3_keys or {}).get(qr.question_id, "")
                        cur.execute(
                            """
                            INSERT INTO question_results (
                                run_id, question_id, score, latency_ms, cost_usd, s3_key, details
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (run_id, question_id) DO UPDATE SET
                                score = EXCLUDED.score,
                                latency_ms = EXCLUDED.latency_ms,
                                cost_usd = EXCLUDED.cost_usd,
                                s3_key = EXCLUDED.s3_key,
                                details = EXCLUDED.details
                            """,
                            (
                                str(run.run_id),
                                qr.question_id,
                                qr.scores.get("faithfulness", 0.0),
                                qr.latency_ms,
                                qr.cost_usd,
                                s3_key,
                                json.dumps(qr.model_dump()),
                            ),
                        )
            conn.commit()

    def get_cost_baseline(self, days: int = 7) -> float | None:
        try:
            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT AVG(rm.value)
                        FROM run_metrics rm
                        JOIN eval_runs er ON er.run_id = rm.run_id
                        WHERE rm.metric_name = 'cost_per_query_usd'
                          AND er.started_at >= NOW() - (%s * INTERVAL '1 day')
                        """,
                        (days,),
                    )
                    row = cur.fetchone()
                    return float(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    def close(self) -> None:
        if self._pool:
            self._pool.close()
