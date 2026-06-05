-- LLM Eval CI/CD — initial PostgreSQL schema

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    git_sha       CHAR(40) NOT NULL,
    git_branch    TEXT,
    trigger_type  TEXT NOT NULL,
    model_version TEXT,
    config_hash   TEXT,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at   TIMESTAMPTZ,
    status        TEXT NOT NULL DEFAULT 'running',
    gate_result   JSONB DEFAULT '[]'::jsonb,
    scope         TEXT DEFAULT 'full'
);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id      UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    p50         DOUBLE PRECISION,
    p95         DOUBLE PRECISION,
    PRIMARY KEY (run_id, metric_name)
);

CREATE TABLE IF NOT EXISTS question_results (
    run_id      UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    question_id TEXT NOT NULL,
    score       DOUBLE PRECISION DEFAULT 0,
    latency_ms  DOUBLE PRECISION DEFAULT 0,
    cost_usd    DOUBLE PRECISION DEFAULT 0,
    s3_key      TEXT,
    details     JSONB,
    PRIMARY KEY (run_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_git_sha ON eval_runs(git_sha);
CREATE INDEX IF NOT EXISTS idx_eval_runs_started_at ON eval_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_metrics_name ON run_metrics(metric_name);
