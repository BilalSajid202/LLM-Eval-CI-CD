"""Configuration loading for eval gates, pipeline, and runtime settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GateThreshold(BaseModel):
    block_threshold: float
    warn_threshold: float


class GatesConfig(BaseModel):
    hallucination_rate: GateThreshold
    p95_latency_ms: GateThreshold
    cost_regression_pct: GateThreshold
    faithfulness: GateThreshold
    answer_relevancy: GateThreshold
    prompt_injection_resistance: GateThreshold | None = None
    jailbreak_resistance: GateThreshold | None = None
    accuracy: GateThreshold | None = None


class EvalSettings(BaseModel):
    dataset_path: str
    scope_on_pr: str = "full"
    parallel_workers: int = 10
    timeout_per_question_s: int = 30
    batch_size: int = 10


class ScorerConfig(BaseModel):
    enabled: bool = True
    model: str | None = None
    model_name: str | None = None
    judge_model: str | None = None


class ScorersConfig(BaseModel):
    hallucination: ScorerConfig = Field(default_factory=ScorerConfig)
    relevancy: ScorerConfig = Field(default_factory=ScorerConfig)
    ragas: ScorerConfig = Field(default_factory=ScorerConfig)
    security: ScorerConfig = Field(default_factory=ScorerConfig)
    accuracy: ScorerConfig = Field(default_factory=ScorerConfig)


class ReportingConfig(BaseModel):
    enabled: bool = True
    auto_export: bool = True
    formats: list[str] = Field(default_factory=lambda: ["excel", "pdf"])
    baseline_days: int = 7


class AgentsConfig(BaseModel):
    enabled: bool = True
    model: str = "claude-sonnet-4-5"
    run_on: str = "failure"


class EvalConfig(BaseModel):
    gates: GatesConfig
    eval: EvalSettings
    scorers: ScorersConfig = Field(default_factory=ScorersConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


class ModelConfig(BaseModel):
    provider: str = "anthropic"
    name: str = "claude-haiku-4-5"
    temperature: float = 0.0
    max_tokens: int = 1024


class RetrieverConfig(BaseModel):
    type: str = "local"
    top_k: int = 5
    chunk_size: int = 512


class PricingConfig(BaseModel):
    input_per_million: float = 0.80
    output_per_million: float = 4.00


class RagConfig(BaseModel):
    knowledge_base_path: str = "data/knowledge_base"
    system_prompt_path: str = "prompts/system_prompt.yaml"


class PipelineConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    rag: RagConfig = Field(default_factory=RagConfig)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    database_url: str = ""
    s3_bucket: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    local_storage_path: str = "./.eval-storage"
    github_token: str = ""
    github_repository: str = ""
    dashboard_url: str = "http://localhost:3000"
    modal_token_id: str = ""
    modal_token_secret: str = ""
    slack_webhook_url: str = ""
    eval_mode: str = "local"
    log_level: str = "INFO"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_eval_config(root: Path | None = None) -> EvalConfig:
    root = root or _project_root()
    data = _load_yaml(root / "config" / "eval.yaml")
    return EvalConfig.model_validate(data)


def load_pipeline_config(root: Path | None = None) -> PipelineConfig:
    root = root or _project_root()
    data = _load_yaml(root / "config" / "pipeline.yaml")
    return PipelineConfig.model_validate(data)


def compute_config_hash(*configs: dict[str, Any]) -> str:
    payload = json.dumps(configs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_settings(root: Path | None = None) -> tuple[AppSettings, EvalConfig, PipelineConfig]:
    root = root or _project_root()
    return AppSettings(), load_eval_config(root), load_pipeline_config(root)
