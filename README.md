# LLM Eval CI/CD

Automated evaluation framework for LLM-powered systems. Runs quality checks on every prompt, model, or RAG change — like unit tests for LLM behavior — and blocks merges when gates fail.

## Architecture

```
Triggers (GitHub Actions)
    → Orchestration (EvalService)
        → Execution (local asyncio / Modal workers)
            → Scoring (hallucination, relevancy, RAGAS, latency, cost)
                → Storage (PostgreSQL + S3/local)
                    → Merge Gate (GitHub status) + Dashboard (Grafana)
                    → Agents (CrewAI failure analysis on regressions)
```

## Directory Structure

```
LLM-Eval-CI-CD/
├── .github/workflows/          # CI/CD pipelines
├── config/                     # Team-owned config (gates + pipeline)
│   ├── eval.yaml               # Quality gate thresholds
│   └── pipeline.yaml           # Model, retriever, pricing
├── data/
│   ├── golden_dataset/         # Version-controlled test set
│   └── knowledge_base/         # Reference RAG documents
├── src/llm_eval/               # Main Python package (src layout)
│   ├── agents/                 # CrewAI failure analysis
│   ├── config/                 # Settings loader
│   ├── dataset/                # Golden dataset load/validate
│   ├── execution/              # Runner, collector, Modal
│   ├── gates/                  # Quality gates + GitHub status
│   ├── models/                 # Pydantic domain types
│   ├── pipeline/               # RAG pipeline under test
│   ├── scorers/                # Metric scorers
│   ├── storage/                # Postgres, S3, local fallback
│   ├── cli.py                  # CLI entry point
│   └── service.py              # Top-level orchestrator
├── prompts/                    # Prompt templates (eval triggers)
├── migrations/                 # PostgreSQL schema
├── infrastructure/
│   ├── docker/                 # Container image
│   └── grafana/                # Dashboard exports
├── scripts/                    # Setup and smoke test helpers
└── tests/                      # Unit tests
```

### Improvements over the original spec

| Area | Change | Why |
|------|--------|-----|
| Package layout | `src/llm_eval/` | Standard Python packaging, clean imports |
| Config | `config/eval.yaml` + `config/pipeline.yaml` | Separates gate thresholds from pipeline settings |
| Data | `data/` namespace | Keeps datasets and KB together, out of package code |
| Storage | Local fallback via `.eval-storage/` | Dev-friendly without cloud credentials |
| Service layer | `service.py` orchestrator | Single entry for CLI, Actions, and agents |
| Scorers | Heuristic fallbacks | Works without API keys or heavy ML deps in CI |
| Infrastructure | `infrastructure/` folder | Docker/Grafana separated from app code |

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For full scorer support:
```bash
pip install -e ".[scorers,agents,modal,dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY for live LLM evals (optional for unit tests)
```

### 3. Validate dataset

```bash
llm-eval validate-dataset
```

### 4. Run smoke eval (3 questions)

```bash
llm-eval run --scope smoke --no-gate
```

### 5. Run full eval

```bash
llm-eval run --scope full --trigger manual
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `llm-eval run` | Execute eval pipeline |
| `llm-eval gate` | Post GitHub status for existing run |
| `llm-eval validate-dataset` | Validate golden dataset |
| `llm-eval analyze` | Run eval + CrewAI failure analysis |

```bash
llm-eval run --scope full --trigger pr --sha abc123 --branch feature/x
llm-eval gate --run-id <uuid> --sha abc123
```

## Quality Gates

Configured in `config/eval.yaml`:

| Gate | Default Block | Default Warn |
|------|---------------|--------------|
| Hallucination rate | > 5% | > 3% |
| p95 latency | > 4000ms | > 3000ms |
| Faithfulness | < 0.60 | < 0.75 |
| Answer relevancy | < 0.70 | < 0.78 |
| Cost regression | > 50% vs baseline | > 20% |

## GitHub Actions

Workflow triggers on changes to:
- `prompts/**`
- `config/**`
- `data/golden_dataset/**`
- `data/knowledge_base/**`

Required secrets: `ANTHROPIC_API_KEY`, `GITHUB_TOKEN`

Optional: `DATABASE_URL`, `S3_BUCKET`, AWS credentials

Enable branch protection requiring status check: `llm-eval/quality-gates`

## Database Setup

```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/llm_eval
bash scripts/init_db.sh
```

## Modal (optional)

```bash
pip install -e ".[modal]"
modal token new
modal secret create llm-secrets ANTHROPIC_API_KEY=... DATABASE_URL=...
export EVAL_MODE=modal
llm-eval run --scope full
```

## Development

```bash
pytest tests/ -v
ruff check src tests
```

## License

MIT
