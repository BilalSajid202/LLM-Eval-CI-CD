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
| Concurrency | Per-question semaphore | `parallel_workers` caps concurrent questions, not batches |
| Agents | `agents.run_on` config | CrewAI analysis runs only on configured gate outcomes |

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For full scorer support (embedding relevancy + RAGAS metrics):
```bash
pip install -e ".[scorers,agents,modal,reporting,dev]"
```

For detailed Excel/PDF evaluation reports:
```bash
pip install -e ".[reporting]"
```

Without `[scorers]`, hallucination/relevancy/RAGAS use lightweight heuristics — fine for CI unit tests, but install `[scorers]` for production eval accuracy.

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
| `llm-eval run` | Execute eval pipeline (auto-generates Excel/PDF reports) |
| `llm-eval report` | Generate detailed report for an existing run |
| `llm-eval compare` | Compare run metrics against historical baseline |
| `llm-eval gate` | Post GitHub status for existing run |
| `llm-eval validate-dataset` | Validate golden dataset |
| `llm-eval analyze` | Run eval + CrewAI failure analysis |

```bash
llm-eval run --scope full --trigger pr --sha abc123 --branch feature/x
llm-eval report --run-id <uuid> --format both
llm-eval compare --run-id <uuid> --baseline auto --days 7
llm-eval gate --run-id <uuid> --sha abc123
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Hallucination rate | LLM-as-judge verdict on unsupported claims |
| Answer relevancy | Question–answer semantic similarity |
| Faithfulness | Answer grounded in retrieved context |
| Context recall | Retrieved context covers expected facts |
| Accuracy | Answer vs expected answer overlap |
| Precision | Source retrieval precision (or answer precision) |
| Recall | Expected content coverage in answer |
| F1 score | Harmonic mean of precision and recall |
| Injection resistance | Refusal rate on `prompt_injection` tagged questions |
| Jailbreak resistance | Refusal rate on `jailbreak` tagged questions |

## Detailed Reports

After each eval run, reports are saved to `.eval-storage/reports/<run-id>/`:

| File | Contents |
|------|----------|
| `evaluation_report.xlsx` | Summary, raw results, metric comparisons, category breakdown, charts |
| `evaluation_report.pdf` | Executive summary with metrics table, charts, insights |
| `report.json` | Machine-readable report with comparisons and recommendations |

Excel workbook sheets:
- **Summary** — run metadata, aggregate metrics, gate results, insights
- **Raw Results** — per-question scores (accuracy, relevancy, precision, recall, security)
- **Metric Comparisons** — current vs 7-day baseline with trend indicators
- **Category Breakdown** — performance by factual/reasoning/edge_case/adversarial
- **Visualizations** — bar charts for quality metrics and baseline comparison

## Baseline & History

- **History**: Every run is persisted to `.eval-storage/runs/` (metadata) and `.eval-storage/outputs/` (per-question results). Optional PostgreSQL + Grafana for long-term trends.
- **Baseline**: Rolling 7-day average across all metrics (cost regression gate already used this; now extended to full metric comparison in reports).
- **Comparison**: `llm-eval compare` shows delta and trend (improved/regressed/stable) per metric.

## Quality Gates

Configured in `config/eval.yaml`:

| Gate | Default Block | Default Warn |
|------|---------------|--------------|
| Hallucination rate | > 5% | > 3% |
| p95 latency | > 4000ms | > 3000ms |
| Faithfulness | < 0.60 | < 0.75 |
| Answer relevancy | < 0.70 | < 0.78 |
| Cost regression | > 50% vs baseline | > 20% |
| Accuracy | < 0.55 | < 0.65 |
| Injection resistance | < 0.50 | < 0.75 |
| Jailbreak resistance | < 0.50 | < 0.75 |

Per-question fields in `questions.yaml` (`sla_latency_ms`, `min_relevancy_score`) produce **warn** gates when violated.

### Eval scope on PRs

Set `eval.scope_on_pr` in `config/eval.yaml` (default: `full`). When running with `--trigger pr` and default scope, the CLI uses this value automatically.

### Agent analysis trigger

Set `agents.run_on` in `config/eval.yaml`:

| Value | When CrewAI analysis runs |
|-------|---------------------------|
| `failure` | Block-level gate failures only (default) |
| `warn` | Any warn or block gate |
| `always` | Every eval run |

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
ruff check src tests   # requires pip install -e ".[dev]"
```

### Dataset builder

Regenerate the golden dataset from the knowledge base:

```bash
python scripts/build_golden_dataset.py
```

The script validates all `expected_sources` paths exist before writing.

## License

MIT
