#!/usr/bin/env bash
set -euo pipefail

llm-eval validate-dataset
llm-eval run --scope smoke --no-gate
echo "Smoke test passed."
