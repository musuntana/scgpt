#!/usr/bin/env bash
# Evaluate a saved perturbation model.
#
# Usage:
#   ./scripts/run_evaluate.sh \
#     --bundle-dir data/processed/demo_bundle \
#     --checkpoint-path artifacts/transformer_seen/best_model.pt \
#     --model-type transformer \
#     --output-path artifacts/transformer_seen/test_metrics.json \
#     --deg-artifact-path artifacts/transformer_seen/deg_artifact.csv
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PROJECT_ENVIRONMENT="${ROOT_DIR}/.venv"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${ROOT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

uv run python scripts/evaluate_model.py "$@"
