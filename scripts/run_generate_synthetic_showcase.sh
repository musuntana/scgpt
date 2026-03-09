#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PROJECT_ENVIRONMENT="${ROOT_DIR}/.venv"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${ROOT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

BUNDLE_DIR="${ROOT_DIR}/data/processed/synthetic_demo_bundle"
TRANSFORMER_DIR="${ROOT_DIR}/artifacts/transformer_seen_synthetic_demo"
MLP_DIR="${ROOT_DIR}/artifacts/mlp_seen_synthetic_demo"
XGBOOST_DIR="${ROOT_DIR}/artifacts/xgboost_seen_synthetic_demo"
DATA_CONFIG="${ROOT_DIR}/configs/data_synthetic_demo.yaml"
MODEL_CONFIG="${ROOT_DIR}/configs/model_synthetic_demo.yaml"
TRAIN_CONFIG="${ROOT_DIR}/configs/train_synthetic_demo.yaml"
ASSET_DIR="${ROOT_DIR}/docs/assets"

"${ROOT_DIR}/scripts/run_generate_synthetic_demo.sh"

"${ROOT_DIR}/scripts/run_train_baselines.sh" \
  --bundle-dir "${BUNDLE_DIR}" \
  --output-dir "${MLP_DIR}" \
  --baseline mlp \
  --train-config "${TRAIN_CONFIG}"

"${ROOT_DIR}/scripts/run_summarize_run.sh" \
  --bundle-dir "${BUNDLE_DIR}" \
  --output-dir "${MLP_DIR}" \
  --checkpoint-path "${MLP_DIR}/best_model.pt" \
  --model-type mlp \
  --split-prefix seen \
  --data-config "${DATA_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --seen-metrics-path "${MLP_DIR}/mlp_seen_test_metrics.json" \
  --unseen-metrics-path "${MLP_DIR}/mlp_unseen_test_metrics.json"

"${ROOT_DIR}/scripts/run_train_baselines.sh" \
  --bundle-dir "${BUNDLE_DIR}" \
  --output-dir "${XGBOOST_DIR}" \
  --baseline xgboost \
  --train-config "${TRAIN_CONFIG}"

"${ROOT_DIR}/scripts/run_generate_results_assets.sh" \
  --bundle-dir "${BUNDLE_DIR}" \
  --transformer-artifact-dir "${TRANSFORMER_DIR}" \
  --mlp-artifact-dir "${MLP_DIR}" \
  --xgboost-artifact-dir "${XGBOOST_DIR}" \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --comparison-title "Synthetic Demo Bundle: Model Comparison" \
  --comparison-output-name "model_comparison_seen_synthetic_demo.png" \
  --preview-output-name "transformer_inference_preview_synthetic_demo.png"
