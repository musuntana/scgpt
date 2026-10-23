#!/usr/bin/env bash
# Evaluate each trained model only on the test split matching its training protocol.
#
# Usage:
#   ./scripts/run_full_evaluation.sh [--bundle-dir DIR] [--split-prefix seen|unseen] [--deg-artifact-path CSV]
#
# Defaults:
#   --bundle-dir       data/processed/norman2019_demo_bundle
#   --split-prefix     seen
#   --deg-artifact-path auto-detected for the matching Transformer run when present
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PROJECT_ENVIRONMENT="${ROOT_DIR}/.venv"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${ROOT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

BUNDLE_DIR="data/processed/norman2019_demo_bundle"
SPLIT_PREFIX="seen"
DEG_ARTIFACT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle-dir)
            BUNDLE_DIR="$2"; shift 2 ;;
        --split-prefix)
            SPLIT_PREFIX="$2"; shift 2 ;;
        --deg-artifact-path)
            DEG_ARTIFACT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${SPLIT_PREFIX}" != "seen" && "${SPLIT_PREFIX}" != "unseen" ]]; then
    echo "--split-prefix must be 'seen' or 'unseen'." >&2
    exit 1
fi

TEST_SPLIT="${SPLIT_PREFIX}_test"
TRANSFORMER_DIR="artifacts/transformer_${SPLIT_PREFIX}_norman2019_demo"
MLP_DIR="artifacts/mlp_${SPLIT_PREFIX}_norman2019_demo"
XGB_DIR="artifacts/xgboost_${SPLIT_PREFIX}_norman2019_demo"

if [[ -z "${DEG_ARTIFACT}" ]]; then
    CANDIDATE="${TRANSFORMER_DIR}/deg_artifact.csv"
    if [[ -f "${CANDIDATE}" ]]; then
        DEG_ARTIFACT="${CANDIDATE}"
    fi
fi

echo "=== Protocol-matched evaluation ==="
echo "  bundle-dir   : ${BUNDLE_DIR}"
echo "  split-prefix : ${SPLIT_PREFIX}"
echo "  test-split   : ${TEST_SPLIT}"
echo "  deg-artifact : ${DEG_ARTIFACT:-<none>}"
echo ""

deg_flag() {
    if [[ -n "${DEG_ARTIFACT}" && -f "${DEG_ARTIFACT}" ]]; then
        echo "--deg-artifact-path ${DEG_ARTIFACT}"
    fi
}

TRANSFORMER_CKPT="${TRANSFORMER_DIR}/best_model.pt"
if [[ -f "${TRANSFORMER_CKPT}" ]]; then
    echo "--- Transformer: ${TEST_SPLIT} ---"
    uv run python scripts/evaluate_model.py \
        --bundle-dir "${BUNDLE_DIR}" \
        --checkpoint-path "${TRANSFORMER_CKPT}" \
        --model-type transformer \
        --split-name "${TEST_SPLIT}" \
        --output-path "${TRANSFORMER_DIR}/${TEST_SPLIT}_metrics.json" \
        --per-perturbation-output-path "${TRANSFORMER_DIR}/${TEST_SPLIT}_per_perturbation.csv" \
        --error-summary-output-path "${TRANSFORMER_DIR}/${TEST_SPLIT}_error_summary.json" \
        $(deg_flag)
else
    echo "[SKIP] Transformer checkpoint not found: ${TRANSFORMER_CKPT}"
fi

MLP_CKPT="${MLP_DIR}/best_model.pt"
if [[ -f "${MLP_CKPT}" ]]; then
    echo "--- MLP: ${TEST_SPLIT} ---"
    uv run python scripts/evaluate_model.py \
        --bundle-dir "${BUNDLE_DIR}" \
        --checkpoint-path "${MLP_CKPT}" \
        --model-type mlp \
        --model-config configs/model.yaml \
        --split-name "${TEST_SPLIT}" \
        --output-path "${MLP_DIR}/mlp_${TEST_SPLIT}_metrics.json" \
        --per-perturbation-output-path "${MLP_DIR}/mlp_${TEST_SPLIT}_per_perturbation.csv" \
        --error-summary-output-path "${MLP_DIR}/mlp_${TEST_SPLIT}_error_summary.json" \
        $(deg_flag)
else
    echo "[SKIP] MLP checkpoint not found: ${MLP_CKPT}"
fi

XGB_SUMMARY="${XGB_DIR}/xgboost_run_summary.json"
if [[ -f "${XGB_SUMMARY}" ]]; then
    echo "--- XGBoost: metrics already recorded in ${XGB_SUMMARY} ---"
else
    echo "[SKIP] XGBoost run summary not found: ${XGB_SUMMARY}"
fi

echo ""
echo "=== Protocol-matched evaluation complete. ==="
