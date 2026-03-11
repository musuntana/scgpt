#!/usr/bin/env bash
# Evaluate all three trained models (Transformer, MLP, XGBoost) on the
# Norman2019 bundle and write per-model test metric JSON files.
#
# Usage:
#   ./scripts/run_full_evaluation.sh [--bundle-dir DIR] [--deg-artifact-path CSV]
#
# Defaults:
#   --bundle-dir            data/processed/norman2019_demo_bundle
#   --deg-artifact-path     artifacts/transformer_seen_norman2019_demo/deg_artifact.csv
#                           (passed only to the Transformer and MLP evaluations)
#
# Outputs (one file per model / split):
#   artifacts/transformer_seen_norman2019_demo/seen_test_metrics.json
#   artifacts/transformer_seen_norman2019_demo/unseen_test_metrics.json
#   artifacts/transformer_seen_norman2019_demo/seen_test_per_perturbation.csv
#   artifacts/transformer_seen_norman2019_demo/unseen_test_error_summary.json
#   artifacts/mlp_seen_norman2019_demo/mlp_seen_test_metrics.json
#   artifacts/mlp_seen_norman2019_demo/mlp_unseen_test_metrics.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PROJECT_ENVIRONMENT="${ROOT_DIR}/.venv"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${ROOT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

# ── Defaults ─────────────────────────────────────────────────────────────────
BUNDLE_DIR="data/processed/norman2019_demo_bundle"
DEG_ARTIFACT=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle-dir)
            BUNDLE_DIR="$2"; shift 2 ;;
        --deg-artifact-path)
            DEG_ARTIFACT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Auto-detect DEG artifact if not provided
if [[ -z "${DEG_ARTIFACT}" ]]; then
    CANDIDATE="artifacts/transformer_seen_norman2019_demo/deg_artifact.csv"
    if [[ -f "${CANDIDATE}" ]]; then
        DEG_ARTIFACT="${CANDIDATE}"
    fi
fi

echo "=== Full evaluation: Norman2019 bundle ==="
echo "  bundle-dir       : ${BUNDLE_DIR}"
echo "  deg-artifact     : ${DEG_ARTIFACT:-<none>}"
echo ""

# ── Helper: build optional --deg-artifact-path flag ──────────────────────────
deg_flag() {
    if [[ -n "${DEG_ARTIFACT}" && -f "${DEG_ARTIFACT}" ]]; then
        echo "--deg-artifact-path ${DEG_ARTIFACT}"
    fi
}

# ── Transformer ───────────────────────────────────────────────────────────────
TRANSFORMER_DIR="artifacts/transformer_seen_norman2019_demo"
TRANSFORMER_CKPT="${TRANSFORMER_DIR}/best_model.pt"

if [[ -f "${TRANSFORMER_CKPT}" ]]; then
    echo "--- Transformer: seen test ---"
    uv run python scripts/evaluate_model.py \
        --bundle-dir "${BUNDLE_DIR}" \
        --checkpoint-path "${TRANSFORMER_CKPT}" \
        --model-type transformer \
        --split-name seen_test \
        --output-path "${TRANSFORMER_DIR}/seen_test_metrics.json" \
        --per-perturbation-output-path "${TRANSFORMER_DIR}/seen_test_per_perturbation.csv" \
        --error-summary-output-path "${TRANSFORMER_DIR}/seen_test_error_summary.json" \
        $(deg_flag)

    echo "--- Transformer: unseen test ---"
    uv run python scripts/evaluate_model.py \
        --bundle-dir "${BUNDLE_DIR}" \
        --checkpoint-path "${TRANSFORMER_CKPT}" \
        --model-type transformer \
        --split-name unseen_test \
        --output-path "${TRANSFORMER_DIR}/unseen_test_metrics.json" \
        --per-perturbation-output-path "${TRANSFORMER_DIR}/unseen_test_per_perturbation.csv" \
        --error-summary-output-path "${TRANSFORMER_DIR}/unseen_test_error_summary.json" \
        $(deg_flag)
else
    echo "[SKIP] Transformer checkpoint not found: ${TRANSFORMER_CKPT}"
fi

# ── MLP ───────────────────────────────────────────────────────────────────────
MLP_DIR="artifacts/mlp_seen_norman2019_demo"
MLP_CKPT="${MLP_DIR}/best_model.pt"

if [[ -f "${MLP_CKPT}" ]]; then
    echo "--- MLP: seen test ---"
    uv run python scripts/evaluate_model.py \
        --bundle-dir "${BUNDLE_DIR}" \
        --checkpoint-path "${MLP_CKPT}" \
        --model-type mlp \
        --model-config configs/model.yaml \
        --split-name seen_test \
        --output-path "${MLP_DIR}/mlp_seen_test_metrics.json" \
        --per-perturbation-output-path "${MLP_DIR}/mlp_seen_test_per_perturbation.csv" \
        --error-summary-output-path "${MLP_DIR}/mlp_seen_test_error_summary.json" \
        $(deg_flag)

    echo "--- MLP: unseen test ---"
    uv run python scripts/evaluate_model.py \
        --bundle-dir "${BUNDLE_DIR}" \
        --checkpoint-path "${MLP_CKPT}" \
        --model-type mlp \
        --model-config configs/model.yaml \
        --split-name unseen_test \
        --output-path "${MLP_DIR}/mlp_unseen_test_metrics.json" \
        --per-perturbation-output-path "${MLP_DIR}/mlp_unseen_test_per_perturbation.csv" \
        --error-summary-output-path "${MLP_DIR}/mlp_unseen_test_error_summary.json" \
        $(deg_flag)
else
    echo "[SKIP] MLP checkpoint not found: ${MLP_CKPT}"
fi

# ── XGBoost (no re-evaluation needed — metrics saved at train time) ───────────
XGB_DIR="artifacts/xgboost_seen_norman2019_demo"
XGB_SUMMARY="${XGB_DIR}/xgboost_run_summary.json"

if [[ -f "${XGB_SUMMARY}" ]]; then
    echo "--- XGBoost: metrics already in ${XGB_SUMMARY} (no checkpoint to reload) ---"
else
    echo "[SKIP] XGBoost run summary not found: ${XGB_SUMMARY}"
fi

echo ""
echo "=== Evaluation complete. ==="
