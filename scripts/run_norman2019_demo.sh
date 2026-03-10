#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PROJECT_ENVIRONMENT="${ROOT_DIR}/.venv"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${ROOT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

RAW_PATH="${1:-data/raw/NormanWeissman2019_filtered.h5ad}"
OUTPUT_DIR="${2:-data/processed/norman2019_demo_bundle}"
SCHEMA_JSON="${3:-data/interim/norman2019_schema.json}"

if [[ ! -f "${RAW_PATH}" ]]; then
  echo "Missing raw dataset: ${RAW_PATH}" >&2
  echo "Try: ./scripts/download_norman2019.sh ${RAW_PATH}" >&2
  echo "If curl is unstable, retry with: ./scripts/download_norman2019.sh --backend wget ${RAW_PATH}" >&2
  echo "If you placed the file manually, verify it with: ./scripts/download_norman2019.sh --verify-only ${RAW_PATH}" >&2
  echo "If you need an immediate demo, continue with: ./scripts/run_generate_synthetic_showcase.sh" >&2
  exit 1
fi

uv run python - <<'PY' "${RAW_PATH}"
from __future__ import annotations

import sys

from src.data.io import validate_h5ad_file

try:
    validate_h5ad_file(sys.argv[1])
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1) from exc
PY

./scripts/run_inspect_anndata.sh \
  --input-path "${RAW_PATH}" \
  --output-json "${SCHEMA_JSON}"

./scripts/run_preprocess_demo.sh \
  --input-path "${RAW_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --hvg-top-genes 512 \
  --max-cells-per-perturbation 100
