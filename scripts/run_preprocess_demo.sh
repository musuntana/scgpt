#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PROJECT_ENVIRONMENT="${ROOT_DIR}/.venv"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${ROOT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

uv run python scripts/preprocess_data.py "$@"
