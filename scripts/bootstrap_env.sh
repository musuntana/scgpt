#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"

if [[ ! -d ".venv" ]]; then
  uv venv --python 3.11 .venv
fi

uv sync --dev --python .venv/bin/python

echo
echo "Environment ready."
echo "Activate with: source .venv/bin/activate"
echo "Or run commands with: uv run <command>"

