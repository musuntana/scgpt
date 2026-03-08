#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./scripts/download_norman2019.sh [target_path]"
  echo "Default target_path: data/raw/NormanWeissman2019_filtered.h5ad"
  exit 0
fi

TARGET_PATH="${1:-data/raw/NormanWeissman2019_filtered.h5ad}"
SOURCE_URL="https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad?download=1"
EXPECTED_MD5="c870e6967d91c017d9da827bab183cd6"

mkdir -p "$(dirname "${TARGET_PATH}")"

if [[ -f "${TARGET_PATH}" ]]; then
  ACTUAL_MD5="$(python3 - <<'PY' "${TARGET_PATH}"
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
md5 = hashlib.md5()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        md5.update(chunk)
print(md5.hexdigest())
PY
)"
  if [[ "${ACTUAL_MD5}" == "${EXPECTED_MD5}" ]]; then
    echo "File already exists and passed MD5: ${TARGET_PATH}"
    exit 0
  fi
  echo "Existing file does not match expected MD5. Resuming download: ${TARGET_PATH}"
fi

curl \
  -L \
  -C - \
  --retry 8 \
  --retry-all-errors \
  --retry-delay 5 \
  --fail \
  "${SOURCE_URL}" \
  -o "${TARGET_PATH}"

ACTUAL_MD5="$(python3 - <<'PY' "${TARGET_PATH}"
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
md5 = hashlib.md5()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        md5.update(chunk)
print(md5.hexdigest())
PY
)"

if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
  echo "MD5 mismatch for ${TARGET_PATH}: expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}" >&2
  exit 1
fi

echo "Downloaded and verified ${TARGET_PATH}"
