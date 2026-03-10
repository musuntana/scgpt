#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_TARGET_PATH="data/raw/NormanWeissman2019_filtered.h5ad"
DEFAULT_SOURCE_URL="https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad?download=1"
TARGET_PATH="${DEFAULT_TARGET_PATH}"
SOURCE_URL="${NORMAN2019_SOURCE_URL:-${DEFAULT_SOURCE_URL}}"
EXPECTED_MD5="${NORMAN2019_EXPECTED_MD5:-c870e6967d91c017d9da827bab183cd6}"
DOWNLOAD_BACKEND="${NORMAN2019_DOWNLOAD_BACKEND:-auto}"
VERIFY_ONLY=0

usage() {
  cat <<EOF
Usage: ./scripts/download_norman2019.sh [options] [target_path]
Options:

  --backend <auto|curl|wget|python>  Choose a download backend. Default: auto
  --url <source_url>          Override the source URL for manual mirrors.
  --verify-only               Only verify the local file checksum and exit.
  -h, --help                  Show this help text.

Default target_path: ${DEFAULT_TARGET_PATH}

Environment overrides:
  NORMAN2019_SOURCE_URL
  NORMAN2019_EXPECTED_MD5
  NORMAN2019_DOWNLOAD_BACKEND
EOF
}

compute_md5() {
  python3 - <<'PY' "$1"
from __future__ import annotations

import sys

from src.data.io import compute_file_md5

print(compute_file_md5(sys.argv[1]))
PY
}

verify_existing_file() {
  local path="$1"
  local quiet="${2:-0}"

  if [[ ! -f "${path}" ]]; then
    if [[ "${quiet}" -eq 0 ]]; then
      echo "File not found: ${path}" >&2
    fi
    return 1
  fi

  local actual_md5
  actual_md5="$(compute_md5 "${path}")"
  if [[ "${actual_md5}" != "${EXPECTED_MD5}" ]]; then
    if [[ "${quiet}" -eq 0 ]]; then
      echo "MD5 mismatch for ${path}: expected ${EXPECTED_MD5}, got ${actual_md5}" >&2
    fi
    return 1
  fi

  if [[ "${quiet}" -eq 0 ]]; then
    echo "Verified Norman2019 dataset: ${path}"
  fi
  return 0
}

download_with_curl() {
  if ! command -v curl >/dev/null 2>&1; then
    return 127
  fi

  echo "Downloading with curl: ${SOURCE_URL}"
  curl \
    --http1.1 \
    --location \
    --continue-at - \
    --retry 12 \
    --retry-all-errors \
    --retry-delay 5 \
    --connect-timeout 30 \
    --speed-time 30 \
    --speed-limit 1024 \
    --fail \
    "${SOURCE_URL}" \
    --output "${TARGET_PATH}"
}

download_with_wget() {
  if ! command -v wget >/dev/null 2>&1; then
    return 127
  fi

  echo "Downloading with wget: ${SOURCE_URL}"
  wget \
    --continue \
    --tries=12 \
    --waitretry=5 \
    --retry-connrefused \
    --timeout=30 \
    --read-timeout=30 \
    --output-document="${TARGET_PATH}" \
    "${SOURCE_URL}"
}
download_with_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    return 127
  fi

  echo "Downloading with python urllib: ${SOURCE_URL}"
  python3 - <<'PY' "${SOURCE_URL}" "${TARGET_PATH}"
from __future__ import annotations

import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

url = sys.argv[1]
target_path = Path(sys.argv[2])
headers = {"User-Agent": "Mozilla/5.0"}


def download(range_start: int | None, mode: str) -> None:
    request_headers = dict(headers)
    if range_start is not None and range_start > 0:
        request_headers["Range"] = f"bytes={range_start}-"

    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(
        request,
        timeout=60,
        context=ssl.create_default_context(),
    ) as response:
        status = getattr(response, "status", response.getcode())
        if range_start is not None and range_start > 0 and status != 206:
            raise RuntimeError(f"Range resume not supported (status={status})")
        with target_path.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


existing_size = target_path.stat().st_size if target_path.exists() else 0
if existing_size > 0:
    try:
        download(existing_size, "ab")
    except Exception:
        download(None, "wb")
else:
    download(None, "wb")
PY
}

print_manual_next_steps() {
  local alternate_backend="wget"
  case "${DOWNLOAD_BACKEND}" in
    wget)
      alternate_backend="python"
      ;;
    python)
      alternate_backend="curl"
      ;;
  esac
  cat >&2 <<EOF
Download could not be completed automatically.

Next steps:
  1. Retry with the alternate backend if available:
       ./scripts/download_norman2019.sh --backend ${alternate_backend}
  2. Or download the file manually from:
       ${SOURCE_URL}
  3. Place it at:
       ${TARGET_PATH}
  4. Verify it locally:
       ./scripts/download_norman2019.sh --verify-only ${TARGET_PATH}
  5. Then continue with:
       ./scripts/run_norman2019_demo.sh ${TARGET_PATH}

If you only need a local demo right now, you can continue with:
  ./scripts/run_generate_synthetic_showcase.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --backend)
      DOWNLOAD_BACKEND="${2:-}"
      shift 2
      ;;
    --url)
      SOURCE_URL="${2:-}"
      shift 2
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      TARGET_PATH="$1"
      shift
      ;;
  esac
done

mkdir -p "$(dirname "${TARGET_PATH}")"

if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
  verify_existing_file "${TARGET_PATH}"
  exit $?
fi

if verify_existing_file "${TARGET_PATH}" 1; then
  echo "File already exists and passed MD5: ${TARGET_PATH}"
  exit 0
fi

if [[ -f "${TARGET_PATH}" ]]; then
  echo "Existing file failed checksum verification. Attempting resume/re-download: ${TARGET_PATH}" >&2
fi

download_succeeded=0
case "${DOWNLOAD_BACKEND}" in
  auto)
    if download_with_curl; then
      download_succeeded=1
    elif download_with_wget; then
      download_succeeded=1
    elif download_with_python; then
      download_succeeded=1
    fi
    ;;
  curl)
    if download_with_curl; then
      download_succeeded=1
    fi
    ;;
  wget)
    if download_with_wget; then
      download_succeeded=1
    fi
    ;;
  python)
    if download_with_python; then
      download_succeeded=1
    fi
    ;;
  *)
    echo "Unsupported backend: ${DOWNLOAD_BACKEND}" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "${download_succeeded}" -ne 1 ]]; then
  print_manual_next_steps
  exit 1
fi

if ! verify_existing_file "${TARGET_PATH}"; then
  print_manual_next_steps
  exit 1
fi

echo "Downloaded and verified ${TARGET_PATH}"
