from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_anndata(path: str | Path, backed: str | None = None):
    """Load an AnnData object from disk."""
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required to load AnnData files. Install dependencies first."
        ) from exc

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix != ".h5ad":
        raise ValueError(f"Only .h5ad files are currently supported, got {suffix}")
    return ad.read_h5ad(file_path, backed=backed)


def validate_h5ad_file(path: str | Path) -> None:
    """Validate that an H5AD file can be opened successfully."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to validate H5AD files. Install dependencies first."
        ) from exc

    file_path = Path(path)
    try:
        with h5py.File(file_path, "r"):
            return
    except OSError as exc:
        raise ValueError(
            f"H5AD file appears incomplete or unreadable: {file_path}. "
            "Resume the download and try again."
        ) from exc


def compute_file_md5(path: str | Path) -> str:
    """Compute the MD5 checksum for a local file."""
    file_path = Path(path)
    md5 = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def file_matches_md5(path: str | Path, expected_md5: str) -> bool:
    """Check whether a local file matches an expected MD5 checksum."""
    return compute_file_md5(path).lower() == expected_md5.strip().lower()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a JSON payload with stable formatting."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {file_path}")
    return data
