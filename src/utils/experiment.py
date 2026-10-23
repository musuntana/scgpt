from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import load_yaml


def _find_repo_root(start_path: str | Path) -> Path | None:
    """Find the nearest parent containing Git metadata."""
    path = Path(start_path).resolve()
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _fingerprint_file(
    path: str | Path | None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Describe a file and hash it in bounded-memory chunks."""
    if path is None:
        return {
            "path": None,
            "status": "not_provided",
            "size_bytes": None,
            "sha256": None,
        }

    display_path = str(path)
    candidate = Path(path)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = Path(base_dir) / candidate

    try:
        if not candidate.exists():
            status = "not_found"
            return {
                "path": display_path,
                "status": status,
                "size_bytes": None,
                "sha256": None,
            }
        if not candidate.is_file():
            return {
                "path": display_path,
                "status": "not_a_file",
                "size_bytes": None,
                "sha256": None,
            }

        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": display_path,
            "status": "available",
            "size_bytes": candidate.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    except OSError as exc:
        return {
            "path": display_path,
            "status": "unreadable",
            "size_bytes": None,
            "sha256": None,
            "error_type": type(exc).__name__,
        }


def _git_provenance(repo_root: Path | None) -> dict[str, Any]:
    """Collect Git revision and dirty state without requiring Git metadata."""
    if repo_root is None:
        return {
            "status": "unavailable",
            "commit": None,
            "working_tree": "unknown",
            "reason": "repository_root_not_found",
        }

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
        return {
            "status": "available",
            "commit": commit,
            "working_tree": "dirty" if status_output.strip() else "clean",
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "unavailable",
            "commit": None,
            "working_tree": "unknown",
            "reason": type(exc).__name__,
        }


def _runtime_provenance() -> dict[str, Any]:
    """Record runtime versions when the corresponding distributions exist."""
    packages = ("numpy", "torch", "scanpy", "scikit-learn", "xgboost")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _build_provenance(
    *,
    bundle_dir: str | Path,
    checkpoint_path: str | Path,
    data_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    data_config: dict[str, Any],
    train_config: dict[str, Any],
    model_type: str,
    split_prefix: str,
    seed: int | None,
) -> dict[str, Any]:
    config_path = Path(data_config_path).resolve()
    repo_root = _find_repo_root(config_path.parent)
    # Configs conventionally live in <project-root>/configs/.
    project_root = repo_root or config_path.parent.parent
    split_config = data_config["split"]
    training_config = train_config.get("train", {})
    split_protocol_key = f"{split_prefix}_protocol"

    return {
        "schema_version": 1,
        # Keep timezone.utc for the declared Python 3.10 compatibility.
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),  # noqa: UP017
        "source": {"git": _git_provenance(repo_root)},
        "runtime": _runtime_provenance(),
        "experiment": {
            "model_type": model_type,
            "split_prefix": split_prefix,
            "split_protocol": split_config.get(split_protocol_key),
            "split_seed": split_config.get("random_seed"),
            "training_seed": training_config.get("seed"),
            "seed_override": seed,
        },
        "inputs": {
            "raw_dataset": _fingerprint_file(
                data_config.get("dataset", {}).get("raw_path"),
                base_dir=project_root,
            ),
            "processed_bundle": {
                name: _fingerprint_file(Path(bundle_dir) / name)
                for name in ("arrays.npz", "splits.npz", "metadata.json")
            },
            "configs": {
                "data": _fingerprint_file(data_config_path),
                "model": _fingerprint_file(model_config_path),
                "train": _fingerprint_file(train_config_path),
            },
        },
        "outputs": {
            "checkpoint": _fingerprint_file(checkpoint_path),
        },
    }


def load_json(path: str | Path) -> dict[str, Any] | list[Any]:
    """Load a JSON file into a Python object."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_history(
    history: list[dict[str, Any]],
    checkpoint_metric: str,
) -> dict[str, Any]:
    """Summarize the tracked validation history for a training run."""
    if not history:
        return {
            "history_length": 0,
            "best_epoch": None,
            "best_validation": {},
            "last_epoch": {},
        }

    best_record = max(history, key=lambda record: float(record.get(checkpoint_metric, float("-inf"))))
    return {
        "history_length": len(history),
        "best_epoch": int(best_record["epoch"]),
        "best_validation": best_record,
        "last_epoch": history[-1],
    }


def _load_bundle_overview(bundle_dir: str | Path) -> dict[str, Any]:
    bundle_path = Path(bundle_dir)
    raw_metadata = load_json(bundle_path / "metadata.json")
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"Expected metadata.json to be a dict, got {type(raw_metadata)}")
    metadata: dict[str, Any] = raw_metadata
    arrays = np.load(bundle_path / "arrays.npz", allow_pickle=False)
    splits = np.load(bundle_path / "splits.npz", allow_pickle=False)

    split_sizes = {
        split_name: int(split_indices.shape[0])
        for split_name, split_indices in splits.items()
    }

    return {
        "bundle_dir": str(bundle_path),
        "num_samples": int(arrays["perturbation_index"].shape[0]),
        "num_genes": int(len(metadata["gene_names"])),
        "num_perturbations": int(len(metadata["perturbation_names"])),
        "split_sizes": split_sizes,
    }


def build_run_summary(
    *,
    bundle_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_type: str,
    split_prefix: str,
    data_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    history_path: str | Path | None = None,
    seen_metrics_path: str | Path | None = None,
    unseen_metrics_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a structured local summary artifact for a completed training run."""
    data_config = load_yaml(data_config_path)
    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)
    if seed is not None:
        train_config.setdefault("train", {})["seed"] = int(seed)
    checkpoint_metric = str(train_config["train"]["checkpoint_metric"])

    history: list[dict[str, Any]] = []
    if history_path is not None and Path(history_path).exists():
        loaded_history = load_json(history_path)
        if isinstance(loaded_history, list):
            history = loaded_history

    metrics: dict[str, Any] = {}
    if seen_metrics_path is not None and Path(seen_metrics_path).exists():
        metrics["seen_test"] = load_json(seen_metrics_path)
    if unseen_metrics_path is not None and Path(unseen_metrics_path).exists():
        metrics["unseen_test"] = load_json(unseen_metrics_path)

    split_config = data_config["split"]
    split_protocol_key = f"{split_prefix}_protocol"

    return {
        "dataset": {
            "name": data_config["dataset"]["name"],
            "source": data_config["dataset"]["source"],
            "raw_path": data_config["dataset"]["raw_path"],
            "cell_context": data_config["dataset"]["cell_context"],
            "single_gene_only": data_config["dataset"]["include_single_gene_only"],
        },
        "preprocessing": data_config["preprocess"],
        "pairing": data_config["pairing"],
        "split": {
            "train_protocol": split_prefix,
            "protocol_name": split_config[split_protocol_key],
            "val_fraction": split_config["val_fraction"],
            "test_fraction": split_config["test_fraction"],
            "random_seed": split_config["random_seed"],
        },
        "model": {
            "model_type": model_type,
            **model_config,
        },
        "training": train_config["train"],
        "artifacts": {
            "bundle": _load_bundle_overview(bundle_dir),
            "output_dir": str(Path(output_dir)),
            "checkpoint_path": str(Path(checkpoint_path)),
            "history_path": str(Path(history_path)) if history_path is not None else None,
        },
        "validation": summarize_history(history, checkpoint_metric),
        "test_metrics": metrics,
        "provenance": _build_provenance(
            bundle_dir=bundle_dir,
            checkpoint_path=checkpoint_path,
            data_config_path=data_config_path,
            model_config_path=model_config_path,
            train_config_path=train_config_path,
            data_config=data_config,
            train_config=train_config,
            model_type=model_type,
            split_prefix=split_prefix,
            seed=seed,
        ),
    }


def write_run_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write a run summary artifact to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return destination
