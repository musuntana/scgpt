from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import load_yaml


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
) -> dict[str, Any]:
    """Build a structured local summary artifact for a completed training run."""
    data_config = load_yaml(data_config_path)
    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)
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
    }


def write_run_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write a run summary artifact to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return destination
