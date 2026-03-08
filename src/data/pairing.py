from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

from src.data.io import read_json, write_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ProcessedBundle:
    control_expression: np.ndarray
    target_delta: np.ndarray
    perturbation_index: np.ndarray
    gene_names: list[str]
    perturbation_names: list[str]
    sample_ids: list[str]
    splits: dict[str, np.ndarray]


def _row_to_dense(matrix, row_index: int) -> np.ndarray:
    row = matrix[row_index]
    if sparse.issparse(row):
        row = row.toarray()
    return np.asarray(row).reshape(-1).astype(np.float32)


def _matrix_to_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _make_group_key(row: pd.Series, group_columns: list[str]) -> tuple[str, ...]:
    if not group_columns:
        return tuple()
    return tuple(str(row[column]) for column in group_columns)


def build_control_mean_lookup(
    adata,
    perturbation_col: str,
    control_label: str,
    batch_col: str | None = None,
    context_cols: list[str] | None = None,
) -> tuple[dict[tuple[str, ...], np.ndarray], np.ndarray]:
    """Build batch-aware control means with a global fallback."""
    context_cols = context_cols or []
    group_columns = [column for column in [batch_col, *context_cols] if column]

    control_mask = adata.obs[perturbation_col].astype(str) == control_label
    if int(control_mask.sum()) == 0:
        raise ValueError("No control samples found; cannot build control mean lookup.")

    control_subset = adata[control_mask].copy()
    global_control_mean = _matrix_to_dense(control_subset.X).mean(axis=0).astype(np.float32)

    lookup: dict[tuple[str, ...], np.ndarray] = {}
    if not group_columns:
        return lookup, global_control_mean

    for group_values, group_frame in control_subset.obs.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (str(group_values),)
        else:
            group_values = tuple(str(value) for value in group_values)
        group_indices = control_subset.obs.index.isin(group_frame.index)
        group_matrix = control_subset[group_indices].X
        lookup[group_values] = _matrix_to_dense(group_matrix).mean(axis=0).astype(np.float32)

    return lookup, global_control_mean


def _can_stratify(labels: np.ndarray) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return counts.min(initial=0) >= 2


def _three_way_stratified_split(
    labels: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    indices = np.arange(len(labels))
    temp_fraction = val_fraction + test_fraction
    if temp_fraction <= 0:
        return {
            "seen_train": indices,
            "seen_val": np.array([], dtype=np.int64),
            "seen_test": np.array([], dtype=np.int64),
        }

    stratify_labels = labels if _can_stratify(labels) else None
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_fraction,
        random_state=random_seed,
        stratify=stratify_labels,
    )

    if len(temp_idx) == 0:
        return {
            "seen_train": np.sort(train_idx.astype(np.int64)),
            "seen_val": np.array([], dtype=np.int64),
            "seen_test": np.array([], dtype=np.int64),
        }

    val_ratio_within_temp = val_fraction / temp_fraction if temp_fraction else 0.0
    temp_labels = labels[temp_idx]
    temp_stratify = temp_labels if _can_stratify(temp_labels) else None
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=max(0.0, 1.0 - val_ratio_within_temp),
        random_state=random_seed,
        stratify=temp_stratify,
    )

    return {
        "seen_train": np.sort(train_idx.astype(np.int64)),
        "seen_val": np.sort(val_idx.astype(np.int64)),
        "seen_test": np.sort(test_idx.astype(np.int64)),
    }


def _three_way_group_split(
    labels: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    unique_groups = np.unique(labels)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_test = int(round(n_groups * test_fraction))
    n_val = int(round(n_groups * val_fraction))
    if n_groups >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
    n_test = min(n_test, max(0, n_groups - 2))
    n_val = min(n_val, max(0, n_groups - n_test - 1))

    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test : n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val :])

    indices = np.arange(len(labels))
    return {
        "unseen_train": indices[np.isin(labels, list(train_groups))].astype(np.int64),
        "unseen_val": indices[np.isin(labels, list(val_groups))].astype(np.int64),
        "unseen_test": indices[np.isin(labels, list(test_groups))].astype(np.int64),
    }


def create_split_indices(
    perturbation_index: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Create both seen and unseen perturbation split indices."""
    seen = _three_way_stratified_split(
        labels=perturbation_index,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    unseen = _three_way_group_split(
        labels=perturbation_index,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    return {**seen, **unseen}


def build_training_bundle(
    adata,
    perturbation_col: str,
    control_label: str,
    batch_col: str | None = None,
    context_cols: list[str] | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    random_seed: int = 42,
) -> ProcessedBundle:
    """Create training arrays from preprocessed AnnData."""
    context_cols = context_cols or []
    required_columns = [perturbation_col, *[column for column in [batch_col, *context_cols] if column]]
    for column in required_columns:
        if column not in adata.obs.columns:
            raise ValueError(f"Missing required obs column: {column}")

    control_lookup, global_control_mean = build_control_mean_lookup(
        adata=adata,
        perturbation_col=perturbation_col,
        control_label=control_label,
        batch_col=batch_col,
        context_cols=context_cols,
    )
    group_columns = [column for column in [batch_col, *context_cols] if column]

    perturb_mask = adata.obs[perturbation_col].astype(str) != control_label
    perturbed = adata[perturb_mask].copy()
    if perturbed.n_obs == 0:
        raise ValueError("No perturbed samples found after filtering.")

    gene_names = perturbed.var_names.astype(str).tolist()
    perturbation_names = sorted(perturbed.obs[perturbation_col].astype(str).unique().tolist())
    perturb_to_idx = {name: idx for idx, name in enumerate(perturbation_names)}

    control_expression = np.zeros((perturbed.n_obs, perturbed.n_vars), dtype=np.float32)
    target_delta = np.zeros((perturbed.n_obs, perturbed.n_vars), dtype=np.float32)
    perturbation_index = np.zeros(perturbed.n_obs, dtype=np.int64)
    sample_ids: list[str] = []

    for row_idx in range(perturbed.n_obs):
        obs_row = perturbed.obs.iloc[row_idx]
        perturbation_name = str(obs_row[perturbation_col])
        group_key = _make_group_key(obs_row, group_columns)
        matched_control = control_lookup.get(group_key, global_control_mean)
        perturbed_expression = _row_to_dense(perturbed.X, row_idx)

        control_expression[row_idx] = matched_control
        target_delta[row_idx] = perturbed_expression - matched_control
        perturbation_index[row_idx] = perturb_to_idx[perturbation_name]
        sample_ids.append(str(perturbed.obs_names[row_idx]))

    splits = create_split_indices(
        perturbation_index=perturbation_index,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    LOGGER.info(
        "Built processed bundle with %s samples, %s genes, and %s perturbations.",
        len(sample_ids),
        len(gene_names),
        len(perturbation_names),
    )
    return ProcessedBundle(
        control_expression=control_expression,
        target_delta=target_delta,
        perturbation_index=perturbation_index,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        sample_ids=sample_ids,
        splits=splits,
    )


def save_processed_bundle(bundle: ProcessedBundle, output_dir: str | Path) -> None:
    """Save processed arrays and metadata for training."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        destination / "arrays.npz",
        control_expression=bundle.control_expression,
        target_delta=bundle.target_delta,
        perturbation_index=bundle.perturbation_index,
        sample_ids=np.asarray(bundle.sample_ids, dtype=np.str_),
    )
    np.savez_compressed(destination / "splits.npz", **bundle.splits)
    write_json(
        destination / "metadata.json",
        {
            "gene_names": bundle.gene_names,
            "perturbation_names": bundle.perturbation_names,
        },
    )


def load_processed_bundle(output_dir: str | Path) -> dict[str, Any]:
    """Load processed arrays and metadata from disk."""
    source = Path(output_dir)
    arrays = np.load(source / "arrays.npz", allow_pickle=False)
    splits = np.load(source / "splits.npz", allow_pickle=False)
    metadata = read_json(source / "metadata.json")
    return {
        "control_expression": arrays["control_expression"],
        "target_delta": arrays["target_delta"],
        "perturbation_index": arrays["perturbation_index"],
        "sample_ids": arrays["sample_ids"].tolist(),
        "splits": {key: splits[key] for key in splits.files},
        "metadata": metadata,
    }

