from __future__ import annotations

from dataclasses import dataclass, field
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
    split_diagnostics: dict[str, Any] = field(default_factory=dict)
    preprocessing_diagnostics: dict[str, Any] = field(default_factory=dict)


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
        return ()
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


def _validate_split_fractions(val_fraction: float, test_fraction: float) -> None:
    """Validate fractions before any split helper calls scikit-learn."""
    for name, value in (("val_fraction", val_fraction), ("test_fraction", test_fraction)):
        if not np.isfinite(value) or value < 0.0 or value >= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1); got {value!r}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            "val_fraction + test_fraction must be less than 1.0; "
            f"got {val_fraction + test_fraction!r}"
        )


def _stratification_fallback_reason(
    labels: np.ndarray,
    *,
    train_size: int,
    test_size: int,
) -> str | None:
    _, counts = np.unique(labels, return_counts=True)
    if counts.size == 0:
        return "no labels are available"
    if int(counts.min()) < 2:
        return "at least one class has fewer than two samples"
    num_classes = int(counts.size)
    if train_size < num_classes:
        return f"train partition size {train_size} is smaller than {num_classes} classes"
    if test_size < num_classes:
        return f"test partition size {test_size} is smaller than {num_classes} classes"
    return None


def _can_stratify(
    labels: np.ndarray,
    *,
    train_size: int | None = None,
    test_size: int | None = None,
) -> bool:
    """Return whether every class can be represented in both split partitions."""
    if train_size is None or test_size is None:
        _, counts = np.unique(labels, return_counts=True)
        return counts.size > 0 and int(counts.min()) >= 2
    return _stratification_fallback_reason(
        labels,
        train_size=train_size,
        test_size=test_size,
    ) is None


def _three_way_stratified_split(
    labels: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    _validate_split_fractions(val_fraction, test_fraction)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Seen split requires a non-empty one-dimensional labels array.")

    indices = np.arange(len(labels), dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    if val_fraction == 0.0 and test_fraction == 0.0:
        return {"seen_train": indices, "seen_val": empty, "seen_test": empty}

    if val_fraction == 0.0 or test_fraction == 0.0:
        heldout_fraction = test_fraction if val_fraction == 0.0 else val_fraction
        heldout_size = int(np.ceil(len(indices) * heldout_fraction))
        train_size = len(indices) - heldout_size
        if train_size < 1 or heldout_size < 1:
            raise ValueError(
                "Seen split needs at least one training sample and one held-out sample; "
                f"got {len(indices)} samples at fraction {heldout_fraction}."
            )
        reason = _stratification_fallback_reason(
            labels,
            train_size=train_size,
            test_size=heldout_size,
        )
        stratify_labels = labels if reason is None else None
        if reason is not None:
            LOGGER.warning("Seen split uses unstratified sampling: %s.", reason)
        train_idx, heldout_idx = train_test_split(
            indices,
            test_size=heldout_size,
            random_state=random_seed,
            stratify=stratify_labels,
        )
        return {
            "seen_train": np.sort(train_idx.astype(np.int64)),
            "seen_val": np.sort(heldout_idx.astype(np.int64)) if val_fraction > 0 else empty,
            "seen_test": np.sort(heldout_idx.astype(np.int64)) if test_fraction > 0 else empty,
        }

    temp_fraction = val_fraction + test_fraction
    temp_size = int(np.ceil(len(indices) * temp_fraction))
    train_size = len(indices) - temp_size
    if train_size < 1 or temp_size < 2:
        raise ValueError(
            "Seen split needs at least one training sample and two held-out samples "
            "when both validation and test fractions are positive; "
            f"got {len(indices)} samples with fractions {val_fraction} and {test_fraction}."
        )

    reason = _stratification_fallback_reason(
        labels,
        train_size=train_size,
        test_size=temp_size,
    )
    stratify_labels = labels if reason is None else None
    if reason is not None:
        LOGGER.warning("Seen split uses unstratified first-stage sampling: %s.", reason)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_size,
        random_state=random_seed,
        stratify=stratify_labels,
    )

    temp_labels = labels[temp_idx]
    temp_test_size = int(np.ceil(len(temp_idx) * test_fraction / temp_fraction))
    temp_test_size = min(max(1, temp_test_size), len(temp_idx) - 1)
    temp_val_size = len(temp_idx) - temp_test_size
    reason = _stratification_fallback_reason(
        temp_labels,
        train_size=temp_val_size,
        test_size=temp_test_size,
    )
    temp_stratify = temp_labels if reason is None else None
    if reason is not None:
        LOGGER.warning("Seen split uses unstratified second-stage sampling: %s.", reason)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=temp_test_size,
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
    _validate_split_fractions(val_fraction, test_fraction)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Unseen split requires a non-empty one-dimensional labels array.")

    indices = np.arange(len(labels), dtype=np.int64)
    if val_fraction == 0.0 and test_fraction == 0.0:
        return {
            "unseen_train": indices,
            "unseen_val": np.array([], dtype=np.int64),
            "unseen_test": np.array([], dtype=np.int64),
        }

    unique_groups = np.unique(labels)
    n_groups = len(unique_groups)
    required_groups = 1 + int(val_fraction > 0.0) + int(test_fraction > 0.0)
    if n_groups < required_groups:
        raise ValueError(
            "Unseen split needs at least "
            f"{required_groups} distinct perturbation groups for the requested non-empty "
            f"partitions; got {n_groups}."
        )

    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_groups)

    n_test = 0
    if test_fraction > 0.0:
        max_test_groups = n_groups - 1 - int(val_fraction > 0.0)
        n_test = min(max(1, int(round(n_groups * test_fraction))), max_test_groups)

    n_val = 0
    if val_fraction > 0.0:
        max_val_groups = n_groups - n_test - 1
        n_val = min(max(1, int(round(n_groups * val_fraction))), max_val_groups)

    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test : n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val :])

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
    """Create sample-level seen and perturbation-group unseen split indices."""
    _validate_split_fractions(val_fraction, test_fraction)
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


def _summarize_split_integrity(
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Summarize sample coverage and perturbation-group separation."""
    labels = np.asarray(labels)
    sample_count = int(len(labels))
    split_names = ("train", "val", "test")
    sample_counts: dict[str, int] = {}
    perturbation_group_counts: dict[str, int | None] = {}
    integrity: dict[str, dict[str, Any]] = {}

    for prefix in ("seen", "unseen"):
        arrays: dict[str, np.ndarray] = {}
        present = True
        for split_name in split_names:
            key = f"{prefix}_{split_name}"
            if key not in splits:
                present = False
                indices = np.array([], dtype=np.int64)
            else:
                indices = np.asarray(splits[key], dtype=np.int64).reshape(-1)
            arrays[split_name] = indices
            sample_counts[key] = int(len(indices))

            in_range = bool(np.all((indices >= 0) & (indices < sample_count)))
            perturbation_group_counts[key] = (
                int(len(np.unique(labels[indices]))) if in_range else None
            )

        combined = np.concatenate([arrays[name] for name in split_names])
        in_range = bool(np.all((combined >= 0) & (combined < sample_count)))
        pairwise_disjoint = bool(len(np.unique(combined)) == len(combined))
        coverage_complete = bool(
            in_range
            and np.array_equal(np.sort(combined), np.arange(sample_count, dtype=np.int64))
        )
        group_disjoint: bool | None = None
        if prefix == "unseen" and present and in_range:
            group_sets = {
                name: set(np.unique(labels[arrays[name]]).tolist())
                for name in split_names
            }
            group_disjoint = bool(
                pairwise_disjoint
                and coverage_complete
                and group_sets["train"].isdisjoint(group_sets["val"])
                and group_sets["train"].isdisjoint(group_sets["test"])
                and group_sets["val"].isdisjoint(group_sets["test"])
            )
        integrity[prefix] = {
            "split_indices_present": present,
            "indices_in_range": in_range,
            "pairwise_disjoint": pairwise_disjoint,
            "complete_sample_coverage": coverage_complete,
            "perturbation_groups_disjoint": group_disjoint,
        }

    return {
        "schema_version": 1,
        "sample_count": sample_count,
        "sample_counts": sample_counts,
        "perturbation_group_counts": perturbation_group_counts,
        "integrity": integrity,
    }


def _summarize_split_diagnostics(
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    val_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    """Persist split integrity plus the reason stratification was or was not used."""
    diagnostics = _summarize_split_integrity(labels, splits)
    seen_train = np.asarray(splits["seen_train"], dtype=np.int64)
    seen_val = np.asarray(splits["seen_val"], dtype=np.int64)
    seen_test = np.asarray(splits["seen_test"], dtype=np.int64)
    labels = np.asarray(labels)

    if val_fraction == 0.0 and test_fraction == 0.0:
        stratification = {
            "status": "not_requested",
            "fallback_stage": None,
            "fallback_reason": None,
        }
    elif not diagnostics["integrity"]["seen"]["indices_in_range"]:
        stratification = {
            "status": "unavailable",
            "fallback_stage": None,
            "fallback_reason": "split indices are out of range",
        }
    elif val_fraction == 0.0 or test_fraction == 0.0:
        heldout = seen_test if test_fraction > 0.0 else seen_val
        reason = _stratification_fallback_reason(
            labels,
            train_size=len(seen_train),
            test_size=len(heldout),
        )
        stratification = {
            "status": "stratified" if reason is None else "unstratified_fallback",
            "fallback_stage": "single_stage" if reason is not None else None,
            "fallback_reason": reason,
        }
    else:
        temp_indices = np.concatenate([seen_val, seen_test])
        first_reason = _stratification_fallback_reason(
            labels,
            train_size=len(seen_train),
            test_size=len(temp_indices),
        )
        if first_reason is not None:
            stratification = {
                "status": "unstratified_fallback",
                "fallback_stage": "train_vs_holdout",
                "fallback_reason": first_reason,
            }
        else:
            second_reason = _stratification_fallback_reason(
                labels[temp_indices],
                train_size=len(seen_val),
                test_size=len(seen_test),
            )
            stratification = {
                "status": (
                    "stratified" if second_reason is None else "partially_stratified_fallback"
                ),
                "fallback_stage": "validation_vs_test" if second_reason is not None else None,
                "fallback_reason": second_reason,
            }

    unseen_integrity = diagnostics["integrity"]["unseen"]
    unseen_group_split = {
        "status": (
            "grouped"
            if (
                unseen_integrity["split_indices_present"]
                and unseen_integrity["indices_in_range"]
                and unseen_integrity["pairwise_disjoint"]
                and unseen_integrity["complete_sample_coverage"]
                and unseen_integrity["perturbation_groups_disjoint"]
            )
            else "invalid_or_unavailable"
        ),
        "fallback_reason": None
        if unseen_integrity["perturbation_groups_disjoint"]
        else "perturbation groups overlap or indices are out of range",
    }
    diagnostics["seen_stratification"] = stratification
    diagnostics["unseen_group_split"] = unseen_group_split
    return diagnostics


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
    split_diagnostics = _summarize_split_diagnostics(
        labels=perturbation_index,
        splits=splits,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    preprocessing_diagnostics = dict(
        getattr(adata, "uns", {}).get("scgpt_preprocessing", {})
    )
    if not preprocessing_diagnostics:
        preprocessing_diagnostics = {
            "schema_version": 1,
            "status": "not_recorded_legacy_or_external",
            "hvg_selection_method": "unknown",
        }
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
        split_diagnostics=split_diagnostics,
        preprocessing_diagnostics=preprocessing_diagnostics,
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
    np.savez_compressed(destination / "splits.npz", **bundle.splits)  # type: ignore[arg-type]
    write_json(
        destination / "metadata.json",
        {
            "gene_names": bundle.gene_names,
            "perturbation_names": bundle.perturbation_names,
            "split_diagnostics": bundle.split_diagnostics
            or {
                "schema_version": 1,
                "status": "not_recorded",
                **_summarize_split_integrity(
                    bundle.perturbation_index,
                    bundle.splits,
                ),
            },
            "preprocessing_diagnostics": bundle.preprocessing_diagnostics
            or {
                "schema_version": 1,
                "status": "not_recorded",
                "hvg_selection_method": "unknown",
            },
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

