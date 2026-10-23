from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import sparse

from src.evaluation.metrics import mean_squared_error, pearson_correlation
from src.integrations.gears_contract import PARTITIONS, _as_1d_int_array


def attach_train_only_loss_metadata(adata: Any) -> dict[str, Any]:
    """Attach GEARS' required loss masks without computing expression-derived DE.

    The upstream GEARS trainer assumes condition-level DE/dropout metadata.
    This adapter deliberately supplies only all-gene loss masks and leaves
    DE-ranking metadata absent. The custom trainer must evaluate all genes
    directly and must not call GEARS.train(), which labels its metrics as DE.
    """
    required_obs = {"condition", "cell_type", "scgpt_split"}
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    if missing_obs:
        raise ValueError(f"GEARS input is missing obs columns: {missing_obs}.")
    if "gene_name" not in adata.var:
        raise ValueError("GEARS input is missing var['gene_name'].")
    if not adata.var_names.is_unique or not adata.var["gene_name"].is_unique:
        raise ValueError("GEARS input gene IDs and gene symbols must both be unique.")
    if adata.n_vars < 1 or adata.n_obs < 1:
        raise ValueError("GEARS input must contain at least one cell and one gene.")

    export_info = getattr(adata, "uns", {}).get("scgpt_gears_export", {})
    if (
        export_info.get("status") != "INPUT_ONLY_NO_MODEL_RUN"
        or export_info.get("de_statistics_computed") is not False
    ):
        raise ValueError(
            "Expected a pristine INPUT_ONLY_NO_MODEL_RUN AnnData export without DE metadata."
        )

    conditions = adata.obs["condition"].astype(str).to_numpy()
    cell_types = adata.obs["cell_type"].astype(str).to_numpy()
    splits = adata.obs["scgpt_split"].astype(str).to_numpy()
    if set(splits.tolist()) - set(PARTITIONS):
        raise ValueError("scgpt_split must contain only train, val and test.")
    if len(set(cell_types.tolist())) != 1:
        raise ValueError("GEARS runtime supports one cell type per run.")
    expression_matrix = adata.X
    finite_values = (
        expression_matrix.data
        if sparse.issparse(expression_matrix)
        else np.asarray(expression_matrix)
    )
    if not np.all(np.isfinite(finite_values)):
        raise ValueError("GEARS expression matrix contains non-finite values.")

    if not np.any((conditions == "ctrl") & (splits == "train")):
        raise ValueError("GEARS training partition must contain control cells.")
    if np.any((conditions == "ctrl") & (splits != "train")):
        raise ValueError("Control cells must remain in train; controls cannot enter val/test.")

    condition_to_split: dict[str, str] = {}
    for condition, split in zip(conditions, splits, strict=True):
        previous = condition_to_split.setdefault(condition, split)
        if previous != split:
            raise ValueError(
                f"Condition {condition!r} appears in both {previous!r} and {split!r}."
            )
    if not {"train", "val", "test"}.issubset(set(condition_to_split.values())):
        raise ValueError("GEARS runtime requires non-empty train, val and test partitions.")

    # Match GEARS data_utils.get_DE_genes() condition_name encoding, but do not
    # compute any condition-specific DE statistics from held-out expression.
    condition_name_to_condition: dict[str, str] = {}
    for cell_type, condition in zip(cell_types, conditions, strict=True):
        dose = "1+1" if len(condition.split("+")) == 2 else "1"
        condition_name = f"{cell_type}_{condition}_{dose}"
        existing = condition_name_to_condition.setdefault(condition_name, condition)
        if existing != condition:
            raise ValueError(f"GEARS condition-name collision for {condition_name!r}.")

    all_gene_indices = np.arange(adata.n_vars, dtype=np.int64)
    adata.uns["non_zeros_gene_idx"] = {
        condition_name: all_gene_indices.copy()
        for condition_name in condition_name_to_condition
    }
    # Fail closed if any caller tries to reuse DE statistics from another path.
    adata.uns.pop("rank_genes_groups_cov_all", None)
    adata.uns.pop("top_non_dropout_de_20", None)
    adata.uns.pop("top_non_zero_de_20", None)
    adata.uns.pop("non_dropout_gene_idx", None)

    export_info = dict(export_info)
    export_info["status"] = "TRAIN_ONLY_RUNTIME_READY"
    export_info["de_statistics_computed"] = False
    export_info["runtime_metadata"] = {
        "method": "all_gene_loss_mask_no_DE_or_dropout_statistics",
        "condition_name_count": len(condition_name_to_condition),
        "gene_count": int(adata.n_vars),
        "heldout_expression_used_for_metadata": False,
        "heldout_condition_DE_ranking": False,
        "expression_dependent_gene_mask": False,
        "evaluation_required": "custom all-gene absolute-expression metrics",
        "must_not_call_official_GEARS_train": True,
    }
    adata.uns["scgpt_gears_export"] = export_info

    return {
        "status": "TRAIN_ONLY_RUNTIME_READY",
        "condition_name_count": len(condition_name_to_condition),
        "gene_count": int(adata.n_vars),
        "training_cell_count": int(np.sum(splits == "train")),
        "validation_cell_count": int(np.sum(splits == "val")),
        "test_cell_count": int(np.sum(splits == "test")),
        "heldout_expression_used_for_metadata": False,
        "de_statistics_computed": False,
        "all_gene_loss_mask": True,
    }


def validate_gears_graph_coverage(
    perturbation_names: list[str],
    gears_perturbation_names: set[str] | list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Fail closed when GEARS' GO graph cannot represent any held-out target."""
    available = {str(name) for name in gears_perturbation_names}
    required = set(perturbation_names)
    missing = sorted(required - available)
    return {
        "verified": not missing,
        "required_perturbation_count": len(required),
        "available_perturbation_count": len(available),
        "missing_perturbations": missing,
    }


def validate_gears_gene_order(
    bundle_gene_ids: list[str],
    input_gene_ids: list[str],
    input_gene_symbols: list[str],
    gears_gene_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Verify the ordered gene identity contract used by GEARS and the bundle."""
    def normalize(values: list[str], label: str) -> list[str]:
        normalized = [str(value).strip() for value in values]
        if not normalized:
            raise ValueError(f"{label} must not be empty.")
        if any(value.lower() in {"", "nan", "none"} for value in normalized):
            raise ValueError(f"{label} contains an empty or invalid gene identifier.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{label} must contain unique values.")
        return normalized

    bundle_ids = normalize(bundle_gene_ids, "Bundle gene IDs")
    input_ids = normalize(input_gene_ids, "GEARS input gene IDs")
    input_symbols = normalize(input_gene_symbols, "GEARS input gene symbols")
    if bundle_ids != input_ids:
        raise ValueError("GEARS input gene ID order differs from the processed bundle.")
    if len(input_symbols) != len(bundle_ids):
        raise ValueError("GEARS gene-symbol count differs from the processed bundle.")

    gears_symbols: list[str] | None = None
    if gears_gene_symbols is not None:
        gears_symbols = normalize(gears_gene_symbols, "GEARS runtime gene symbols")
        if gears_symbols != input_symbols:
            raise ValueError(
                "GEARS runtime gene-symbol order differs from the input AnnData."
            )

    def digest(values: list[str]) -> str:
        payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "verified": gears_symbols is not None,
        "bundle_input_gene_order_verified": True,
        "gears_symbol_order_verified": gears_symbols is not None,
        "gene_count": len(bundle_ids),
        "bundle_gene_ids_sha256": digest(bundle_ids),
        "input_gene_symbols_sha256": digest(input_symbols),
        "gears_gene_symbols_sha256": (
            digest(gears_symbols) if gears_symbols is not None else None
        ),
    }


def compute_control_relative_delta_metrics(
    predictions: Mapping[str, np.ndarray],
    bundle: Mapping[str, Any],
    *,
    control_reference_mean: np.ndarray,
    split_name: str = "unseen_test",
) -> dict[str, Any]:
    """Convert GEARS absolute predictions to the bundle's control-relative delta space.

    GEARS predicts expression by applying the perturbation to its control cells.
    Subtracting the mean expression of those exact control cells yields a
    perturbation delta. Targets are the bundle's matched-control deltas, averaged
    over each held-out perturbation group. Only per-perturbation MSE and Pearson
    are marked comparable to the repository's corresponding aggregated metrics.
    """
    splits = bundle.get("splits")
    if not isinstance(splits, Mapping) or split_name not in splits:
        raise ValueError(f"Processed bundle has no split {split_name!r}.")
    metadata = bundle.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Processed bundle metadata is missing.")
    names = metadata.get("perturbation_names")
    if not isinstance(names, list) or not names:
        raise ValueError("Bundle perturbation_names metadata is missing.")

    labels = _as_1d_int_array(bundle.get("perturbation_index"), "perturbation_index")
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= len(names)):
        raise ValueError(
            "Perturbation indices are empty or outside perturbation_names range."
        )
    indices = _as_1d_int_array(splits[split_name], split_name)
    if indices.size == 0:
        raise ValueError(f"{split_name} has no samples to evaluate.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{split_name} contains duplicate sample indices.")
    if np.any(indices < 0) or np.any(indices >= labels.size):
        raise ValueError(f"{split_name} contains out-of-range sample indices.")

    if bundle.get("target_delta") is None:
        raise ValueError("Bundle target_delta array is missing.")
    target_delta = np.asarray(bundle["target_delta"], dtype=np.float64)
    if (
        target_delta.ndim != 2
        or target_delta.shape[0] != labels.size
        or target_delta.shape[1] < 1
    ):
        raise ValueError("Bundle target_delta shape is inconsistent or has no genes.")
    if not np.all(np.isfinite(target_delta)):
        raise ValueError("Bundle target_delta contains non-finite values.")

    baseline = np.asarray(control_reference_mean, dtype=np.float64)
    if baseline.ndim != 1 or baseline.shape != (target_delta.shape[1],):
        raise ValueError(
            "GEARS control reference must be a one-dimensional vector in bundle gene order."
        )
    if not np.all(np.isfinite(baseline)):
        raise ValueError("GEARS control reference contains non-finite values.")

    per_perturbation: dict[str, dict[str, float | int]] = {}
    matched_control_mean_hashes: dict[str, str] = {}
    for perturbation_index in sorted(set(labels[indices].tolist())):
        perturbation = names[int(perturbation_index)]
        if perturbation not in predictions:
            raise ValueError(
                f"Missing GEARS prediction for held-out perturbation {perturbation!r}."
            )
        predicted_absolute = np.asarray(predictions[perturbation], dtype=np.float64)
        if predicted_absolute.ndim != 1 or predicted_absolute.shape != baseline.shape:
            raise ValueError(
                f"Prediction for {perturbation!r} must be a one-dimensional vector "
                f"with {baseline.size} genes; got shape {predicted_absolute.shape}."
            )
        if not np.all(np.isfinite(predicted_absolute)):
            raise ValueError(
                f"Prediction for {perturbation!r} contains non-finite values."
            )

        group_indices = indices[labels[indices] == perturbation_index]
        predicted_delta = predicted_absolute - baseline
        target = target_delta[group_indices].mean(axis=0, dtype=np.float64)
        per_perturbation[perturbation] = {
            "sample_count": int(group_indices.size),
            "mse_delta_expression": mean_squared_error(predicted_delta, target),
            "pearson_delta_expression": pearson_correlation(predicted_delta, target),
        }

    if not per_perturbation:
        raise ValueError(f"{split_name} has no perturbations to evaluate.")
    return {
        "split": split_name,
        "metric_space": "control_relative_delta_expression",
        "prediction_conversion": "absolute_prediction_minus_gears_control_mean",
        "target_source": "bundle.target_delta",
        "target_aggregation": "mean_by_perturbation_over_selected_split",
        "directly_comparable_to_repository_delta_metrics": True,
        "comparable_metric_keys": ["mse_per_perturbation", "pearson_per_perturbation"],
        "non_comparable_metric_keys": ["overall_mse", "pearson_per_gene"],
        "perturbation_count": len(per_perturbation),
        "mse_per_perturbation": float(
            np.mean([row["mse_delta_expression"] for row in per_perturbation.values()])
        ),
        "pearson_per_perturbation": float(
            np.mean([row["pearson_delta_expression"] for row in per_perturbation.values()])
        ),
        "per_perturbation": per_perturbation,
        "matched_control_mean_sha256": matched_control_mean_hashes,
        "comparison_eligible": False,
        "interpretation": (
            "Target-aligned diagnostic metrics; multi-seed and real-data review are "
            "still required before a model-performance claim."
        ),
    }


def compute_absolute_expression_metrics(
    predictions: Mapping[str, np.ndarray],
    bundle: Mapping[str, Any],
    *,
    split_name: str = "unseen_test",
) -> dict[str, Any]:
    """Evaluate GEARS' absolute-expression predictions, explicitly not delta metrics."""
    if split_name not in bundle.get("splits", {}):
        raise ValueError(f"Processed bundle has no split {split_name!r}.")
    names = bundle["metadata"].get("perturbation_names")
    if not isinstance(names, list) or not names:
        raise ValueError("Bundle perturbation_names metadata is missing.")
    labels = _as_1d_int_array(bundle.get("perturbation_index"), "perturbation_index")
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= len(names)):
        raise ValueError(
            "Perturbation indices are empty or outside perturbation_names range."
        )
    indices = _as_1d_int_array(bundle["splits"][split_name], split_name)
    if indices.size == 0:
        raise ValueError(f"{split_name} has no samples to evaluate.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{split_name} contains duplicate sample indices.")
    control = np.asarray(bundle["control_expression"], dtype=np.float32)
    delta = np.asarray(bundle["target_delta"], dtype=np.float32)
    if (
        control.ndim != 2
        or control.shape != delta.shape
        or control.shape[0] != labels.size
    ):
        raise ValueError("Bundle expression arrays and perturbation indices are inconsistent.")
    if control.shape[1] < 1:
        raise ValueError("Bundle expression arrays must contain at least one gene.")
    if not np.all(np.isfinite(control)) or not np.all(np.isfinite(delta)):
        raise ValueError("Bundle control/delta expression arrays contain non-finite values.")
    if np.any(indices < 0) or np.any(indices >= labels.size):
        raise ValueError(f"{split_name} contains out-of-range sample indices.")
    targets = control + delta
    if not np.all(np.isfinite(targets)):
        raise ValueError("Absolute-expression targets contain non-finite values.")

    per_perturbation: dict[str, dict[str, float | int]] = {}
    for perturbation_index in sorted(set(labels[indices].tolist())):
        perturbation = names[int(perturbation_index)]
        if perturbation not in predictions:
            raise ValueError(f"Missing GEARS prediction for held-out perturbation {perturbation!r}.")
        predicted = np.asarray(predictions[perturbation], dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(predicted)):
            raise ValueError(
                f"GEARS prediction for {perturbation!r} contains non-finite values."
            )
        group_indices = indices[labels[indices] == perturbation_index]
        # Accumulate in float64 to avoid overflow when averaging finite float32 values.
        target = targets[group_indices].mean(axis=0, dtype=np.float64)
        if predicted.shape != target.shape:
            raise ValueError(
                f"Prediction for {perturbation!r} has {predicted.size} genes; "
                f"expected {target.size}."
            )
        per_perturbation[perturbation] = {
            "sample_count": int(group_indices.size),
            "mse_absolute_expression": mean_squared_error(predicted, target),
            "pearson_absolute_expression": pearson_correlation(predicted, target),
        }

    if not per_perturbation:
        raise ValueError(f"{split_name} has no perturbations to evaluate.")
    return {
        "split": split_name,
        "metric_space": "absolute_expression",
        "target_is_delta_expression": False,
        "directly_comparable_to_repository_delta_metrics": False,
        "perturbation_count": len(per_perturbation),
        "mse_absolute_expression": float(
            np.mean([row["mse_absolute_expression"] for row in per_perturbation.values()])
        ),
        "pearson_absolute_expression": float(
            np.mean([row["pearson_absolute_expression"] for row in per_perturbation.values()])
        ),
        "per_perturbation": per_perturbation,
        "interpretation": (
            "Diagnostic absolute-expression baseline only. This does not establish "
            "equivalence with batch/context-aware control-relative delta metrics."
        ),
    }
