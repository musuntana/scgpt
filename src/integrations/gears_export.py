from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import sparse

from src.integrations.gears_contract import (
    EXPECTED_HVG_METHOD,
    PARTITIONS,
    _as_1d_int_array,
    _condition_for_gene,
    build_gears_split_contract,
)


def _dense_float32(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def validate_rebuilt_bundle(
    rebuilt: Any,
    saved: Mapping[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Require raw-source preprocessing to reproduce the saved bundle exactly enough."""
    comparisons = (
        ("sample_ids", list(rebuilt.sample_ids), list(saved["sample_ids"])),
        ("gene_names", list(rebuilt.gene_names), list(saved["metadata"]["gene_names"])),
        (
            "perturbation_names",
            list(rebuilt.perturbation_names),
            list(saved["metadata"]["perturbation_names"]),
        ),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(
                f"Reprocessed source does not reproduce bundle {name}; "
                "check source H5AD, config, schema overrides and preprocessing version."
            )

    if not np.array_equal(rebuilt.perturbation_index, saved["perturbation_index"]):
        raise ValueError(
            "Reprocessed source does not reproduce bundle perturbation_index; "
            "refusing to export a mismatched GEARS input."
        )
    for name in ("control_expression", "target_delta"):
        actual_array = np.asarray(getattr(rebuilt, name))
        expected_array = np.asarray(saved[name])
        if actual_array.shape != expected_array.shape or not np.allclose(
            actual_array, expected_array, rtol=rtol, atol=atol, equal_nan=False
        ):
            max_abs = (
                float(np.max(np.abs(actual_array - expected_array)))
                if actual_array.shape == expected_array.shape and actual_array.size
                else None
            )
            raise ValueError(
                f"Reprocessed source does not reproduce bundle {name} "
                f"(max absolute difference {max_abs}); refusing export."
            )

    rebuilt_splits = rebuilt.splits
    saved_splits = saved["splits"]
    split_names = (
        "seen_train", "seen_val", "seen_test",
        "unseen_train", "unseen_val", "unseen_test",
    )
    for name in split_names:
        if name not in saved_splits or not np.array_equal(
            np.asarray(rebuilt_splits[name]), np.asarray(saved_splits[name])
        ):
            raise ValueError(
                f"Reprocessed source does not reproduce bundle split {name}; "
                "refusing to export a mismatched GEARS input."
            )

    return {
        "verified": True,
        "sample_ids_match": True,
        "perturbation_index_match": True,
        "gene_order_match": True,
        "split_indices_match": True,
        "control_expression_match": True,
        "target_delta_match": True,
        "expression_comparison_rtol": rtol,
        "expression_comparison_atol": atol,
    }


def build_gears_input_adata(
    prepared_adata: Any,
    bundle: Mapping[str, Any],
    *,
    perturbation_col: str,
    control_label: str,
    context_cols: list[str],
) -> tuple[Any, dict[str, Any]]:
    """Build a GEARS-schema AnnData preserving processed cells and split identities.

    Only the bundle's perturbation cells and the filtered control reference are
    included. No DE statistics are computed here; a leakage-audited GEARS
    runtime must be used for any subsequent training.
    """
    if perturbation_col not in prepared_adata.obs:
        raise ValueError(f"Prepared AnnData has no {perturbation_col!r} column.")
    if not prepared_adata.obs_names.is_unique:
        raise ValueError("Prepared AnnData sample IDs must be unique.")
    if not prepared_adata.var_names.is_unique:
        raise ValueError("Prepared AnnData gene IDs must be unique.")

    preprocessing = getattr(prepared_adata, "uns", {}).get("scgpt_preprocessing", {})
    if (
        preprocessing.get("hvg_selection_method") != EXPECTED_HVG_METHOD
        or preprocessing.get("heldout_perturbation_samples_excluded_from_hvg_fit") is not True
        or not {"seen", "unseen"}.issubset(set(preprocessing.get("protocols_protected", [])))
    ):
        raise ValueError(
            "Prepared AnnData must carry the current training-only gene-filter/HVG provenance."
        )

    if len(context_cols) != 1:
        raise ValueError(
            "GEARS baseline requires exactly one verified cell-context column; "
            f"got {context_cols!r}."
        )
    context_col = context_cols[0]
    if context_col not in prepared_adata.obs:
        raise ValueError(f"Prepared AnnData has no cell-context column {context_col!r}.")
    if "gene_name" not in prepared_adata.var:
        raise ValueError(
            "GEARS requires var['gene_name']; provide a verified gene-symbol mapping."
        )

    gene_symbols = prepared_adata.var["gene_name"]
    if gene_symbols.isna().any():
        raise ValueError("GEARS gene_name annotation contains missing values.")
    gene_symbols = gene_symbols.astype(str)
    if any(not value or value.lower() == "nan" for value in gene_symbols.tolist()):
        raise ValueError("GEARS gene_name annotation contains empty or invalid values.")
    if not gene_symbols.is_unique:
        raise ValueError("GEARS gene_name annotation must be unique.")
    if prepared_adata.obs[context_col].isna().any():
        raise ValueError("Cell-context metadata contains missing values.")
    cell_types = prepared_adata.obs[context_col].astype(str).unique().tolist()
    if len(cell_types) != 1:
        raise ValueError(
            "Official GEARS does not support cross-cell-type transfer; expected one "
            f"cell type, found {cell_types!r}."
        )

    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Processed bundle metadata is missing.")
    if list(prepared_adata.var_names.astype(str)) != metadata.get("gene_names"):
        raise ValueError("Prepared AnnData gene order differs from the processed bundle.")

    names = metadata.get("perturbation_names")
    sample_ids = bundle.get("sample_ids")
    labels = _as_1d_int_array(bundle.get("perturbation_index"), "perturbation_index")
    if not isinstance(names, list) or not isinstance(sample_ids, list):
        raise ValueError("Bundle perturbation names and sample IDs must be lists.")
    if labels.size != len(sample_ids) or labels.size == 0:
        raise ValueError("Bundle perturbation indices and sample IDs are inconsistent.")
    if np.any(labels < 0) or np.any(labels >= len(names)):
        raise ValueError("Bundle perturbation index is outside the perturbation-name list.")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Bundle sample IDs must be unique.")

    split_conditions, split_manifest = build_gears_split_contract(bundle)
    obs_ids = prepared_adata.obs_names.astype(str).tolist()
    obs_id_set = set(obs_ids)
    missing_ids = sorted(set(sample_ids) - obs_id_set)
    if missing_ids:
        raise ValueError(
            f"Prepared AnnData is missing {len(missing_ids)} bundle sample IDs "
            f"(example {missing_ids[0]!r})."
        )

    source_labels = prepared_adata.obs.loc[sample_ids, perturbation_col].astype(str).tolist()
    expected_labels = [names[int(index)] for index in labels]
    if source_labels != expected_labels:
        mismatch_index = next(
            index
            for index, (actual, expected) in enumerate(
                zip(source_labels, expected_labels, strict=True)
            )
            if actual != expected
        )
        raise ValueError(
            "Prepared AnnData perturbation labels differ from the bundle "
            f"at sample {sample_ids[mismatch_index]!r}."
        )

    all_obs_labels = prepared_adata.obs[perturbation_col].astype(str).tolist()
    source_perturbed_ids = {
        obs_id
        for obs_id, label in zip(obs_ids, all_obs_labels, strict=True)
        if label != control_label
    }
    if source_perturbed_ids != set(sample_ids):
        raise ValueError(
            "Prepared AnnData contains perturbation cells outside the saved bundle "
            "or the bundle omits processed perturbation cells."
        )

    expected_expression = _dense_float32(bundle["control_expression"]) + _dense_float32(
        bundle["target_delta"]
    )
    actual_expression = _dense_float32(prepared_adata[sample_ids].X)
    if expected_expression.shape != actual_expression.shape or not np.allclose(
        expected_expression, actual_expression, rtol=1e-5, atol=1e-6, equal_nan=False
    ):
        raise ValueError(
            "Prepared cell expression does not match control_expression + target_delta; "
            "refusing to export a possibly mismatched source or preprocessing configuration."
        )

    split_by_sample: dict[str, str] = {}
    for partition in PARTITIONS:
        for sample_index in bundle["splits"][f"unseen_{partition}"]:
            split_by_sample[sample_ids[int(sample_index)]] = partition
    if set(split_by_sample) != set(sample_ids):
        raise ValueError("Unseen split labels do not cover every bundle sample exactly.")

    name_to_condition = {name: _condition_for_gene(name) for name in names}
    sample_to_condition = {
        sample_id: name_to_condition[names[int(index)]]
        for sample_id, index in zip(sample_ids, labels, strict=True)
    }
    control_ids = {
        obs_id
        for obs_id, label in zip(obs_ids, all_obs_labels, strict=True)
        if label == control_label
    }
    if not control_ids:
        raise ValueError("Prepared AnnData has no filtered control cells.")

    keep_ids = set(sample_ids) | control_ids
    keep_mask = np.asarray([obs_id in keep_ids for obs_id in obs_ids], dtype=bool)
    exported = prepared_adata[keep_mask].copy()
    exported_ids = exported.obs_names.astype(str).tolist()
    conditions: list[str] = []
    partition_labels: list[str] = []
    for sample_id in exported_ids:
        if sample_id in control_ids:
            conditions.append("ctrl")
            partition_labels.append("train")
        else:
            conditions.append(sample_to_condition[sample_id])
            partition_labels.append(split_by_sample[sample_id])

    exported.obs["condition"] = conditions
    exported.obs["cell_type"] = exported.obs[context_col].astype(str).to_numpy()
    exported.obs["scgpt_sample_id"] = exported_ids
    exported.obs["scgpt_split"] = partition_labels
    exported.var["gene_name"] = gene_symbols.to_numpy()
    exported.uns["scgpt_gears_export"] = {
        "schema_version": 1,
        "status": "INPUT_ONLY_NO_MODEL_RUN",
        "source_protocol": "unseen_group_by_perturbation",
        "expression_space": "training_only_gene_filter_and_HVG_normalized_log1p",
        "control_assignment": "all filtered controls assigned to train",
        "de_statistics_computed": False,
        "warning": (
            "Do not call GEARS new_data_process with its default skip_calc_de=False; "
            "that computes condition-level DE and dropout metadata before splitting. "
            "Use a reviewed train-only metadata/evaluation adapter."
        ),
        "condition_split": split_conditions,
        "preflight_comparison_eligible": False,
    }

    report = {
        "exported": True,
        "status": "INPUT_ONLY_NO_MODEL_RUN",
        "cell_count": int(exported.n_obs),
        "perturbed_sample_count": int(len(sample_ids)),
        "control_cell_count": int(len(control_ids)),
        "gene_count": int(exported.n_vars),
        "cell_type": cell_types[0],
        "condition_count": int(len(names) + 1),
        "partition_sample_counts": {
            part: int(sum(value == part for value in split_by_sample.values()))
            for part in PARTITIONS
        },
        "expression_reconstruction_verified": True,
        "gene_order_verified": True,
        "cell_ids_verified": True,
        "de_statistics_computed": False,
        "comparison_eligible": False,
        "blocking_checks": split_manifest["blocking_checks"]
        + [
            "train-only GEARS metadata and evaluation adapter",
            "GEARS GO-graph coverage",
            "target equivalence and real model execution",
        ],
    }
    return exported, report
