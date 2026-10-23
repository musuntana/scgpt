from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

PARTITIONS = ("train", "val", "test")
EXPECTED_HVG_METHOD = "shared_seen_unseen_training_intersection"


def _as_1d_int_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in ("i", "u"):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    return array.astype(np.int64, copy=False)


def _condition_for_gene(name: str) -> str:
    if not name or name != name.strip():
        raise ValueError(f"Invalid perturbation label: {name!r}.")
    if name.lower() in {"ctrl", "control", "ntc", "unperturbed"}:
        raise ValueError(f"Control cannot be a perturbation label: {name!r}.")
    if any(marker in name for marker in (";", "+", ",", "|", "/", "\\", " ", "\t", "\n")):
        raise ValueError(
            f"GEARS single-gene preflight rejects multi-gene or ambiguous label {name!r}."
        )
    return f"{name}+ctrl"


def _require_training_only_preprocessing(metadata: Mapping[str, Any]) -> None:
    diagnostics = metadata.get("preprocessing_diagnostics", {})
    if not isinstance(diagnostics, dict) or (
        diagnostics.get("hvg_selection_method") != EXPECTED_HVG_METHOD
        or diagnostics.get("heldout_perturbation_samples_excluded_from_hvg_fit") is not True
        or not {"seen", "unseen"}.issubset(set(diagnostics.get("protocols_protected", [])))
    ):
        raise ValueError(
            "GEARS preflight requires a freshly regenerated bundle with training-only "
            "gene filtering/HVG selection. Legacy or unverified preprocessing is not accepted."
        )


def build_gears_split_contract(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Convert a *group-disjoint* unseen bundle into GEARS custom condition lists.

    GEARS' custom split is condition-level, not cell-level. The conversion is
    deliberately unavailable for seen/sampled splits, and is a preflight only:
    GEARS can still filter conditions absent from its perturbation/GO graph.
    """
    metadata = bundle.get("metadata")
    splits = bundle.get("splits")
    if not isinstance(metadata, dict) or not isinstance(splits, dict):
        raise ValueError("Processed bundle must include metadata and split dictionaries.")
    _require_training_only_preprocessing(metadata)

    perturbation_names = metadata.get("perturbation_names")
    if (
        not isinstance(perturbation_names, list)
        or not perturbation_names
        or any(not isinstance(name, str) for name in perturbation_names)
        or len(set(perturbation_names)) != len(perturbation_names)
    ):
        raise ValueError("Bundle perturbation_names must be a non-empty, unique string list.")

    conditions = [_condition_for_gene(name) for name in perturbation_names]
    if len(set(conditions)) != len(conditions):
        raise ValueError("Perturbation names map to colliding GEARS condition names.")

    labels = _as_1d_int_array(bundle.get("perturbation_index"), "perturbation_index")
    sample_ids = bundle.get("sample_ids")
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= len(conditions)):
        raise ValueError("Perturbation indices are empty or outside metadata range.")
    if not isinstance(sample_ids, list) or len(sample_ids) != labels.size:
        raise ValueError("Sample IDs must exist and match perturbation indices.")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique.")

    partition_indices: dict[str, np.ndarray] = {}
    partition_groups: dict[str, set[int]] = {}
    for part in PARTITIONS:
        key = f"unseen_{part}"
        if key not in splits:
            raise ValueError(f"Missing split {key!r}; GEARS requires an unseen-group split.")
        indices = _as_1d_int_array(splits[key], key)
        if indices.size == 0:
            raise ValueError(f"{key} has no samples; GEARS baseline requires train/val/test.")
        if np.any(indices < 0) or np.any(indices >= labels.size):
            raise ValueError(f"{key} contains out-of-range sample indices.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{key} contains duplicate sample indices.")
        partition_indices[part] = indices
        partition_groups[part] = set(labels[indices].tolist())

    all_indices = np.concatenate([partition_indices[part] for part in PARTITIONS])
    if not np.array_equal(np.sort(all_indices), np.arange(labels.size)):
        raise ValueError(
            "Unseen partitions must be pairwise disjoint and cover every perturbation sample."
        )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if partition_groups[left].intersection(partition_groups[right]):
            raise ValueError(
                f"GEARS custom condition split cannot represent overlapping {left}/{right} "
                "perturbation groups; sample-level seen splits are not supported."
            )

    all_groups = set().union(*partition_groups.values())
    if all_groups != set(range(len(conditions))):
        raise ValueError("Every perturbation in metadata must belong to one unseen partition.")

    custom_split = {
        part: sorted(conditions[index] for index in partition_groups[part])
        for part in PARTITIONS
    }
    # GEARS needs control examples in its train loader; never put ctrl in val/test.
    custom_split["train"].insert(0, "ctrl")
    sample_id_digest = hashlib.sha256(
        json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PREFLIGHT_ONLY",
        "baseline": "official_GEARS",
        "source_protocol": "unseen_group_by_perturbation",
        "upstream_custom_split_kind": "condition_name_lists",
        "condition_encoding": "single_gene_plus_ctrl",
        "control_scope": "shared_filtered_controls_in_train_only",
        "sample_count": int(labels.size),
        "perturbation_count": len(conditions),
        "sample_id_sha256": sample_id_digest,
        "partitions": {
            part: {
                "sample_count": int(partition_indices[part].size),
                "perturbation_count": len(partition_groups[part]),
                "condition_names": custom_split[part],
            }
            for part in PARTITIONS
        },
        "preprocessing_method": EXPECTED_HVG_METHOD,
        "go_graph_coverage_verified": False,
        "source_cell_identity_verified": False,
        "target_and_gene_universe_equivalence_verified": False,
        "protocol_matched_real_metrics_available": False,
        "comparison_eligible": False,
        "blocking_checks": [
            "GEARS GO-graph inclusion of every held-out perturbation",
            "source AnnData cell and gene identity mapping",
            "identical sample split and expression target",
            "real model execution and protocol-matched evaluation",
        ],
    }
    return custom_split, manifest


def validate_source_identity(
    adata: Any,
    bundle: Mapping[str, Any],
    *,
    perturbation_col: str,
    control_label: str,
) -> dict[str, Any]:
    """Validate observable raw/source identities without claiming model compatibility.

    This does not verify GEARS' GO filtering, normalization, training target or
    whether its built-in loader can reproduce the precise subset of cells.
    """
    if perturbation_col not in adata.obs:
        raise ValueError(f"Source AnnData has no {perturbation_col!r} column.")
    if not adata.obs_names.is_unique:
        raise ValueError("Source AnnData cell/sample identifiers are not unique.")
    ids = list(bundle["sample_ids"])
    missing = sorted(set(ids) - set(adata.obs_names.astype(str)))
    if missing:
        raise ValueError(
            f"Source AnnData is missing {len(missing)} processed sample IDs "
            f"(example: {missing[0]!r})."
        )
    source_labels = adata.obs.loc[ids, perturbation_col].astype(str).tolist()
    meta_names = bundle["metadata"]["perturbation_names"]
    indices = _as_1d_int_array(bundle["perturbation_index"], "perturbation_index")
    expected_labels = [meta_names[int(index)] for index in indices]
    mismatches = [
        (sample_id, observed, expected)
        for sample_id, observed, expected in zip(ids, source_labels, expected_labels, strict=True)
        if observed != expected
    ]
    if mismatches:
        example = mismatches[0]
        raise ValueError(
            "Source and bundle perturbation labels differ for "
            f"{len(mismatches)} samples (example {example!r})."
        )

    source_var = set(adata.var_names.astype(str).tolist())
    expected_genes = bundle["metadata"].get("gene_names", [])
    missing_genes = sorted(set(expected_genes) - source_var)
    if missing_genes:
        raise ValueError(
            f"Source AnnData is missing {len(missing_genes)} selected bundle genes "
            f"(example: {missing_genes[0]!r})."
        )
    if control_label not in set(adata.obs[perturbation_col].astype(str)):
        raise ValueError("Source AnnData has no control cells for GEARS.")
    if "gene_name" not in adata.var.columns:
        raise ValueError(
            "GEARS requires an explicit var['gene_name'] mapping; source lacks this "
            "annotation. Provide a verified gene-symbol mapping before proceeding."
        )
    if not adata.var["gene_name"].is_unique:
        raise ValueError("GEARS gene_name annotation must be unique.")
    if "cell_type" not in adata.obs.columns:
        raise ValueError(
            "GEARS requires obs['cell_type']; add a verified cell-context mapping "
            "(for example, from celltype_new) before proceeding."
        )
    return {
        "source_cell_identity_verified": True,
        "source_selected_gene_ids_verified": True,
        "source_sample_count": int(adata.n_obs),
        "matched_perturbed_sample_count": len(ids),
        "source_control_sample_count": int(
            (adata.obs[perturbation_col].astype(str) == control_label).sum()
        ),
        "gears_condition_reencoding_required": True,
        "go_graph_coverage_verified": False,
        "target_and_gene_universe_equivalence_verified": False,
        "comparison_eligible": False,
    }
