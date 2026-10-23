from __future__ import annotations

from typing import Any

import numpy as np

from src.data.gene_identity import build_unique_gene_name_index
from src.data.pairing import create_split_indices
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _require_scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError(
            "scanpy is required for preprocessing. Install dependencies first."
        ) from exc
    return sc


def _is_single_gene_label(label: str, control_label: str) -> bool:
    if label == control_label:
        return True
    multi_gene_separators = (";", "+", ",", "|", "_")
    return not any(separator in label for separator in multi_gene_separators)


def filter_single_gene_perturbations(
    adata,
    perturbation_col: str,
    control_label: str,
):
    """Keep only control and single-gene perturbation samples."""
    labels = adata.obs[perturbation_col].astype(str)
    mask = labels.map(lambda label: _is_single_gene_label(label, control_label)).to_numpy()
    filtered = adata[mask].copy()
    LOGGER.info("Filtered to %s cells after removing multi-gene perturbations.", filtered.n_obs)
    return filtered


def cap_cells_per_perturbation(
    adata,
    perturbation_col: str,
    max_cells_per_perturbation: int | None,
    random_seed: int,
):
    """Subsample over-represented perturbation conditions for local-first training."""
    if not max_cells_per_perturbation or max_cells_per_perturbation <= 0:
        return adata

    rng = np.random.default_rng(random_seed)
    selected_indices: list[int] = []
    labels = adata.obs[perturbation_col].astype(str).to_numpy()

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) > max_cells_per_perturbation:
            label_indices = np.sort(
                rng.choice(label_indices, size=max_cells_per_perturbation, replace=False)
            )
        selected_indices.extend(label_indices.tolist())

    selected_indices = sorted(selected_indices)
    subset = adata[selected_indices].copy()
    LOGGER.info("Subsampled to %s cells for local-first iteration.", subset.n_obs)
    return subset


def extend_hvg_mask_with_target_genes(
    *,
    gene_names: list[str],
    hvg_mask: np.ndarray,
    perturbation_names: list[str],
    max_panel_genes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add known target identities to a training-fitted HVG mask, without heldout expression.

    Target labels are metadata available before predicting unseen perturbations.
    Fail closed if any label is missing/ambiguous after training-only gene filtering,
    or if the resulting feature panel would exceed its explicit memory budget.
    """
    selected = np.asarray(hvg_mask, dtype=bool)
    if selected.ndim != 1 or selected.shape[0] != len(gene_names):
        raise ValueError("HVG mask and available gene names have inconsistent shapes.")
    if not selected.any():
        raise ValueError("The training-fitted HVG mask is empty.")
    if max_panel_genes < 1:
        raise ValueError("hvg_upper_bound must be positive for target retention.")

    index_by_gene, ambiguous = build_unique_gene_name_index(gene_names)
    targets = sorted(set(perturbation_names))
    if not targets:
        raise ValueError("Cannot retain target genes without perturbation identities.")
    missing = sorted(name for name in targets if name not in index_by_gene)
    if missing:
        ambiguous_targets = sorted(set(missing) & ambiguous)
        raise ValueError(
            "Target-gene retention requires exact, unique gene names after "
            "training-only filtering. "
            f"Unmapped targets ({len(missing)}): {missing[:10]}; "
            f"ambiguous targets: {ambiguous_targets[:10]}. "
            "Inspect gene symbols or filtering; do not guess aliases."
        )

    output = selected.copy()
    before = int(output.sum())
    for name in targets:
        output[index_by_gene[name]] = True
    after = int(output.sum())
    panel_names = [gene_names[index] for index in np.flatnonzero(output)]
    if len(panel_names) != len(set(panel_names)):
        raise ValueError(
            "Target-aware feature panel has ambiguous duplicate gene names; "
            "resolve stable gene identifiers before using shared-gene conditioning."
        )
    if after > max_panel_genes:
        raise ValueError(
            "Target-aware feature panel exceeds hvg_upper_bound: "
            f"{after} genes (HVGs={before}, targets={len(targets)}, "
            f"upper_bound={max_panel_genes}). Increase the explicit bound or "
            "change the preprocessing design; no target will be silently dropped."
        )
    return output, {
        "known_target_gene_count": len(targets),
        "target_genes_added_to_hvgs": after - before,
        "target_genes_retained": len(targets),
        "feature_panel_gene_count": after,
        "target_selection_scope": "known_perturbation_names_only",
        "heldout_perturbation_expression_used_for_target_retention": False,
    }


def select_hvgs_from_training_partitions(
    adata,
    preprocess_config: dict[str, Any],
    perturbation_col: str,
    control_label: str,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
):
    """Fit HVG selection only on perturbations in both training partitions.

    All filtered control cells are included as the shared reference population.
    The intersection of seen/unseen training indices prevents either protocol's
    validation or test perturbation samples from influencing feature selection.
    """
    sc = _require_scanpy()
    if perturbation_col not in adata.obs.columns:
        raise ValueError(f"Missing perturbation column: {perturbation_col}")

    hvg_top_genes = int(preprocess_config.get("hvg_top_genes", 512))
    if hvg_top_genes < 1:
        raise ValueError(f"hvg_top_genes must be positive; got {hvg_top_genes}.")

    labels = adata.obs[perturbation_col].astype(str).to_numpy()
    control_mask = labels == control_label
    control_positions = np.flatnonzero(control_mask)
    perturbed_positions = np.flatnonzero(~control_mask)
    if control_positions.size == 0:
        raise ValueError("No control samples found; cannot fit training-only HVGs.")
    if perturbed_positions.size == 0:
        raise ValueError("No perturbed samples found; cannot fit training-only HVGs.")

    perturbation_names = sorted(np.unique(labels[perturbed_positions]).tolist())
    perturbation_to_index = {
        name: index for index, name in enumerate(perturbation_names)
    }
    perturbation_index = np.asarray(
        [perturbation_to_index[name] for name in labels[perturbed_positions]],
        dtype=np.int64,
    )
    splits = create_split_indices(
        perturbation_index=perturbation_index,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    shared_train_indices = np.intersect1d(
        splits["seen_train"],
        splits["unseen_train"],
        assume_unique=True,
    )
    if shared_train_indices.size == 0:
        raise ValueError(
            "No perturbed samples remain in the intersection of seen_train and "
            "unseen_train; cannot select HVGs without using held-out samples. "
            "Review split fractions or the number of perturbation groups."
        )

    fit_positions = np.sort(
        np.concatenate([control_positions, perturbed_positions[shared_train_indices]])
    )
    fit_adata = adata[fit_positions].copy()
    min_cells_per_gene = int(preprocess_config.get("min_cells_per_gene", 3))
    if min_cells_per_gene < 1:
        raise ValueError(
            f"min_cells_per_gene must be positive; got {min_cells_per_gene}."
        )

    original_position_key = "_scgpt_original_gene_position"
    fit_adata.var[original_position_key] = np.arange(adata.n_vars, dtype=np.int64)
    sc.pp.filter_genes(fit_adata, min_cells=min_cells_per_gene)
    retained_positions = fit_adata.var[original_position_key].to_numpy(dtype=np.int64)
    if retained_positions.size == 0:
        raise ValueError(
            "Training-only gene filtering removed every gene; lower min_cells_per_gene "
            "or review the training/reference cells."
        )

    retained_gene_mask = np.zeros(adata.n_vars, dtype=bool)
    retained_gene_mask[retained_positions] = True
    adata = adata[:, retained_gene_mask].copy()

    normalize_total_target_sum = float(
        preprocess_config.get("normalize_total_target_sum", 10_000)
    )
    if normalize_total_target_sum <= 0:
        raise ValueError(
            "normalize_total_target_sum must be positive; "
            f"got {normalize_total_target_sum}."
        )
    sc.pp.normalize_total(adata, target_sum=normalize_total_target_sum)
    log1p_enabled = bool(preprocess_config.get("log1p", True))
    if log1p_enabled:
        sc.pp.log1p(adata)

    fit_adata = adata[fit_positions].copy()
    n_top_genes = min(hvg_top_genes, int(fit_adata.n_vars))
    if n_top_genes < 1:
        raise ValueError("Cannot select HVGs from an AnnData object with zero genes.")

    sc.pp.highly_variable_genes(fit_adata, n_top_genes=n_top_genes)
    if "highly_variable" not in fit_adata.var.columns:
        raise RuntimeError("Scanpy did not produce the 'highly_variable' gene annotation.")
    selected_mask = fit_adata.var["highly_variable"].to_numpy(dtype=bool)
    if selected_mask.shape != (adata.n_vars,) or not selected_mask.any():
        raise ValueError(
            "Training-only HVG selection produced an empty or invalid gene mask."
        )

    retain_targets = bool(
        preprocess_config.get("retain_perturbation_target_genes", False)
    )
    original_hvg_mask = selected_mask.copy()
    target_panel_diagnostics: dict[str, Any] = {}
    if retain_targets:
        selected_mask, target_panel_diagnostics = extend_hvg_mask_with_target_genes(
            gene_names=adata.var_names.astype(str).tolist(),
            hvg_mask=selected_mask,
            perturbation_names=perturbation_names,
            max_panel_genes=int(preprocess_config.get("hvg_upper_bound", 800)),
        )

    # Preserve the original training-fitted HVG flag. Added target genes are
    # selected features, not necessarily highly variable genes.
    adata.var["highly_variable"] = original_hvg_mask
    adata.var["scgpt_feature_selected"] = selected_mask
    adata = adata[:, selected_mask].copy()
    adata.uns["scgpt_preprocessing"] = {
        "schema_version": 2 if retain_targets else 1,
        "hvg_selection_method": "shared_seen_unseen_training_intersection",
        "hvg_fit_perturbed_sample_count": int(shared_train_indices.size),
        "hvg_fit_perturbation_count": int(
            np.unique(perturbation_index[shared_train_indices]).size
        ),
        "hvg_fit_control_sample_count": int(control_positions.size),
        "hvg_fit_control_scope": "all_filtered_control_cells",
        "min_cells_per_gene": min_cells_per_gene,
        "normalize_total_target_sum": normalize_total_target_sum,
        "log1p": log1p_enabled,
        "hvg_gene_count_after_training_filter": int(fit_adata.n_vars),
        "hvg_requested_gene_count": hvg_top_genes,
        "hvg_selected_gene_count": int(original_hvg_mask.sum()),
        "retain_perturbation_target_genes": retain_targets,
        **target_panel_diagnostics,
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "split_seed": int(random_seed),
        "heldout_perturbation_samples_excluded_from_hvg_fit": True,
        "protocols_protected": ["seen", "unseen"],
    }
    LOGGER.info(
        "Selected %s HVGs using %s shared training perturbation samples, %s perturbation "
        "groups, and %s control cells; protected both seen/unseen holdouts.",
        int(selected_mask.sum()),
        int(shared_train_indices.size),
        int(np.unique(perturbation_index[shared_train_indices]).size),
        int(control_positions.size),
    )
    return adata


def prepare_adata(
    adata,
    preprocess_config: dict[str, Any],
    perturbation_col: str,
    control_label: str,
    random_seed: int,
    select_hvg: bool = True,
    filter_genes_before_split: bool = True,
    normalize_expression: bool = True,
):
    """Run cell QC, optional normalization/gene filtering/HVG selection, and subsampling."""
    sc = _require_scanpy()

    if perturbation_col not in adata.obs.columns:
        raise ValueError(f"Missing perturbation column: {perturbation_col}")

    adata = filter_single_gene_perturbations(
        adata=adata,
        perturbation_col=perturbation_col,
        control_label=control_label,
    )

    min_genes_per_cell = int(preprocess_config.get("min_genes_per_cell", 200))
    min_cells_per_gene = int(preprocess_config.get("min_cells_per_gene", 3))
    normalize_total_target_sum = float(
        preprocess_config.get("normalize_total_target_sum", 10_000)
    )
    hvg_top_genes = int(preprocess_config.get("hvg_top_genes", 512))
    max_cells_per_perturbation = preprocess_config.get("max_cells_per_perturbation")
    max_cells_per_perturbation = (
        int(max_cells_per_perturbation) if max_cells_per_perturbation else None
    )

    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    if filter_genes_before_split:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    if normalize_expression:
        sc.pp.normalize_total(adata, target_sum=normalize_total_target_sum)
        if bool(preprocess_config.get("log1p", True)):
            sc.pp.log1p(adata)

    if select_hvg:
        n_top_genes = min(hvg_top_genes, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        if "highly_variable" in adata.var.columns:
            adata = adata[:, adata.var["highly_variable"]].copy()

    adata = cap_cells_per_perturbation(
        adata=adata,
        perturbation_col=perturbation_col,
        max_cells_per_perturbation=max_cells_per_perturbation,
        random_seed=random_seed,
    )
    LOGGER.info("Prepared AnnData with %s cells and %s genes.", adata.n_obs, adata.n_vars)
    return adata
