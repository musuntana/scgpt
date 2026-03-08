from __future__ import annotations

from typing import Any

import numpy as np

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


def prepare_adata(
    adata,
    preprocess_config: dict[str, Any],
    perturbation_col: str,
    control_label: str,
    random_seed: int,
):
    """Run QC, normalization, HVG selection, and local-first subsampling."""
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
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    sc.pp.normalize_total(adata, target_sum=normalize_total_target_sum)
    sc.pp.log1p(adata)

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
