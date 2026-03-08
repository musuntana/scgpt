from __future__ import annotations

import numpy as np
import pandas as pd


def compute_true_deg(
    adata,
    groupby: str,
    target_group: str,
    reference_group: str,
    method: str = "wilcoxon",
    adjusted_pvalue_threshold: float = 0.05,
    abs_logfoldchange_threshold: float = 0.25,
) -> pd.DataFrame:
    """Compute true DEG results for a perturbation condition."""
    try:
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError(
            "scanpy is required for DEG computation. Install dependencies first."
        ) from exc

    adata = adata.copy()
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        groups=[target_group],
        reference=reference_group,
        method=method,
    )
    deg_df = sc.get.rank_genes_groups_df(adata, group=target_group)
    deg_df = deg_df.rename(
        columns={
            "names": "gene",
            "logfoldchanges": "logfoldchange",
            "pvals_adj": "adjusted_p_value",
            "scores": "score",
        }
    )
    filtered = deg_df[
        (deg_df["adjusted_p_value"] < adjusted_pvalue_threshold)
        & (deg_df["logfoldchange"].abs() > abs_logfoldchange_threshold)
    ].copy()
    filtered["deg_significance"] = -np.log10(filtered["adjusted_p_value"] + 1e-12)
    filtered = filtered.sort_values(["score", "deg_significance"], ascending=False)
    return filtered.reset_index(drop=True)

