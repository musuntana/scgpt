from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import write_json

DEG_ARTIFACT_FILENAME = "deg_artifact.csv"
DEG_METADATA_FILENAME = "deg_artifact_metadata.json"


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
    if "logfoldchange" not in deg_df.columns:
        deg_df["logfoldchange"] = 0.0
    if "adjusted_p_value" not in deg_df.columns:
        deg_df["adjusted_p_value"] = 1.0
    if "score" not in deg_df.columns:
        deg_df["score"] = 0.0

    deg_df["adjusted_p_value"] = (
        pd.to_numeric(deg_df["adjusted_p_value"], errors="coerce").fillna(1.0)
    )
    deg_df["logfoldchange"] = (
        pd.to_numeric(deg_df["logfoldchange"], errors="coerce").fillna(0.0)
    )
    deg_df["score"] = pd.to_numeric(deg_df["score"], errors="coerce").fillna(0.0)

    filtered = deg_df[
        (deg_df["adjusted_p_value"] < adjusted_pvalue_threshold)
        & (deg_df["logfoldchange"].abs() > abs_logfoldchange_threshold)
    ].copy()
    filtered["deg_significance"] = -np.log10(filtered["adjusted_p_value"] + 1e-12)
    filtered = filtered.sort_values(["score", "deg_significance"], ascending=False)
    return filtered.reset_index(drop=True)


def compute_deg_artifact(
    adata,
    perturbation_col: str,
    control_label: str,
    perturbation_names: list[str] | None = None,
    method: str = "wilcoxon",
    adjusted_pvalue_threshold: float = 0.05,
    abs_logfoldchange_threshold: float = 0.25,
) -> pd.DataFrame:
    """Compute DEG results for every perturbation against the control group."""
    labels = adata.obs[perturbation_col].astype(str)
    if perturbation_names is None:
        perturbation_names = sorted(
            perturbation
            for perturbation in labels.unique().tolist()
            if perturbation != control_label
        )

    frames: list[pd.DataFrame] = []
    for perturbation_name in perturbation_names:
        subset_mask = labels.isin([control_label, perturbation_name]).to_numpy()
        subset = adata[subset_mask].copy()
        subset_labels = subset.obs[perturbation_col].astype(str)
        perturbation_cell_count = int((subset_labels == perturbation_name).sum())
        control_cell_count = int((subset_labels == control_label).sum())
        if perturbation_cell_count == 0 or control_cell_count == 0:
            continue

        deg_df = compute_true_deg(
            adata=subset,
            groupby=perturbation_col,
            target_group=perturbation_name,
            reference_group=control_label,
            method=method,
            adjusted_pvalue_threshold=adjusted_pvalue_threshold,
            abs_logfoldchange_threshold=abs_logfoldchange_threshold,
        )
        if deg_df.empty:
            continue
        deg_df["perturbation"] = perturbation_name
        deg_df["perturbation_cell_count"] = perturbation_cell_count
        deg_df["control_cell_count"] = control_cell_count
        deg_df["rank"] = np.arange(1, len(deg_df) + 1, dtype=np.int64)
        frames.append(
            deg_df[
                [
                    "perturbation",
                    "rank",
                    "gene",
                    "logfoldchange",
                    "adjusted_p_value",
                    "score",
                    "deg_significance",
                    "perturbation_cell_count",
                    "control_cell_count",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "perturbation",
                "rank",
                "gene",
                "logfoldchange",
                "adjusted_p_value",
                "score",
                "deg_significance",
                "perturbation_cell_count",
                "control_cell_count",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def save_deg_artifact(
    deg_df: pd.DataFrame,
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save a DEG artifact CSV and sidecar metadata JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    csv_path = destination / DEG_ARTIFACT_FILENAME
    metadata_path = destination / DEG_METADATA_FILENAME
    deg_df.to_csv(csv_path, index=False)
    write_json(metadata_path, metadata)
    return csv_path, metadata_path


def load_deg_artifact(path: str | Path) -> pd.DataFrame:
    """Load a DEG artifact CSV."""
    artifact_path = Path(path)
    return pd.read_csv(artifact_path)
