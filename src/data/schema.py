from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SchemaResolution:
    perturbation_col: str
    control_label: str
    batch_col: str | None
    context_cols: list[str]


def _normalize_string(value: Any) -> str:
    return str(value).strip()


def normalize_perturbation_label(
    label: str,
    control_labels: set[str],
    multi_gene_delimiters: list[str],
) -> str:
    """Convert raw perturbation labels into a consistent canonical representation."""
    label = _normalize_string(label)
    if label.lower() in control_labels:
        return "control"

    normalized = label
    for delimiter in multi_gene_delimiters:
        if delimiter == ";":
            continue
        normalized = normalized.replace(delimiter, ";")
    normalized = ";".join(part.strip() for part in normalized.split(";") if part.strip())
    return normalized or label


def enrich_adata_for_preset(
    adata,
    preset: str,
    control_label_candidates: list[str],
    multi_gene_delimiters: list[str],
):
    """Add canonical metadata columns for known dataset presets."""
    preset = preset.strip().lower()
    control_set = {candidate.lower() for candidate in control_label_candidates}

    if preset == "scperturb_norman2019":
        if "perturbation_canonical" not in adata.obs.columns:
            source_col = None
            if "perturbation_new" in adata.obs.columns:
                source_col = "perturbation_new"
            elif "perturbation" in adata.obs.columns:
                source_col = "perturbation"
            if source_col is not None:
                adata.obs["perturbation_canonical"] = (
                    adata.obs[source_col]
                    .astype(str)
                    .map(
                        lambda value: normalize_perturbation_label(
                            value,
                            control_labels=control_set,
                            multi_gene_delimiters=multi_gene_delimiters,
                        )
                    )
                )
        if "celltype_new" not in adata.obs.columns and "celltype" in adata.obs.columns:
            adata.obs["celltype_new"] = adata.obs["celltype"].astype(str)
        if "batch" not in adata.obs.columns and "gemgroup" in adata.obs.columns:
            adata.obs["batch"] = adata.obs["gemgroup"].astype(str)
    return adata


def infer_column(columns: list[str], candidates: list[str], label: str) -> str:
    """Select the first matching column from candidate names."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    available = ", ".join(columns)
    wanted = ", ".join(candidates)
    raise ValueError(f"Could not infer {label}. Tried [{wanted}] in [{available}]")


def infer_control_label(
    labels: pd.Series,
    configured_label: str,
    candidates: list[str],
) -> str:
    """Resolve the control label from config or observed values."""
    if configured_label and configured_label != "auto":
        return configured_label

    lower_to_original: dict[str, str] = {}
    for value in labels.astype(str).tolist():
        lowered = value.strip().lower()
        lower_to_original.setdefault(lowered, value)

    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in lower_to_original:
            return lower_to_original[candidate_lower]

    raise ValueError(
        "Unable to infer control label from dataset. "
        f"Tried candidates: {', '.join(candidates)}"
    )


def _resolve_context_cols(
    configured_context_cols: list[str] | str,
    columns: list[str],
    context_candidates: list[str],
) -> list[str]:
    if isinstance(configured_context_cols, str):
        if configured_context_cols == "auto":
            return [column for column in context_candidates if column in columns][:2]
        return [configured_context_cols] if configured_context_cols in columns else []
    return [column for column in configured_context_cols if column in columns]


def resolve_schema(
    adata,
    dataset_config: dict[str, Any],
    cli_perturbation_col: str | None,
    cli_control_label: str | None,
    cli_batch_col: str | None,
    cli_context_cols: list[str] | None,
) -> SchemaResolution:
    """Resolve dataset columns and control labels from config plus CLI overrides."""
    schema_config = dataset_config.get("schema", {})
    columns = adata.obs.columns.astype(str).tolist()

    perturbation_col = cli_perturbation_col
    if not perturbation_col or perturbation_col == "auto":
        configured = schema_config.get("perturbation_col", "auto")
        if configured == "auto":
            perturbation_col = infer_column(
                columns,
                schema_config.get("perturbation_col_candidates", ["perturbation"]),
                label="perturbation column",
            )
        else:
            perturbation_col = str(configured)

    batch_col = cli_batch_col
    if batch_col == "none":
        batch_col = None
    if batch_col is None or batch_col == "auto":
        configured = schema_config.get("batch_col", "auto")
        if configured == "auto":
            batch_candidates = schema_config.get("batch_col_candidates", [])
            batch_col = next((column for column in batch_candidates if column in columns), None)
        elif configured == "none":
            batch_col = None
        else:
            batch_col = str(configured)

    context_cols = cli_context_cols if cli_context_cols else None
    if context_cols is None or context_cols == ["auto"]:
        configured = schema_config.get("context_cols", "auto")
        context_cols = _resolve_context_cols(
            configured_context_cols=configured,
            columns=columns,
            context_candidates=schema_config.get("context_col_candidates", []),
        )

    control_label = infer_control_label(
        labels=adata.obs[perturbation_col],
        configured_label=cli_control_label or dataset_config["dataset"].get("control_label", "auto"),
        candidates=schema_config.get("control_label_candidates", ["control"]),
    )
    return SchemaResolution(
        perturbation_col=perturbation_col,
        control_label=control_label,
        batch_col=batch_col,
        context_cols=context_cols,
    )


def summarize_anndata(adata, max_examples: int = 10) -> dict[str, Any]:
    """Generate a lightweight schema summary for inspection."""
    obs_columns = adata.obs.columns.astype(str).tolist()
    var_columns = adata.var.columns.astype(str).tolist()
    summary: dict[str, Any] = {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_columns": obs_columns,
        "var_columns": var_columns,
        "obs_examples": {},
    }
    for column in obs_columns[: min(len(obs_columns), 20)]:
        try:
            unique_values = adata.obs[column].astype(str).value_counts().head(max_examples)
            summary["obs_examples"][column] = unique_values.to_dict()
        except Exception:
            summary["obs_examples"][column] = {"preview": "unavailable"}
    return summary
