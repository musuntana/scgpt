from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    aggregate_by_label,
    mean_squared_error,
    pearson_correlation,
    topk_overlap,
)


def _top_gene_names(values: np.ndarray, gene_names: list[str], top_n: int) -> list[str]:
    top_indices = np.argsort(np.abs(np.asarray(values, dtype=np.float64).ravel()))[::-1][:top_n]
    return [gene_names[int(index)] for index in top_indices]


def _infer_failure_mode(
    *,
    sample_count: int,
    pearson: float,
    observed_abs_mean: float,
    predicted_abs_mean: float,
    residual_abs_mean: float,
) -> str:
    if sample_count <= 3:
        return "low_sample_support"
    if observed_abs_mean < 0.05:
        return "low_signal_condition"
    if pearson < 0.0:
        return "directional_mismatch"

    magnitude_ratio = predicted_abs_mean / max(observed_abs_mean, 1e-8)
    error_to_signal_ratio = residual_abs_mean / max(observed_abs_mean, 1e-8)

    if magnitude_ratio < 0.67:
        return "underestimates_response_magnitude"
    if magnitude_ratio > 1.5:
        return "overestimates_response_magnitude"
    if error_to_signal_ratio > 1.0:
        return "high_residual_condition"
    return "mostly_aligned"


def _true_topk_genes(
    deg_df: pd.DataFrame | None,
    *,
    perturbation_name: str,
    k: int,
) -> list[str]:
    if deg_df is None or deg_df.empty:
        return []
    subset = deg_df[deg_df["perturbation"] == perturbation_name]
    if subset.empty:
        return []
    return subset["gene"].astype(str).tolist()[:k]


def build_per_perturbation_error_table(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    perturbation_index: np.ndarray,
    perturbation_names: list[str],
    gene_names: list[str],
    deg_df: pd.DataFrame | None = None,
    k_values: list[int] | None = None,
    top_gene_count: int = 5,
) -> pd.DataFrame:
    """Build a perturbation-level error report with lightweight failure heuristics."""
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    perturbation_index = np.asarray(perturbation_index, dtype=np.int64)

    unique_labels, aggregated_predictions = aggregate_by_label(predictions, perturbation_index)
    _, aggregated_targets = aggregate_by_label(targets, perturbation_index)
    counts = {
        int(label): int(np.sum(perturbation_index == label))
        for label in np.unique(perturbation_index)
    }

    rows: list[dict[str, Any]] = []
    requested_k_values = sorted(set(k_values or []))

    for label, predicted_mean, target_mean in zip(
        unique_labels, aggregated_predictions, aggregated_targets, strict=True
    ):
        perturbation_name = perturbation_names[int(label)]
        residual = predicted_mean - target_mean
        observed_abs_mean = float(np.mean(np.abs(target_mean)))
        predicted_abs_mean = float(np.mean(np.abs(predicted_mean)))
        residual_abs_mean = float(np.mean(np.abs(residual)))
        pearson = pearson_correlation(predicted_mean, target_mean)
        mse = mean_squared_error(predicted_mean, target_mean)

        row: dict[str, Any] = {
            "perturbation": perturbation_name,
            "perturbation_index": int(label),
            "sample_count": counts[int(label)],
            "pearson": pearson,
            "mse": mse,
            "observed_abs_mean": observed_abs_mean,
            "predicted_abs_mean": predicted_abs_mean,
            "residual_abs_mean": residual_abs_mean,
            "error_to_signal_ratio": residual_abs_mean / max(observed_abs_mean, 1e-8),
            "failure_mode": _infer_failure_mode(
                sample_count=counts[int(label)],
                pearson=pearson,
                observed_abs_mean=observed_abs_mean,
                predicted_abs_mean=predicted_abs_mean,
                residual_abs_mean=residual_abs_mean,
            ),
            "top_observed_genes": ",".join(
                _top_gene_names(target_mean, gene_names, top_gene_count)
            ),
            "top_predicted_genes": ",".join(
                _top_gene_names(predicted_mean, gene_names, top_gene_count)
            ),
            "top_residual_genes": ",".join(
                _top_gene_names(residual, gene_names, top_gene_count)
            ),
        }

        for k in requested_k_values:
            predicted_top = _top_gene_names(predicted_mean, gene_names, k)
            true_top = _true_topk_genes(deg_df, perturbation_name=perturbation_name, k=k)
            row[f"topk_deg_overlap_{k}"] = (
                topk_overlap(predicted_top, true_top, k) if true_top else None
            )

        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["pearson", "mse", "sample_count"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_error_summary(
    error_table: pd.DataFrame,
    *,
    split_name: str,
    model_type: str,
    worst_n: int = 10,
) -> dict[str, Any]:
    """Summarize the worst perturbation conditions for a split-specific evaluation."""
    if error_table.empty:
        return {
            "split_name": split_name,
            "model_type": model_type,
            "num_perturbations": 0,
            "failure_mode_counts": {},
            "worst_by_pearson": [],
            "worst_by_mse": [],
            "notes": [
                "Failure modes are heuristic and intended for qualitative debugging only.",
            ],
        }

    summary_columns = [
        "perturbation",
        "sample_count",
        "pearson",
        "mse",
        "failure_mode",
        "error_to_signal_ratio",
        "top_residual_genes",
    ]
    available_columns = [column for column in summary_columns if column in error_table.columns]

    worst_by_pearson = (
        error_table.nsmallest(worst_n, "pearson")[available_columns].to_dict(orient="records")
    )
    worst_by_mse = (
        error_table.nlargest(worst_n, "mse")[available_columns].to_dict(orient="records")
    )

    return {
        "split_name": split_name,
        "model_type": model_type,
        "num_perturbations": int(len(error_table)),
        "failure_mode_counts": {
            str(key): int(value)
            for key, value in error_table["failure_mode"].value_counts().to_dict().items()
        },
        "worst_by_pearson": worst_by_pearson,
        "worst_by_mse": worst_by_mse,
        "notes": [
            "Failure modes are heuristic and intended for qualitative debugging only.",
            "Low-signal conditions can have unstable Pearson values even when absolute residuals are small.",
        ],
    }


def build_failure_mode_count_frame(error_summary: dict[str, Any]) -> pd.DataFrame:
    """Convert failure-mode counts from an error summary into a sorted DataFrame."""
    counts = error_summary.get("failure_mode_counts", {})
    if not isinstance(counts, dict) or not counts:
        return pd.DataFrame(columns=["failure_mode", "count"])
    frame = pd.DataFrame(
        [
            {"failure_mode": str(name), "count": int(count)}
            for name, count in counts.items()
        ]
    )
    return frame.sort_values(by=["count", "failure_mode"], ascending=[False, True]).reset_index(drop=True)


def build_worst_conditions_frame(
    error_summary: dict[str, Any],
    *,
    rank_by: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Convert one of the worst-condition lists in an error summary into a DataFrame."""
    rows = error_summary.get(rank_by, [])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([row for row in rows if isinstance(row, dict)])
    if frame.empty:
        return frame
    return frame.head(top_n).reset_index(drop=True)

def format_failure_mode_label(failure_mode: Any) -> str:
    """Render a human-readable failure-mode label for CLI/UI summaries."""
    if failure_mode is None:
        return "n/a"
    label = str(failure_mode).replace("_", " ")
    return label.replace("low signal", "low-signal")


def build_error_highlights(error_summary: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact, display-ready summary from a saved error summary artifact."""
    failure_mode_frame = build_failure_mode_count_frame(error_summary)
    worst_pearson_frame = build_worst_conditions_frame(
        error_summary,
        rank_by="worst_by_pearson",
        top_n=1,
    )
    worst_mse_frame = build_worst_conditions_frame(
        error_summary,
        rank_by="worst_by_mse",
        top_n=1,
    )

    dominant_failure_mode = None
    dominant_failure_mode_count = None
    if not failure_mode_frame.empty:
        dominant_failure_mode = failure_mode_frame.iloc[0]["failure_mode"]
        dominant_failure_mode_count = int(failure_mode_frame.iloc[0]["count"])

    highlights: dict[str, Any] = {
        "split_name": error_summary.get("split_name"),
        "num_perturbations": int(error_summary.get("num_perturbations", 0) or 0),
        "dominant_failure_mode": dominant_failure_mode,
        "dominant_failure_mode_label": format_failure_mode_label(dominant_failure_mode),
        "dominant_failure_mode_count": dominant_failure_mode_count,
    }

    if not worst_pearson_frame.empty:
        row = worst_pearson_frame.iloc[0]
        highlights.update(
            {
                "worst_pearson_perturbation": row.get("perturbation"),
                "worst_pearson_value": row.get("pearson"),
                "worst_pearson_failure_mode": row.get("failure_mode"),
                "worst_pearson_failure_mode_label": format_failure_mode_label(
                    row.get("failure_mode")
                ),
            }
        )

    if not worst_mse_frame.empty:
        row = worst_mse_frame.iloc[0]
        highlights.update(
            {
                "worst_mse_perturbation": row.get("perturbation"),
                "worst_mse_value": row.get("mse"),
                "worst_mse_failure_mode": row.get("failure_mode"),
                "worst_mse_failure_mode_label": format_failure_mode_label(
                    row.get("failure_mode")
                ),
            }
        )

    return highlights


def _find_condition_rank(
    error_summary: dict[str, Any],
    *,
    perturbation_name: str,
    rank_by: str,
) -> tuple[int | None, dict[str, Any] | None]:
    rows = error_summary.get(rank_by, [])
    if not isinstance(rows, list):
        return None, None
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        if row.get("perturbation") == perturbation_name:
            return index, row
    return None, None


def build_selected_condition_story(
    *,
    perturbation_name: str,
    diagnostics: dict[str, Any],
    error_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a concise, display-ready explanation for one perturbation condition."""
    if not diagnostics:
        return {}

    error_summary = error_summary or {}
    failure_mode = diagnostics.get("failure_mode")
    failure_mode_label = format_failure_mode_label(failure_mode)
    pearson_rank, _ = _find_condition_rank(
        error_summary,
        perturbation_name=perturbation_name,
        rank_by="worst_by_pearson",
    )
    mse_rank, _ = _find_condition_rank(
        error_summary,
        perturbation_name=perturbation_name,
        rank_by="worst_by_mse",
    )

    if pearson_rank == 1 and mse_rank == 1:
        status = "error"
        headline = (
            f"{perturbation_name} is the hardest saved condition on this split "
            f"with {failure_mode_label} behavior."
        )
    elif pearson_rank == 1:
        status = "error"
        headline = (
            f"{perturbation_name} is the worst saved Pearson case on this split "
            f"and is currently tagged as {failure_mode_label}."
        )
    elif mse_rank == 1:
        status = "error"
        headline = (
            f"{perturbation_name} is the worst saved MSE case on this split "
            f"and is currently tagged as {failure_mode_label}."
        )
    elif pearson_rank is not None or mse_rank is not None:
        status = "warning"
        headline = (
            f"{perturbation_name} is one of the harder saved conditions on this split "
            f"with {failure_mode_label} behavior."
        )
    elif failure_mode in {
        "directional_mismatch",
        "high_residual_condition",
        "underestimates_response_magnitude",
        "overestimates_response_magnitude",
        "low_sample_support",
    }:
        status = "warning"
        headline = (
            f"{perturbation_name} needs attention on this split because it shows "
            f"{failure_mode_label} behavior."
        )
    elif failure_mode == "mostly_aligned":
        status = "success"
        headline = f"{perturbation_name} looks mostly aligned on this split."
    else:
        status = "info"
        headline = (
            f"{perturbation_name} is primarily a {failure_mode_label} condition on this split."
        )

    details: list[str] = []
    sample_count = diagnostics.get("sample_count")
    if sample_count is not None:
        details.append(f"Saved diagnostics cover {int(sample_count)} matched samples.")
    pearson = diagnostics.get("pearson")
    mse = diagnostics.get("mse")
    if pearson is not None and mse is not None:
        details.append(
            f"Saved split metrics: Pearson={float(pearson):.4f}, MSE={float(mse):.4f}."
        )
    error_to_signal_ratio = diagnostics.get("error_to_signal_ratio")
    if error_to_signal_ratio is not None:
        details.append(
            f"Error-to-signal ratio is {float(error_to_signal_ratio):.4f}."
        )
    if pearson_rank is not None:
        details.append(f"Ranked #{pearson_rank} in the saved worst-Pearson list.")
    if mse_rank is not None:
        details.append(f"Ranked #{mse_rank} in the saved worst-MSE list.")
    top_residual_genes = diagnostics.get("top_residual_genes")
    if top_residual_genes:
        details.append(f"Top residual genes: {top_residual_genes}.")

    return {
        "status": status,
        "headline": headline,
        "details": details,
        "failure_mode_label": failure_mode_label,
        "worst_pearson_rank": pearson_rank,
        "worst_mse_rank": mse_rank,
    }

def select_perturbation_diagnostics(
    error_table: pd.DataFrame,
    *,
    perturbation_name: str,
) -> dict[str, Any]:
    """Return a compact diagnostics dict for a single perturbation from a per-perturbation table."""
    if error_table.empty or "perturbation" not in error_table.columns:
        return {}
    subset = error_table[error_table["perturbation"] == perturbation_name]
    if subset.empty:
        return {}
    row = subset.iloc[0]
    fields = [
        "perturbation",
        "sample_count",
        "pearson",
        "mse",
        "failure_mode",
        "error_to_signal_ratio",
        "top_residual_genes",
    ]
    diagnostics: dict[str, Any] = {}
    for field in fields:
        if field in subset.columns:
            diagnostics[field] = row[field]
    return diagnostics
