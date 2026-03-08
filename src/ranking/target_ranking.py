from __future__ import annotations

import numpy as np
import pandas as pd


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_value = float(values.min(initial=0.0))
    max_value = float(values.max(initial=0.0))
    if max_value == min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def build_target_ranking(
    gene_names: list[str],
    predicted_delta: np.ndarray,
    deg_df: pd.DataFrame | None = None,
    abs_predicted_delta_weight: float = 0.5,
    deg_significance_weight: float = 0.5,
) -> pd.DataFrame:
    """Combine predicted delta magnitude and DEG significance into a ranking."""
    predicted_delta = np.asarray(predicted_delta, dtype=np.float32).reshape(-1)
    if len(gene_names) != len(predicted_delta):
        raise ValueError("gene_names and predicted_delta must have the same length")

    deg_scores = np.zeros(len(gene_names), dtype=np.float32)
    if deg_df is not None and not deg_df.empty:
        deg_lookup = {
            str(row["gene"]): float(row["deg_significance"])
            for _, row in deg_df.iterrows()
        }
        deg_scores = np.asarray(
            [deg_lookup.get(gene, 0.0) for gene in gene_names],
            dtype=np.float32,
        )

    abs_delta = np.abs(predicted_delta)
    normalized_abs_delta = _minmax_normalize(abs_delta)
    normalized_deg = _minmax_normalize(deg_scores)
    importance_score = (
        abs_predicted_delta_weight * normalized_abs_delta
        + deg_significance_weight * normalized_deg
    )

    ranking_df = pd.DataFrame(
        {
            "gene": gene_names,
            "predicted_delta": predicted_delta,
            "abs_predicted_delta": abs_delta,
            "deg_significance": deg_scores,
            "importance_score": importance_score,
        }
    )
    ranking_df = ranking_df.sort_values("importance_score", ascending=False)
    ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)
    return ranking_df.reset_index(drop=True)

