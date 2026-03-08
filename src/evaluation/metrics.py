from __future__ import annotations

from typing import Iterable

import numpy as np


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation with zero-variance safeguards."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute the elementwise mean squared error."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean((predictions - targets) ** 2))


def aggregate_by_label(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average samples by label and return ordered labels plus aggregated values."""
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    aggregated = np.vstack([values[labels == label].mean(axis=0) for label in unique_labels])
    return unique_labels, aggregated


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    perturbation_index: np.ndarray,
) -> dict[str, float]:
    """Compute local-first regression metrics for perturbation prediction."""
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    perturbation_index = np.asarray(perturbation_index, dtype=np.int64)

    _, aggregated_predictions = aggregate_by_label(predictions, perturbation_index)
    _, aggregated_targets = aggregate_by_label(targets, perturbation_index)

    per_perturbation_pearsons = [
        pearson_correlation(pred, target)
        for pred, target in zip(aggregated_predictions, aggregated_targets, strict=True)
    ]
    per_gene_pearsons = [
        pearson_correlation(predictions[:, gene_idx], targets[:, gene_idx])
        for gene_idx in range(predictions.shape[1])
    ]

    return {
        "overall_mse": mean_squared_error(predictions, targets),
        "mse_per_perturbation": mean_squared_error(
            aggregated_predictions, aggregated_targets
        ),
        "pearson_per_perturbation": float(np.mean(per_perturbation_pearsons)),
        "pearson_per_gene": float(np.mean(per_gene_pearsons)),
    }


def topk_overlap(
    predicted_genes: Iterable[str],
    true_genes: Iterable[str],
    k: int,
) -> float:
    """Compute overlap ratio between top-k predicted and true genes."""
    predicted_topk = list(predicted_genes)[:k]
    true_topk = list(true_genes)[:k]
    if k <= 0:
        raise ValueError("k must be positive")
    if not predicted_topk or not true_topk:
        return 0.0
    overlap = set(predicted_topk).intersection(true_topk)
    return float(len(overlap) / min(k, len(predicted_topk), len(true_topk)))

