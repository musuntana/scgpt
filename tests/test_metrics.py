from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_topk_deg_metrics,
    topk_overlap,
)


def test_compute_regression_metrics_returns_expected_keys():
    predictions = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 11.0], [11.0, 12.0]])
    targets = np.array([[1.1, 2.1], [1.9, 2.9], [9.9, 10.9], [11.2, 12.2]])
    perturbation_index = np.array([0, 0, 1, 1])

    metrics = compute_regression_metrics(predictions, targets, perturbation_index)

    assert "overall_mse" in metrics
    assert "pearson_per_perturbation" in metrics
    assert metrics["overall_mse"] >= 0.0


def test_topk_overlap_computes_fraction():
    score = topk_overlap(["g1", "g2", "g3"], ["g2", "g3", "g4"], k=2)
    assert score == 0.5


def test_compute_topk_deg_metrics_returns_expected_keys():
    """Top-k DEG overlap should return a dict with one key per k value."""
    # 4 samples, 3 genes, 2 perturbations
    # Perturbation 0: genes g1=large, g2=medium, g3=small
    # Perturbation 1: genes g3=large, g2=medium, g1=small
    predictions = np.array(
        [
            [5.0, 2.0, 0.1],
            [4.0, 1.5, 0.2],
            [0.1, 1.0, 4.0],
            [0.2, 1.5, 5.0],
        ],
        dtype=np.float32,
    )
    perturbation_index = np.array([0, 0, 1, 1], dtype=np.int64)
    gene_names = ["g1", "g2", "g3"]
    perturbation_names = ["pertA", "pertB"]

    # DEG artifact: pertA top genes are g1, g2; pertB top gene is g3
    deg_df = pd.DataFrame(
        {
            "perturbation": ["pertA", "pertA", "pertB"],
            "gene": ["g1", "g2", "g3"],
            "deg_significance": [5.0, 3.0, 6.0],
        }
    )

    results = compute_topk_deg_metrics(
        predictions=predictions,
        perturbation_index=perturbation_index,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        deg_df=deg_df,
        k_values=[2],
    )

    assert "topk_deg_overlap_2" in results
    # pertA predicted top-2 = [g1, g2] vs true top-2 = [g1, g2] -> overlap 1.0
    # pertB predicted top-2 = [g3, g2] vs true top-1 padded to top-2 = [g3] -> overlap 1.0
    assert results["topk_deg_overlap_2"] > 0.0

