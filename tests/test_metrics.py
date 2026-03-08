from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_regression_metrics, topk_overlap


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

