from __future__ import annotations

import numpy as np

from src.evaluation.inference import (
    build_gene_comparison_frame,
    build_perturbation_batch,
    summarize_perturbation_fit,
)


def test_build_perturbation_batch_aggregates_matching_rows():
    bundle = {
        "control_expression": np.array(
            [[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]],
            dtype=np.float32,
        ),
        "target_delta": np.array(
            [[0.1, 0.2], [0.3, 0.4], [1.0, 2.0]],
            dtype=np.float32,
        ),
        "perturbation_index": np.array([0, 0, 1], dtype=np.int64),
        "metadata": {
            "gene_names": ["g1", "g2"],
            "perturbation_names": ["p1", "p2"],
        },
    }

    batch = build_perturbation_batch(bundle, "p1")

    assert batch.sample_count == 2
    np.testing.assert_allclose(batch.control_mean, np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(
        batch.observed_delta_mean, np.array([0.2, 0.3], dtype=np.float32)
    )


def test_gene_comparison_and_fit_summary_are_consistent():
    comparison = build_gene_comparison_frame(
        gene_names=["g1", "g2", "g3"],
        predicted_delta=np.array([0.5, -0.2, 0.1], dtype=np.float32),
        observed_delta=np.array([0.4, -0.1, 0.0], dtype=np.float32),
    )
    fit = summarize_perturbation_fit(
        predicted_delta=np.array([0.5, -0.2, 0.1], dtype=np.float32),
        observed_delta=np.array([0.4, -0.1, 0.0], dtype=np.float32),
    )

    assert list(comparison.columns) == [
        "gene",
        "predicted_delta",
        "observed_delta",
        "abs_predicted_delta",
        "abs_observed_delta",
        "residual",
        "abs_residual",
    ]
    assert fit["pearson"] > 0.9
    assert fit["mse"] >= 0.0
