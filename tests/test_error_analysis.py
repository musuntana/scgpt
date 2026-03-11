from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.error_analysis import (
    build_error_summary,
    build_failure_mode_count_frame,
    build_per_perturbation_error_table,
    build_worst_conditions_frame,
    select_perturbation_diagnostics,
)


def test_build_per_perturbation_error_table_reports_failure_modes_and_top_genes():
    predictions = np.array(
        [
            [0.10, 0.02, 0.00],
            [0.12, 0.01, 0.01],
            [0.11, 0.02, 0.00],
            [0.09, 0.01, 0.01],
            [0.02, 0.01, 0.00],
            [0.01, 0.00, 0.00],
            [0.02, 0.01, 0.00],
            [0.01, 0.01, 0.00],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [0.80, 0.20, 0.05],
            [0.75, 0.18, 0.04],
            [0.78, 0.19, 0.05],
            [0.76, 0.21, 0.05],
            [0.02, 0.01, 0.00],
            [0.01, 0.01, 0.00],
            [0.02, 0.01, 0.00],
            [0.01, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    perturbation_index = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    deg_df = pd.DataFrame(
        {
            "perturbation": ["pertA", "pertA", "pertB"],
            "gene": ["g1", "g2", "g1"],
        }
    )

    frame = build_per_perturbation_error_table(
        predictions=predictions,
        targets=targets,
        perturbation_index=perturbation_index,
        perturbation_names=["pertA", "pertB"],
        gene_names=["g1", "g2", "g3"],
        deg_df=deg_df,
        k_values=[2],
        top_gene_count=2,
    )

    assert list(frame["perturbation"]) == ["pertA", "pertB"]
    assert "topk_deg_overlap_2" in frame.columns
    assert frame.loc[0, "failure_mode"] == "underestimates_response_magnitude"
    assert frame.loc[0, "top_residual_genes"].startswith("g1")
    assert frame.loc[0, "sample_count"] == 4


def test_build_error_summary_surfaces_worst_conditions():
    frame = pd.DataFrame(
        [
            {
                "perturbation": "pert_bad",
                "sample_count": 4,
                "pearson": -0.20,
                "mse": 0.50,
                "failure_mode": "directional_mismatch",
                "error_to_signal_ratio": 2.0,
                "top_residual_genes": "g1,g2",
            },
            {
                "perturbation": "pert_ok",
                "sample_count": 6,
                "pearson": 0.85,
                "mse": 0.02,
                "failure_mode": "mostly_aligned",
                "error_to_signal_ratio": 0.2,
                "top_residual_genes": "g3,g4",
            },
        ]
    )

    summary = build_error_summary(frame, split_name="unseen_test", model_type="transformer")

    assert summary["num_perturbations"] == 2
    assert summary["failure_mode_counts"]["directional_mismatch"] == 1
    assert summary["worst_by_pearson"][0]["perturbation"] == "pert_bad"
    assert summary["worst_by_mse"][0]["perturbation"] == "pert_bad"


def test_error_summary_display_helpers_build_sorted_frames():
    summary = {
        "num_perturbations": 3,
        "failure_mode_counts": {
            "low_signal_condition": 2,
            "mostly_aligned": 1,
        },
        "worst_by_pearson": [
            {
                "perturbation": "pert_bad",
                "sample_count": 4,
                "pearson": -0.4,
                "failure_mode": "directional_mismatch",
                "top_residual_genes": "g1,g2",
            },
            {
                "perturbation": "pert_mid",
                "sample_count": 5,
                "pearson": 0.1,
                "failure_mode": "low_signal_condition",
                "top_residual_genes": "g3,g4",
            },
        ],
        "worst_by_mse": [
            {
                "perturbation": "pert_bad",
                "sample_count": 4,
                "mse": 0.5,
                "failure_mode": "directional_mismatch",
                "top_residual_genes": "g1,g2",
            },
            {
                "perturbation": "pert_mid",
                "sample_count": 5,
                "mse": 0.2,
                "failure_mode": "low_signal_condition",
                "top_residual_genes": "g3,g4",
            },
        ],
    }

    failure_mode_frame = build_failure_mode_count_frame(summary)
    worst_pearson_frame = build_worst_conditions_frame(summary, rank_by="worst_by_pearson", top_n=1)
    worst_mse_frame = build_worst_conditions_frame(summary, rank_by="worst_by_mse", top_n=2)

    assert list(failure_mode_frame["failure_mode"]) == ["low_signal_condition", "mostly_aligned"]
    assert list(failure_mode_frame["count"]) == [2, 1]
    assert list(worst_pearson_frame["perturbation"]) == ["pert_bad"]
    assert list(worst_mse_frame["perturbation"]) == ["pert_bad", "pert_mid"]


def test_select_perturbation_diagnostics_returns_compact_row_dict():
    frame = pd.DataFrame(
        [
            {
                "perturbation": "pert_bad",
                "sample_count": 4,
                "pearson": -0.2,
                "mse": 0.5,
                "failure_mode": "directional_mismatch",
                "error_to_signal_ratio": 2.0,
                "top_residual_genes": "g1,g2",
            },
            {
                "perturbation": "pert_ok",
                "sample_count": 6,
                "pearson": 0.85,
                "mse": 0.02,
                "failure_mode": "mostly_aligned",
                "error_to_signal_ratio": 0.2,
                "top_residual_genes": "g3,g4",
            },
        ]
    )

    diagnostics = select_perturbation_diagnostics(frame, perturbation_name="pert_ok")
    missing = select_perturbation_diagnostics(frame, perturbation_name="missing")

    assert diagnostics["perturbation"] == "pert_ok"
    assert diagnostics["failure_mode"] == "mostly_aligned"
    assert diagnostics["sample_count"] == 6
    assert missing == {}
