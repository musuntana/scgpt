from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.integrations.gears_export import build_gears_input_adata
from src.integrations.gears_runtime import (
    attach_train_only_loss_metadata,
    compute_absolute_expression_metrics,
    validate_gears_graph_coverage,
)


def _bundle() -> dict:
    labels = np.repeat(np.arange(4, dtype=np.int64), 2)
    return {
        "metadata": {
            "gene_names": ["g1", "g2"],
            "perturbation_names": ["A", "B", "C", "D"],
            "preprocessing_diagnostics": {
                "hvg_selection_method": "shared_seen_unseen_training_intersection",
                "heldout_perturbation_samples_excluded_from_hvg_fit": True,
                "protocols_protected": ["seen", "unseen"],
            },
        },
        "control_expression": np.ones((8, 2), dtype=np.float32),
        "target_delta": np.zeros((8, 2), dtype=np.float32),
        "perturbation_index": labels,
        "sample_ids": [f"s{i}" for i in range(8)],
        "splits": {
            "seen_train": np.array([0, 1, 2, 3], dtype=np.int64),
            "seen_val": np.array([4, 5], dtype=np.int64),
            "seen_test": np.array([6, 7], dtype=np.int64),
            "unseen_train": np.array([0, 1, 2, 3], dtype=np.int64),
            "unseen_val": np.array([4, 5], dtype=np.int64),
            "unseen_test": np.array([6, 7], dtype=np.int64),
        },
    }


def _exported() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "pert": ["control"] * 3 + ["A", "A", "B", "B", "C", "C", "D", "D"],
            "celltype": ["K562"] * 11,
        },
        index=["c0", "c1", "c2"] + [f"s{i}" for i in range(8)],
    )
    var = pd.DataFrame({"gene_name": ["G1", "G2"]}, index=["g1", "g2"])
    prepared = ad.AnnData(
        X=np.ones((11, 2), dtype=np.float32),
        obs=obs,
        var=var,
    )
    prepared.uns["scgpt_preprocessing"] = {
        "hvg_selection_method": "shared_seen_unseen_training_intersection",
        "heldout_perturbation_samples_excluded_from_hvg_fit": True,
        "protocols_protected": ["seen", "unseen"],
    }
    exported, _ = build_gears_input_adata(
        prepared,
        _bundle(),
        perturbation_col="pert",
        control_label="control",
        context_cols=["celltype"],
    )
    return exported


def test_train_only_metadata_uses_all_gene_masks_without_de_statistics():
    adata = _exported()
    adata.uns["rank_genes_groups_cov_all"] = {"leaked": ["g1"]}
    report = attach_train_only_loss_metadata(adata)

    assert report["status"] == "TRAIN_ONLY_RUNTIME_READY"
    assert report["heldout_expression_used_for_metadata"] is False
    assert report["de_statistics_computed"] is False
    assert report["all_gene_loss_mask"] is True
    assert "rank_genes_groups_cov_all" not in adata.uns
    assert "top_non_dropout_de_20" not in adata.uns
    assert set(adata.uns["non_zeros_gene_idx"]) == {
        "K562_ctrl_1",
        "K562_A+ctrl_1+1",
        "K562_B+ctrl_1+1",
        "K562_C+ctrl_1+1",
        "K562_D+ctrl_1+1",
    }
    for gene_indices in adata.uns["non_zeros_gene_idx"].values():
        np.testing.assert_array_equal(gene_indices, np.array([0, 1]))
    assert adata.uns["scgpt_gears_export"]["status"] == "TRAIN_ONLY_RUNTIME_READY"


def test_train_only_metadata_rejects_conditions_crossing_partitions():
    adata = _exported()
    adata.obs.loc["s0", "scgpt_split"] = "val"
    with pytest.raises(ValueError, match="appears in both"):
        attach_train_only_loss_metadata(adata)


def test_train_only_metadata_rejects_controls_outside_train():
    adata = _exported()
    adata.obs.loc["c0", "scgpt_split"] = "test"
    with pytest.raises(ValueError, match="Control cells must remain in train"):
        attach_train_only_loss_metadata(adata)


def test_graph_coverage_reports_missing_perturbation_genes():
    result = validate_gears_graph_coverage(["A", "B", "C"], {"A", "C"})
    assert result["verified"] is False
    assert result["missing_perturbations"] == ["B"]


def test_absolute_expression_metrics_are_explicitly_not_delta_comparable():
    bundle = _bundle()
    predictions = {
        "C": np.ones(2, dtype=np.float32),
        "D": np.ones(2, dtype=np.float32),
    }
    result = compute_absolute_expression_metrics(predictions, bundle)

    assert result["metric_space"] == "absolute_expression"
    assert result["target_is_delta_expression"] is False
    assert result["directly_comparable_to_repository_delta_metrics"] is False
    assert result["perturbation_count"] == 2
    assert result["mse_absolute_expression"] == 0.0
    assert set(result["per_perturbation"]) == {"C", "D"}


def test_absolute_expression_metrics_fail_if_prediction_is_missing():
    with pytest.raises(ValueError, match="Missing GEARS prediction"):
        compute_absolute_expression_metrics(
            {"C": np.ones(2, dtype=np.float32)},
            _bundle(),
        )
