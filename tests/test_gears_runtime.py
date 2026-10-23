from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.integrations.gears_export import build_gears_input_adata
from src.integrations.gears_runtime import (
    attach_train_only_loss_metadata,
    compute_absolute_expression_metrics,
    compute_control_relative_delta_metrics,
    validate_gears_gene_order,
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


def test_train_only_metadata_checks_sparse_matrix_without_densifying(monkeypatch):
    adata = _exported()
    adata.X = sparse.csr_matrix(adata.X)

    def forbidden_toarray(*args, **kwargs):
        raise AssertionError("sparse expression validation must not densify the matrix")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden_toarray)
    report = attach_train_only_loss_metadata(adata)

    assert report["status"] == "TRAIN_ONLY_RUNTIME_READY"
    assert report["de_statistics_computed"] is False


def test_train_only_metadata_rejects_non_finite_sparse_values():
    adata = _exported()
    adata.X = sparse.csr_matrix(adata.X)
    adata.X.data[0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        attach_train_only_loss_metadata(adata)


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
    predictions = {"D": np.ones(2, dtype=np.float32)}
    result = compute_absolute_expression_metrics(predictions, bundle)

    assert result["metric_space"] == "absolute_expression"
    assert result["target_is_delta_expression"] is False
    assert result["directly_comparable_to_repository_delta_metrics"] is False
    assert result["perturbation_count"] == 1
    assert result["mse_absolute_expression"] == 0.0
    assert set(result["per_perturbation"]) == {"D"}


def test_absolute_expression_metrics_reject_non_finite_prediction():
    with pytest.raises(ValueError, match="prediction.*non-finite"):
        compute_absolute_expression_metrics(
            {"D": np.array([1.0, np.nan], dtype=np.float32)},
            _bundle(),
        )


def test_absolute_expression_metrics_reject_non_finite_bundle_arrays():
    bundle = _bundle()
    bundle["target_delta"][6, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        compute_absolute_expression_metrics(
            {"D": np.ones(2, dtype=np.float32)},
            bundle,
        )


@pytest.mark.parametrize("invalid_index", [-1, 4])
def test_absolute_expression_metrics_reject_out_of_range_perturbation_indices(
    invalid_index: int,
) -> None:
    bundle = _bundle()
    bundle["perturbation_index"][6] = invalid_index

    with pytest.raises(ValueError, match="outside perturbation_names range"):
        compute_absolute_expression_metrics({"D": np.ones(2, dtype=np.float32)}, bundle)


def test_absolute_expression_metrics_reject_duplicate_split_indices() -> None:
    bundle = _bundle()
    bundle["splits"]["unseen_test"] = np.array([6, 6], dtype=np.int64)

    with pytest.raises(ValueError, match="duplicate sample indices"):
        compute_absolute_expression_metrics({"D": np.ones(2, dtype=np.float32)}, bundle)


def test_absolute_expression_metrics_reject_empty_gene_universe() -> None:
    bundle = _bundle()
    bundle["control_expression"] = np.empty((8, 0), dtype=np.float32)
    bundle["target_delta"] = np.empty((8, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one gene"):
        compute_absolute_expression_metrics({"D": np.empty(0, dtype=np.float32)}, bundle)


def test_absolute_expression_metrics_fail_if_prediction_is_missing():
    with pytest.raises(ValueError, match="Missing GEARS prediction"):
        compute_absolute_expression_metrics(
            {"C": np.ones(2, dtype=np.float32)},
            _bundle(),
        )


def test_gene_order_validation_records_exact_order_hashes() -> None:
    report = validate_gears_gene_order(
        bundle_gene_ids=["gene-1", "gene-2"],
        input_gene_ids=["gene-1", "gene-2"],
        input_gene_symbols=["G1", "G2"],
        gears_gene_symbols=["G1", "G2"],
    )

    assert report["verified"] is True
    assert report["bundle_input_gene_order_verified"] is True
    assert report["gears_symbol_order_verified"] is True
    assert report["gene_count"] == 2
    assert len(report["bundle_gene_ids_sha256"]) == 64
    assert report["input_gene_symbols_sha256"] == report["gears_gene_symbols_sha256"]


def test_gene_order_validation_rejects_bundle_order_mismatch() -> None:
    with pytest.raises(ValueError, match="gene ID order"):
        validate_gears_gene_order(
            bundle_gene_ids=["gene-1", "gene-2"],
            input_gene_ids=["gene-2", "gene-1"],
            input_gene_symbols=["G2", "G1"],
        )


def test_gene_order_validation_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_gears_gene_order(
            bundle_gene_ids=["gene-1", "gene-2"],
            input_gene_ids=["gene-1", "gene-2"],
            input_gene_symbols=["DUP", "DUP"],
        )


def test_gene_order_validation_rejects_gears_symbol_reordering() -> None:
    with pytest.raises(ValueError, match="symbol order"):
        validate_gears_gene_order(
            bundle_gene_ids=["gene-1", "gene-2"],
            input_gene_ids=["gene-1", "gene-2"],
            input_gene_symbols=["G1", "G2"],
            gears_gene_symbols=["G2", "G1"],
        )


def test_control_relative_delta_metrics_align_prediction_and_target_spaces() -> None:
    bundle = _bundle()
    bundle["target_delta"][6:8] = np.array([[0.5, -0.5], [0.5, -0.5]])

    result = compute_control_relative_delta_metrics(
        {"D": np.array([1.5, 0.5], dtype=np.float32)},
        bundle,
        control_reference_mean=np.array([1.0, 1.0], dtype=np.float32),
    )

    assert result["metric_space"] == "control_relative_delta_expression"
    assert result["directly_comparable_to_repository_delta_metrics"] is True
    assert result["mse_per_perturbation"] == pytest.approx(0.0)
    assert result["pearson_per_perturbation"] == pytest.approx(1.0)
    assert result["comparable_metric_keys"] == [
        "mse_per_perturbation",
        "pearson_per_perturbation",
    ]
    assert result["comparison_eligible"] is False


@pytest.mark.parametrize(
    ("baseline", "message"),
    [
        (np.array([1.0]), "one-dimensional vector"),
        (np.array([1.0, np.nan]), "non-finite"),
    ],
)
def test_control_relative_delta_metrics_reject_invalid_control_reference(
    baseline: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_control_relative_delta_metrics(
            {"D": np.ones(2, dtype=np.float32)},
            _bundle(),
            control_reference_mean=baseline,
        )


def test_control_relative_delta_metrics_rejects_non_vector_prediction() -> None:
    with pytest.raises(ValueError, match="one-dimensional vector"):
        compute_control_relative_delta_metrics(
            {"D": np.ones((1, 2), dtype=np.float32)},
            _bundle(),
            control_reference_mean=np.ones(2, dtype=np.float32),
        )


def test_control_relative_delta_metrics_rejects_non_finite_target_delta() -> None:
    bundle = _bundle()
    bundle["target_delta"][6, 0] = np.inf

    with pytest.raises(ValueError, match="target_delta contains non-finite"):
        compute_control_relative_delta_metrics(
            {"D": np.ones(2, dtype=np.float32)},
            bundle,
            control_reference_mean=np.ones(2, dtype=np.float32),
        )
