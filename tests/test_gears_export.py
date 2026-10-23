from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.integrations.gears_export import build_gears_input_adata, validate_rebuilt_bundle


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


def _prepared() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "pert": ["control"] * 3 + ["A", "A", "B", "B", "C", "C", "D", "D"],
            "celltype": ["K562"] * 11,
        },
        index=["c0", "c1", "c2"] + [f"s{i}" for i in range(8)],
    )
    var = pd.DataFrame({"gene_name": ["G1", "G2"]}, index=["g1", "g2"])
    adata = ad.AnnData(X=np.ones((11, 2), dtype=np.float32), obs=obs, var=var)
    adata.uns["scgpt_preprocessing"] = {
        "hvg_selection_method": "shared_seen_unseen_training_intersection",
        "heldout_perturbation_samples_excluded_from_hvg_fit": True,
        "protocols_protected": ["seen", "unseen"],
    }
    return adata


def test_gears_input_preserves_exact_cells_and_group_split():
    exported, report = build_gears_input_adata(
        _prepared(),
        _bundle(),
        perturbation_col="pert",
        control_label="control",
        context_cols=["celltype"],
    )

    assert exported.n_obs == 11
    assert exported.n_vars == 2
    assert exported.obs["condition"].value_counts().to_dict() == {
        "ctrl": 3,
        "A+ctrl": 2,
        "B+ctrl": 2,
        "C+ctrl": 2,
        "D+ctrl": 2,
    }
    assert set(exported.obs.loc[exported.obs["condition"] == "ctrl", "scgpt_split"]) == {
        "train"
    }
    for sample_id in ("s0", "s1", "s2", "s3"):
        assert exported.obs.loc[sample_id, "scgpt_split"] == "train"
    for sample_id in ("s4", "s5"):
        assert exported.obs.loc[sample_id, "scgpt_split"] == "val"
    for sample_id in ("s6", "s7"):
        assert exported.obs.loc[sample_id, "scgpt_split"] == "test"

    assert exported.obs["cell_type"].unique().tolist() == ["K562"]
    assert exported.var["gene_name"].tolist() == ["G1", "G2"]
    assert report["expression_reconstruction_verified"] is True
    assert report["de_statistics_computed"] is False
    assert report["comparison_eligible"] is False
    assert exported.uns["scgpt_gears_export"]["status"] == "INPUT_ONLY_NO_MODEL_RUN"


def test_gears_input_rejects_expression_that_does_not_match_saved_bundle():
    adata = _prepared()
    adata.X[3, 0] = 9.0
    with pytest.raises(ValueError, match="expression does not match"):
        build_gears_input_adata(
            adata,
            _bundle(),
            perturbation_col="pert",
            control_label="control",
            context_cols=["celltype"],
        )


def test_gears_input_rejects_unbundled_perturbed_cells():
    adata = _prepared()
    adata.obs.loc["s7", "pert"] = "E"
    with pytest.raises(ValueError, match="labels differ|outside the saved bundle"):
        build_gears_input_adata(
            adata,
            _bundle(),
            perturbation_col="pert",
            control_label="control",
            context_cols=["celltype"],
        )


def test_gears_input_rejects_multiple_cell_types():
    adata = _prepared()
    adata.obs.loc["s0", "celltype"] = "another-cell-line"
    with pytest.raises(ValueError, match="does not support cross-cell-type"):
        build_gears_input_adata(
            adata,
            _bundle(),
            perturbation_col="pert",
            control_label="control",
            context_cols=["celltype"],
        )


def test_gears_input_requires_unique_verified_gene_symbols():
    adata = _prepared()
    adata.var.loc["g2", "gene_name"] = "G1"
    with pytest.raises(ValueError, match="must be unique"):
        build_gears_input_adata(
            adata,
            _bundle(),
            perturbation_col="pert",
            control_label="control",
            context_cols=["celltype"],
        )


def test_gears_input_rejects_legacy_preprocessing():
    adata = _prepared()
    adata.uns["scgpt_preprocessing"] = {}
    with pytest.raises(ValueError, match="training-only"):
        build_gears_input_adata(
            adata,
            _bundle(),
            perturbation_col="pert",
            control_label="control",
            context_cols=["celltype"],
        )


def test_rebuilt_bundle_validation_records_all_identity_checks():
    bundle = _bundle()
    rebuilt = SimpleNamespace(
        sample_ids=list(bundle["sample_ids"]),
        gene_names=list(bundle["metadata"]["gene_names"]),
        perturbation_names=list(bundle["metadata"]["perturbation_names"]),
        perturbation_index=bundle["perturbation_index"].copy(),
        control_expression=bundle["control_expression"].copy(),
        target_delta=bundle["target_delta"].copy(),
        splits={key: value.copy() for key, value in bundle["splits"].items()},
    )
    report = validate_rebuilt_bundle(rebuilt, bundle)
    assert report["verified"] is True
    assert report["split_indices_match"] is True
    assert report["target_delta_match"] is True


def test_rebuilt_bundle_validation_rejects_changed_split_membership():
    bundle = _bundle()
    rebuilt = SimpleNamespace(
        sample_ids=list(bundle["sample_ids"]),
        gene_names=list(bundle["metadata"]["gene_names"]),
        perturbation_names=list(bundle["metadata"]["perturbation_names"]),
        perturbation_index=bundle["perturbation_index"].copy(),
        control_expression=bundle["control_expression"].copy(),
        target_delta=bundle["target_delta"].copy(),
        splits={key: value.copy() for key, value in bundle["splits"].items()},
    )
    rebuilt.splits["unseen_test"] = np.array([5, 6], dtype=np.int64)
    with pytest.raises(ValueError, match="split unseen_test"):
        validate_rebuilt_bundle(rebuilt, bundle)
