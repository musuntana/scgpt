from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.prepare_gears_baseline import prepare_gears_preflight
from src.data.io import load_anndata
from src.data.pairing import build_training_bundle, save_processed_bundle
from src.data.preprocess import prepare_adata, select_hvgs_from_training_partitions
from src.data.schema import resolve_schema
from src.integrations.gears_export import build_gears_input_adata, validate_rebuilt_bundle
from src.utils.config import load_yaml


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


def test_source_h5ad_reprocessing_exports_exact_gears_input(tmp_path):
    rng = np.random.default_rng(19)
    perturbations = ["control"] * 8
    sample_ids = [f"control-{index}" for index in range(8)]
    for perturbation in ("A", "B", "C", "D"):
        perturbations.extend([perturbation] * 8)
        sample_ids.extend(f"{perturbation}-{index}" for index in range(8))

    raw = ad.AnnData(
        X=rng.poisson(5, size=(len(perturbations), 12)).astype(np.float32),
        obs=pd.DataFrame(
            {"pert": perturbations, "celltype": ["K562"] * len(perturbations)},
            index=sample_ids,
        ),
        var=pd.DataFrame(
            {"gene_name": [f"GENE{index}" for index in range(12)]},
            index=[f"g{index}" for index in range(12)],
        ),
    )
    raw_path = tmp_path / "raw.h5ad"
    raw.write_h5ad(raw_path)
    config = {
        "dataset": {"control_label": "control"},
        "schema": {
            "preset": "",
            "perturbation_col": "pert",
            "batch_col": "none",
            "context_cols": ["celltype"],
            "control_label_candidates": ["control"],
            "perturbation_col_candidates": ["pert"],
            "batch_col_candidates": [],
            "context_col_candidates": ["celltype"],
            "multi_gene_delimiters": [";", "+", ",", "|", "_"],
        },
        "preprocess": {
            "min_genes_per_cell": 1,
            "min_cells_per_gene": 1,
            "normalize_total_target_sum": 10000,
            "log1p": True,
            "hvg_top_genes": 4,
            "max_cells_per_perturbation": None,
        },
        "pairing": {"target": "delta_expression"},
        "split": {"val_fraction": 0.25, "test_fraction": 0.25, "random_seed": 42},
    }
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded_config = load_yaml(config_path)
    schema = resolve_schema(
        adata=load_anndata(raw_path),
        dataset_config=loaded_config,
        cli_perturbation_col="auto",
        cli_control_label=None,
        cli_batch_col="auto",
        cli_context_cols=["auto"],
    )
    prepared = prepare_adata(
        adata=load_anndata(raw_path),
        preprocess_config=dict(loaded_config["preprocess"]),
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        random_seed=42,
        select_hvg=False,
        filter_genes_before_split=False,
        normalize_expression=False,
    )
    prepared = select_hvgs_from_training_partitions(
        adata=prepared,
        preprocess_config=dict(loaded_config["preprocess"]),
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        val_fraction=0.25,
        test_fraction=0.25,
        random_seed=42,
    )
    initial_bundle = build_training_bundle(
        adata=prepared,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        batch_col=schema.batch_col,
        context_cols=schema.context_cols,
        val_fraction=0.25,
        test_fraction=0.25,
        random_seed=42,
    )
    bundle_dir = tmp_path / "bundle"
    save_processed_bundle(initial_bundle, bundle_dir)

    output_dir = tmp_path / "export"
    manifest = prepare_gears_preflight(
        bundle_dir,
        output_dir,
        source_h5ad=raw_path,
        data_config_path=config_path,
    )

    assert manifest["source_preprocessing_reproduction_verified"] is True
    assert manifest["gears_input_h5ad_exported"] is True
    assert manifest["source_validation"]["source_cell_identity_verified"] is True
    assert manifest["source_preprocessing_reproduction"]["split_indices_match"] is True
    assert manifest["gears_input"]["expression_reconstruction_verified"] is True
    assert len(manifest["gears_input_h5ad_sha256"]) == 64

    exported = ad.read_h5ad(output_dir / "gears_input.h5ad")
    assert exported.n_obs == len(initial_bundle.sample_ids) + 8
    assert exported.n_vars == len(initial_bundle.gene_names)
    assert {"condition", "cell_type", "scgpt_sample_id", "scgpt_split"}.issubset(
        exported.obs.columns
    )
    assert set(exported.obs.loc[exported.obs["condition"] == "ctrl", "scgpt_split"]) == {
        "train"
    }
    assert exported.uns["scgpt_gears_export"]["de_statistics_computed"] is False


def test_preflight_rejects_existing_artifact_before_loading_bundle(tmp_path, monkeypatch):
    output_dir = tmp_path / "existing-export"
    output_dir.mkdir()
    (output_dir / "gears_preflight.json").write_text("{}", encoding="utf-8")

    def unexpected_bundle_load(*args, **kwargs):
        raise AssertionError("bundle loading must not start when output is already occupied")

    monkeypatch.setattr(
        "scripts.prepare_gears_baseline.load_processed_bundle",
        unexpected_bundle_load,
    )
    with pytest.raises(FileExistsError, match="gears_preflight.json"):
        prepare_gears_preflight(tmp_path / "missing-bundle", output_dir)


def test_preflight_rejects_file_output_path_before_loading_bundle(tmp_path, monkeypatch):
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("keep me", encoding="utf-8")

    def unexpected_bundle_load(*args, **kwargs):
        raise AssertionError("bundle loading must not start when output path is a file")

    monkeypatch.setattr(
        "scripts.prepare_gears_baseline.load_processed_bundle",
        unexpected_bundle_load,
    )
    with pytest.raises(NotADirectoryError, match="not a directory"):
        prepare_gears_preflight(tmp_path / "missing-bundle", output_path)
    assert output_path.read_text(encoding="utf-8") == "keep me"
