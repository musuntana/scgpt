from __future__ import annotations

import json
import pickle
from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scripts.prepare_gears_baseline import prepare_gears_preflight
from src.integrations.gears_contract import (
    build_gears_split_contract,
    validate_source_identity,
)

NAMES = ["A", "B", "C", "D"]


def _bundle() -> dict:
    labels = np.repeat(np.arange(4, dtype=np.int64), 2)
    return {
        "metadata": {
            "gene_names": ["g1", "g2"],
            "perturbation_names": NAMES,
            "preprocessing_diagnostics": {
                "hvg_selection_method": "shared_seen_unseen_training_intersection",
                "heldout_perturbation_samples_excluded_from_hvg_fit": True,
                "protocols_protected": ["seen", "unseen"],
            },
        },
        "perturbation_index": labels,
        "sample_ids": [f"s{i}" for i in range(8)],
        "splits": {
            "unseen_train": np.array([0, 1, 2, 3], dtype=np.int64),
            "unseen_val": np.array([4, 5], dtype=np.int64),
            "unseen_test": np.array([6, 7], dtype=np.int64),
        },
    }


def test_gears_custom_split_has_group_disjoint_train_val_test():
    split, manifest = build_gears_split_contract(_bundle())

    assert split == {
        "train": ["ctrl", "A+ctrl", "B+ctrl"],
        "val": ["C+ctrl"],
        "test": ["D+ctrl"],
    }
    assert manifest["status"] == "PREFLIGHT_ONLY"
    assert manifest["sample_count"] == 8
    assert manifest["partitions"]["test"]["sample_count"] == 2
    assert manifest["comparison_eligible"] is False
    assert manifest["go_graph_coverage_verified"] is False


def test_gears_split_rejects_sample_level_group_overlap():
    bundle = _bundle()
    bundle["splits"]["unseen_train"] = np.array([0, 1, 2, 4])
    bundle["splits"]["unseen_val"] = np.array([3, 5])
    with pytest.raises(ValueError, match="overlapping train/val"):
        build_gears_split_contract(bundle)


def test_gears_split_rejects_missing_samples_and_duplicates():
    bundle = _bundle()
    bundle["splits"]["unseen_test"] = np.array([6, 6])
    with pytest.raises(ValueError, match="duplicate sample indices"):
        build_gears_split_contract(bundle)
    bundle["splits"]["unseen_test"] = np.array([6])
    with pytest.raises(ValueError, match="cover every perturbation sample"):
        build_gears_split_contract(bundle)


@pytest.mark.parametrize(
    "name",
    ["A+B", "A;B", "A B", "ctrl", "control", " A", "A|B"],
)
def test_gears_split_rejects_ambiguous_condition_names(name):
    bundle = _bundle()
    bundle["metadata"]["perturbation_names"][0] = name
    with pytest.raises(ValueError, match="label"):
        build_gears_split_contract(bundle)


def test_gears_split_rejects_legacy_preprocessing():
    bundle = _bundle()
    bundle["metadata"].pop("preprocessing_diagnostics")
    with pytest.raises(ValueError, match="training-only"):
        build_gears_split_contract(bundle)


def test_gears_split_rejects_out_of_range_perturbation_index():
    bundle = _bundle()
    bundle["perturbation_index"][0] = 99
    with pytest.raises(ValueError, match="outside metadata range"):
        build_gears_split_contract(bundle)


def _source_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "pert": ["control"] * 3 + ["A", "A", "B", "B", "C", "C", "D", "D"],
            "cell_type": ["K562"] * 11,
        },
        index=["c0", "c1", "c2"] + [f"s{i}" for i in range(8)],
    )
    var = pd.DataFrame(
        {"gene_name": ["G1", "G2", "G3"]}, index=["g1", "g2", "g3"]
    )
    return ad.AnnData(X=np.ones((11, 3)), obs=obs, var=var)


def test_source_identity_validation_is_explicitly_not_metric_validation():
    report = validate_source_identity(
        _source_adata(), _bundle(),
        perturbation_col="pert",
        control_label="control",
    )
    assert report["source_cell_identity_verified"] is True
    assert report["matched_perturbed_sample_count"] == 8
    assert report["source_control_sample_count"] == 3
    assert report["comparison_eligible"] is False


def test_source_identity_detects_corrupted_cell_labels():
    source = _source_adata()
    source.obs.loc["s0", "pert"] = "D"
    with pytest.raises(ValueError, match="labels differ"):
        validate_source_identity(
            source, _bundle(), perturbation_col="pert", control_label="control"
        )


def test_source_identity_rejects_missing_gene_symbol_annotation():
    source = _source_adata()
    source.var.drop(columns=["gene_name"], inplace=True)
    with pytest.raises(ValueError, match="gene_name"):
        validate_source_identity(
            source, _bundle(), perturbation_col="pert", control_label="control"
        )


def test_preflight_export_creates_pickle_json_and_hashes_without_gears(tmp_path):
    bundle = _bundle()
    source = tmp_path / "bundle"
    source.mkdir()
    np.savez_compressed(
        source / "arrays.npz",
        control_expression=np.ones((8, 2), dtype=np.float32),
        target_delta=np.zeros((8, 2), dtype=np.float32),
        perturbation_index=bundle["perturbation_index"],
        sample_ids=np.array(bundle["sample_ids"]),
    )
    np.savez_compressed(source / "splits.npz", **bundle["splits"])
    (source / "metadata.json").write_text(
        json.dumps(bundle["metadata"]), encoding="utf-8"
    )
    output = tmp_path / "gears_export"

    manifest = prepare_gears_preflight(source, output)

    with (output / "gears_custom_split.pkl").open("rb") as handle:
        loaded_split = pickle.load(handle)
    assert loaded_split["test"] == ["D+ctrl"]
    json_split = json.loads((output / "gears_custom_split.json").read_text())
    assert json_split == loaded_split
    saved = json.loads((output / "gears_preflight.json").read_text())
    assert saved == manifest
    assert len(manifest["bundle_artifacts_sha256"]["splits.npz"]) == 64
    assert manifest["source_cell_identity_verified"] is False
    assert manifest["comparison_eligible"] is False

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_gears_preflight(source, output)


def test_preflight_does_not_export_when_group_split_is_invalid(tmp_path):
    bundle = deepcopy(_bundle())
    bundle["splits"]["unseen_val"] = np.array([2, 5])
    source = tmp_path / "bundle"
    source.mkdir()
    np.savez_compressed(
        source / "arrays.npz",
        perturbation_index=bundle["perturbation_index"],
        sample_ids=np.array(bundle["sample_ids"]),
    )
    np.savez_compressed(source / "splits.npz", **bundle["splits"])
    (source / "metadata.json").write_text(
        json.dumps(bundle["metadata"]), encoding="utf-8"
    )
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="pairwise disjoint"):
        prepare_gears_preflight(source, output)
    assert not output.exists()
