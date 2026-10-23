from __future__ import annotations

import hashlib
import json

import numpy as np

from src.utils.experiment import (
    _fingerprint_file,
    _git_provenance,
    build_run_summary,
    write_run_summary,
)


def test_build_run_summary_collects_required_fields(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    np.savez_compressed(
        bundle_dir / "arrays.npz",
        control_expression=np.array([[1.0, 2.0]], dtype=np.float32),
        target_delta=np.array([[0.1, 0.2]], dtype=np.float32),
        perturbation_index=np.array([0], dtype=np.int64),
        sample_ids=np.array(["s1"]),
    )
    np.savez_compressed(
        bundle_dir / "splits.npz",
        seen_train=np.array([0], dtype=np.int64),
        seen_val=np.array([], dtype=np.int64),
        seen_test=np.array([], dtype=np.int64),
        unseen_train=np.array([], dtype=np.int64),
        unseen_val=np.array([], dtype=np.int64),
        unseen_test=np.array([], dtype=np.int64),
    )
    (bundle_dir / "metadata.json").write_text(
        json.dumps({"gene_names": ["g1", "g2"], "perturbation_names": ["p1"]}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps(
            [
                {"epoch": 1.0, "pearson_per_perturbation": 0.2, "train_loss": 0.5},
                {"epoch": 2.0, "pearson_per_perturbation": 0.4, "train_loss": 0.4},
            ]
        ),
        encoding="utf-8",
    )
    seen_metrics_path = output_dir / "seen_test_metrics.json"
    seen_metrics_path.write_text(
        json.dumps({"pearson_per_perturbation": 0.35, "overall_mse": 0.1}),
        encoding="utf-8",
    )
    checkpoint_path = output_dir / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    summary = build_run_summary(
        bundle_dir=bundle_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        model_type="transformer",
        split_prefix="seen",
        data_config_path="configs/data.yaml",
        model_config_path="configs/model.yaml",
        train_config_path="configs/train.yaml",
        history_path=history_path,
        seen_metrics_path=seen_metrics_path,
    )

    assert summary["dataset"]["name"] == "scperturb_norman2019"
    assert summary["validation"]["best_epoch"] == 2
    assert summary["artifacts"]["bundle"]["num_samples"] == 1
    assert summary["test_metrics"]["seen_test"]["overall_mse"] == 0.1
    provenance = summary["provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["runtime"]["python"]
    assert provenance["experiment"]["split_protocol"] == "stratified_within_perturbation_condition"
    assert provenance["inputs"]["processed_bundle"]["arrays.npz"]["sha256"]
    assert provenance["inputs"]["configs"]["data"]["status"] == "available"
    assert provenance["outputs"]["checkpoint"]["sha256"] == hashlib.sha256(
        b"checkpoint"
    ).hexdigest()
    assert provenance["inputs"]["raw_dataset"]["status"] in {"available", "not_found"}
    split_audit = summary["split"]["audit"]
    assert split_audit["sample_count"] == 1
    assert split_audit["sample_counts"]["seen_train"] == 1
    assert split_audit["integrity"]["seen"]["complete_sample_coverage"] is True
    assert split_audit["seen_stratification"]["status"] == "not_recorded_legacy_bundle"

    output_path = write_run_summary(summary, output_dir / "run_summary.json")
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["model"]["model_type"] == "transformer"


def test_build_run_summary_can_override_recorded_seed(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    np.savez_compressed(
        bundle_dir / "arrays.npz",
        control_expression=np.array([[1.0, 2.0]], dtype=np.float32),
        target_delta=np.array([[0.1, 0.2]], dtype=np.float32),
        perturbation_index=np.array([0], dtype=np.int64),
        sample_ids=np.array(["s1"]),
    )
    np.savez_compressed(
        bundle_dir / "splits.npz",
        seen_train=np.array([0], dtype=np.int64),
        seen_val=np.array([], dtype=np.int64),
        seen_test=np.array([], dtype=np.int64),
        unseen_train=np.array([], dtype=np.int64),
        unseen_val=np.array([], dtype=np.int64),
        unseen_test=np.array([], dtype=np.int64),
    )
    (bundle_dir / "metadata.json").write_text(
        json.dumps({"gene_names": ["g1", "g2"], "perturbation_names": ["p1"]}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    checkpoint_path = output_dir / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    summary = build_run_summary(
        bundle_dir=bundle_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        model_type="transformer",
        split_prefix="seen",
        data_config_path="configs/data.yaml",
        model_config_path="configs/model.yaml",
        train_config_path="configs/train.yaml",
        seed=7,
    )

    assert summary["training"]["seed"] == 7



def test_fingerprint_file_is_deterministic_and_reports_missing_files(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"streamed-hash-payload")

    first = _fingerprint_file(path)
    second = _fingerprint_file(path)

    assert first["status"] == "available"
    assert first["sha256"] == second["sha256"]
    assert first["sha256"] == hashlib.sha256(b"streamed-hash-payload").hexdigest()
    assert first["size_bytes"] == len(b"streamed-hash-payload")

    missing = _fingerprint_file(tmp_path / "missing.bin")
    assert missing["status"] == "not_found"
    assert missing["sha256"] is None

    not_provided = _fingerprint_file(None)
    assert not_provided["status"] == "not_provided"


def test_git_provenance_is_optional_outside_a_repository(tmp_path):
    git = _git_provenance(tmp_path)

    assert git["status"] == "unavailable"
    assert git["commit"] is None
    assert git["working_tree"] == "unknown"
