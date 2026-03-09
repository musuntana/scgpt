from __future__ import annotations

import json

import numpy as np

from src.utils.experiment import build_run_summary, write_run_summary


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

    output_path = write_run_summary(summary, output_dir / "run_summary.json")
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["model"]["model_type"] == "transformer"
