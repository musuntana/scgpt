from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluation.inference import (
    build_gene_comparison_frame,
    build_perturbation_batch,
    predict_delta_for_batch,
    load_torch_model_for_bundle,
    summarize_perturbation_fit,
)
from src.models.shared_gene_transformer import SharedGeneIdentityTransformer


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


def test_shared_gene_identity_inference_uses_exact_gene_mapping():
    bundle = {
        "control_expression": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "target_delta": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "perturbation_index": np.array([0, 1], dtype=np.int64),
        "metadata": {
            "gene_names": ["p1", "g2"],
            "perturbation_names": ["p1", "p2"],
        },
    }
    model = SharedGeneIdentityTransformer(
        num_genes=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        dropout=0.0,
    )

    mapped_batch = build_perturbation_batch(bundle, "p1")
    assert mapped_batch.perturbation_gene_index == 0
    prediction = predict_delta_for_batch(model, mapped_batch)
    assert prediction.shape == (2,)

    unmapped_batch = build_perturbation_batch(bundle, "p2")
    assert unmapped_batch.perturbation_gene_index == -1
    with pytest.raises(ValueError, match="does not exactly match a saved gene name"):
        predict_delta_for_batch(model, unmapped_batch)


def test_shared_gene_identity_inference_rejects_ambiguous_gene_names():
    bundle = {
        "control_expression": np.array([[1.0, 2.0]], dtype=np.float32),
        "target_delta": np.array([[0.1, 0.2]], dtype=np.float32),
        "perturbation_index": np.array([0], dtype=np.int64),
        "metadata": {
            "gene_names": ["p1", "p1"],
            "perturbation_names": ["p1"],
        },
    }

    batch = build_perturbation_batch(bundle, "p1")

    assert batch.perturbation_gene_index == -1


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


def test_checkpoint_loading_requires_matching_bundle_before_model_init(tmp_path):
    from src.evaluation.protocol import fingerprint_processed_bundle

    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    (bundle_path / "arrays.npz").write_bytes(b"arrays")
    (bundle_path / "splits.npz").write_bytes(b"splits")
    (bundle_path / "metadata.json").write_bytes(b"metadata")
    protocol = {
        "schema_version": 2,
        "bundle_artifacts_sha256": fingerprint_processed_bundle(bundle_path),
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    checkpoint_path = run_dir / "best_model.pt"

    bundle = {
        "control_expression": np.ones((1, 2), dtype=np.float32),
        "metadata": {"gene_names": ["G1", "G2"], "perturbation_names": ["P1"]},
    }
    with pytest.raises(ValueError, match="bundle_dir is required"):
        load_torch_model_for_bundle(
            bundle=bundle, checkpoint_path=checkpoint_path,
            model_config_path="configs/model.yaml", model_type="transformer",
        )

    (bundle_path / "metadata.json").write_bytes(b"altered")
    with pytest.raises(ValueError, match="identity mismatch for metadata.json"):
        load_torch_model_for_bundle(
            bundle=bundle, checkpoint_path=checkpoint_path,
            model_config_path="configs/model.yaml", model_type="transformer",
            bundle_dir=bundle_path,
        )
