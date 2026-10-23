from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluation.protocol import (
    fingerprint_processed_bundle,
    matching_test_split,
    validate_checkpoint_bundle_identity,
    validate_split_matches_training_protocol,
)


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [("seen", "seen_test"), ("unseen", "unseen_test")],
)
def test_matching_test_split_returns_only_the_selected_protocol(prefix, expected):
    assert matching_test_split(prefix, {"seen_test", "unseen_test"}) == expected


def test_matching_test_split_rejects_missing_matching_split():
    with pytest.raises(ValueError, match="does not contain the matching test split"):
        matching_test_split("unseen", {"seen_train", "seen_test"})


def test_matching_test_split_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="expected 'seen' or 'unseen'"):
        matching_test_split("cross_validation", {"seen_test", "unseen_test"})



@pytest.mark.parametrize(
    ("train_protocol", "split_name"),
    [
        ("seen", "unseen_test"),
        ("unseen", "seen_test"),
        ("seen", "unseen_val"),
    ],
)
def test_checkpoint_protocol_validation_rejects_cross_protocol_split(
    train_protocol, split_name
):
    with pytest.raises(ValueError, match="Checkpoint was trained with"):
        validate_split_matches_training_protocol(train_protocol, split_name)


@pytest.mark.parametrize(
    ("train_protocol", "split_name"),
    [("seen", "seen_test"), ("unseen", "unseen_test"), (None, "unseen_test")],
)
def test_checkpoint_protocol_validation_allows_matching_or_legacy_protocol(
    train_protocol, split_name
):
    validate_split_matches_training_protocol(train_protocol, split_name)


def _create_identity_bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    np.savez_compressed(
        root / "arrays.npz",
        control_expression=np.array([[1.0, 2.0]], dtype=np.float32),
    )
    np.savez_compressed(root / "splits.npz", unseen_train=np.array([0]))
    (root / "metadata.json").write_text(
        json.dumps({"gene_names": ["G1", "G2"], "perturbation_names": ["P1"]}),
        encoding="utf-8",
    )
    return root


def test_checkpoint_bundle_identity_roundtrip(tmp_path):
    root = _create_identity_bundle(tmp_path)
    hashes = fingerprint_processed_bundle(root)
    protocol = {"schema_version": 2, "bundle_artifacts_sha256": hashes}

    assert set(hashes) == {"arrays.npz", "splits.npz", "metadata.json"}
    assert validate_checkpoint_bundle_identity(protocol, root) is True


@pytest.mark.parametrize("artifact", ["arrays.npz", "splits.npz", "metadata.json"])
def test_checkpoint_bundle_identity_rejects_same_shape_changed_data(tmp_path, artifact):
    root = _create_identity_bundle(tmp_path)
    protocol = {
        "schema_version": 2,
        "bundle_artifacts_sha256": fingerprint_processed_bundle(root),
    }
    # Alter contents without changing the file name or tensor dimensions.
    with (root / artifact).open("ab") as handle:
        handle.write(b"changed-without-changing-shape")

    with pytest.raises(ValueError, match=f"identity mismatch for {artifact}"):
        validate_checkpoint_bundle_identity(protocol, root)


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"arrays.npz": "0" * 64},
        {"arrays.npz": "invalid", "splits.npz": "0" * 64, "metadata.json": "0" * 64},
    ],
)
def test_checkpoint_bundle_identity_rejects_incomplete_hash_contract(tmp_path, invalid):
    root = _create_identity_bundle(tmp_path)
    with pytest.raises(ValueError, match="must contain exactly three valid"):
        validate_checkpoint_bundle_identity(
            {"schema_version": 2, "bundle_artifacts_sha256": invalid},
            root,
        )


def test_checkpoint_bundle_identity_marks_legacy_unverified(tmp_path):
    root = _create_identity_bundle(tmp_path)
    assert validate_checkpoint_bundle_identity({"schema_version": 1}, root) is False
    with pytest.raises(ValueError, match="missing required bundle_artifacts_sha256"):
        validate_checkpoint_bundle_identity({"schema_version": 2}, root)
