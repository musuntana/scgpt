from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.torch_dataset import ProcessedDataset


def test_processed_dataset_reads_saved_bundle(tmp_path):
    arrays_path = tmp_path / "arrays.npz"
    splits_path = tmp_path / "splits.npz"
    metadata_path = tmp_path / "metadata.json"

    np.savez_compressed(
        arrays_path,
        control_expression=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        target_delta=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        perturbation_index=np.array([0, 1], dtype=np.int64),
        sample_ids=np.array(["s1", "s2"]),
    )
    np.savez_compressed(
        splits_path,
        seen_train=np.array([0], dtype=np.int64),
        seen_val=np.array([1], dtype=np.int64),
        seen_test=np.array([1], dtype=np.int64),
        unseen_train=np.array([0], dtype=np.int64),
        unseen_val=np.array([], dtype=np.int64),
        unseen_test=np.array([1], dtype=np.int64),
    )
    metadata_path.write_text(
        json.dumps(
            {"gene_names": ["g1", "g2"], "perturbation_names": ["p1", "p2"]},
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = ProcessedDataset(tmp_path, "seen_train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert tuple(sample["control_expression"].shape) == (2,)
    assert int(sample["perturbation_index"].item()) == 0
    assert tuple(sample["target_delta"].shape) == (2,)
    assert int(sample["perturbation_gene_index"].item()) == -1
    assert dataset.num_mapped_perturbations == 0
    assert dataset.unmapped_perturbation_names == ["p1", "p2"]


def test_processed_dataset_maps_exact_perturbation_gene_names(tmp_path):
    np.savez_compressed(
        tmp_path / "arrays.npz",
        control_expression=np.array([[1.0, 2.0]], dtype=np.float32),
        target_delta=np.array([[0.1, 0.2]], dtype=np.float32),
        perturbation_index=np.array([0], dtype=np.int64),
        sample_ids=np.array(["s1"]),
    )
    np.savez_compressed(
        tmp_path / "splits.npz",
        seen_train=np.array([0], dtype=np.int64),
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"gene_names": ["p1", "g2"], "perturbation_names": ["p1"]}),
        encoding="utf-8",
    )

    dataset = ProcessedDataset(tmp_path, "seen_train")
    sample = dataset[0]

    assert int(sample["perturbation_gene_index"].item()) == 0
    assert dataset.num_mapped_perturbations == 1
    assert dataset.unmapped_perturbation_names == []


def test_processed_dataset_raises_for_unknown_split(tmp_path):
    arrays_path = tmp_path / "arrays.npz"
    splits_path = tmp_path / "splits.npz"
    metadata_path = tmp_path / "metadata.json"

    np.savez_compressed(
        arrays_path,
        control_expression=np.array([[1.0, 2.0]], dtype=np.float32),
        target_delta=np.array([[0.1, 0.2]], dtype=np.float32),
        perturbation_index=np.array([0], dtype=np.int64),
        sample_ids=np.array(["s1"]),
    )
    np.savez_compressed(
        splits_path,
        seen_train=np.array([0], dtype=np.int64),
    )
    metadata_path.write_text(
        json.dumps({"gene_names": ["g1", "g2"], "perturbation_names": ["p1"]}),
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        ProcessedDataset(tmp_path, "missing_split")
