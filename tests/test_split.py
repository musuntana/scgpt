from __future__ import annotations

import numpy as np

from src.data.pairing import _can_stratify, _three_way_stratified_split


def test_can_stratify_requires_nonempty_labels_with_two_samples_per_class():
    assert _can_stratify(np.array([0, 0, 1, 1, 2, 2]))
    assert not _can_stratify(np.array([0, 0, 1, 2]))
    assert not _can_stratify(np.array([], dtype=np.int64))
    labels = np.repeat(np.arange(3), 2)
    assert not _can_stratify(labels, train_size=2, test_size=4)
    assert _can_stratify(labels, train_size=3, test_size=3)


def test_seen_split_preserves_class_counts_when_stratification_is_possible():
    labels = np.repeat(np.arange(10), 20)

    splits = _three_way_stratified_split(
        labels=labels,
        val_fraction=0.1,
        test_fraction=0.1,
        random_seed=42,
    )

    expected_counts = {
        "seen_train": 16,
        "seen_val": 2,
        "seen_test": 2,
    }
    for split_name, expected_per_class in expected_counts.items():
        counts = np.bincount(labels[splits[split_name]], minlength=10)
        np.testing.assert_array_equal(counts, np.full(10, expected_per_class))
