from __future__ import annotations

import numpy as np
import pytest

from src.data.pairing import (
    _can_stratify,
    _three_way_stratified_split,
    create_split_indices,
)


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



def test_split_indices_are_disjoint_complete_and_unseen_groups_do_not_overlap():
    labels = np.repeat(np.arange(6), 5)
    splits = create_split_indices(
        perturbation_index=labels,
        val_fraction=0.2,
        test_fraction=0.2,
        random_seed=17,
    )
    expected_indices = np.arange(len(labels))

    for prefix in ("seen", "unseen"):
        train = splits[f"{prefix}_train"]
        val = splits[f"{prefix}_val"]
        test = splits[f"{prefix}_test"]
        combined = np.concatenate([train, val, test])

        assert len(np.unique(combined)) == len(combined)
        np.testing.assert_array_equal(np.sort(combined), expected_indices)

    unseen_groups = {
        split: set(np.unique(labels[splits[f"unseen_{split}"]]).tolist())
        for split in ("train", "val", "test")
    }
    assert unseen_groups["train"].isdisjoint(unseen_groups["val"])
    assert unseen_groups["train"].isdisjoint(unseen_groups["test"])
    assert unseen_groups["val"].isdisjoint(unseen_groups["test"])


def test_split_indices_support_zero_validation_or_test_fraction():
    labels = np.repeat(np.arange(3), 10)

    no_validation = create_split_indices(
        perturbation_index=labels,
        val_fraction=0.0,
        test_fraction=0.2,
        random_seed=5,
    )
    assert no_validation["seen_val"].size == 0
    assert no_validation["unseen_val"].size == 0
    assert no_validation["seen_test"].size > 0
    assert no_validation["unseen_test"].size > 0

    no_test = create_split_indices(
        perturbation_index=labels,
        val_fraction=0.2,
        test_fraction=0.0,
        random_seed=5,
    )
    assert no_test["seen_test"].size == 0
    assert no_test["unseen_test"].size == 0
    assert no_test["seen_val"].size > 0
    assert no_test["unseen_val"].size > 0


@pytest.mark.parametrize(
    ("val_fraction", "test_fraction"),
    [
        (-0.1, 0.1),
        (0.1, -0.1),
        (0.6, 0.4),
        (float("nan"), 0.1),
        (0.1, float("inf")),
    ],
)
def test_invalid_split_fractions_fail_with_actionable_errors(val_fraction, test_fraction):
    with pytest.raises(ValueError, match="fraction"):
        create_split_indices(
            perturbation_index=np.repeat(np.arange(3), 5),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            random_seed=42,
        )


def test_unseen_split_rejects_too_few_perturbation_groups():
    labels = np.repeat(np.arange(2), 10)

    with pytest.raises(ValueError, match="at least 3 distinct perturbation groups"):
        create_split_indices(
            perturbation_index=labels,
            val_fraction=0.1,
            test_fraction=0.1,
            random_seed=42,
        )


def test_seen_split_rejects_too_few_samples_for_requested_partitions():
    with pytest.raises(ValueError, match="at least one training sample and two held-out samples"):
        _three_way_stratified_split(
            labels=np.array([0, 1, 2]),
            val_fraction=0.1,
            test_fraction=0.1,
            random_seed=42,
        )
