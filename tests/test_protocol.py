from __future__ import annotations

import pytest

from src.evaluation.protocol import (
    matching_test_split,
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
