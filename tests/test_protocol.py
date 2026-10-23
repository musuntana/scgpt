from __future__ import annotations

import pytest

from src.evaluation.protocol import matching_test_split


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
