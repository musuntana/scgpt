from __future__ import annotations

from collections.abc import Collection


def matching_test_split(split_prefix: str, available_splits: Collection[str]) -> str:
    """Return only the test split matching the protocol used for training."""
    if split_prefix not in {"seen", "unseen"}:
        raise ValueError(
            f"Unsupported split prefix {split_prefix!r}; expected 'seen' or 'unseen'."
        )

    test_split = f"{split_prefix}_test"
    if test_split not in available_splits:
        raise ValueError(
            f"Bundle does not contain the matching test split {test_split!r}. "
            "Regenerate the processed bundle or check the selected protocol."
        )
    return test_split
