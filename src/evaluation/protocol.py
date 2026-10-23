from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

CHECKPOINT_METRIC_DIRECTION = {
    "pearson_per_perturbation": "max",
    "pearson_per_gene": "max",
    "overall_mse": "min",
    "mse_per_perturbation": "min",
}

BUNDLE_ARTIFACT_NAMES = ("arrays.npz", "splits.npz", "metadata.json")


def fingerprint_processed_bundle(bundle_dir: str | Path) -> dict[str, str]:
    """Record byte-exact input identity without loading expression arrays."""
    root = Path(bundle_dir)
    digests: dict[str, str] = {}
    for filename in BUNDLE_ARTIFACT_NAMES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required processed-bundle artifact missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[filename] = digest.hexdigest()
    return digests


def validate_checkpoint_bundle_identity(
    protocol: Mapping[str, Any],
    bundle_dir: str | Path,
) -> bool:
    """Reject input bundles inconsistent with the checkpoint training protocol.

    Return False only for historical training records that predate bundle hashes.
    New records (schema >= 2) must carry a complete hash contract.
    """
    hashes = protocol.get("bundle_artifacts_sha256")
    if hashes is None:
        if protocol.get("schema_version", 1) >= 2:
            raise ValueError(
                "Checkpoint protocol is missing required bundle_artifacts_sha256."
            )
        return False
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(BUNDLE_ARTIFACT_NAMES)
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in hashes.values()
        )
    ):
        raise ValueError(
            "Checkpoint bundle_artifacts_sha256 must contain exactly three valid "
            "SHA-256 hashes (arrays.npz, splits.npz, metadata.json)."
        )

    actual = fingerprint_processed_bundle(bundle_dir)
    for filename in BUNDLE_ARTIFACT_NAMES:
        if actual[filename] != hashes[filename]:
            raise ValueError(
                f"Checkpoint bundle identity mismatch for {filename}: "
                "the evaluation input differs from the training bundle."
            )
    return True




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



def validate_split_matches_training_protocol(
    train_protocol: str | None,
    split_name: str,
) -> None:
    """Reject evaluation splits that conflict with a checkpoint's recorded protocol."""
    requested_protocol = split_name.split("_", maxsplit=1)[0]
    if (
        train_protocol in {"seen", "unseen"}
        and requested_protocol in {"seen", "unseen"}
        and requested_protocol != train_protocol
    ):
        raise ValueError(
            f"Checkpoint was trained with the {train_protocol!r} protocol, but "
            f"{split_name!r} belongs to {requested_protocol!r}. Evaluate the matching "
            "protocol split or provide a checkpoint trained for that protocol."
        )
