from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def build_go_graph_coverage_report(
    *,
    required_perturbations: Iterable[str],
    available_perturbations: Iterable[str],
    input_h5ad_sha256: str | None = None,
    preflight_manifest_sha256: str | None = None,
    bundle_artifacts_sha256: dict[str, str] | None = None,
    go_annotation_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed node-coverage report without claiming model validation."""
    required_values = [str(value) for value in required_perturbations]
    available_values = [str(value) for value in available_perturbations]
    if not required_values:
        raise ValueError("Required perturbation targets must not be empty.")
    if any(not value or value != value.strip() for value in required_values):
        raise ValueError("Required perturbation targets must be non-empty and trimmed.")
    if len(set(required_values)) != len(required_values):
        raise ValueError("Required perturbation targets must be unique.")
    if any(not value or value != value.strip() for value in available_values):
        raise ValueError("Available GO-node identifiers must be non-empty and trimmed.")
    if len(set(available_values)) != len(available_values):
        raise ValueError("Available GO-node identifiers must be unique.")

    available = set(available_values)
    missing = sorted(set(required_values) - available)
    verified = not missing
    return {
        "schema_version": 1,
        "status": (
            "GO_GRAPH_NODE_COVERAGE_ONLY_PASSED"
            if verified
            else "STOPPED_GO_GRAPH_COVERAGE"
        ),
        "go_graph_node_coverage": {
            "verified": verified,
            "required_perturbation_count": len(required_values),
            "available_go_node_count": len(available),
            "missing_perturbations": missing,
        },
        "input_h5ad_sha256": input_h5ad_sha256,
        "preflight_manifest_sha256": preflight_manifest_sha256,
        "bundle_artifacts_sha256": bundle_artifacts_sha256,
        "go_annotation_sha256": go_annotation_sha256,
        "model_initialized": False,
        "training_started": False,
        "metrics_generated": False,
        "target_equivalence_verified": False,
        "comparison_eligible": False,
        "limitations": [
            "Checks perturbation membership in the official GEARS GO annotation node set only.",
            "Does not initialize a model or build weighted GO edges.",
            "Does not validate expression-target equivalence or model performance.",
            "A passing report does not authorize training or scientific comparison.",
        ],
    }
