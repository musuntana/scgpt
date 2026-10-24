from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.integrations.gears_coverage import build_go_graph_coverage_report


def test_go_graph_node_coverage_pass_is_explicitly_non_scientific() -> None:
    report = build_go_graph_coverage_report(
        required_perturbations=["GENE_A", "GENE_B"],
        available_perturbations=["GENE_A", "GENE_B", "GENE_C"],
    )

    assert report["status"] == "GO_GRAPH_NODE_COVERAGE_ONLY_PASSED"
    assert report["go_graph_node_coverage"] == {
        "verified": True,
        "required_perturbation_count": 2,
        "available_go_node_count": 3,
        "missing_perturbations": [],
    }
    assert report["model_initialized"] is False
    assert report["training_started"] is False
    assert report["metrics_generated"] is False
    assert report["target_equivalence_verified"] is False
    assert report["comparison_eligible"] is False
    assert "performance" not in report


def test_go_graph_node_coverage_fails_closed_for_missing_target() -> None:
    report = build_go_graph_coverage_report(
        required_perturbations=["GENE_A", "GENE_MISSING"],
        available_perturbations=["GENE_A", "GENE_B"],
    )

    assert report["status"] == "STOPPED_GO_GRAPH_COVERAGE"
    assert report["go_graph_node_coverage"]["verified"] is False
    assert report["go_graph_node_coverage"]["missing_perturbations"] == [
        "GENE_MISSING"
    ]
    assert report["training_started"] is False
    assert report["metrics_generated"] is False
    assert report["comparison_eligible"] is False


@pytest.mark.parametrize(
    "required",
    [[], ["GENE_A", "GENE_A"], [" GENE_A"]],
)
def test_go_graph_node_coverage_rejects_invalid_required_targets(
    required: list[str],
) -> None:
    with pytest.raises(ValueError):
        build_go_graph_coverage_report(
            required_perturbations=required,
            available_perturbations=["GENE_A", "GENE_B"],
        )


def test_coverage_cli_help_does_not_require_optional_gears_installation() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_gears_coverage.py"), "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--preflight-dir" in completed.stdout
    assert "--bundle-dir" in completed.stdout
    assert "--output-dir" in completed.stdout
