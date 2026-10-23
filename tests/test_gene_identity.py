from __future__ import annotations

import pytest

from src.data.gene_identity import (
    build_gene_identity_coverage_report,
    build_unique_gene_name_index,
)


def test_unique_gene_name_index_excludes_ambiguous_names():
    index, ambiguous = build_unique_gene_name_index(["A", "B", "A", "C"])

    assert index == {"B": 1, "C": 3}
    assert ambiguous == {"A"}


def test_gene_identity_coverage_report_passes_exact_complete_mapping():
    report = build_gene_identity_coverage_report(
        gene_names=["A", "B", "C"],
        perturbation_names=["A", "C"],
    )

    assert report["status"] == "PASS"
    assert report["mapping_method"] == "exact_match_to_saved_gene_names"
    assert report["mapped_perturbation_count"] == 2
    assert report["unmapped_perturbation_count"] == 0
    assert report["mapping_coverage"] == 1.0
    assert len(report["gene_names_sha256"]) == 64
    assert len(report["perturbation_names_sha256"]) == 64


def test_gene_identity_coverage_report_blocks_unmapped_names():
    report = build_gene_identity_coverage_report(
        gene_names=["A", "B"],
        perturbation_names=["A", "C", "D"],
    )

    assert report["status"] == "BLOCKED_INCOMPLETE_OR_AMBIGUOUS_MAPPING"
    assert report["mapped_perturbation_count"] == 1
    assert report["unmapped_perturbation_count"] == 2
    assert report["mapping_coverage"] == pytest.approx(1 / 3)
    assert report["unmapped_perturbation_names"] == ["C", "D"]


def test_gene_identity_coverage_report_blocks_ambiguous_gene_names():
    report = build_gene_identity_coverage_report(
        gene_names=["A", "A", "B"],
        perturbation_names=["A", "B"],
    )

    assert report["status"] == "BLOCKED_INCOMPLETE_OR_AMBIGUOUS_MAPPING"
    assert report["ambiguous_gene_names"] == ["A"]
    assert report["ambiguous_perturbation_names"] == ["A"]
    assert report["mapped_perturbation_count"] == 1
    assert report["unmapped_perturbation_names"] == ["A"]


@pytest.mark.parametrize(
    ("gene_names", "perturbation_names", "message"),
    [
        ([], ["A"], "gene_names must be a non-empty"),
        (["A"], [], "perturbation_names must be a non-empty"),
        (["A", ""], ["A"], "gene_names must contain only non-empty"),
        (["A"], ["A", "A"], "perturbation_names must be unique"),
    ],
)
def test_gene_identity_coverage_report_rejects_invalid_name_lists(
    gene_names: list[str],
    perturbation_names: list[str],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        build_gene_identity_coverage_report(gene_names, perturbation_names)
