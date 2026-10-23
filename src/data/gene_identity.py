from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


def build_unique_gene_name_index(
    gene_names: Sequence[str],
) -> tuple[dict[str, int], set[str]]:
    """Index only unambiguous gene names; duplicate names are deliberately unmappable."""
    gene_to_index: dict[str, int] = {}
    ambiguous_gene_names: set[str] = set()
    for index, name in enumerate(gene_names):
        if name in gene_to_index:
            ambiguous_gene_names.add(name)
        else:
            gene_to_index[name] = index

    for name in ambiguous_gene_names:
        gene_to_index.pop(name, None)
    return gene_to_index, ambiguous_gene_names


def build_gene_identity_coverage_report(
    gene_names: Sequence[str],
    perturbation_names: Sequence[str],
) -> dict[str, Any]:
    """Summarize exact perturbation-to-gene coverage without guessing aliases."""
    for label, values in (
        ("gene_names", gene_names),
        ("perturbation_names", perturbation_names),
    ):
        if len(values) == 0:
            raise ValueError(f"{label} must be a non-empty sequence.")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{label} must contain only non-empty strings.")

    if len(set(perturbation_names)) != len(perturbation_names):
        raise ValueError("perturbation_names must be unique for coverage reporting.")

    gene_to_index, ambiguous_gene_names = build_unique_gene_name_index(gene_names)
    mapped_names = [name for name in perturbation_names if name in gene_to_index]
    unmapped_names = [name for name in perturbation_names if name not in gene_to_index]
    ambiguous_perturbations = [
        name for name in unmapped_names if name in ambiguous_gene_names
    ]
    coverage = len(mapped_names) / len(perturbation_names)
    status = (
        "PASS"
        if not unmapped_names and not ambiguous_gene_names
        else "BLOCKED_INCOMPLETE_OR_AMBIGUOUS_MAPPING"
    )

    def digest(values: Sequence[str]) -> str:
        payload = json.dumps(
            list(values),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return {
        "schema_version": 1,
        "status": status,
        "mapping_method": "exact_match_to_saved_gene_names",
        "gene_count": len(gene_names),
        "unique_gene_name_count": len(gene_to_index),
        "ambiguous_gene_name_count": len(ambiguous_gene_names),
        "perturbation_count": len(perturbation_names),
        "mapped_perturbation_count": len(mapped_names),
        "unmapped_perturbation_count": len(unmapped_names),
        "mapping_coverage": coverage,
        "gene_names_sha256": digest(gene_names),
        "perturbation_names_sha256": digest(perturbation_names),
        "ambiguous_gene_names": sorted(ambiguous_gene_names),
        "ambiguous_perturbation_names": ambiguous_perturbations,
        "unmapped_perturbation_names": unmapped_names,
    }
