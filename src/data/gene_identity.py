from __future__ import annotations

from collections.abc import Sequence


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
