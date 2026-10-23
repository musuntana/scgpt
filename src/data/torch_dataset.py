from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.gene_identity import build_unique_gene_name_index
from src.data.pairing import load_processed_bundle


class ProcessedDataset(Dataset):
    """PyTorch dataset backed by a processed perturbation bundle."""

    def __init__(self, bundle_dir: str | Path, split_name: str | None = None):
        bundle = load_processed_bundle(bundle_dir)
        all_indices = np.arange(len(bundle["perturbation_index"]), dtype=np.int64)
        if split_name is None:
            self.indices = all_indices
        else:
            if split_name not in bundle["splits"]:
                raise KeyError(f"Unknown split_name={split_name!r} for bundle {bundle_dir}")
            self.indices = bundle["splits"][split_name]
        self.control_expression = bundle["control_expression"]
        self.target_delta = bundle["target_delta"]
        self.perturbation_index = bundle["perturbation_index"]
        self.sample_ids = bundle["sample_ids"]
        self.metadata = bundle["metadata"]
        gene_names = [str(name) for name in self.metadata["gene_names"]]
        perturbation_names = [
            str(name) for name in self.metadata["perturbation_names"]
        ]
        gene_to_index, ambiguous_gene_names = build_unique_gene_name_index(gene_names)
        self.ambiguous_gene_names = sorted(ambiguous_gene_names)
        self.ambiguous_perturbation_names = [
            name for name in perturbation_names if name in ambiguous_gene_names
        ]
        self.perturbation_gene_indices = np.asarray(
            [gene_to_index.get(name, -1) for name in perturbation_names],
            dtype=np.int64,
        )
        self.unmapped_perturbation_names = [
            name
            for name, gene_index in zip(
                perturbation_names, self.perturbation_gene_indices, strict=True
            )
            if gene_index < 0
        ]

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_index = int(self.indices[index])
        return {
            "control_expression": torch.from_numpy(
                self.control_expression[sample_index]
            ).float(),
            "perturbation_index": torch.tensor(
                int(self.perturbation_index[sample_index]), dtype=torch.long
            ),
            "perturbation_gene_index": torch.tensor(
                int(self.perturbation_gene_indices[self.perturbation_index[sample_index]]),
                dtype=torch.long,
            ),
            "target_delta": torch.from_numpy(self.target_delta[sample_index]).float(),
            "sample_id": self.sample_ids[sample_index],
        }

    @property
    def num_genes(self) -> int:
        return int(self.control_expression.shape[1])

    @property
    def num_perturbations(self) -> int:
        return int(len(self.metadata["perturbation_names"]))

    @property
    def num_mapped_perturbations(self) -> int:
        return int(self.num_perturbations - len(self.unmapped_perturbation_names))
