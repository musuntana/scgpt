from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Simple baseline that concatenates control expression with perturbation embedding."""

    def __init__(
        self,
        num_genes: int,
        num_perturbations: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.perturbation_embedding = nn.Embedding(num_perturbations, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(num_genes + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_genes),
        )

    def forward(
        self,
        control_expression: torch.Tensor,
        perturbation_index: torch.Tensor,
    ) -> torch.Tensor:
        perturbation_features = self.perturbation_embedding(perturbation_index)
        features = torch.cat([control_expression, perturbation_features], dim=1)
        return self.network(features)

