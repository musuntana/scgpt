from __future__ import annotations

import warnings

import torch
from torch import nn


class TransformerPerturbationModel(nn.Module):
    """Local-first Transformer for perturbation response prediction."""

    def __init__(
        self,
        num_genes: int,
        num_perturbations: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.gene_embedding = nn.Embedding(num_genes, d_model)
        self.value_encoder = nn.Linear(1, d_model)
        self.perturbation_embedding = nn.Embedding(num_perturbations, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # norm_first=True disables nested-tensor optimisation; suppress the
        # resulting UserWarning which is expected and non-actionable here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="enable_nested_tensor is True",
                category=UserWarning,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        control_expression: torch.Tensor,
        perturbation_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_genes = control_expression.shape
        if num_genes != self.num_genes:
            raise ValueError(f"Expected {self.num_genes} genes, got {num_genes}")

        gene_ids = torch.arange(self.num_genes, device=control_expression.device)
        gene_ids = gene_ids.unsqueeze(0).expand(batch_size, -1)

        gene_tokens = self.gene_embedding(gene_ids)
        value_tokens = self.value_encoder(control_expression.unsqueeze(-1))
        perturbation_tokens = self.perturbation_embedding(perturbation_index).unsqueeze(1)
        tokens = gene_tokens + value_tokens + perturbation_tokens

        encoded = self.encoder(tokens)
        return self.output_head(encoded).squeeze(-1)

