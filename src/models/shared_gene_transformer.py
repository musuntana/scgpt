from __future__ import annotations

import warnings

import torch
from torch import nn


class SharedGeneIdentityTransformer(nn.Module):
    """Transformer using the shared output-gene embedding as perturbation identity.

    Unlike the categorical-ID model, this model does not allocate a dedicated,
    untrained embedding row to each held-out perturbation. A perturbation must
    map exactly to a gene in the saved output-gene universe; the training entry
    point is responsible for enforcing complete mapping coverage.
    """

    requires_perturbation_gene_index = True

    def __init__(
        self,
        num_genes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_genes < 1:
            raise ValueError(f"num_genes must be positive; got {num_genes}.")
        self.num_genes = num_genes
        self.gene_embedding = nn.Embedding(num_genes, d_model)
        self.value_encoder = nn.Linear(1, d_model)

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
        perturbation_gene_index: torch.Tensor,
    ) -> torch.Tensor:
        if control_expression.ndim != 2:
            raise ValueError(
                "control_expression must have shape (batch_size, num_genes)."
            )
        batch_size, num_genes = control_expression.shape
        if num_genes != self.num_genes:
            raise ValueError(f"Expected {self.num_genes} genes, got {num_genes}.")
        if perturbation_gene_index.ndim != 1 or perturbation_gene_index.shape[0] != batch_size:
            raise ValueError(
                "perturbation_gene_index must have shape (batch_size,)."
            )
        if perturbation_gene_index.dtype != torch.long:
            raise ValueError("perturbation_gene_index must use torch.long indices.")
        if torch.any(perturbation_gene_index < 0) or torch.any(
            perturbation_gene_index >= self.num_genes
        ):
            raise ValueError(
                "Every perturbation must map to a gene in the saved gene universe."
            )

        gene_ids = torch.arange(self.num_genes, device=control_expression.device)
        gene_ids = gene_ids.unsqueeze(0).expand(batch_size, -1)
        gene_tokens = self.gene_embedding(gene_ids)
        value_tokens = self.value_encoder(control_expression.unsqueeze(-1))
        perturbation_tokens = self.gene_embedding(perturbation_gene_index).unsqueeze(1)
        tokens = gene_tokens + value_tokens + perturbation_tokens

        encoded = self.encoder(tokens)
        return self.output_head(encoded).squeeze(-1)
