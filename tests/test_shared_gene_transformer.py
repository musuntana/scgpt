from __future__ import annotations

import pytest
import torch

from src.models.shared_gene_transformer import SharedGeneIdentityTransformer


def _model() -> SharedGeneIdentityTransformer:
    return SharedGeneIdentityTransformer(
        num_genes=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        dropout=0.0,
    )


def test_shared_gene_identity_transformer_predicts_all_output_genes():
    model = _model()
    control = torch.tensor(
        [[1.0, 0.5, 0.1, 0.0], [0.8, 0.4, 0.2, 0.1]],
        dtype=torch.float32,
    )
    target_gene_index = torch.tensor([0, 2], dtype=torch.long)

    predictions = model(control, target_gene_index)

    assert predictions.shape == control.shape
    assert torch.isfinite(predictions).all()
    assert not hasattr(model, "perturbation_embedding")


def test_shared_gene_identity_embedding_is_shared_with_output_gene_tokens():
    torch.manual_seed(7)
    model = _model()
    control = torch.tensor([[1.0, 0.5, 0.1, 0.0]], dtype=torch.float32)
    target_gene_index = torch.tensor([0], dtype=torch.long)

    model(control, target_gene_index).square().mean().backward()

    gradient = model.gene_embedding.weight.grad
    assert gradient is not None
    # Gene 3 was not the target perturbation, but still receives learning signal
    # through the shared output-gene token path.
    assert torch.linalg.vector_norm(gradient[3]).item() > 0.0


@pytest.mark.parametrize(
    ("gene_indices", "message"),
    [
        (torch.tensor([-1], dtype=torch.long), "map to a gene"),
        (torch.tensor([4], dtype=torch.long), "map to a gene"),
        (torch.tensor([[0]], dtype=torch.long), "shape"),
        (torch.tensor([0.0]), "torch.long"),
    ],
)
def test_shared_gene_identity_transformer_rejects_invalid_gene_indices(
    gene_indices: torch.Tensor,
    message: str,
):
    model = _model()
    control = torch.ones((1, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match=message):
        model(control, gene_indices)


def test_shared_gene_identity_transformer_rejects_wrong_gene_dimension():
    model = _model()
    control = torch.ones((1, 3), dtype=torch.float32)
    target_gene_index = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="Expected 4 genes"):
        model(control, target_gene_index)
