from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    model: torch.nn.Module,
    l1_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute MSE with optional L1 regularization."""
    mse = F.mse_loss(predictions, targets)
    if l1_lambda <= 0:
        return mse, {"mse": float(mse.item()), "l1": 0.0}

    l1_term = torch.tensor(0.0, device=predictions.device)
    for parameter in model.parameters():
        l1_term = l1_term + parameter.abs().sum()
    total_loss = mse + (l1_lambda * l1_term)
    return total_loss, {"mse": float(mse.item()), "l1": float(l1_term.item())}

