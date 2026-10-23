from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.run_gears_baseline import train_with_all_gene_validation


class _Batch:
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y
        self.pert = ["GENE"] * int(x.shape[0])

    def to(self, device: str) -> _Batch:
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self


def _batch(target: float) -> _Batch:
    return _Batch(
        x=torch.ones((1, 1), dtype=torch.float32),
        y=torch.full((1, 1), target, dtype=torch.float32),
    )


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()

    def forward(self, batch: _Batch) -> torch.Tensor:
        return self.linear(batch.x)


def _gears_model() -> SimpleNamespace:
    model = _ToyModel()
    return SimpleNamespace(
        model=model,
        ctrl_expression=torch.zeros(1),
        dict_filter={},
        config={"direction_lambda": 0.1},
    )


def _mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    perts: Any,
    **kwargs: Any,
) -> torch.Tensor:
    del perts, kwargs
    return torch.mean((prediction - target) ** 2)


def _train(
    *,
    gears_model: SimpleNamespace,
    train_loader: list[_Batch],
    val_loader: list[_Batch],
    epochs: int = 1,
    loss_fct: Any = _mse_loss,
) -> dict[str, Any]:
    return train_with_all_gene_validation(
        gears_model,
        train_loader,
        val_loader,
        torch=torch,
        loss_fct=loss_fct,
        device="cpu",
        epochs=epochs,
        learning_rate=0.1,
        weight_decay=0.0,
    )


def test_custom_trainer_restores_lowest_validation_checkpoint() -> None:
    torch.manual_seed(7)
    gears_model = _gears_model()

    report = _train(
        gears_model=gears_model,
        train_loader=[_batch(1.0)],
        val_loader=[_batch(0.0)],
        epochs=3,
    )

    validation_scores = [
        float(row["validation_mse_absolute_expression"]) for row in report["history"]
    ]
    assert report["best_validation_mse_absolute_expression"] == min(validation_scores)
    assert validation_scores[0] < validation_scores[1] < validation_scores[2]
    assert torch.allclose(
        gears_model.model.linear.weight.detach(),
        torch.tensor([[0.1]]),
        atol=1e-5,
    )
    assert torch.equal(
        gears_model.model.linear.weight.detach(),
        gears_model.best_model.linear.weight.detach(),
    )
    assert report["official_gears_train_called"] is False
    assert report["DE_specific_metrics_used_for_selection"] is False


def test_custom_trainer_rejects_non_positive_epoch_count() -> None:
    with pytest.raises(ValueError, match="epochs must be positive"):
        _train(
            gears_model=_gears_model(),
            train_loader=[_batch(1.0)],
            val_loader=[_batch(1.0)],
            epochs=0,
        )


def test_custom_trainer_rejects_empty_training_loader() -> None:
    with pytest.raises(RuntimeError, match="no batches"):
        _train(
            gears_model=_gears_model(),
            train_loader=[],
            val_loader=[_batch(1.0)],
        )


def test_custom_trainer_rejects_empty_validation_loader() -> None:
    with pytest.raises(RuntimeError, match="no expression targets"):
        _train(
            gears_model=_gears_model(),
            train_loader=[_batch(1.0)],
            val_loader=[],
        )


def test_custom_trainer_stops_on_non_finite_loss() -> None:
    def non_finite_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        perts: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del target, perts, kwargs
        return prediction.sum() * torch.tensor(float("nan"))

    with pytest.raises(RuntimeError, match="Non-finite GEARS training loss"):
        _train(
            gears_model=_gears_model(),
            train_loader=[_batch(1.0)],
            val_loader=[_batch(1.0)],
            loss_fct=non_finite_loss,
        )
