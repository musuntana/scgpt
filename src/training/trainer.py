from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_regression_metrics
from src.training.losses import mse_l1_loss
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainerConfig:
    device: str = "auto"
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    l1_lambda: float = 0.0
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    checkpoint_metric: str = "pearson_per_perturbation"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainerConfig":
        train_cfg = payload.get("train", payload)
        return cls(
            device=str(train_cfg.get("device", "auto")),
            batch_size=int(train_cfg.get("batch_size", 16)),
            num_workers=int(train_cfg.get("num_workers", 0)),
            epochs=int(train_cfg.get("epochs", 20)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            l1_lambda=float(train_cfg.get("l1_lambda", 0.0)),
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            early_stopping_patience=int(train_cfg.get("early_stopping_patience", 5)),
            checkpoint_metric=str(
                train_cfg.get("checkpoint_metric", "pearson_per_perturbation")
            ),
        )


class Trainer:
    """Minimal trainer for local-first perturbation models."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        best_metric = float("-inf")
        patience = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            epoch_record = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics}
            history.append(epoch_record)

            current_metric = float(val_metrics.get(self.config.checkpoint_metric, 0.0))
            LOGGER.info(
                "Epoch %s | train_loss=%.4f | %s=%.4f",
                epoch,
                train_loss,
                self.config.checkpoint_metric,
                current_metric,
            )
            if current_metric > best_metric:
                best_metric = current_metric
                patience = 0
                self.save_checkpoint("best_model.pt")
            else:
                patience += 1
            self._write_history(history)
            if patience >= self.config.early_stopping_patience:
                LOGGER.info("Early stopping after %s epochs.", epoch)
                break

        return history

    def _train_one_epoch(self, loader: DataLoader) -> float:
        if len(loader.dataset) == 0:
            return 0.0

        self.model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            control_expression = batch["control_expression"].to(self.device)
            perturbation_index = batch["perturbation_index"].to(self.device)
            target_delta = batch["target_delta"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(control_expression, perturbation_index)
            loss, _ = mse_l1_loss(
                predictions=predictions,
                targets=target_delta,
                model=self.model,
                l1_lambda=self.config.l1_lambda,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()

            batch_size = int(control_expression.size(0))
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

        return total_loss / max(total_examples, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        if len(loader.dataset) == 0:
            return {
                "overall_mse": 0.0,
                "mse_per_perturbation": 0.0,
                "pearson_per_perturbation": 0.0,
                "pearson_per_gene": 0.0,
            }

        self.model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        perturbations: list[np.ndarray] = []

        for batch in loader:
            control_expression = batch["control_expression"].to(self.device)
            perturbation_index = batch["perturbation_index"].to(self.device)
            target_delta = batch["target_delta"].to(self.device)

            batch_predictions = self.model(control_expression, perturbation_index)
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(target_delta.detach().cpu().numpy())
            perturbations.append(perturbation_index.detach().cpu().numpy())

        return compute_regression_metrics(
            predictions=np.concatenate(predictions, axis=0),
            targets=np.concatenate(targets, axis=0),
            perturbation_index=np.concatenate(perturbations, axis=0),
        )

    @torch.no_grad()
    def collect_predictions(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect raw predictions and perturbation indices from a loader."""
        self.model.eval()
        predictions: list[np.ndarray] = []
        perturbations: list[np.ndarray] = []
        for batch in loader:
            control_expression = batch["control_expression"].to(self.device)
            perturbation_index = batch["perturbation_index"].to(self.device)
            batch_predictions = self.model(control_expression, perturbation_index)
            predictions.append(batch_predictions.detach().cpu().numpy())
            perturbations.append(perturbation_index.detach().cpu().numpy())
        return (
            np.concatenate(predictions, axis=0),
            np.concatenate(perturbations, axis=0),
        )

    def save_checkpoint(self, filename: str) -> None:
        checkpoint_path = self.output_dir / filename
        torch.save(self.model.state_dict(), checkpoint_path)

    def _write_history(self, history: list[dict[str, float]]) -> None:
        history_path = self.output_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
