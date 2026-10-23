from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.protocol import CHECKPOINT_METRIC_DIRECTION
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
    def from_dict(cls, payload: dict[str, Any]) -> TrainerConfig:
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
        direction = CHECKPOINT_METRIC_DIRECTION.get(self.config.checkpoint_metric)
        if direction is None:
            choices = ", ".join(sorted(CHECKPOINT_METRIC_DIRECTION))
            raise ValueError(
                f"Unsupported checkpoint_metric={self.config.checkpoint_metric!r}; "
                f"select one of: {choices}."
            )
        if self.config.epochs < 1:
            raise ValueError("epochs must be at least 1 to select a checkpoint.")
        if self.config.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if len(train_loader) == 0 or len(train_loader.dataset) == 0:  # type: ignore[arg-type]
            raise ValueError("Training split is empty; cannot fit a model.")
        if len(val_loader) == 0 or len(val_loader.dataset) == 0:  # type: ignore[arg-type]
            raise ValueError(
                "Validation split is empty; cannot select a reliable checkpoint."
            )

        history: list[dict[str, float]] = []
        best_metric = float("-inf") if direction == "max" else float("inf")
        patience = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch=epoch)
            val_metrics = self.evaluate(val_loader)
            epoch_record = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics}
            history.append(epoch_record)

            current_metric = float(val_metrics[self.config.checkpoint_metric])
            if not np.isfinite(current_metric):
                raise ValueError(
                    f"Validation checkpoint metric {self.config.checkpoint_metric!r} "
                    "is non-finite; refusing checkpoint selection."
                )
            LOGGER.info(
                "Epoch %s | train_loss=%.4f | %s=%.4f",
                epoch,
                train_loss,
                self.config.checkpoint_metric,
                current_metric,
            )
            improved = (
                current_metric > best_metric
                if direction == "max"
                else current_metric < best_metric
            )
            if improved:
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

    def _train_one_epoch(self, loader: DataLoader, epoch: int = 0) -> float:
        if len(loader) == 0 or len(loader.dataset) == 0:  # type: ignore[arg-type]
            raise ValueError("Training split is empty; cannot train an epoch.")

        self.model.train()
        total_loss = 0.0
        total_examples = 0
        n_batches = len(loader)
        log_every = max(1, n_batches // 5)

        for batch_idx, batch in enumerate(loader):
            control_expression = batch["control_expression"].to(self.device)
            target_delta = batch["target_delta"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self._forward_batch(batch)
            if predictions.shape != target_delta.shape:
                raise ValueError(
                    "Training prediction and target shapes differ: "
                    f"{tuple(predictions.shape)} vs {tuple(target_delta.shape)}."
                )
            loss, _ = mse_l1_loss(
                predictions=predictions,
                targets=target_delta,
                model=self.model,
                l1_lambda=self.config.l1_lambda,
            )
            if not bool(torch.isfinite(loss).item()):
                raise ValueError(
                    f"Non-finite training loss at epoch {epoch}, batch {batch_idx + 1}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()

            batch_size = int(control_expression.size(0))
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            if (batch_idx + 1) % log_every == 0 or batch_idx == n_batches - 1:
                running_loss = total_loss / max(total_examples, 1)
                LOGGER.info(
                    "Epoch %s | batch %s/%s | running_loss=%.4f",
                    epoch,
                    batch_idx + 1,
                    n_batches,
                    running_loss,
                )

        return total_loss / max(total_examples, 1)

    def _forward_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        """Route batches to categorical-ID or shared-gene-identity models."""
        control_expression = batch["control_expression"].to(self.device)
        if getattr(self.model, "requires_perturbation_gene_index", False):
            if "perturbation_gene_index" not in batch:
                raise ValueError(
                    "This model requires perturbation_gene_index in each dataset batch."
                )
            perturbation_gene_index = batch["perturbation_gene_index"].to(self.device)
            return self.model(control_expression, perturbation_gene_index)

        perturbation_index = batch["perturbation_index"].to(self.device)
        return self.model(control_expression, perturbation_index)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        if len(loader) == 0 or len(loader.dataset) == 0:  # type: ignore[arg-type]
            raise ValueError(
                "Evaluation split is empty; no metrics can be reported."
            )
        predictions, targets, perturbations = self.collect_outputs(loader)

        metrics = compute_regression_metrics(
            predictions=predictions,
            targets=targets,
            perturbation_index=perturbations,
        )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Non-finite evaluation metrics; refusing to report scores.")
        return metrics

    @torch.no_grad()
    def collect_outputs(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect raw predictions, targets, and perturbation indices from a loader."""
        if len(loader) == 0 or len(loader.dataset) == 0:  # type: ignore[arg-type]
            raise ValueError("Evaluation split is empty; no outputs can be collected.")
        self.model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        perturbations: list[np.ndarray] = []
        for batch in loader:
            perturbation_index = batch["perturbation_index"].to(self.device)
            target_delta = batch["target_delta"].to(self.device)
            batch_predictions = self._forward_batch(batch)
            if batch_predictions.shape != target_delta.shape:
                raise ValueError(
                    "Evaluation prediction and target shapes differ: "
                    f"{tuple(batch_predictions.shape)} vs {tuple(target_delta.shape)}."
                )
            if not bool(torch.isfinite(batch_predictions).all().item()):
                raise ValueError("Non-finite evaluation predictions.")
            if not bool(torch.isfinite(target_delta).all().item()):
                raise ValueError("Non-finite evaluation targets.")
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(target_delta.detach().cpu().numpy())
            perturbations.append(perturbation_index.detach().cpu().numpy())
        return (
            np.concatenate(predictions, axis=0),
            np.concatenate(targets, axis=0),
            np.concatenate(perturbations, axis=0),
        )

    @torch.no_grad()
    def collect_predictions(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect raw predictions and perturbation indices from a loader."""
        predictions, _, perturbations = self.collect_outputs(loader)
        return predictions, perturbations

    def save_checkpoint(self, filename: str) -> None:
        checkpoint_path = self.output_dir / filename
        torch.save(self.model.state_dict(), checkpoint_path)

    def _write_history(self, history: list[dict[str, float]]) -> None:
        history_path = self.output_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
