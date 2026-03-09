from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.mlp import MLPBaseline
from src.training.trainer import Trainer, TrainerConfig


class TinyPerturbationDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        control = torch.tensor([1.0, 0.5], dtype=torch.float32)
        perturbation = torch.tensor(index % 2, dtype=torch.long)
        target = torch.tensor([0.2, -0.1], dtype=torch.float32)
        return {
            "control_expression": control,
            "perturbation_index": perturbation,
            "target_delta": target,
        }


def test_trainer_writes_full_history(tmp_path):
    dataset = TinyPerturbationDataset()
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MLPBaseline(num_genes=2, num_perturbations=2, hidden_dim=8)
    trainer = Trainer(
        model=model,
        config=TrainerConfig(
            device="cpu",
            batch_size=2,
            epochs=3,
            learning_rate=1e-2,
            early_stopping_patience=10,
        ),
        output_dir=tmp_path,
    )

    history = trainer.fit(train_loader, val_loader)

    history_path = tmp_path / "history.json"
    saved_history = json.loads(history_path.read_text(encoding="utf-8"))

    assert len(history) == 3
    assert len(saved_history) == len(history)
    assert saved_history[-1]["epoch"] == 3.0


def test_trainer_collect_predictions(tmp_path):
    dataset = TinyPerturbationDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MLPBaseline(num_genes=2, num_perturbations=2, hidden_dim=8)
    trainer = Trainer(
        model=model,
        config=TrainerConfig(device="cpu", batch_size=2, epochs=1),
        output_dir=tmp_path,
    )

    predictions, perturbations = trainer.collect_predictions(loader)

    assert predictions.shape == (4, 2)
    assert perturbations.shape == (4,)
