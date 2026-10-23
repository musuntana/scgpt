from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.models.mlp import MLPBaseline
from src.models.shared_gene_transformer import SharedGeneIdentityTransformer
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
            "perturbation_gene_index": perturbation,
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


def test_trainer_supports_shared_gene_identity_model(tmp_path):
    dataset = TinyPerturbationDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = SharedGeneIdentityTransformer(
        num_genes=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        dropout=0.0,
    )
    trainer = Trainer(
        model=model,
        config=TrainerConfig(
            device="cpu",
            batch_size=2,
            epochs=1,
            learning_rate=1e-2,
        ),
        output_dir=tmp_path,
    )

    history = trainer.fit(loader, loader)
    predictions, perturbations = trainer.collect_predictions(loader)

    assert len(history) == 1
    assert predictions.shape == (4, 2)
    assert perturbations.shape == (4,)


@pytest.mark.parametrize(
    ("metric", "scores", "expected_epoch"),
    [
        ("overall_mse", [0.2, 0.4, 0.3], 1),
        ("mse_per_perturbation", [0.2, 0.4, 0.3], 1),
        ("pearson_per_perturbation", [0.2, 0.4, 0.3], 2),
    ],
)
def test_trainer_checkpoint_selection_uses_metric_direction(
    tmp_path, monkeypatch, metric, scores, expected_epoch
):
    dataset = TinyPerturbationDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MLPBaseline(num_genes=2, num_perturbations=2, hidden_dim=8)
    trainer = Trainer(
        model=model,
        config=TrainerConfig(
            device="cpu",
            epochs=3,
            checkpoint_metric=metric,
            early_stopping_patience=4,
        ),
        output_dir=tmp_path,
    )
    first_parameter = next(model.parameters())

    def fake_train_one_epoch(_loader, epoch=0):
        with torch.no_grad():
            first_parameter.fill_(float(epoch))
        return float(epoch)

    values = iter(scores)
    monkeypatch.setattr(trainer, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(trainer, "evaluate", lambda _loader: {metric: next(values)})

    trainer.fit(loader, loader)

    checkpoint = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=True)
    assert torch.all(
        checkpoint[next(iter(checkpoint))] == float(expected_epoch)
    )


@pytest.mark.parametrize(
    ("train_empty", "val_empty", "message"),
    [
        (True, False, "Training split is empty"),
        (False, True, "Validation split is empty"),
    ],
)
def test_trainer_rejects_empty_fit_splits(tmp_path, train_empty, val_empty, message):
    dataset = TinyPerturbationDataset()
    full_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    empty_loader = DataLoader(Subset(dataset, []), batch_size=2, shuffle=False)
    trainer = Trainer(
        model=MLPBaseline(num_genes=2, num_perturbations=2, hidden_dim=8),
        config=TrainerConfig(device="cpu", epochs=1),
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match=message):
        trainer.fit(
            empty_loader if train_empty else full_loader,
            empty_loader if val_empty else full_loader,
        )
    assert not (tmp_path / "best_model.pt").exists()


def test_trainer_does_not_report_perfect_scores_for_empty_evaluation(tmp_path):
    dataset = TinyPerturbationDataset()
    empty_loader = DataLoader(Subset(dataset, []), batch_size=2, shuffle=False)
    trainer = Trainer(
        model=MLPBaseline(num_genes=2, num_perturbations=2, hidden_dim=8),
        config=TrainerConfig(device="cpu"),
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="Evaluation split is empty"):
        trainer.evaluate(empty_loader)
    with pytest.raises(ValueError, match="Evaluation split is empty"):
        trainer.collect_predictions(empty_loader)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_metric", "unknown_score", "Unsupported checkpoint_metric"),
        ("epochs", 0, "epochs must be at least 1"),
        ("early_stopping_patience", 0, "early_stopping_patience must be at least 1"),
    ],
)
def test_trainer_rejects_invalid_selection_config(tmp_path, field, value, message):
    dataset = TinyPerturbationDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    config = TrainerConfig(device="cpu", epochs=1)
    setattr(config, field, value)
    trainer = Trainer(
        model=MLPBaseline(num_genes=2, num_perturbations=2, hidden_dim=8),
        config=config,
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match=message):
        trainer.fit(loader, loader)
    assert not (tmp_path / "best_model.pt").exists()


def test_trainer_rejects_nonfinite_training_loss(tmp_path):
    class NonFiniteDataset(TinyPerturbationDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["target_delta"] = torch.full((2,), float("nan"))
            return sample

    dataset = NonFiniteDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    trainer = Trainer(
        model=MLPBaseline(num_genes=2, num_perturbations=2, hidden_dim=8),
        config=TrainerConfig(device="cpu", epochs=1),
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="Non-finite training loss"):
        trainer.fit(loader, loader)
    assert not (tmp_path / "best_model.pt").exists()


def test_trainer_rejects_nonfinite_validation_outputs(tmp_path):
    class NonFiniteDataset(TinyPerturbationDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["target_delta"] = torch.full((2,), float("nan"))
            return sample

    loader = DataLoader(NonFiniteDataset(), batch_size=2, shuffle=False)
    trainer = Trainer(
        model=MLPBaseline(num_genes=2, num_perturbations=2, hidden_dim=8),
        config=TrainerConfig(device="cpu"),
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="Non-finite evaluation targets"):
        trainer.evaluate(loader)
