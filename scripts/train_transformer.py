from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.data.torch_dataset import ProcessedDataset
from src.models.transformer import TransformerPerturbationModel
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import ensure_keys, load_yaml
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Transformer perturbation model.")
    parser.add_argument("--bundle-dir", required=True, help="Directory with processed bundle files.")
    parser.add_argument("--output-dir", required=True, help="Directory to store checkpoints.")
    parser.add_argument(
        "--model-config", default="configs/model.yaml", help="Path to model configuration YAML."
    )
    parser.add_argument(
        "--train-config", default="configs/train.yaml", help="Path to training configuration YAML."
    )
    parser.add_argument(
        "--split-prefix",
        default="seen",
        choices=["seen", "unseen"],
        help="Which split protocol to train against.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_yaml(args.model_config)
    train_config_payload = load_yaml(args.train_config)
    ensure_keys(
        model_config,
        ["transformer.d_model", "transformer.n_heads", "transformer.n_layers"],
    )
    ensure_keys(train_config_payload, ["train.seed", "train.batch_size", "train.epochs"])

    seed_everything(int(train_config_payload["train"]["seed"]))
    trainer_config = TrainerConfig.from_dict(train_config_payload)

    train_dataset = ProcessedDataset(args.bundle_dir, f"{args.split_prefix}_train")
    val_dataset = ProcessedDataset(args.bundle_dir, f"{args.split_prefix}_val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
    )

    transformer_cfg = model_config["transformer"]
    model = TransformerPerturbationModel(
        num_genes=train_dataset.num_genes,
        num_perturbations=train_dataset.num_perturbations,
        d_model=int(transformer_cfg["d_model"]),
        n_heads=int(transformer_cfg["n_heads"]),
        n_layers=int(transformer_cfg["n_layers"]),
        ffn_dim=int(transformer_cfg["ffn_dim"]),
        dropout=float(transformer_cfg["dropout"]),
    )

    trainer = Trainer(model=model, config=trainer_config, output_dir=Path(args.output_dir))
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    LOGGER.info("Training complete. History length: %s", len(history))


if __name__ == "__main__":
    main()
