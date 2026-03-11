from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pairing import load_processed_bundle
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for train.seed without editing the YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_yaml(args.model_config)
    train_config_payload = load_yaml(args.train_config)
    if args.seed is not None:
        train_config_payload.setdefault("train", {})["seed"] = int(args.seed)
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

    output_dir = Path(args.output_dir)
    trainer = Trainer(model=model, config=trainer_config, output_dir=output_dir)
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    LOGGER.info("Training complete. History length: %s", len(history))

    # Reload best checkpoint for test evaluation
    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.to(trainer.device)

    # Evaluate on both seen and unseen test splits
    bundle = load_processed_bundle(args.bundle_dir)
    for split_name in ("seen_test", "unseen_test"):
        if split_name not in bundle["splits"]:
            continue
        eval_dataset = ProcessedDataset(args.bundle_dir, split_name)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=trainer_config.batch_size,
            shuffle=False,
            num_workers=trainer_config.num_workers,
        )
        metrics = trainer.evaluate(eval_loader)
        metrics_path = output_dir / f"{split_name}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        LOGGER.info("Saved %s metrics to %s: %s", split_name, metrics_path, metrics)


if __name__ == "__main__":
    main()
