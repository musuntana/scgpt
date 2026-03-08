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

from src.data.torch_dataset import ProcessedDataset
from src.models.mlp import MLPBaseline
from src.models.transformer import TransformerPerturbationModel
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved perturbation model.")
    parser.add_argument("--bundle-dir", required=True, help="Directory with processed bundle files.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to a model checkpoint.")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["transformer", "mlp"],
        help="Which model architecture to load.",
    )
    parser.add_argument("--output-path", required=True, help="Where to write evaluation JSON.")
    parser.add_argument(
        "--model-config", default="configs/model.yaml", help="Path to model configuration YAML."
    )
    parser.add_argument(
        "--train-config", default="configs/train.yaml", help="Path to training configuration YAML."
    )
    parser.add_argument(
        "--split-name",
        default="seen_test",
        help="Processed bundle split name to evaluate.",
    )
    return parser.parse_args()


def _build_model(model_type: str, dataset: ProcessedDataset, model_config: dict):
    if model_type == "mlp":
        return MLPBaseline(
            num_genes=dataset.num_genes,
            num_perturbations=dataset.num_perturbations,
        )
    transformer_cfg = model_config["transformer"]
    return TransformerPerturbationModel(
        num_genes=dataset.num_genes,
        num_perturbations=dataset.num_perturbations,
        d_model=int(transformer_cfg["d_model"]),
        n_heads=int(transformer_cfg["n_heads"]),
        n_layers=int(transformer_cfg["n_layers"]),
        ffn_dim=int(transformer_cfg["ffn_dim"]),
        dropout=float(transformer_cfg["dropout"]),
    )


def main() -> None:
    args = parse_args()
    dataset = ProcessedDataset(args.bundle_dir, args.split_name)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = _build_model(args.model_type, dataset, load_yaml(args.model_config))
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)

    trainer = Trainer(
        model=model,
        config=TrainerConfig.from_dict(load_yaml(args.train_config)),
        output_dir=Path(args.output_path).parent,
    )
    metrics = trainer.evaluate(loader)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
