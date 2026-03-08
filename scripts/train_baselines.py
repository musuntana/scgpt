from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pairing import load_processed_bundle
from src.data.torch_dataset import ProcessedDataset
from src.evaluation.metrics import compute_regression_metrics
from src.models.mlp import MLPBaseline
from src.models.xgboost_baseline import build_xgboost_baseline
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local baseline models.")
    parser.add_argument("--bundle-dir", required=True, help="Directory with processed bundle files.")
    parser.add_argument("--output-dir", required=True, help="Directory to store baseline outputs.")
    parser.add_argument(
        "--train-config", default="configs/train.yaml", help="Path to training configuration YAML."
    )
    parser.add_argument(
        "--baseline",
        default="mlp",
        choices=["mlp", "xgboost"],
        help="Baseline model to train.",
    )
    parser.add_argument(
        "--split-prefix",
        default="seen",
        choices=["seen", "unseen"],
        help="Which split protocol to use.",
    )
    return parser.parse_args()


def _build_numpy_features(
    control_expression: np.ndarray,
    perturbation_index: np.ndarray,
    num_perturbations: int,
) -> np.ndarray:
    one_hot = np.eye(num_perturbations, dtype=np.float32)[perturbation_index]
    return np.concatenate([control_expression, one_hot], axis=1)


def _train_xgboost(args: argparse.Namespace) -> None:
    bundle = load_processed_bundle(args.bundle_dir)
    splits = bundle["splits"]
    metadata = bundle["metadata"]

    train_idx = splits[f"{args.split_prefix}_train"]
    test_idx = splits[f"{args.split_prefix}_test"]

    features = _build_numpy_features(
        bundle["control_expression"],
        bundle["perturbation_index"],
        num_perturbations=len(metadata["perturbation_names"]),
    )
    targets = bundle["target_delta"]

    model = build_xgboost_baseline()
    model.fit(features[train_idx], targets[train_idx])
    predictions = model.predict(features[test_idx])
    metrics = compute_regression_metrics(
        predictions=predictions,
        targets=targets[test_idx],
        perturbation_index=bundle["perturbation_index"][test_idx],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "xgboost_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("Saved XGBoost metrics to %s", output_dir / "xgboost_metrics.json")


def _train_mlp(args: argparse.Namespace, train_config_path: str) -> None:
    train_config_payload = load_yaml(train_config_path)
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

    model = MLPBaseline(
        num_genes=train_dataset.num_genes,
        num_perturbations=train_dataset.num_perturbations,
    )
    trainer = Trainer(model=model, config=trainer_config, output_dir=args.output_dir)
    trainer.fit(train_loader, val_loader)


def main() -> None:
    args = parse_args()
    if args.baseline == "xgboost":
        _train_xgboost(args)
        return
    _train_mlp(args, args.train_config)


if __name__ == "__main__":
    main()
