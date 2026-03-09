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

from src.data.torch_dataset import ProcessedDataset
from src.evaluation.deg import load_deg_artifact
from src.evaluation.inference import load_torch_model_for_bundle
from src.evaluation.metrics import compute_topk_deg_metrics
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import load_yaml
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


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
    parser.add_argument(
        "--deg-artifact-path",
        default=None,
        help="Optional path to a DEG artifact CSV for top-k overlap metrics.",
    )
    parser.add_argument(
        "--topk-values",
        type=int,
        nargs="+",
        default=None,
        help="k values for top-k DEG overlap. Defaults to train config topk_values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.train_config)
    dataset = ProcessedDataset(args.bundle_dir, args.split_name)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = load_torch_model_for_bundle(
        bundle={
            "control_expression": dataset.control_expression,
            "metadata": dataset.metadata,
        },
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config,
        model_type=args.model_type,
    )

    trainer = Trainer(
        model=model,
        config=TrainerConfig.from_dict(train_config),
        output_dir=Path(args.output_path).parent,
    )
    metrics = trainer.evaluate(loader)

    # Optionally compute top-k DEG overlap
    deg_artifact_path = args.deg_artifact_path
    if deg_artifact_path is not None and Path(deg_artifact_path).exists():
        k_values = args.topk_values or train_config.get("evaluation", {}).get(
            "topk_values", [20, 50]
        )
        deg_df = load_deg_artifact(deg_artifact_path)
        all_predictions, all_perturbations = trainer.collect_predictions(loader)
        topk_metrics = compute_topk_deg_metrics(
            predictions=all_predictions,
            perturbation_index=all_perturbations,
            gene_names=dataset.metadata["gene_names"],
            perturbation_names=dataset.metadata["perturbation_names"],
            deg_df=deg_df,
            k_values=k_values,
        )
        metrics.update(topk_metrics)
        LOGGER.info("Added top-k DEG overlap metrics: %s", topk_metrics)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
