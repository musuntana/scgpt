from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.torch_dataset import ProcessedDataset
from src.evaluation.deg import load_deg_artifact
from src.evaluation.error_analysis import (
    build_error_summary,
    build_per_perturbation_error_table,
)
from src.evaluation.inference import load_torch_model_for_bundle
from src.evaluation.metrics import compute_regression_metrics, compute_topk_deg_metrics
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
    parser.add_argument(
        "--per-perturbation-output-path",
        default=None,
        help="Optional CSV path for the perturbation-level error table.",
    )
    parser.add_argument(
        "--error-summary-output-path",
        default=None,
        help="Optional JSON path for the summarized worst-case perturbations.",
    )
    parser.add_argument(
        "--top-gene-count",
        type=int,
        default=5,
        help="How many genes to include in perturbation-level residual summaries.",
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
    predictions, targets, perturbations = trainer.collect_outputs(loader)
    metrics = compute_regression_metrics(
        predictions=predictions,
        targets=targets,
        perturbation_index=perturbations,
    )

    # Optionally compute top-k DEG overlap
    deg_df = pd.DataFrame()
    k_values = args.topk_values or train_config.get("evaluation", {}).get("topk_values", [20, 50])
    deg_artifact_path = args.deg_artifact_path
    if deg_artifact_path is not None and Path(deg_artifact_path).exists():
        deg_df = load_deg_artifact(deg_artifact_path)
        topk_metrics = compute_topk_deg_metrics(
            predictions=predictions,
            perturbation_index=perturbations,
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

    error_table = build_per_perturbation_error_table(
        predictions=predictions,
        targets=targets,
        perturbation_index=perturbations,
        perturbation_names=dataset.metadata["perturbation_names"],
        gene_names=dataset.metadata["gene_names"],
        deg_df=deg_df if not deg_df.empty else None,
        k_values=k_values,
        top_gene_count=args.top_gene_count,
    )

    if args.per_perturbation_output_path is not None:
        per_perturbation_output_path = Path(args.per_perturbation_output_path)
        per_perturbation_output_path.parent.mkdir(parents=True, exist_ok=True)
        error_table.to_csv(per_perturbation_output_path, index=False)
        LOGGER.info("Wrote perturbation-level error table to %s", per_perturbation_output_path)

    if args.error_summary_output_path is not None:
        error_summary_output_path = Path(args.error_summary_output_path)
        error_summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        with error_summary_output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                build_error_summary(
                    error_table,
                    split_name=args.split_name,
                    model_type=args.model_type,
                ),
                handle,
                indent=2,
            )
        LOGGER.info("Wrote error summary to %s", error_summary_output_path)


if __name__ == "__main__":
    main()
