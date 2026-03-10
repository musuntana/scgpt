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

from src.data.pairing import save_processed_bundle
from src.data.synthetic import (
    SyntheticDemoConfig,
    build_synthetic_deg_artifact,
    generate_synthetic_processed_bundle,
)
from src.data.torch_dataset import ProcessedDataset
from src.evaluation.deg import save_deg_artifact
from src.models.transformer import TransformerPerturbationModel
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import load_yaml
from src.utils.experiment import build_run_summary, write_run_summary
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an offline synthetic demo bundle and artifacts.")
    parser.add_argument(
        "--bundle-dir",
        default="data/processed/synthetic_demo_bundle",
        help="Directory to write the synthetic processed bundle.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/transformer_seen_synthetic_demo",
        help="Directory to write the synthetic checkpoint, metrics, DEG artifact, and summary.",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_synthetic_demo.yaml",
        help="Path to the synthetic data config YAML.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model_synthetic_demo.yaml",
        help="Path to the synthetic model config YAML.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train_synthetic_demo.yaml",
        help="Path to the synthetic train config YAML.",
    )
    parser.add_argument("--num-genes", type=int, default=64, help="Number of synthetic genes.")
    parser.add_argument(
        "--samples-per-perturbation",
        type=int,
        default=48,
        help="Number of synthetic samples per perturbation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for bundle generation and training.",
    )
    return parser.parse_args()


def _build_model(num_genes: int, num_perturbations: int, model_config: dict) -> TransformerPerturbationModel:
    transformer_cfg = model_config["transformer"]
    return TransformerPerturbationModel(
        num_genes=num_genes,
        num_perturbations=num_perturbations,
        d_model=int(transformer_cfg["d_model"]),
        n_heads=int(transformer_cfg["n_heads"]),
        n_layers=int(transformer_cfg["n_layers"]),
        ffn_dim=int(transformer_cfg["ffn_dim"]),
        dropout=float(transformer_cfg["dropout"]),
    )


def _write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    args = parse_args()
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    train_config = load_yaml(args.train_config)
    seed_everything(int(args.random_seed))

    synthetic_config = SyntheticDemoConfig(
        num_genes=int(args.num_genes),
        samples_per_perturbation=int(args.samples_per_perturbation),
        random_seed=int(args.random_seed),
        val_fraction=float(data_config["split"]["val_fraction"]),
        test_fraction=float(data_config["split"]["test_fraction"]),
    )
    bundle, perturbation_effects = generate_synthetic_processed_bundle(synthetic_config)
    save_processed_bundle(bundle, args.bundle_dir)
    LOGGER.info("Saved synthetic bundle to %s", args.bundle_dir)

    trainer_config = TrainerConfig.from_dict(train_config)
    train_dataset = ProcessedDataset(args.bundle_dir, "seen_train")
    val_dataset = ProcessedDataset(args.bundle_dir, "seen_val")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_model(
        num_genes=train_dataset.num_genes,
        num_perturbations=train_dataset.num_perturbations,
        model_config=model_config,
    )
    trainer = Trainer(model=model, config=trainer_config, output_dir=output_dir)
    trainer.fit(train_loader=train_loader, val_loader=val_loader)

    checkpoint_path = output_dir / "best_model.pt"
    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    for split_name in ("seen_test", "unseen_test"):
        eval_dataset = ProcessedDataset(args.bundle_dir, split_name)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=trainer_config.batch_size,
            shuffle=False,
            num_workers=trainer_config.num_workers,
        )
        metrics = trainer.evaluate(eval_loader)
        metrics_path = output_dir / f"{split_name}_metrics.json"
        _write_metrics(metrics_path, metrics)
        LOGGER.info("Saved synthetic %s metrics to %s", split_name, metrics_path)

    deg_df = build_synthetic_deg_artifact(
        gene_names=bundle.gene_names,
        perturbation_effects=perturbation_effects,
        perturbation_cell_count=synthetic_config.samples_per_perturbation,
        min_abs_logfoldchange=float(train_config["deg"]["abs_logfoldchange_threshold"]),
    )
    save_deg_artifact(
        deg_df=deg_df,
        output_dir=output_dir,
        metadata={
            "dataset_name": data_config["dataset"]["name"],
            "source": data_config["dataset"]["source"],
            "bundle_dir": str(args.bundle_dir),
            "num_rows": int(len(deg_df)),
            "num_genes": int(len(bundle.gene_names)),
            "num_perturbations": int(len(bundle.perturbation_names)),
            "deg": train_config["deg"],
        },
    )

    summary = build_run_summary(
        bundle_dir=args.bundle_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        model_type="transformer",
        split_prefix="seen",
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        history_path=output_dir / "history.json",
        seen_metrics_path=output_dir / "seen_test_metrics.json",
        unseen_metrics_path=output_dir / "unseen_test_metrics.json",
    )
    write_run_summary(summary, output_dir / "run_summary.json")
    LOGGER.info("Saved synthetic run summary to %s", output_dir / "run_summary.json")


if __name__ == "__main__":
    main()
