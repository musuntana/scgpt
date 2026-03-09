from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import read_json
from src.data.pairing import load_processed_bundle
from src.evaluation.deg import DEG_ARTIFACT_FILENAME, load_deg_artifact
from src.evaluation.inference import (
    build_gene_comparison_frame,
    build_perturbation_batch,
    load_torch_model_for_bundle,
    predict_delta_for_batch,
    summarize_perturbation_fit,
)
from src.ranking.target_ranking import build_target_ranking
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README result assets.")
    parser.add_argument(
        "--bundle-dir",
        default="data/processed/norman2019_demo_bundle",
        help="Processed bundle directory.",
    )
    parser.add_argument(
        "--transformer-artifact-dir",
        default="artifacts/transformer_seen_norman2019_demo",
        help="Transformer artifact directory.",
    )
    parser.add_argument(
        "--mlp-artifact-dir",
        default="artifacts/mlp_seen_norman2019_demo",
        help="MLP artifact directory.",
    )
    parser.add_argument(
        "--xgboost-artifact-dir",
        default="artifacts/xgboost_seen_norman2019_demo",
        help="XGBoost artifact directory.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
        help="Model config path for loading torch checkpoints.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="Train config path for ranking weights.",
    )
    parser.add_argument(
        "--perturbation-name",
        default="JUN",
        help="Perturbation to visualize in the README preview figure.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/assets",
        help="Directory to write generated image assets.",
    )
    return parser.parse_args()


def _load_summary(path: str | Path) -> dict:
    return read_json(path)


def generate_model_comparison_figure(
    transformer_summary: dict,
    mlp_summary: dict,
    xgboost_summary: dict,
    output_path: Path,
) -> None:
    models = ["Transformer", "MLP", "XGBoost"]
    seen_pearson = [
        transformer_summary["test_metrics"]["seen_test"]["pearson_per_perturbation"],
        mlp_summary["test_metrics"]["seen_test"]["pearson_per_perturbation"],
        xgboost_summary["metrics"]["seen_test"]["pearson_per_perturbation"],
    ]
    unseen_pearson = [
        transformer_summary["test_metrics"]["unseen_test"]["pearson_per_perturbation"],
        mlp_summary["test_metrics"]["unseen_test"]["pearson_per_perturbation"],
        xgboost_summary["metrics"]["unseen_test"]["pearson_per_perturbation"],
    ]

    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(models))
    width = 0.34

    axis.bar(x - width / 2, seen_pearson, width, label="seen_test")
    axis.bar(x + width / 2, unseen_pearson, width, label="unseen_test")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Per-perturbation Pearson")
    axis.set_title("Norman2019 Demo Bundle: Model Comparison")
    axis.set_xticks(x)
    axis.set_xticklabels(models)
    axis.legend()

    for x_pos, value in zip(x - width / 2, seen_pearson, strict=True):
        axis.text(x_pos, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for x_pos, value in zip(x + width / 2, unseen_pearson, strict=True):
        axis.text(x_pos, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_inference_preview_figure(
    *,
    bundle_dir: str | Path,
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    deg_artifact_path: str | Path,
    perturbation_name: str,
    output_path: Path,
) -> str:
    bundle = load_processed_bundle(bundle_dir)
    perturbation_names = bundle["metadata"]["perturbation_names"]
    if perturbation_name not in perturbation_names:
        perturbation_name = perturbation_names[0]

    model = load_torch_model_for_bundle(
        bundle=bundle,
        checkpoint_path=checkpoint_path,
        model_config_path=model_config_path,
        model_type="transformer",
    )
    batch = build_perturbation_batch(bundle, perturbation_name)
    predicted_delta = predict_delta_for_batch(model, batch)
    comparison_df = build_gene_comparison_frame(
        gene_names=bundle["metadata"]["gene_names"],
        predicted_delta=predicted_delta,
        observed_delta=batch.observed_delta_mean,
    )
    ranking_config = load_yaml(train_config_path).get("ranking", {})
    deg_df = pd.DataFrame()
    artifact_path = Path(deg_artifact_path)
    if artifact_path.exists():
        all_deg = load_deg_artifact(artifact_path)
        deg_df = all_deg[all_deg["perturbation"] == perturbation_name].copy()

    ranking_df = build_target_ranking(
        gene_names=bundle["metadata"]["gene_names"],
        predicted_delta=predicted_delta,
        deg_df=deg_df if not deg_df.empty else None,
        abs_predicted_delta_weight=(
            float(ranking_config.get("abs_predicted_delta_weight", 0.5))
            if not deg_df.empty
            else 1.0
        ),
        deg_significance_weight=(
            float(ranking_config.get("deg_significance_weight", 0.5))
            if not deg_df.empty
            else 0.0
        ),
    )
    fit_metrics = summarize_perturbation_fit(
        predicted_delta=predicted_delta,
        observed_delta=batch.observed_delta_mean,
    )

    top_up = comparison_df.sort_values("predicted_delta", ascending=False).head(10)
    top_down = comparison_df.sort_values("predicted_delta", ascending=True).head(10)
    top_ranked = ranking_df.head(10).iloc[::-1]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].scatter(
        comparison_df["observed_delta"],
        comparison_df["predicted_delta"],
        alpha=0.55,
        s=14,
    )
    axes[0, 0].axline((0, 0), slope=1.0, linestyle="--", color="black", linewidth=1.0)
    axes[0, 0].set_xlabel("Observed mean delta")
    axes[0, 0].set_ylabel("Predicted delta")
    axes[0, 0].set_title(
        f"{perturbation_name}: Predicted vs Observed\n"
        f"Pearson={fit_metrics['pearson']:.3f}, MSE={fit_metrics['mse']:.4f}, n={batch.sample_count}"
    )

    axes[0, 1].barh(top_up["gene"][::-1], top_up["predicted_delta"][::-1])
    axes[0, 1].set_title("Top Predicted Up Genes")
    axes[0, 1].set_xlabel("Predicted delta")

    axes[1, 0].barh(top_down["gene"], top_down["predicted_delta"])
    axes[1, 0].set_title("Top Predicted Down Genes")
    axes[1, 0].set_xlabel("Predicted delta")

    axes[1, 1].barh(top_ranked["gene"], top_ranked["importance_score"])
    axes[1, 1].set_title(
        "Predicted + DEG Ranking" if not deg_df.empty else "Prediction-Only Target Ranking"
    )
    axes[1, 1].set_xlabel("Importance score")

    figure.suptitle("PerturbScope-GPT Result Preview", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return perturbation_name


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformer_summary = _load_summary(
        Path(args.transformer_artifact_dir) / "run_summary.json"
    )
    mlp_summary = _load_summary(Path(args.mlp_artifact_dir) / "run_summary.json")
    xgboost_summary = _load_summary(
        Path(args.xgboost_artifact_dir) / "xgboost_run_summary.json"
    )

    generate_model_comparison_figure(
        transformer_summary=transformer_summary,
        mlp_summary=mlp_summary,
        xgboost_summary=xgboost_summary,
        output_path=output_dir / "model_comparison_seen_norman2019_demo.png",
    )
    used_perturbation = generate_inference_preview_figure(
        bundle_dir=args.bundle_dir,
        checkpoint_path=Path(args.transformer_artifact_dir) / "best_model.pt",
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        deg_artifact_path=Path(args.transformer_artifact_dir) / DEG_ARTIFACT_FILENAME,
        perturbation_name=args.perturbation_name,
        output_path=output_dir / "transformer_inference_preview.png",
    )

    print(f"Generated README assets in {output_dir}")
    print(f"Inference preview perturbation: {used_perturbation}")


if __name__ == "__main__":
    main()
