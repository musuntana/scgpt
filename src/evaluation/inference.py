from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import mean_squared_error, pearson_correlation
from src.models.mlp import MLPBaseline
from src.models.transformer import TransformerPerturbationModel
from src.utils.config import load_yaml


@dataclass
class PerturbationInferenceBatch:
    """Aggregated bundle view for one perturbation condition."""

    perturbation_name: str
    perturbation_index: int
    sample_count: int
    control_mean: np.ndarray
    observed_delta_mean: np.ndarray


def build_torch_model(
    *,
    model_type: str,
    num_genes: int,
    num_perturbations: int,
    model_config: dict[str, Any],
) -> torch.nn.Module:
    """Construct a torch model for local inference."""
    if model_type == "mlp":
        return MLPBaseline(
            num_genes=num_genes,
            num_perturbations=num_perturbations,
        )
    if model_type != "transformer":
        raise ValueError(f"Unsupported torch model_type={model_type!r}")

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


def load_torch_model_for_bundle(
    *,
    bundle: dict[str, Any],
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    model_type: str = "transformer",
) -> torch.nn.Module:
    """Load a checkpointed torch model for a processed bundle."""
    model = build_torch_model(
        model_type=model_type,
        num_genes=int(bundle["control_expression"].shape[1]),
        num_perturbations=int(len(bundle["metadata"]["perturbation_names"])),
        model_config=load_yaml(model_config_path),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_perturbation_batch(
    bundle: dict[str, Any],
    perturbation_name: str,
) -> PerturbationInferenceBatch:
    """Aggregate reference control and observed deltas for one perturbation."""
    perturbation_names = bundle["metadata"]["perturbation_names"]
    if perturbation_name not in perturbation_names:
        raise KeyError(f"Unknown perturbation_name={perturbation_name!r}")

    perturbation_index = int(perturbation_names.index(perturbation_name))
    mask = bundle["perturbation_index"] == perturbation_index
    if int(mask.sum()) == 0:
        raise ValueError(f"No samples found for perturbation={perturbation_name}")

    return PerturbationInferenceBatch(
        perturbation_name=perturbation_name,
        perturbation_index=perturbation_index,
        sample_count=int(mask.sum()),
        control_mean=bundle["control_expression"][mask].mean(axis=0).astype(np.float32),
        observed_delta_mean=bundle["target_delta"][mask].mean(axis=0).astype(np.float32),
    )


@torch.no_grad()
def predict_delta_for_batch(
    model: torch.nn.Module,
    batch: PerturbationInferenceBatch,
) -> np.ndarray:
    """Run one aggregated perturbation inference pass."""
    control_tensor = torch.from_numpy(batch.control_mean).float().unsqueeze(0)
    perturbation_tensor = torch.tensor([batch.perturbation_index], dtype=torch.long)
    predicted = model(control_tensor, perturbation_tensor)
    return predicted.squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_gene_comparison_frame(
    *,
    gene_names: list[str],
    predicted_delta: np.ndarray,
    observed_delta: np.ndarray,
) -> pd.DataFrame:
    """Create a gene-level comparison frame for app display."""
    predicted_delta = np.asarray(predicted_delta, dtype=np.float32).reshape(-1)
    observed_delta = np.asarray(observed_delta, dtype=np.float32).reshape(-1)
    if len(gene_names) != len(predicted_delta) or len(gene_names) != len(observed_delta):
        raise ValueError("gene_names, predicted_delta, and observed_delta must align")

    frame = pd.DataFrame(
        {
            "gene": gene_names,
            "predicted_delta": predicted_delta,
            "observed_delta": observed_delta,
        }
    )
    frame["abs_predicted_delta"] = frame["predicted_delta"].abs()
    frame["abs_observed_delta"] = frame["observed_delta"].abs()
    frame["residual"] = frame["predicted_delta"] - frame["observed_delta"]
    frame["abs_residual"] = frame["residual"].abs()
    return frame.sort_values("abs_predicted_delta", ascending=False).reset_index(drop=True)


def summarize_perturbation_fit(
    *,
    predicted_delta: np.ndarray,
    observed_delta: np.ndarray,
) -> dict[str, float]:
    """Compute per-perturbation summary metrics for aggregated inference."""
    return {
        "pearson": pearson_correlation(predicted_delta, observed_delta),
        "mse": mean_squared_error(predicted_delta, observed_delta),
    }
