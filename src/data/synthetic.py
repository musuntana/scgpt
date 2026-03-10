from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.pairing import ProcessedBundle, create_split_indices


@dataclass(frozen=True)
class SyntheticDemoConfig:
    num_genes: int = 64
    samples_per_perturbation: int = 48
    effect_genes_per_perturbation: int = 8
    effect_size: float = 1.5
    control_loc: float = 1.0
    control_scale: float = 0.25
    noise_scale: float = 0.05
    val_fraction: float = 0.1
    test_fraction: float = 0.2
    random_seed: int = 42
    perturbation_names: tuple[str, ...] = ("JUN", "STAT1", "CEBPB", "IRF1")


def _build_gene_names(num_genes: int) -> list[str]:
    return [f"GENE_{gene_idx:03d}" for gene_idx in range(num_genes)]


def _build_effect_vector(
    *,
    num_genes: int,
    perturbation_index: int,
    effect_genes_per_perturbation: int,
    effect_size: float,
) -> np.ndarray:
    effect = np.zeros(num_genes, dtype=np.float32)
    positive_start = (perturbation_index * effect_genes_per_perturbation * 2) % num_genes
    negative_start = (positive_start + effect_genes_per_perturbation) % num_genes

    positive_indices = [
        (positive_start + offset) % num_genes for offset in range(effect_genes_per_perturbation)
    ]
    negative_indices = [
        (negative_start + offset) % num_genes for offset in range(effect_genes_per_perturbation)
    ]
    effect[positive_indices] = effect_size
    effect[negative_indices] = -0.7 * effect_size
    return effect


def generate_synthetic_processed_bundle(
    config: SyntheticDemoConfig | None = None,
) -> tuple[ProcessedBundle, dict[str, np.ndarray]]:
    """生成一个离线可跑的合成 processed bundle。"""
    config = config or SyntheticDemoConfig()
    rng = np.random.default_rng(config.random_seed)

    gene_names = _build_gene_names(config.num_genes)
    perturbation_names = list(config.perturbation_names)
    perturbation_effects = {
        perturbation_name: _build_effect_vector(
            num_genes=config.num_genes,
            perturbation_index=perturbation_index,
            effect_genes_per_perturbation=config.effect_genes_per_perturbation,
            effect_size=config.effect_size,
        )
        for perturbation_index, perturbation_name in enumerate(perturbation_names)
    }

    num_samples = len(perturbation_names) * config.samples_per_perturbation
    control_expression = np.zeros((num_samples, config.num_genes), dtype=np.float32)
    target_delta = np.zeros((num_samples, config.num_genes), dtype=np.float32)
    perturbation_index = np.zeros(num_samples, dtype=np.int64)
    sample_ids: list[str] = []

    sample_cursor = 0
    for perturbation_idx, perturbation_name in enumerate(perturbation_names):
        effect = perturbation_effects[perturbation_name]
        for sample_idx in range(config.samples_per_perturbation):
            sampled_control = rng.normal(
                loc=config.control_loc,
                scale=config.control_scale,
                size=config.num_genes,
            ).astype(np.float32)
            sampled_noise = rng.normal(
                loc=0.0,
                scale=config.noise_scale,
                size=config.num_genes,
            ).astype(np.float32)
            sampled_delta = effect + (0.1 * np.tanh(sampled_control - config.control_loc)) + sampled_noise

            control_expression[sample_cursor] = sampled_control
            target_delta[sample_cursor] = sampled_delta.astype(np.float32)
            perturbation_index[sample_cursor] = perturbation_idx
            sample_ids.append(f"{perturbation_name}_sample_{sample_idx:03d}")
            sample_cursor += 1

    splits = create_split_indices(
        perturbation_index=perturbation_index,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        random_seed=config.random_seed,
    )
    bundle = ProcessedBundle(
        control_expression=control_expression,
        target_delta=target_delta,
        perturbation_index=perturbation_index,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        sample_ids=sample_ids,
        splits=splits,
    )
    return bundle, perturbation_effects


def build_synthetic_deg_artifact(
    *,
    gene_names: Sequence[str],
    perturbation_effects: dict[str, np.ndarray],
    perturbation_cell_count: int,
    control_cell_count: int | None = None,
    min_abs_logfoldchange: float = 0.25,
) -> pd.DataFrame:
    """根据合成扰动效应生成 app 可消费的 DEG artifact。"""
    rows: list[dict[str, float | int | str]] = []
    resolved_control_count = (
        int(control_cell_count)
        if control_cell_count is not None
        else int(perturbation_cell_count)
    )

    for perturbation_name, effect in perturbation_effects.items():
        effect = np.asarray(effect, dtype=np.float32).reshape(-1)
        ranked_indices = np.argsort(np.abs(effect))[::-1]
        rank = 1
        for gene_idx in ranked_indices:
            logfoldchange = float(effect[gene_idx])
            if abs(logfoldchange) < float(min_abs_logfoldchange):
                continue

            score = abs(logfoldchange)
            adjusted_p_value = float(min(0.99, 10 ** (-(1.0 + score))))
            rows.append(
                {
                    "perturbation": perturbation_name,
                    "rank": rank,
                    "gene": str(gene_names[gene_idx]),
                    "logfoldchange": logfoldchange,
                    "adjusted_p_value": adjusted_p_value,
                    "score": score,
                    "deg_significance": float(-np.log10(adjusted_p_value + 1e-12)),
                    "perturbation_cell_count": int(perturbation_cell_count),
                    "control_cell_count": resolved_control_count,
                }
            )
            rank += 1

    return pd.DataFrame(
        rows,
        columns=[
            "perturbation",
            "rank",
            "gene",
            "logfoldchange",
            "adjusted_p_value",
            "score",
            "deg_significance",
            "perturbation_cell_count",
            "control_cell_count",
        ],
    )
