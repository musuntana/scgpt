from __future__ import annotations

import numpy as np

from src.data.synthetic import (
    SyntheticDemoConfig,
    build_synthetic_deg_artifact,
    generate_synthetic_processed_bundle,
)


def test_generate_synthetic_processed_bundle_returns_expected_shapes():
    bundle, perturbation_effects = generate_synthetic_processed_bundle(
        SyntheticDemoConfig(
            num_genes=12,
            samples_per_perturbation=6,
            effect_genes_per_perturbation=2,
            random_seed=7,
            perturbation_names=("JUN", "STAT1", "IRF1"),
        )
    )

    assert bundle.control_expression.shape == (18, 12)
    assert bundle.target_delta.shape == (18, 12)
    assert len(bundle.gene_names) == 12
    assert set(bundle.perturbation_names) == {"JUN", "STAT1", "IRF1"}
    assert set(bundle.splits) == {
        "seen_train",
        "seen_val",
        "seen_test",
        "unseen_train",
        "unseen_val",
        "unseen_test",
    }
    assert set(perturbation_effects) == {"JUN", "STAT1", "IRF1"}


def test_build_synthetic_deg_artifact_contains_expected_schema():
    gene_names = [f"GENE_{idx:03d}" for idx in range(6)]
    perturbation_effects = {
        "JUN": np.array([1.0, 0.8, 0.0, 0.0, -0.7, -0.9], dtype=np.float32),
        "STAT1": np.array([0.0, 0.0, 1.2, 0.9, -0.8, 0.0], dtype=np.float32),
    }

    deg_df = build_synthetic_deg_artifact(
        gene_names=gene_names,
        perturbation_effects=perturbation_effects,
        perturbation_cell_count=10,
        min_abs_logfoldchange=0.25,
    )

    assert not deg_df.empty
    assert {
        "perturbation",
        "rank",
        "gene",
        "logfoldchange",
        "adjusted_p_value",
        "score",
        "deg_significance",
        "perturbation_cell_count",
        "control_cell_count",
    }.issubset(deg_df.columns)
    assert set(deg_df["perturbation"]) == {"JUN", "STAT1"}
