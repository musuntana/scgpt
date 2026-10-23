from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_results_assets import generate_model_comparison_figure


def test_model_comparison_uses_only_the_shared_training_protocol(tmp_path):
    transformer_summary = {
        "split": {"train_protocol": "seen"},
        "test_metrics": {"seen_test": {"pearson_per_perturbation": 0.31}},
    }
    mlp_summary = {
        "split": {"train_protocol": "seen"},
        "test_metrics": {"seen_test": {"pearson_per_perturbation": 0.42}},
    }
    xgboost_summary = {
        "train_split_prefix": "seen",
        "metrics": {"seen_test": {"pearson_per_perturbation": 0.53}},
    }
    output_path = tmp_path / "comparison.png"

    generate_model_comparison_figure(
        transformer_summary=transformer_summary,
        mlp_summary=mlp_summary,
        xgboost_summary=xgboost_summary,
        protocol="seen",
        title="Seen protocol comparison",
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_model_comparison_rejects_missing_matching_protocol_metrics(tmp_path):
    summary = {
        "split": {"train_protocol": "seen"},
        "test_metrics": {"seen_test": {"pearson_per_perturbation": 0.31}},
    }

    with pytest.raises(ValueError, match="must contain 'unseen_test' metrics"):
        generate_model_comparison_figure(
            transformer_summary=summary,
            mlp_summary=summary,
            xgboost_summary={
                "train_split_prefix": "unseen",
                "metrics": {"seen_test": {"pearson_per_perturbation": 0.53}},
            },
            protocol="unseen",
            title="Unseen protocol comparison",
            output_path=tmp_path / "invalid.png",
        )
