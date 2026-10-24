from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking.target_ranking import build_target_ranking


def test_build_target_ranking_includes_expected_columns():
    deg_df = pd.DataFrame(
        {
            "gene": ["g1", "g3"],
            "deg_significance": [5.0, 1.0],
        }
    )
    ranking = build_target_ranking(
        gene_names=["g1", "g2", "g3"],
        predicted_delta=np.array([0.1, 0.5, 0.2]),
        deg_df=deg_df,
    )

    assert list(ranking.columns) == [
        "gene",
        "predicted_delta",
        "abs_predicted_delta",
        "deg_significance",
        "importance_score",
        "rank",
    ]
    assert ranking.iloc[0]["rank"] == 1


def test_prediction_only_ranking_does_not_use_observed_deg_labels_by_default():
    """A held-out true DEG must not change the output of an independent predictor."""
    observed_deg = pd.DataFrame(
        {"gene": ["g2"], "deg_significance": [100.0]}
    )
    ranking_with_observed_labels = build_target_ranking(
        gene_names=["g1", "g2"],
        predicted_delta=np.array([3.0, 1.0], dtype=np.float32),
        deg_df=observed_deg,
        abs_predicted_delta_weight=0.5,
        deg_significance_weight=0.5,
    )
    ranking_without_labels = build_target_ranking(
        gene_names=["g1", "g2"],
        predicted_delta=np.array([3.0, 1.0], dtype=np.float32),
    )

    assert ranking_with_observed_labels["gene"].tolist() == ["g1", "g2"]
    assert ranking_with_observed_labels["gene"].tolist() == ranking_without_labels["gene"].tolist()
    np.testing.assert_allclose(
        ranking_with_observed_labels["importance_score"],
        ranking_without_labels["importance_score"],
    )
    assert ranking_with_observed_labels["deg_significance"].eq(0).all()


def test_observed_deg_assisted_mode_is_explicitly_diagnostic_only():
    observed_deg = pd.DataFrame(
        {"gene": ["g2"], "deg_significance": [100.0]}
    )
    ranking = build_target_ranking(
        gene_names=["g1", "g2"],
        predicted_delta=np.array([3.0, 1.0], dtype=np.float32),
        deg_df=observed_deg,
        ranking_mode="observed_deg_diagnostic",
    )

    assert ranking["gene"].tolist() == ["g2", "g1"]
    assert ranking.loc[0, "deg_significance"] == 100.0


def test_observed_deg_assisted_mode_rejects_missing_evidence():
    import pytest

    with pytest.raises(ValueError, match="requires a non-empty observed DEG artifact"):
        build_target_ranking(
            gene_names=["g1", "g2"],
            predicted_delta=np.array([3.0, 1.0], dtype=np.float32),
            ranking_mode="observed_deg_diagnostic",
        )


def test_unknown_ranking_mode_is_not_silently_accepted():
    import pytest

    with pytest.raises(ValueError, match="Unknown ranking_mode"):
        build_target_ranking(
            gene_names=["g1"],
            predicted_delta=np.array([1.0], dtype=np.float32),
            ranking_mode="auto",
        )
