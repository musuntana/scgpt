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
