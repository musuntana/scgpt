from __future__ import annotations

from src.models.xgboost_baseline import build_xgboost_baseline


def test_build_xgboost_baseline_sets_random_state():
    model = build_xgboost_baseline()
    assert model.estimator.get_params()["random_state"] == 42
