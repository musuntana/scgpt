from __future__ import annotations

from typing import Any

from sklearn.multioutput import MultiOutputRegressor


def build_xgboost_baseline(params: dict[str, Any] | None = None):
    """Build a multi-output XGBoost regressor."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required for the XGBoost baseline. Install dependencies first."
        ) from exc

    default_params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "n_jobs": 1,
    }
    if params:
        default_params.update(params)
    return MultiOutputRegressor(XGBRegressor(**default_params))

