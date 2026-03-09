"""Utilities for comparing multiple trained models from their artifact directories.

This module contains pure functions with no UI or framework dependencies so they
can be tested independently from the Streamlit app.
"""
from __future__ import annotations

from pathlib import Path

from src.data.io import read_json


def extract_summary_row(summary: dict, label: str) -> dict | None:
    """Return a flat metrics dict from a run_summary.json or xgboost_run_summary.json.

    Args:
        summary: Parsed JSON content of a run_summary or xgboost_run_summary file.
        label: Display label for this model (usually the artifact sub-directory name).

    Returns:
        Flat dict with ``model``, ``seen_pearson``, ``seen_mse``, etc., or ``None``
        when the summary contains no recognisable test-metrics section.
    """
    # Transformer / MLP layout uses "test_metrics"; XGBoost uses "metrics"
    test_metrics = summary.get("test_metrics") or summary.get("metrics")
    if not test_metrics:
        return None
    seen = test_metrics.get("seen_test", {})
    unseen = test_metrics.get("unseen_test", {})
    return {
        "model": label,
        "seen_pearson": seen.get("pearson_per_perturbation"),
        "seen_mse": seen.get("mse_per_perturbation"),
        "unseen_pearson": unseen.get("pearson_per_perturbation"),
        "unseen_mse": unseen.get("mse_per_perturbation"),
        "seen_top20_deg": seen.get("topk_deg_overlap_20"),
        "seen_top100_deg": seen.get("topk_deg_overlap_100"),
        "unseen_top20_deg": unseen.get("topk_deg_overlap_20"),
        "unseen_top100_deg": unseen.get("topk_deg_overlap_100"),
    }


def scan_artifact_comparison_rows(artifact_root: str | Path) -> list[dict]:
    """Scan *artifact_root* sub-directories for run_summary files.

    Tries ``run_summary.json`` first (Transformer / MLP), then falls back to
    ``xgboost_run_summary.json``.  Directories without either file are skipped.

    Args:
        artifact_root: Parent directory that contains per-model artifact directories.

    Returns:
        List of flat metric dicts suitable for building a comparison DataFrame,
        sorted by sub-directory name.
    """
    root = Path(artifact_root)
    rows: list[dict] = []
    if not root.exists():
        return rows
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        label = sub.name
        rs_path = sub / "run_summary.json"
        if rs_path.exists():
            row = extract_summary_row(read_json(rs_path), label)
            if row:
                rows.append(row)
            continue
        xgb_path = sub / "xgboost_run_summary.json"
        if xgb_path.exists():
            row = extract_summary_row(read_json(xgb_path), label)
            if row:
                rows.append(row)
    return rows
