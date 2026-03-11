"""Utilities for comparing multiple trained models from their artifact directories.

This module contains pure functions with no UI or framework dependencies so they
can be tested independently from the Streamlit app.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from src.data.io import read_json


def _extract_model_type(summary: dict, label: str) -> str:
    model_section = summary.get("model", {})
    if isinstance(model_section, dict) and model_section.get("model_type") is not None:
        return str(model_section["model_type"])
    if summary.get("model_type") is not None:
        return str(summary["model_type"])
    return shorten_model_label(label)


def _extract_dataset_name(summary: dict) -> str | None:
    dataset_section = summary.get("dataset", {})
    if isinstance(dataset_section, dict) and dataset_section.get("name") is not None:
        return str(dataset_section["name"])
    return None


def _extract_train_protocol(summary: dict) -> str | None:
    split_section = summary.get("split", {})
    if isinstance(split_section, dict) and split_section.get("train_protocol") is not None:
        return str(split_section["train_protocol"])
    if summary.get("train_split_prefix") is not None:
        return str(summary["train_split_prefix"])
    return None


def _extract_seed(summary: dict) -> int | None:
    training_section = summary.get("training", {})
    if isinstance(training_section, dict) and training_section.get("seed") is not None:
        return int(training_section["seed"])
    xgboost_params = summary.get("xgboost_params", {})
    if isinstance(xgboost_params, dict) and xgboost_params.get("random_state") is not None:
        return int(xgboost_params["random_state"])
    if summary.get("seed") is not None:
        return int(summary["seed"])
    return None


def normalize_seeded_label(label: str) -> str:
    """Strip trailing seed and demo suffixes from artifact labels."""
    normalized = re.sub(r"([_-](seed|s)\d+)$", "", label)
    return normalized.removesuffix("_demo")


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
        "base_model_label": normalize_seeded_label(label),
        "model_type": _extract_model_type(summary, label),
        "dataset_name": _extract_dataset_name(summary),
        "train_protocol": _extract_train_protocol(summary),
        "seed": _extract_seed(summary),
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


def shorten_model_label(label: str) -> str:
    """Collapse artifact directory names into short model labels for charts."""
    shortened = label
    for suffix in ("_seen_norman2019_demo", "_seen_synthetic_demo"):
        shortened = shortened.removesuffix(suffix)
    return shortened


def _annotation_offset(values: Sequence[float]) -> float:
    peak = max((abs(float(value)) for value in values), default=0.0)
    if peak == 0.0:
        return 0.05
    if peak >= 1.0:
        return max(peak * 0.03, 0.02)
    if peak >= 0.1:
        return max(peak * 0.03, 0.01)
    if peak >= 0.01:
        return max(peak * 0.03, 0.001)
    return max(peak * 0.03, 0.0002)


def plot_grouped_metric_bars(
    ax: Axes,
    models: Sequence[str],
    seen_vals: Sequence[float],
    unseen_vals: Sequence[float],
    *,
    ylabel: str,
    title: str,
    seen_label: str = "Seen test",
    unseen_label: str = "Unseen test",
    seen_color: str = "steelblue",
    unseen_color: str = "coral",
    annotate: bool = False,
    value_format: str = "{:.3f}",
    x_tick_rotation: int = 15,
) -> None:
    """Draw a grouped bar chart with enough headroom for value labels."""
    x = np.arange(len(models))
    width = 0.35
    seen_bars = ax.bar(x - width / 2, seen_vals, width, label=seen_label, color=seen_color)
    unseen_bars = ax.bar(x + width / 2, unseen_vals, width, label=unseen_label, color=unseen_color)

    numeric_values = [float(value) for value in [*seen_vals, *unseen_vals]]
    offset = _annotation_offset(numeric_values)
    y_min = min(0.0, min(numeric_values, default=0.0))
    if y_min < 0.0:
        y_min -= offset * 2.0
    y_max = max(numeric_values, default=0.0) + offset * 2.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=x_tick_rotation, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(fontsize=8)

    if not annotate:
        return

    for bars in (seen_bars, unseen_bars):
        for bar in bars:
            value = float(bar.get_height())
            y_pos = value + offset if value >= 0.0 else value - offset
            vertical_alignment = "bottom" if value >= 0.0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                value_format.format(value),
                ha="center",
                va=vertical_alignment,
                fontsize=9,
            )
