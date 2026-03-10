"""Tests for src/utils/comparison — model comparison utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.comparison import extract_summary_row, scan_artifact_comparison_rows

# ---------------------------------------------------------------------------
# extract_summary_row
# ---------------------------------------------------------------------------

_TORCH_SUMMARY = {
    "test_metrics": {
        "seen_test": {
            "pearson_per_perturbation": 0.604,
            "mse_per_perturbation": 0.007,
            "topk_deg_overlap_20": 0.816,
            "topk_deg_overlap_100": 0.964,
        },
        "unseen_test": {
            "pearson_per_perturbation": 0.824,
            "mse_per_perturbation": 0.001,
            "topk_deg_overlap_20": 0.930,
            "topk_deg_overlap_100": 0.976,
        },
    }
}

_XGBOOST_SUMMARY = {
    "metrics": {
        "seen_test": {
            "pearson_per_perturbation": 0.618,
            "mse_per_perturbation": 0.006,
        },
        "unseen_test": {
            "pearson_per_perturbation": 0.840,
            "mse_per_perturbation": 0.0008,
        },
    }
}


def test_extract_summary_row_torch_layout():
    row = extract_summary_row(_TORCH_SUMMARY, "transformer_seen_norman2019_demo")
    assert row is not None
    assert row["model"] == "transformer_seen_norman2019_demo"
    assert abs(row["seen_pearson"] - 0.604) < 1e-6
    assert abs(row["unseen_pearson"] - 0.824) < 1e-6
    assert abs(row["seen_top20_deg"] - 0.816) < 1e-6
    assert abs(row["unseen_top100_deg"] - 0.976) < 1e-6


def test_extract_summary_row_xgboost_layout():
    row = extract_summary_row(_XGBOOST_SUMMARY, "xgboost_seen_norman2019_demo")
    assert row is not None
    assert row["model"] == "xgboost_seen_norman2019_demo"
    assert abs(row["seen_pearson"] - 0.618) < 1e-6
    assert row["seen_top20_deg"] is None  # not present in xgboost summary


def test_extract_summary_row_returns_none_for_empty_summary():
    assert extract_summary_row({}, "empty") is None
    assert extract_summary_row({"other_key": {}}, "no_metrics") is None


def test_extract_summary_row_partial_metrics():
    """Partial metrics dict: missing keys should come back as None, not raise."""
    summary = {"test_metrics": {"seen_test": {"pearson_per_perturbation": 0.5}}}
    row = extract_summary_row(summary, "partial")
    assert row is not None
    assert abs(row["seen_pearson"] - 0.5) < 1e-6
    assert row["seen_mse"] is None
    assert row["unseen_pearson"] is None


# ---------------------------------------------------------------------------
# scan_artifact_comparison_rows
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_scan_returns_empty_for_missing_root(tmp_path):
    rows = scan_artifact_comparison_rows(tmp_path / "nonexistent")
    assert rows == []


def test_scan_returns_empty_for_empty_directory(tmp_path):
    rows = scan_artifact_comparison_rows(tmp_path)
    assert rows == []


def test_scan_picks_up_torch_and_xgboost_summaries(tmp_path):
    # Write a torch summary and an xgboost summary in separate sub-dirs
    _write_json(tmp_path / "mlp_seen" / "run_summary.json", _TORCH_SUMMARY)
    _write_json(tmp_path / "xgboost_seen" / "xgboost_run_summary.json", _XGBOOST_SUMMARY)
    # Subdir with no recognized file should be skipped
    (tmp_path / "other_dir").mkdir()

    rows = scan_artifact_comparison_rows(tmp_path)
    assert len(rows) == 2
    labels = {r["model"] for r in rows}
    assert labels == {"mlp_seen", "xgboost_seen"}


def test_scan_results_are_sorted_by_directory_name(tmp_path):
    _write_json(tmp_path / "zzz_model" / "run_summary.json", _TORCH_SUMMARY)
    _write_json(tmp_path / "aaa_model" / "run_summary.json", _TORCH_SUMMARY)
    rows = scan_artifact_comparison_rows(tmp_path)
    assert [r["model"] for r in rows] == ["aaa_model", "zzz_model"]


def test_scan_prefers_run_summary_over_xgboost_summary(tmp_path):
    """If both files exist in the same dir, run_summary.json takes precedence."""
    subdir = tmp_path / "model"
    _write_json(subdir / "run_summary.json", _TORCH_SUMMARY)
    _write_json(subdir / "xgboost_run_summary.json", _XGBOOST_SUMMARY)

    rows = scan_artifact_comparison_rows(tmp_path)
    assert len(rows) == 1
    # Pearson from torch summary, not xgboost
    assert abs(rows[0]["seen_pearson"] - 0.604) < 1e-6


def test_scan_skips_subdirs_without_metrics(tmp_path):
    _write_json(tmp_path / "bad_model" / "run_summary.json", {"no_metrics": {}})
    rows = scan_artifact_comparison_rows(tmp_path)
    assert rows == []
