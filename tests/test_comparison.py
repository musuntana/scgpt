"""Tests for src/utils/comparison — model comparison utilities."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

from src.utils.comparison import (
    extract_gears_summary_row,
    extract_summary_row,
    normalize_seeded_label,
    plot_grouped_metric_bars,
    scan_artifact_comparison_rows,
    shorten_model_label,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# extract_summary_row
# ---------------------------------------------------------------------------

_TORCH_SUMMARY = {
    "split": {"train_protocol": "seen"},
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
    "train_split_prefix": "seen",
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
    assert row["base_model_label"] == "transformer_seen_norman2019"
    assert row["model_type"] == "transformer"
    assert abs(row["seen_pearson"] - 0.604) < 1e-6
    assert row["unseen_pearson"] is None
    assert abs(row["seen_top20_deg"] - 0.816) < 1e-6
    assert row["unseen_top100_deg"] is None


def test_extract_summary_row_xgboost_layout():
    row = extract_summary_row(_XGBOOST_SUMMARY, "xgboost_seen_norman2019_demo")
    assert row is not None
    assert row["model"] == "xgboost_seen_norman2019_demo"
    assert row["model_type"] == "xgboost"
    assert abs(row["seen_pearson"] - 0.618) < 1e-6
    assert row["seen_top20_deg"] is None  # not present in xgboost summary





def test_extract_summary_row_unseen_protocol_does_not_report_seen_metrics():
    summary = {
        "split": {"train_protocol": "unseen"},
        "test_metrics": {
            "seen_test": {"pearson_per_perturbation": 0.99},
            "unseen_test": {"pearson_per_perturbation": 0.42},
        },
    }
    row = extract_summary_row(summary, "transformer_unseen_norman2019_demo")

    assert row is not None
    assert row["seen_pearson"] is None
    assert row["unseen_pearson"] == 0.42


def test_extract_summary_row_returns_none_for_empty_summary():
    assert extract_summary_row({}, "empty") is None
    assert extract_summary_row({"other_key": {}}, "no_metrics") is None


def test_extract_summary_row_partial_metrics():
    """Partial metrics dict: missing keys should come back as None, not raise."""
    summary = {
        "split": {"train_protocol": "seen"},
        "test_metrics": {"seen_test": {"pearson_per_perturbation": 0.5}},
    }
    row = extract_summary_row(summary, "partial")
    assert row is not None
    assert abs(row["seen_pearson"] - 0.5) < 1e-6
    assert row["seen_mse"] is None
    assert row["unseen_pearson"] is None


# ---------------------------------------------------------------------------
# scan_artifact_comparison_rows
# ---------------------------------------------------------------------------


def _gears_manifest(seed: int = 42, *, input_hash: str = "b" * 64) -> dict:
    return {
        "status": "TRAINED_DIAGNOSTIC_DELTA_ALIGNED_NOT_MULTI_SEED_VALIDATED",
        "seed": seed,
        "source_h5ad_sha256": "a" * 64,
        "input_h5ad_sha256": input_hash,
        "preflight_manifest_sha256": "c" * 64,
        "preflight_split_pickle_sha256": "d" * 64,
        "validated_split_pickle_sha256": "e" * 64,
        "split_json_sha256": "f" * 64,
        "bundle_artifacts_sha256": {
            "arrays.npz": "1" * 64,
            "splits.npz": "2" * 64,
            "metadata.json": "3" * 64,
        },
        "model_config": {"epochs": 1, "batch_size": 16, "hidden_size": 64},
        "device": "cpu",
        "runtime_versions": {
            "python": "3.11",
            "torch": "2.5",
            "torch-geometric": "2.5",
            "cell-gears": "0.1",
        },
        "gene_order": {
            "bundle_gene_ids_sha256": "4" * 64,
            "input_gene_symbols_sha256": "5" * 64,
            "gears_gene_symbols_sha256": "6" * 64,
        },
        "graph_provenance": {
            "coexpression_graph_sha256": "7" * 64,
            "training_conditions_sha256": "8" * 64,
        },
        "delta_metrics": {
            "metric_space": "control_relative_delta_expression",
            "prediction_conversion": "matched_control_delta",
            "target_source": "bundle.target_delta",
            "pearson_per_perturbation": 0.8 + seed / 10000,
            "mse_per_perturbation": 0.01 + seed / 100000,
        },
        "target_alignment": {"real_data_target_equivalence_verified": False},
        "target_equivalence_verified": False,
        "comparison_eligible": False,
    }


def test_extract_gears_summary_row_uses_delta_metrics_and_stable_protocol_id():
    first = extract_gears_summary_row(_gears_manifest(42), "gears_runs/seed42")
    second = extract_gears_summary_row(_gears_manifest(43), "gears_runs/seed43")

    assert first is not None
    assert second is not None
    assert first["model_type"] == "gears"
    assert first["train_protocol"] == "unseen"
    assert first["unseen_pearson"] == pytest.approx(0.8042)
    assert first["unseen_mse"] == pytest.approx(0.01042)
    assert first["seen_pearson"] is None
    assert first["base_model_label"] == second["base_model_label"]
    assert first["comparison_eligible"] is False
    assert first["target_equivalence_verified"] is False


def test_extract_gears_summary_row_separates_different_inputs():
    first = extract_gears_summary_row(_gears_manifest(42), "gears_runs/seed42")
    second = extract_gears_summary_row(
        _gears_manifest(43, input_hash="9" * 64),
        "gears_runs/seed43",
    )

    assert first is not None
    assert second is not None
    assert first["base_model_label"] != second["base_model_label"]


@pytest.mark.parametrize(
    "bundle_hashes",
    [
        {},
        {"arrays.npz": "not-a-sha256"},
        {"": "1" * 64},
    ],
)
def test_extract_gears_summary_row_rejects_invalid_bundle_artifact_hashes(bundle_hashes):
    manifest = _gears_manifest()
    manifest["bundle_artifacts_sha256"] = bundle_hashes

    assert extract_gears_summary_row(manifest, "gears_runs/seed42") is None


@pytest.mark.parametrize(
    "bundle_hashes",
    [
        {"arrays.npz": "1" * 64, "splits.npz": "2" * 64},
        {"arrays.npz": "1" * 64, "metadata.json": "3" * 64},
        {"splits.npz": "2" * 64, "metadata.json": "3" * 64},
        {"unrelated.json": "4" * 64},
        {
            "arrays.npz": "1" * 64,
            "splits.npz": "2" * 64,
            "metadata.json": "3" * 64,
            "unrelated.json": "4" * 64,
        },
    ],
)
def test_extract_gears_summary_row_requires_all_bundle_artifacts(bundle_hashes):
    manifest = _gears_manifest()
    manifest["bundle_artifacts_sha256"] = bundle_hashes

    assert extract_gears_summary_row(manifest, "gears_runs/seed42") is None


@pytest.mark.parametrize(
    "hash_key",
    [
        "bundle_gene_ids_sha256",
        "input_gene_symbols_sha256",
        "gears_gene_symbols_sha256",
    ],
)
def test_extract_gears_summary_row_rejects_missing_gene_order_hashes(hash_key):
    manifest = _gears_manifest()
    manifest["gene_order"].pop(hash_key)

    assert extract_gears_summary_row(manifest, "gears_runs/seed42") is None


@pytest.mark.parametrize(
    "hash_key",
    [
        "bundle_gene_ids_sha256",
        "input_gene_symbols_sha256",
        "gears_gene_symbols_sha256",
    ],
)
def test_extract_gears_summary_row_rejects_invalid_gene_order_hashes(hash_key):
    manifest = _gears_manifest()
    manifest["gene_order"][hash_key] = "not-a-sha256"

    assert extract_gears_summary_row(manifest, "gears_runs/seed42") is None


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



def test_scan_discovers_nested_gears_run_manifests(tmp_path):
    for seed in (42, 43, 44):
        _write_json(
            tmp_path / "gears_runs" / f"seed{seed}" / "run_manifest.json",
            _gears_manifest(seed),
        )

    rows = scan_artifact_comparison_rows(tmp_path)

    assert len(rows) == 3
    assert {row["model_type"] for row in rows} == {"gears"}
    assert {row["seed"] for row in rows} == {42, 43, 44}
    assert len({row["base_model_label"] for row in rows}) == 1


def test_shorten_model_label_removes_demo_suffixes():
    assert shorten_model_label("transformer_seen_norman2019_demo") == "transformer"
    assert shorten_model_label("transformer_unseen_norman2019_demo") == "transformer"
    assert shorten_model_label("mlp_seen_synthetic_demo") == "mlp"
    assert shorten_model_label("mlp_unseen_synthetic_demo") == "mlp"
    assert shorten_model_label("custom_model") == "custom_model"


def test_normalize_seeded_label_strips_seed_suffixes():
    assert normalize_seeded_label("transformer_seen_norman2019_seed7") == "transformer_seen_norman2019"
    assert normalize_seeded_label("mlp_seen_norman2019-s42") == "mlp_seen_norman2019"
    assert normalize_seeded_label("transformer_seen_norman2019_demo") == "transformer_seen_norman2019"


def test_plot_grouped_metric_bars_adds_headroom_for_high_values():
    figure, axis = plt.subplots(figsize=(8, 4.8))
    seen_values = [0.999, 0.998, 0.999]
    unseen_values = [0.999, 0.999, 1.0]

    plot_grouped_metric_bars(
        axis,
        ["Transformer", "MLP", "XGBoost"],
        seen_values,
        unseen_values,
        ylabel="Per-perturbation Pearson",
        title="Synthetic Demo Bundle: Model Comparison",
        annotate=True,
    )

    y_min, y_max = axis.get_ylim()
    assert y_min == 0.0
    assert y_max > max([*seen_values, *unseen_values])
    assert len(axis.texts) == 6
    for text in axis.texts:
        assert text.get_position()[1] < y_max

    plt.close(figure)
