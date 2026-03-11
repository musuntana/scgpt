from __future__ import annotations

import json

from src.utils.multiseed import (
    build_multiseed_report,
    format_multiseed_report,
    load_multiseed_report,
    select_multiseed_group,
)


def test_build_multiseed_report_aggregates_mean_and_std():
    rows = [
        {
            "model": "transformer_seen_norman2019_demo",
            "base_model_label": "transformer_seen_norman2019",
            "model_type": "transformer",
            "dataset_name": "scperturb_norman2019",
            "train_protocol": "seen",
            "seed": 42,
            "seen_pearson": 0.61,
            "seen_mse": 0.0069,
            "unseen_pearson": 0.83,
            "unseen_mse": 0.00105,
            "seen_top20_deg": None,
            "seen_top100_deg": None,
            "unseen_top20_deg": 0.94,
            "unseen_top100_deg": 0.975,
        },
        {
            "model": "transformer_seen_norman2019_seed7",
            "base_model_label": "transformer_seen_norman2019",
            "model_type": "transformer",
            "dataset_name": "scperturb_norman2019",
            "train_protocol": "seen",
            "seed": 7,
            "seen_pearson": 0.60,
            "seen_mse": 0.0070,
            "unseen_pearson": 0.82,
            "unseen_mse": 0.0011,
            "seen_top20_deg": None,
            "seen_top100_deg": None,
            "unseen_top20_deg": 0.93,
            "unseen_top100_deg": 0.97,
        },
        {
            "model": "transformer_seen_norman2019_seed21",
            "base_model_label": "transformer_seen_norman2019",
            "model_type": "transformer",
            "dataset_name": "scperturb_norman2019",
            "train_protocol": "seen",
            "seed": 21,
            "seen_pearson": 0.62,
            "seen_mse": 0.0068,
            "unseen_pearson": 0.84,
            "unseen_mse": 0.0010,
            "seen_top20_deg": None,
            "seen_top100_deg": None,
            "unseen_top20_deg": 0.95,
            "unseen_top100_deg": 0.98,
        },
        {
            "model": "transformer_seen_synthetic_demo",
            "base_model_label": "transformer_seen_synthetic",
            "model_type": "transformer",
            "dataset_name": "synthetic_demo",
            "train_protocol": "seen",
            "seed": 3,
            "seen_pearson": 0.99,
            "seen_mse": 0.0001,
            "unseen_pearson": 0.99,
            "unseen_mse": 0.0001,
            "seen_top20_deg": None,
            "seen_top100_deg": None,
            "unseen_top20_deg": 0.99,
            "unseen_top100_deg": 0.99,
        },
        {
            "model": "transformer_seen_synthetic_seed9",
            "base_model_label": "transformer_seen_synthetic",
            "model_type": "transformer",
            "dataset_name": "synthetic_demo",
            "train_protocol": "seen",
            "seed": 9,
            "seen_pearson": 0.98,
            "seen_mse": 0.0002,
            "unseen_pearson": 0.98,
            "unseen_mse": 0.0002,
            "seen_top20_deg": None,
            "seen_top100_deg": None,
            "unseen_top20_deg": 0.98,
            "unseen_top100_deg": 0.98,
        },
    ]

    report = build_multiseed_report(rows)

    assert len(report) == 2

    real_report = next(row for row in report if row["dataset_name"] == "scperturb_norman2019")
    assert real_report["num_runs"] == 3
    assert real_report["seeds"] == [7, 21, 42]
    assert abs(real_report["unseen_pearson_mean"] - 0.83) < 1e-6
    assert real_report["unseen_pearson_std"] > 0.0

    synthetic_report = next(row for row in report if row["dataset_name"] == "synthetic_demo")
    assert synthetic_report["num_runs"] == 2
    assert synthetic_report["seeds"] == [3, 9]


def test_format_multiseed_report_handles_empty_groups():
    report = format_multiseed_report([], artifact_root="artifacts")
    assert "No groups with repeated runs were found." in report


def test_select_multiseed_group_filters_by_dataset_and_model() -> None:
    report_rows = build_multiseed_report(
        [
            {
                "model": "transformer_seen_norman2019_demo",
                "base_model_label": "transformer_seen_norman2019",
                "model_type": "transformer",
                "dataset_name": "scperturb_norman2019",
                "train_protocol": "seen",
                "seed": 42,
                "seen_pearson": 0.61,
                "seen_mse": 0.0069,
                "unseen_pearson": 0.83,
                "unseen_mse": 0.00105,
                "seen_top20_deg": None,
                "seen_top100_deg": None,
                "unseen_top20_deg": 0.94,
                "unseen_top100_deg": 0.975,
            },
            {
                "model": "transformer_seen_norman2019_seed7",
                "base_model_label": "transformer_seen_norman2019",
                "model_type": "transformer",
                "dataset_name": "scperturb_norman2019",
                "train_protocol": "seen",
                "seed": 7,
                "seen_pearson": 0.60,
                "seen_mse": 0.0070,
                "unseen_pearson": 0.82,
                "unseen_mse": 0.0011,
                "seen_top20_deg": None,
                "seen_top100_deg": None,
                "unseen_top20_deg": 0.93,
                "unseen_top100_deg": 0.97,
            },
            {
                "model": "transformer_seen_synthetic_demo",
                "base_model_label": "transformer_seen_synthetic",
                "model_type": "transformer",
                "dataset_name": "synthetic_demo",
                "train_protocol": "seen",
                "seed": 3,
                "seen_pearson": 0.99,
                "seen_mse": 0.0001,
                "unseen_pearson": 0.99,
                "unseen_mse": 0.0001,
                "seen_top20_deg": None,
                "seen_top100_deg": None,
                "unseen_top20_deg": 0.99,
                "unseen_top100_deg": 0.99,
            },
            {
                "model": "transformer_seen_synthetic_seed9",
                "base_model_label": "transformer_seen_synthetic",
                "model_type": "transformer",
                "dataset_name": "synthetic_demo",
                "train_protocol": "seen",
                "seed": 9,
                "seen_pearson": 0.98,
                "seen_mse": 0.0002,
                "unseen_pearson": 0.98,
                "unseen_mse": 0.0002,
                "seen_top20_deg": None,
                "seen_top100_deg": None,
                "unseen_top20_deg": 0.98,
                "unseen_top100_deg": 0.98,
            },
        ]
    )

    real_group = select_multiseed_group(
        report_rows,
        dataset_name="scperturb_norman2019",
        train_protocol="seen",
        model_type="transformer",
    )
    synthetic_group = select_multiseed_group(
        report_rows,
        dataset_name="synthetic_demo",
        train_protocol="seen",
        model_type="transformer",
    )

    assert real_group["seeds"] == [7, 42]
    assert synthetic_group["seeds"] == [3, 9]


def test_load_multiseed_report_returns_only_dict_rows(tmp_path) -> None:
    report_path = tmp_path / "multi_seed_report.json"
    report_path.write_text(
        json.dumps(
            [
                {"dataset_name": "scperturb_norman2019", "model_type": "transformer"},
                ["unexpected"],
                "unexpected",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_multiseed_report(report_path)

    assert rows == [{"dataset_name": "scperturb_norman2019", "model_type": "transformer"}]
