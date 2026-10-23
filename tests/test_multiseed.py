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
            "model": "transformer_unseen_norman2019_demo",
            "base_model_label": "transformer_unseen_norman2019",
            "model_type": "transformer",
            "dataset_name": "scperturb_norman2019",
            "train_protocol": "unseen",
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
            "model": "transformer_unseen_norman2019_seed7",
            "base_model_label": "transformer_unseen_norman2019",
            "model_type": "transformer",
            "dataset_name": "scperturb_norman2019",
            "train_protocol": "unseen",
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
            "model": "transformer_unseen_norman2019_seed21",
            "base_model_label": "transformer_unseen_norman2019",
            "model_type": "transformer",
            "dataset_name": "scperturb_norman2019",
            "train_protocol": "unseen",
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
    assert abs(synthetic_report["seen_pearson_mean"] - 0.985) < 1e-6
    assert "unseen_pearson_mean" not in synthetic_report


def _gears_row(seed: int, *, model: str | None = None) -> dict:
    return {
        "model": model or f"gears_runs/seed{seed}",
        "base_model_label": "gears_unseen_same-protocol",
        "model_type": "gears",
        "dataset_name": "source-aaaaaaaaaaaa",
        "train_protocol": "unseen",
        "seed": seed,
        "seen_pearson": None,
        "seen_mse": None,
        "unseen_pearson": 0.80 + seed / 10000,
        "unseen_mse": 0.01 + seed / 100000,
        "seen_top20_deg": None,
        "seen_top100_deg": None,
        "unseen_top20_deg": None,
        "unseen_top100_deg": None,
        "comparison_protocol_id": "same-protocol",
        "comparison_eligible": False,
        "target_equivalence_verified": False,
        "coexpression_graph_sha256": "7" * 64,
    }


def test_gears_multiseed_report_keeps_scientific_gates_explicit():
    rows = [_gears_row(seed) for seed in (42, 43, 44)]

    report = build_multiseed_report(rows, min_runs=2)

    assert len(report) == 1
    row = report[0]
    assert row["model_type"] == "gears"
    assert row["num_runs"] == 3
    assert row["seeds"] == [42, 43, 44]
    assert row["minimum_seed_gate_passed"] is True
    assert row["coexpression_graph_consistent"] is True
    assert row["target_equivalence_verified"] is False
    assert row["comparison_eligible"] is False
    assert "real-data target equivalence has not been verified" in row["comparison_blockers"]
    text_report = format_multiseed_report(report, artifact_root="artifacts")
    assert "diagnostic-only: comparison_eligible=False" in text_report


def test_gears_multiseed_report_requires_three_distinct_seeds():
    report = build_multiseed_report([_gears_row(42), _gears_row(43)], min_runs=2)

    assert len(report) == 1
    assert report[0]["minimum_seed_gate_passed"] is False
    assert report[0]["comparison_eligible"] is False
    assert "at least three distinct random seeds are required" in report[0]["comparison_blockers"]


def test_gears_multiseed_report_blocks_comparison_with_duplicate_seeds():
    rows = [
        _gears_row(42, model="gears_runs/a_seed42"),
        _gears_row(42, model="gears_runs/b_seed42"),
        _gears_row(43),
        _gears_row(44),
    ]
    for row in rows:
        row["comparison_eligible"] = True
        row["target_equivalence_verified"] = True

    report = build_multiseed_report(rows, min_runs=2)

    assert len(report) == 1
    row = report[0]
    assert row["num_runs"] == 3
    assert row["seeds"] == [42, 43, 44]
    assert row["duplicate_seed_runs_ignored"] == 1
    assert row["duplicate_seeds"] == [42]
    assert row["duplicate_seed_artifacts_ignored"] == ["gears_runs/b_seed42"]
    assert row["duplicate_seed_gate_passed"] is False
    assert row["minimum_seed_gate_passed"] is True
    assert row["comparison_eligible"] is False
    assert "duplicate-seed artifacts detected; resolve duplicates before comparison" in row["comparison_blockers"]


def test_gears_multiseed_report_includes_duplicate_artifact_graph_hashes():
    rows = [
        _gears_row(42, model="gears_runs/a_seed42"),
        _gears_row(42, model="gears_runs/b_seed42"),
        _gears_row(43),
        _gears_row(44),
    ]
    for row in rows:
        row["comparison_eligible"] = True
        row["target_equivalence_verified"] = True
    rows[1]["coexpression_graph_sha256"] = "8" * 64

    report = build_multiseed_report(rows, min_runs=2)

    assert len(report) == 1
    row = report[0]
    assert row["coexpression_graph_hashes"] == sorted(["7" * 64, "8" * 64])
    assert row["coexpression_graph_consistent"] is False
    assert row["comparison_eligible"] is False
    assert "co-expression graph hash is missing or differs across runs" in row["comparison_blockers"]


def test_gears_multiseed_report_requires_graph_hash_for_every_artifact():
    rows = [_gears_row(seed) for seed in (42, 43, 44)]
    for row in rows:
        row["comparison_eligible"] = True
        row["target_equivalence_verified"] = True
    rows[1]["coexpression_graph_sha256"] = None

    report = build_multiseed_report(rows, min_runs=2)

    assert len(report) == 1
    row = report[0]
    assert row["coexpression_graph_hashes"] == ["7" * 64]
    assert row["coexpression_graph_consistent"] is False
    assert row["comparison_eligible"] is False
    assert "co-expression graph hash is missing or differs across runs" in row["comparison_blockers"]


def test_format_multiseed_report_handles_empty_groups():
    report = format_multiseed_report([], artifact_root="artifacts")
    assert "No groups with repeated runs were found." in report


def test_select_multiseed_group_filters_by_dataset_and_model() -> None:
    report_rows = build_multiseed_report(
        [
            {
                "model": "transformer_unseen_norman2019_demo",
                "base_model_label": "transformer_unseen_norman2019",
                "model_type": "transformer",
                "dataset_name": "scperturb_norman2019",
                "train_protocol": "unseen",
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
                "model": "transformer_unseen_norman2019_seed7",
                "base_model_label": "transformer_unseen_norman2019",
                "model_type": "transformer",
                "dataset_name": "scperturb_norman2019",
                "train_protocol": "unseen",
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
        train_protocol="unseen",
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
