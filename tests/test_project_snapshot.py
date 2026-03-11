from __future__ import annotations

import json
from pathlib import Path

from src.utils.project_snapshot import (
    build_project_snapshot,
    format_project_snapshot,
    write_project_snapshot,
)


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text("placeholder", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write_json(root: Path, relative_path: str, payload: dict) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _create_repo_scaffold(root: Path) -> None:
    for relative_path in [
        "README.md",
        "PROJECT_PLAN.md",
        "docs/architecture.md",
        "CHANGELOG.md",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/ci.yml",
        "Makefile",
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_model_comparison.ipynb",
    ]:
        _touch(root, relative_path)
    (root / ".python-version").write_text("3.11\n", encoding="utf-8")
    (root / ".venv").mkdir(parents=True, exist_ok=True)


def _transformer_summary(unseen_pearson: float, top100: float) -> dict:
    return {
        "dataset": {
            "name": "scperturb_norman2019",
            "cell_context": "K562",
        },
        "artifacts": {
            "bundle": {
                "num_samples": 10500,
                "num_genes": 512,
                "num_perturbations": 105,
            }
        },
        "test_metrics": {
            "seen_test": {
                "pearson_per_perturbation": 0.60,
                "mse_per_perturbation": 0.0071,
            },
            "unseen_test": {
                "pearson_per_perturbation": unseen_pearson,
                "mse_per_perturbation": 0.0011,
                "topk_deg_overlap_20": 0.93,
                "topk_deg_overlap_100": top100,
            },
        },
    }


def _generic_summary(unseen_pearson: float, unseen_mse: float) -> dict:
    return {
        "test_metrics": {
            "seen_test": {
                "pearson_per_perturbation": 0.62,
                "mse_per_perturbation": 0.0066,
            },
            "unseen_test": {
                "pearson_per_perturbation": unseen_pearson,
                "mse_per_perturbation": unseen_mse,
            },
        }
    }


def _xgboost_summary(unseen_pearson: float, unseen_mse: float) -> dict:
    return {
        "metrics": {
            "seen_test": {
                "pearson_per_perturbation": 0.61,
                "mse_per_perturbation": 0.0065,
            },
            "unseen_test": {
                "pearson_per_perturbation": unseen_pearson,
                "mse_per_perturbation": unseen_mse,
            },
        }
    }


def test_build_project_snapshot_selects_best_real_model_and_deg_story(
    tmp_path: Path,
) -> None:
    _create_repo_scaffold(tmp_path)
    _write_json(
        tmp_path,
        "artifacts/transformer_seen_norman2019_demo/run_summary.json",
        _transformer_summary(0.8243, 0.9755),
    )
    _write_json(
        tmp_path,
        "artifacts/mlp_seen_norman2019_demo/run_summary.json",
        _generic_summary(0.8374, 0.00085),
    )
    _write_json(
        tmp_path,
        "artifacts/xgboost_seen_norman2019_demo/xgboost_run_summary.json",
        _xgboost_summary(0.8405, 0.00084),
    )

    snapshot = build_project_snapshot(tmp_path)

    assert snapshot["headline"]["best_real_unseen_model"] == "XGBoost"
    assert snapshot["headline"]["best_real_unseen_pearson"] == 0.8405
    assert snapshot["headline"]["transformer_unseen_top100_deg_overlap"] == 0.9755
    assert snapshot["headline"]["all_real_models_unseen_pearson_ge_0_82"] is True
    assert len(snapshot["real_model_rows"]) == 3


def test_build_project_snapshot_separates_real_and_synthetic_rows(
    tmp_path: Path,
) -> None:
    _create_repo_scaffold(tmp_path)
    _write_json(
        tmp_path,
        "artifacts/transformer_seen_norman2019_demo/run_summary.json",
        _transformer_summary(0.8243, 0.9755),
    )
    _write_json(
        tmp_path,
        "artifacts/transformer_seen_synthetic_demo/run_summary.json",
        _transformer_summary(0.9989, 0.9990),
    )
    _write_json(
        tmp_path,
        "artifacts/mlp_seen_synthetic_demo/run_summary.json",
        _generic_summary(0.9995, 0.0036),
    )
    _write_json(
        tmp_path,
        "artifacts/xgboost_seen_synthetic_demo/xgboost_run_summary.json",
        _xgboost_summary(0.9999, 0.0014),
    )

    snapshot = build_project_snapshot(tmp_path)

    assert len(snapshot["real_model_rows"]) == 1
    assert len(snapshot["synthetic_model_rows"]) == 3
    assert snapshot["synthetic_model_rows"][0]["display_name"] == "XGBoost"


def test_format_and_write_project_snapshot(tmp_path: Path) -> None:
    _create_repo_scaffold(tmp_path)
    _write_json(
        tmp_path,
        "artifacts/transformer_seen_norman2019_demo/run_summary.json",
        _transformer_summary(0.8243, 0.9755),
    )
    snapshot = build_project_snapshot(tmp_path)

    report = format_project_snapshot(snapshot)
    output_path = tmp_path / "artifacts/project_snapshot.json"
    destination = write_project_snapshot(snapshot, output_path)

    assert "PerturbScope-GPT snapshot" in report
    assert "Best real unseen Pearson" in report
    assert "make snapshot" in report
    assert destination == output_path
    assert output_path.exists()
