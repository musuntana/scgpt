from __future__ import annotations

from pathlib import Path

from src.utils.project_health import collect_project_health, format_health_report


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if relative_path.endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
    elif path.suffix:
        path.write_text("placeholder", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _create_repository_scaffold(root: Path) -> None:
    _touch(root, "README.md")
    _touch(root, "PROJECT_PLAN.md")
    _touch(root, "docs/architecture.md")
    _touch(root, "CHANGELOG.md")
    _touch(root, "pyproject.toml")
    _touch(root, "uv.lock")
    (root / ".python-version").write_text("3.11\n", encoding="utf-8")
    _touch(root, ".venv")
    _touch(root, ".github/workflows/ci.yml")
    _touch(root, "Makefile")


def test_collect_project_health_reports_repository_and_notebooks_ready(
    tmp_path: Path,
) -> None:
    _create_repository_scaffold(tmp_path)
    _touch(tmp_path, "notebooks/01_data_exploration.ipynb")
    _touch(tmp_path, "notebooks/02_model_comparison.ipynb")

    summary = collect_project_health(tmp_path)

    assert summary["python_version"] == "3.11"
    assert summary["modes"]["bootstrap_ready"] is True
    assert summary["modes"]["notebooks_ready"] is True
    assert summary["modes"]["offline_demo_ready"] is False
    assert summary["modes"]["real_results_ready"] is False


def test_collect_project_health_reports_demo_modes_ready(tmp_path: Path) -> None:
    _create_repository_scaffold(tmp_path)
    _touch(tmp_path, "notebooks/01_data_exploration.ipynb")
    _touch(tmp_path, "notebooks/02_model_comparison.ipynb")

    for relative_path in [
        "data/processed/synthetic_demo_bundle/arrays.npz",
        "data/processed/synthetic_demo_bundle/metadata.json",
        "data/processed/synthetic_demo_bundle/splits.npz",
        "artifacts/transformer_seen_synthetic_demo/best_model.pt",
        "artifacts/mlp_seen_synthetic_demo/mlp_seen_test_metrics.json",
        "artifacts/xgboost_seen_synthetic_demo/xgboost_seen_test_metrics.json",
        "docs/assets/model_comparison_seen_synthetic_demo.png",
        "docs/assets/transformer_inference_preview_synthetic_demo.png",
        "data/processed/norman2019_demo_bundle/arrays.npz",
        "data/processed/norman2019_demo_bundle/metadata.json",
        "artifacts/transformer_seen_norman2019_demo/best_model.pt",
        "artifacts/transformer_seen_norman2019_demo/deg_artifact.csv",
        "artifacts/transformer_seen_norman2019_demo/run_summary.json",
        "docs/assets/model_comparison_seen_norman2019_demo.png",
        "docs/assets/transformer_inference_preview.png",
        "artifacts/mlp_seen_norman2019_demo/run_summary.json",
        "artifacts/xgboost_seen_norman2019_demo/xgboost_run_summary.json",
        "data/raw/NormanWeissman2019_filtered.h5ad",
    ]:
        _touch(tmp_path, relative_path)

    summary = collect_project_health(tmp_path)

    assert summary["modes"]["offline_demo_ready"] is True
    assert summary["modes"]["real_results_ready"] is True
    assert summary["modes"]["model_comparison_ready"] is True
    assert summary["modes"]["raw_data_available"] is True


def test_format_health_report_mentions_modes_and_recommended_order(
    tmp_path: Path,
) -> None:
    _create_repository_scaffold(tmp_path)
    summary = collect_project_health(tmp_path)

    report = format_health_report(summary)

    assert "PerturbScope-GPT doctor" in report
    assert "[OK] bootstrap_ready" in report
    assert "[MISSING] offline_demo_ready" in report
    assert "Recommended order:" in report
