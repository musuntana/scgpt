from __future__ import annotations

from pathlib import Path

from src.utils.interview_script import (
    build_interview_script,
    format_interview_script,
    write_interview_script_text,
)


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text("placeholder", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write_json(root: Path, relative_path: str, payload: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _create_snapshot_backing_files(root: Path) -> None:
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
        "docs/assets/model_comparison_seen_norman2019_demo.png",
        "docs/assets/transformer_inference_preview.png",
        "docs/assets/model_comparison_seen_synthetic_demo.png",
        "docs/assets/transformer_inference_preview_synthetic_demo.png",
    ]:
        _touch(root, relative_path)
    (root / ".python-version").write_text("3.11\n", encoding="utf-8")
    (root / ".venv").mkdir(parents=True, exist_ok=True)
    _write_json(
        root,
        "artifacts/transformer_seen_norman2019_demo/run_summary.json",
        """
{
  "dataset": {"name": "scperturb_norman2019", "cell_context": "K562"},
  "artifacts": {"bundle": {"num_samples": 10500, "num_genes": 512, "num_perturbations": 105}},
  "test_metrics": {
    "seen_test": {"pearson_per_perturbation": 0.6044, "mse_per_perturbation": 0.0071},
    "unseen_test": {
      "pearson_per_perturbation": 0.8243,
      "mse_per_perturbation": 0.0011,
      "topk_deg_overlap_20": 0.9296,
      "topk_deg_overlap_100": 0.9755
    }
  }
}
""".strip(),
    )
    _write_json(
        root,
        "artifacts/mlp_seen_norman2019_demo/run_summary.json",
        """
{
  "test_metrics": {
    "seen_test": {"pearson_per_perturbation": 0.6333, "mse_per_perturbation": 0.0066},
    "unseen_test": {"pearson_per_perturbation": 0.8374, "mse_per_perturbation": 0.00085}
  }
}
""".strip(),
    )
    _write_json(
        root,
        "artifacts/xgboost_seen_norman2019_demo/xgboost_run_summary.json",
        """
{
  "metrics": {
    "seen_test": {"pearson_per_perturbation": 0.6178, "mse_per_perturbation": 0.0066},
    "unseen_test": {"pearson_per_perturbation": 0.8405, "mse_per_perturbation": 0.00084}
  }
}
""".strip(),
    )


def test_build_interview_script_contains_expected_sections(tmp_path: Path) -> None:
    _create_snapshot_backing_files(tmp_path)

    script = build_interview_script(tmp_path)

    assert script["title"] == "PerturbScope-GPT interview script"
    assert script["selected_track"] == "both"
    assert set(script["tracks"]) == {"ai4bio", "ml-engineering"}
    assert len(script["tracks"]["ai4bio"]["resume_bullets"]) == 3
    assert len(script["tracks"]["ml-engineering"]["thirty_second_pitch"]) == 4
    assert "XGBoost" in script["tracks"]["ai4bio"]["two_minute_walkthrough"][4]


def test_build_interview_script_can_select_single_track(tmp_path: Path) -> None:
    _create_snapshot_backing_files(tmp_path)

    script = build_interview_script(tmp_path, track="ml-engineering")

    assert script["selected_track"] == "ml-engineering"
    assert list(script["tracks"]) == ["ml-engineering"]
    assert script["tracks"]["ml-engineering"]["label"] == "ML Engineering"


def test_format_interview_script_mentions_key_metrics_and_sections(tmp_path: Path) -> None:
    _create_snapshot_backing_files(tmp_path)
    script = build_interview_script(tmp_path)

    report = format_interview_script(script)
    assert "Track: AI4Bio" in report
    assert "Track: ML Engineering" in report

    assert "30-second pitch:" in report
    assert "2-minute walkthrough:" in report
    assert "Live demo script:" in report
    assert "0.8405" in report
    assert "0.9755" in report


def test_write_interview_script_text_creates_file(tmp_path: Path) -> None:
    _create_snapshot_backing_files(tmp_path)
    script = build_interview_script(tmp_path)
    output_path = tmp_path / "artifacts/interview_script.txt"

    destination = write_interview_script_text(script, output_path)

    assert destination == output_path
    assert output_path.exists()
    assert "PerturbScope-GPT interview script" in output_path.read_text(encoding="utf-8")
