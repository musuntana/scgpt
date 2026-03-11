from __future__ import annotations

from src.utils.showcase import build_showcase_plan, format_showcase_report


def test_build_showcase_plan_prefers_real_results_and_skips_regen() -> None:
    plan = build_showcase_plan(
        "/tmp/project",
        {
            "real_results_ready": True,
            "offline_demo_ready": True,
        },
        snapshot_output_path="artifacts/project_snapshot.json",
    )

    assert plan["demo_mode"] == "real_norman2019"
    assert plan["prefer_real_results"] is True
    assert plan["prepare_synthetic_showcase"] is False
    assert plan["snapshot_output_path"].endswith("artifacts/project_snapshot.json")


def test_build_showcase_plan_requests_synthetic_regen_when_missing() -> None:
    plan = build_showcase_plan(
        "/tmp/project",
        {
            "real_results_ready": False,
            "offline_demo_ready": False,
        },
        force_refresh_synthetic=False,
        launch_app=True,
    )

    assert plan["demo_mode"] == "synthetic_fallback"
    assert plan["prepare_synthetic_showcase"] is True
    assert plan["launch_app"] is True


def test_format_showcase_report_includes_talk_track_and_commands() -> None:
    snapshot = {
        "headline": {
            "best_real_unseen_model": "XGBoost",
            "best_real_unseen_pearson": 0.8405,
            "transformer_unseen_pearson": 0.8243,
            "transformer_unseen_top100_deg_overlap": 0.9755,
            "transformer_multiseed_num_runs": 3,
            "transformer_multiseed_unseen_pearson_mean": 0.8304,
            "transformer_multiseed_unseen_pearson_std": 0.0067,
            "transformer_multiseed_unseen_top100_deg_mean": 0.9850,
            "transformer_multiseed_unseen_top100_deg_std": 0.0067,
        },
        "assets": {
            "real_comparison_figure": {
                "path": "docs/assets/model_comparison_seen_norman2019_demo.png",
                "exists": True,
            },
            "real_inference_figure": {
                "path": "docs/assets/transformer_inference_preview.png",
                "exists": True,
            },
        },
    }
    plan = {
        "demo_mode": "real_norman2019",
        "snapshot_output_path": "/tmp/project/artifacts/project_snapshot.json",
    }
    actions_taken = {
        "generated_synthetic_showcase": False,
        "snapshot_written": True,
        "launch_app": False,
    }

    report = format_showcase_report(snapshot, plan, actions_taken)

    assert "PerturbScope-GPT showcase" in report
    assert "Best unseen Pearson = XGBoost (0.8405)".lower() in report.lower()
    assert "Anchor stability across 3 real Transformer seeds" in report
    assert "Streamlit" in report
    assert "./scripts/run_showcase.sh --launch-app" in report
