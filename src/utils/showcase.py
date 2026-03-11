from __future__ import annotations

from pathlib import Path
from typing import Any


def build_showcase_plan(
    project_root: str | Path,
    health_modes: dict[str, bool],
    *,
    launch_app: bool = False,
    force_refresh_synthetic: bool = False,
    snapshot_output_path: str | Path = "artifacts/project_snapshot.json",
) -> dict[str, Any]:
    """Build a plan for preparing and presenting the live demo showcase."""
    root = Path(project_root).resolve()
    output_path = Path(snapshot_output_path)
    if not output_path.is_absolute():
        output_path = root / output_path

    real_ready = bool(health_modes.get("real_results_ready"))
    synthetic_ready = bool(health_modes.get("offline_demo_ready"))

    return {
        "demo_mode": "real_norman2019" if real_ready else "synthetic_fallback",
        "prefer_real_results": real_ready,
        "prepare_synthetic_showcase": force_refresh_synthetic or not synthetic_ready,
        "launch_app": launch_app,
        "snapshot_output_path": str(output_path),
    }


def format_showcase_report(
    snapshot: dict[str, Any],
    plan: dict[str, Any],
    actions_taken: dict[str, bool],
) -> str:
    """Render a concise live-demo checklist for local use."""
    headline = snapshot["headline"]
    assets = snapshot["assets"]
    talk_track = [
        (
            f"Lead with best unseen Pearson = "
            f"{headline.get('best_real_unseen_model') or 'n/a'} "
            f"({_format_metric(headline.get('best_real_unseen_pearson'))})"
        ),
        (
            f"Highlight Transformer unseen Pearson / top-100 DEG overlap = "
            f"{_format_metric(headline.get('transformer_unseen_pearson'))} / "
            f"{_format_metric(headline.get('transformer_unseen_top100_deg_overlap'))}"
        ),
    ]
    if headline.get("transformer_multiseed_num_runs") is not None:
        talk_track.append(
            f"Anchor stability across {headline.get('transformer_multiseed_num_runs')} "
            f"real Transformer seeds: unseen Pearson = "
            f"{_format_mean_std(headline.get('transformer_multiseed_unseen_pearson_mean'), headline.get('transformer_multiseed_unseen_pearson_std'))}, "
            f"top-100 DEG overlap = "
            f"{_format_mean_std(headline.get('transformer_multiseed_unseen_top100_deg_mean'), headline.get('transformer_multiseed_unseen_top100_deg_std'))}"
        )
    talk_track.extend(
        [
            f"Show real comparison figure: {assets['real_comparison_figure']['path']}",
            f"Show inference preview: {assets['real_inference_figure']['path']}",
            "Open Streamlit and walk through one perturbation end-to-end",
        ]
    )

    lines = [
        "PerturbScope-GPT showcase",
        f"Demo mode: {plan['demo_mode']}",
        "",
        "Actions taken:",
        (
            "  "
            f"[{'DONE' if actions_taken.get('generated_synthetic_showcase') else 'SKIP'}] "
            "synthetic showcase generation"
        ),
        (
            "  "
            f"[{'DONE' if actions_taken.get('snapshot_written') else 'SKIP'}] "
            f"snapshot written to {plan['snapshot_output_path']}"
        ),
        (
            "  "
            f"[{'DONE' if actions_taken.get('launch_app') else 'SKIP'}] "
            "launch Streamlit app"
        ),
        "",
        "Talk track:",
    ]
    lines.extend([f"  {index}. {item}" for index, item in enumerate(talk_track, start=1)])
    lines.extend(
        [
        "",
        "Recommended commands:",
        "  make doctor",
        "  make snapshot",
        "  ./scripts/run_app.sh",
        "  ./scripts/run_showcase.sh --launch-app",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_mean_std(mean: Any, std: Any) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return _format_metric(mean)
    return f"{float(mean):.4f} +/- {float(std):.4f}"
