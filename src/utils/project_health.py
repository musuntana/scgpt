from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HealthCheck:
    """A single project-health check."""

    name: str
    path: str
    ok: bool
    required: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the check into a JSON-serializable dictionary."""
        return asdict(self)


def _make_check(
    *,
    project_root: Path,
    name: str,
    relative_path: str,
    message: str,
    required: bool = True,
) -> HealthCheck:
    path = project_root / relative_path
    return HealthCheck(
        name=name,
        path=relative_path,
        ok=path.exists(),
        required=required,
        message=message,
    )


def _all_required_ok(checks: list[HealthCheck]) -> bool:
    return all(check.ok for check in checks if check.required)


def collect_project_health(project_root: str | Path = ".") -> dict[str, Any]:
    """Collect local project-health signals for onboarding and demo readiness."""
    root = Path(project_root).resolve()
    python_version_path = root / ".python-version"
    python_version = (
        python_version_path.read_text(encoding="utf-8").strip()
        if python_version_path.exists()
        else None
    )

    groups: dict[str, list[HealthCheck]] = {
        "repository": [
            _make_check(
                project_root=root,
                name="readme",
                relative_path="README.md",
                message="Repository overview and quick-start.",
            ),
            _make_check(
                project_root=root,
                name="project_plan",
                relative_path="PROJECT_PLAN.md",
                message="Project scope and milestone plan.",
            ),
            _make_check(
                project_root=root,
                name="architecture_doc",
                relative_path="docs/architecture.md",
                message="Architecture and data-flow reference.",
            ),
            _make_check(
                project_root=root,
                name="changelog",
                relative_path="CHANGELOG.md",
                message="Milestone and release history.",
            ),
            _make_check(
                project_root=root,
                name="pyproject",
                relative_path="pyproject.toml",
                message="Dependency and project metadata source of truth.",
            ),
            _make_check(
                project_root=root,
                name="uv_lock",
                relative_path="uv.lock",
                message="Locked dependency snapshot for reproducibility.",
            ),
            _make_check(
                project_root=root,
                name="python_version_pin",
                relative_path=".python-version",
                message="Pinned local Python version.",
            ),
            _make_check(
                project_root=root,
                name="virtualenv",
                relative_path=".venv",
                message="Local virtual environment created by bootstrap script.",
            ),
            _make_check(
                project_root=root,
                name="ci_workflow",
                relative_path=".github/workflows/ci.yml",
                message="GitHub Actions workflow for tests, lint, and type checks.",
            ),
            _make_check(
                project_root=root,
                name="makefile",
                relative_path="Makefile",
                message="Common local developer shortcuts.",
            ),
        ],
        "notebooks": [
            _make_check(
                project_root=root,
                name="eda_notebook",
                relative_path="notebooks/01_data_exploration.ipynb",
                message="Exploratory data analysis notebook.",
            ),
            _make_check(
                project_root=root,
                name="comparison_notebook",
                relative_path="notebooks/02_model_comparison.ipynb",
                message="Model-comparison notebook.",
            ),
        ],
        "synthetic_demo": [
            _make_check(
                project_root=root,
                name="synthetic_bundle_arrays",
                relative_path="data/processed/synthetic_demo_bundle/arrays.npz",
                message="Synthetic bundle arrays.",
            ),
            _make_check(
                project_root=root,
                name="synthetic_bundle_metadata",
                relative_path="data/processed/synthetic_demo_bundle/metadata.json",
                message="Synthetic bundle metadata.",
            ),
            _make_check(
                project_root=root,
                name="synthetic_bundle_splits",
                relative_path="data/processed/synthetic_demo_bundle/splits.npz",
                message="Synthetic bundle split indices.",
            ),
            _make_check(
                project_root=root,
                name="synthetic_transformer_checkpoint",
                relative_path="artifacts/transformer_seen_synthetic_demo/best_model.pt",
                message="Synthetic Transformer checkpoint.",
            ),
            _make_check(
                project_root=root,
                name="synthetic_mlp_metrics",
                relative_path="artifacts/mlp_seen_synthetic_demo/mlp_seen_test_metrics.json",
                message="Synthetic MLP metrics.",
            ),
            _make_check(
                project_root=root,
                name="synthetic_xgboost_metrics",
                relative_path="artifacts/xgboost_seen_synthetic_demo/xgboost_seen_test_metrics.json",
                message="Synthetic XGBoost metrics.",
            ),
            _make_check(
                project_root=root,
                name="synthetic_comparison_figure",
                relative_path="docs/assets/model_comparison_seen_synthetic_demo.png",
                message="Synthetic model-comparison figure.",
            ),
            _make_check(
                project_root=root,
                name="synthetic_inference_figure",
                relative_path="docs/assets/transformer_inference_preview_synthetic_demo.png",
                message="Synthetic app preview figure.",
            ),
        ],
        "real_results": [
            _make_check(
                project_root=root,
                name="real_bundle_arrays",
                relative_path="data/processed/norman2019_demo_bundle/arrays.npz",
                message="Real Norman2019 bundle arrays.",
            ),
            _make_check(
                project_root=root,
                name="real_bundle_metadata",
                relative_path="data/processed/norman2019_demo_bundle/metadata.json",
                message="Real Norman2019 bundle metadata.",
            ),
            _make_check(
                project_root=root,
                name="real_transformer_checkpoint",
                relative_path="artifacts/transformer_seen_norman2019_demo/best_model.pt",
                message="Real-data Transformer checkpoint.",
            ),
            _make_check(
                project_root=root,
                name="real_deg_artifact",
                relative_path="artifacts/transformer_seen_norman2019_demo/deg_artifact.csv",
                message="Real-data DEG artifact.",
            ),
            _make_check(
                project_root=root,
                name="real_transformer_summary",
                relative_path="artifacts/transformer_seen_norman2019_demo/run_summary.json",
                message="Real-data Transformer run summary.",
            ),
            _make_check(
                project_root=root,
                name="real_comparison_figure",
                relative_path="docs/assets/model_comparison_seen_norman2019_demo.png",
                message="Real-data model-comparison figure.",
            ),
            _make_check(
                project_root=root,
                name="real_inference_figure",
                relative_path="docs/assets/transformer_inference_preview.png",
                message="Real-data app preview figure.",
            ),
        ],
        "comparison_artifacts": [
            _make_check(
                project_root=root,
                name="mlp_summary",
                relative_path="artifacts/mlp_seen_norman2019_demo/run_summary.json",
                message="Real-data MLP run summary.",
            ),
            _make_check(
                project_root=root,
                name="xgboost_summary",
                relative_path="artifacts/xgboost_seen_norman2019_demo/xgboost_run_summary.json",
                message="Real-data XGBoost run summary.",
            ),
        ],
        "raw_data": [
            _make_check(
                project_root=root,
                name="norman2019_raw_file",
                relative_path="data/raw/NormanWeissman2019_filtered.h5ad",
                message="Optional raw Norman2019 dataset for full rebuilds.",
                required=False,
            ),
        ],
    }

    grouped = {
        group_name: [check.to_dict() for check in checks]
        for group_name, checks in groups.items()
    }

    modes = {
        "bootstrap_ready": _all_required_ok(groups["repository"]),
        "notebooks_ready": _all_required_ok(groups["notebooks"]),
        "offline_demo_ready": _all_required_ok(groups["synthetic_demo"]),
        "real_results_ready": _all_required_ok(groups["real_results"]),
        "model_comparison_ready": _all_required_ok(groups["comparison_artifacts"]),
        "raw_data_available": groups["raw_data"][0].ok,
    }

    return {
        "project_root": str(root),
        "python_version": python_version,
        "modes": modes,
        "groups": grouped,
    }


def format_health_report(summary: dict[str, Any]) -> str:
    """Render a readable plain-text report for local CLI use."""
    lines = [
        "PerturbScope-GPT doctor",
        f"Project root: {summary['project_root']}",
        f"Python pin: {summary.get('python_version') or 'missing'}",
        "",
        "Modes:",
    ]

    for mode_name, ok in summary["modes"].items():
        status = "OK" if ok else "MISSING"
        lines.append(f"  [{status}] {mode_name}")

    group_titles = {
        "repository": "Repository",
        "notebooks": "Notebooks",
        "synthetic_demo": "Offline synthetic demo",
        "real_results": "Real Norman2019 results",
        "comparison_artifacts": "Model comparison artifacts",
        "raw_data": "Optional raw dataset",
    }

    for group_name, checks in summary["groups"].items():
        lines.extend(["", f"{group_titles.get(group_name, group_name.title())}:"])
        for check in checks:
            status = "OK" if check["ok"] else "MISSING"
            suffix = " (optional)" if not check["required"] else ""
            lines.append(
                f"  [{status}] {check['path']}{suffix} — {check['message']}"
            )

    lines.extend(
        [
            "",
            "Recommended order:",
            "  1. make doctor",
            "  2. make snapshot",
            "  3. make test && make lint && make typecheck",
            "  4. ./scripts/run_generate_synthetic_showcase.sh",
            "  5. ./scripts/run_app.sh",
        ]
    )
    return "\n".join(lines)
