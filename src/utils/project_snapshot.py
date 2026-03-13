from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.io import read_json, write_json
from src.evaluation.error_analysis import build_error_highlights
from src.utils.comparison import scan_artifact_comparison_rows
from src.utils.project_health import collect_project_health


def _display_name(label: str) -> str:
    prefix = label.split("_seen_")[0]
    mapping = {
        "transformer": "Transformer",
        "mlp": "MLP",
        "xgboost": "XGBoost",
    }
    return mapping.get(prefix, label)


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("unseen_pearson") is None,
            -(float(row["unseen_pearson"]) if row.get("unseen_pearson") is not None else 0.0),
        ),
    )


def _filter_rows(rows: list[dict[str, Any]], suffix: str) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        label = str(row["model"])
        if not label.endswith(suffix):
            continue
        enriched = dict(row)
        enriched["display_name"] = _display_name(label)
        filtered.append(enriched)
    return _sort_rows(filtered)


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("unseen_pearson") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["unseen_pearson"]))


def _asset_entry(project_root: Path, relative_path: str) -> dict[str, Any]:
    path = project_root / relative_path
    return {
        "path": relative_path,
        "exists": path.exists(),
    }


def _transformer_real_summary(project_root: Path) -> dict[str, Any]:
    summary_path = project_root / "artifacts/transformer_seen_norman2019_demo/run_summary.json"
    if not summary_path.exists():
        return {}
    return read_json(summary_path)


def _transformer_real_multiseed_summary(project_root: Path) -> dict[str, Any]:
    report_path = project_root / "artifacts/multi_seed_report.json"
    if not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        if (
            row.get("dataset_name") == "scperturb_norman2019"
            and row.get("train_protocol") == "seen"
            and row.get("model_type") == "transformer"
        ):
            return row
    return {}


def _transformer_real_error_highlights(project_root: Path) -> dict[str, dict[str, Any]]:
    artifact_dir = project_root / "artifacts/transformer_seen_norman2019_demo"
    highlights: dict[str, dict[str, Any]] = {}
    for split_name in ("seen_test", "unseen_test"):
        summary_path = artifact_dir / f"{split_name}_error_summary.json"
        if not summary_path.exists():
            continue
        highlights[split_name] = build_error_highlights(read_json(summary_path))
    return highlights


def build_project_snapshot(project_root: str | Path = ".") -> dict[str, Any]:
    """Build an interview-friendly snapshot of current project status and results."""
    root = Path(project_root).resolve()
    health = collect_project_health(root)
    all_rows = scan_artifact_comparison_rows(root / "artifacts")
    real_rows = _filter_rows(all_rows, "_norman2019_demo")
    synthetic_rows = _filter_rows(all_rows, "_synthetic_demo")
    best_real = _best_row(real_rows)
    transformer_summary = _transformer_real_summary(root)
    transformer_multiseed = _transformer_real_multiseed_summary(root)
    transformer_error_highlights = _transformer_real_error_highlights(root)
    transformer_unseen = transformer_summary.get("test_metrics", {}).get("unseen_test", {})
    transformer_seen = transformer_summary.get("test_metrics", {}).get("seen_test", {})
    bundle = transformer_summary.get("artifacts", {}).get("bundle", {})
    dataset = transformer_summary.get("dataset", {})

    unseen_values = [
        float(row["unseen_pearson"])
        for row in real_rows
        if row.get("unseen_pearson") is not None
    ]
    min_real_unseen = min(unseen_values) if unseen_values else None

    headline = {
        "dataset_name": dataset.get("name"),
        "cell_context": dataset.get("cell_context"),
        "num_samples": bundle.get("num_samples"),
        "num_genes": bundle.get("num_genes"),
        "num_perturbations": bundle.get("num_perturbations"),
        "best_real_unseen_model": best_real.get("display_name") if best_real else None,
        "best_real_unseen_pearson": best_real.get("unseen_pearson") if best_real else None,
        "transformer_seen_pearson": transformer_seen.get("pearson_per_perturbation"),
        "transformer_unseen_pearson": transformer_unseen.get("pearson_per_perturbation"),
        "transformer_unseen_top20_deg_overlap": transformer_unseen.get("topk_deg_overlap_20"),
        "transformer_unseen_top100_deg_overlap": transformer_unseen.get("topk_deg_overlap_100"),
        "transformer_multiseed_num_runs": transformer_multiseed.get("num_runs"),
        "transformer_multiseed_seeds": transformer_multiseed.get("seeds"),
        "transformer_multiseed_unseen_pearson_mean": transformer_multiseed.get(
            "unseen_pearson_mean"
        ),
        "transformer_multiseed_unseen_pearson_std": transformer_multiseed.get(
            "unseen_pearson_std"
        ),
        "transformer_multiseed_unseen_top100_deg_mean": transformer_multiseed.get(
            "unseen_top100_deg_mean"
        ),
        "transformer_multiseed_unseen_top100_deg_std": transformer_multiseed.get(
            "unseen_top100_deg_std"
        ),
        "all_real_models_unseen_pearson_ge_0_82": (
            min_real_unseen is not None and min_real_unseen >= 0.82
        ),
    }

    assets = {
        "architecture_doc": _asset_entry(root, "docs/architecture.md"),
        "real_comparison_figure": _asset_entry(
            root, "docs/assets/model_comparison_seen_norman2019_demo.png"
        ),
        "real_inference_figure": _asset_entry(
            root, "docs/assets/transformer_inference_preview.png"
        ),
        "synthetic_comparison_figure": _asset_entry(
            root, "docs/assets/model_comparison_seen_synthetic_demo.png"
        ),
        "synthetic_inference_figure": _asset_entry(
            root, "docs/assets/transformer_inference_preview_synthetic_demo.png"
        ),
    }

    commands = {
        "doctor": "make doctor",
        "snapshot": "make snapshot",
        "tests": "make test && make lint && make typecheck",
        "synthetic_showcase": "./scripts/run_generate_synthetic_showcase.sh",
        "app": "./scripts/run_app.sh",
    }

    return {
        "project_root": str(root),
        "health_modes": health["modes"],
        "headline": headline,
        "real_model_rows": real_rows,
        "synthetic_model_rows": synthetic_rows,
        "transformer_error_highlights": transformer_error_highlights,
        "assets": assets,
        "commands": commands,
    }


def write_project_snapshot(snapshot: dict[str, Any], output_path: str | Path) -> Path:
    """Write a project snapshot JSON file to disk."""
    destination = Path(output_path)
    write_json(destination, snapshot)
    return destination


def format_project_snapshot(snapshot: dict[str, Any]) -> str:
    """Render a human-readable project snapshot for local CLI use."""
    headline = snapshot["headline"]
    lines = [
        "PerturbScope-GPT snapshot",
        f"Project root: {snapshot['project_root']}",
        "",
        "Headline:",
        (
            "  Dataset: "
            f"{headline.get('dataset_name') or 'unknown'}"
            f" | cell context={headline.get('cell_context') or 'unknown'}"
            f" | samples={headline.get('num_samples') or 'unknown'}"
            f" | genes={headline.get('num_genes') or 'unknown'}"
            f" | perturbations={headline.get('num_perturbations') or 'unknown'}"
        ),
        (
            "  Best real unseen Pearson: "
            f"{headline.get('best_real_unseen_model') or 'n/a'}"
            f" ({_format_metric(headline.get('best_real_unseen_pearson'))})"
        ),
        (
            "  Transformer unseen Pearson / top-100 DEG overlap: "
            f"{_format_metric(headline.get('transformer_unseen_pearson'))}"
            f" / {_format_metric(headline.get('transformer_unseen_top100_deg_overlap'))}"
        ),
        (
            "  All real models unseen Pearson >= 0.82: "
            f"{'yes' if headline.get('all_real_models_unseen_pearson_ge_0_82') else 'no'}"
        ),
        "",
        "Readiness modes:",
    ]

    if headline.get("transformer_multiseed_num_runs") is not None:
        lines.insert(
            8,
            (
                "  Transformer multi-seed unseen Pearson / top-100 DEG overlap: "
                f"{_format_mean_std(headline.get('transformer_multiseed_unseen_pearson_mean'), headline.get('transformer_multiseed_unseen_pearson_std'))}"
                f" / {_format_mean_std(headline.get('transformer_multiseed_unseen_top100_deg_mean'), headline.get('transformer_multiseed_unseen_top100_deg_std'))}"
                f" across {headline.get('transformer_multiseed_num_runs')} runs"
                f" (seeds={_format_seed_list(headline.get('transformer_multiseed_seeds'))})"
            ),
        )

    error_highlights = snapshot.get("transformer_error_highlights", {})
    if error_highlights:
        lines.extend(["", "Transformer error-analysis highlights:"])
        seen_highlights = error_highlights.get("seen_test")
        unseen_highlights = error_highlights.get("unseen_test")
        if seen_highlights:
            lines.append(_format_error_highlight_line("seen", seen_highlights))
        if unseen_highlights:
            lines.append(_format_error_highlight_line("unseen", unseen_highlights))

    for mode_name, ok in snapshot["health_modes"].items():
        lines.append(f"  [{'OK' if ok else 'MISSING'}] {mode_name}")

    lines.extend(["", "Real Norman2019 model comparison:"])
    if snapshot["real_model_rows"]:
        for row in snapshot["real_model_rows"]:
            lines.append(
                "  "
                f"{row['display_name']}: "
                f"seen={_format_metric(row.get('seen_pearson'))}, "
                f"unseen={_format_metric(row.get('unseen_pearson'))}, "
                f"unseen_mse={_format_metric(row.get('unseen_mse'))}"
            )
    else:
        lines.append("  No real-data model summaries found.")

    lines.extend(["", "Synthetic showcase model comparison:"])
    if snapshot["synthetic_model_rows"]:
        for row in snapshot["synthetic_model_rows"]:
            lines.append(
                "  "
                f"{row['display_name']}: "
                f"seen={_format_metric(row.get('seen_pearson'))}, "
                f"unseen={_format_metric(row.get('unseen_pearson'))}"
            )
    else:
        lines.append("  No synthetic model summaries found.")

    lines.extend(["", "Key assets:"])
    for asset_name, asset in snapshot["assets"].items():
        lines.append(
            f"  [{'OK' if asset['exists'] else 'MISSING'}] {asset['path']} ({asset_name})"
        )

    lines.extend(["", "Suggested demo commands:"])
    for label, command in snapshot["commands"].items():
        lines.append(f"  {label}: {command}")

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


def _format_seed_list(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "n/a"
    return ",".join(str(value) for value in values)


def _format_error_highlight_line(split_label: str, highlights: dict[str, Any]) -> str:
    dominant_label = highlights.get("dominant_failure_mode_label") or "n/a"
    dominant_count = highlights.get("dominant_failure_mode_count")
    num_perturbations = highlights.get("num_perturbations")
    dominant_text = dominant_label
    if dominant_count is not None and num_perturbations:
        dominant_text += f" ({dominant_count}/{num_perturbations})"

    return (
        f"  {split_label}: dominant failure mode={dominant_text}, "
        f"worst Pearson={highlights.get('worst_pearson_perturbation') or 'n/a'} "
        f"({_format_metric(highlights.get('worst_pearson_value'))}), "
        f"worst MSE={highlights.get('worst_mse_perturbation') or 'n/a'} "
        f"({_format_metric(highlights.get('worst_mse_value'))})"
    )
