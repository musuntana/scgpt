from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.comparison import scan_artifact_comparison_rows

METRIC_FIELDS = (
    "seen_pearson",
    "seen_mse",
    "unseen_pearson",
    "unseen_mse",
    "seen_top20_deg",
    "seen_top100_deg",
    "unseen_top20_deg",
    "unseen_top100_deg",
)


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    dataset_name = str(row.get("dataset_name") or "")
    train_protocol = str(row.get("train_protocol") or "unknown_protocol")
    model_type = str(row.get("model_type") or row.get("base_model_label") or row["model"])
    base_model_label = str(row.get("base_model_label") or row["model"])
    dataset_group = dataset_name or base_model_label
    return dataset_group, train_protocol, model_type, base_model_label


def _metric_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.shape[0]),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_multiseed_report(
    rows: list[dict[str, Any]],
    *,
    min_runs: int = 2,
) -> list[dict[str, Any]]:
    """Aggregate repeated runs into mean/std summaries grouped by dataset, split, and model."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    report_rows: list[dict[str, Any]] = []
    for group, group_rows in grouped.items():
        is_gears = group[2] == "gears"
        duplicate_seeds: list[int] = []
        duplicate_seed_artifacts: list[str] = []
        aggregation_rows = group_rows
        if is_gears:
            unique_seed_rows: dict[int, dict[str, Any]] = {}
            for row in sorted(group_rows, key=lambda item: str(item.get("model", ""))):
                seed = row.get("seed")
                if seed is None:
                    continue
                seed_value = int(seed)
                if seed_value in unique_seed_rows:
                    duplicate_seeds.append(seed_value)
                    duplicate_seed_artifacts.append(str(row.get("model", "")))
                    continue
                unique_seed_rows[seed_value] = row
            aggregation_rows = list(unique_seed_rows.values())
        if len(aggregation_rows) < min_runs:
            continue

        metric_summary: dict[str, dict[str, float | int]] = {}
        flat_metrics: dict[str, float] = {}
        train_protocol = group[1]
        for field in METRIC_FIELDS:
            if field.startswith("seen_") and train_protocol != "seen":
                continue
            if field.startswith("unseen_") and train_protocol != "unseen":
                continue
            values = [
                float(row[field])
                for row in aggregation_rows
                if row.get(field) is not None
            ]
            if not values:
                continue
            stats = _metric_stats(values)
            metric_summary[field] = stats
            flat_metrics[f"{field}_mean"] = float(stats["mean"])
            flat_metrics[f"{field}_std"] = float(stats["std"])

        dataset_group, train_protocol, model_type, base_model_label = group
        dataset_name = next(
            (
                str(row["dataset_name"])
                for row in aggregation_rows
                if row.get("dataset_name")
            ),
            "unknown_dataset",
        )
        seeds = sorted(
            {
                int(row["seed"])
                for row in aggregation_rows
                if row.get("seed") is not None
            }
        )
        report_row: dict[str, Any] = {
            "group_key": f"{dataset_group}:{train_protocol}:{model_type}",
            "group_label": base_model_label,
            "dataset_name": dataset_name,
            "train_protocol": train_protocol,
            "model_type": model_type,
            "num_runs": len(aggregation_rows),
            "artifact_labels": [str(row["model"]) for row in aggregation_rows],
            "seeds": seeds,
            "metrics": metric_summary,
            **flat_metrics,
        }

        if is_gears:
            # Gate on every artifact, including duplicate-seed artifacts that are
            # excluded from the metric aggregate. Otherwise a conflicting retry
            # could hide a graph mismatch or an unverified target alignment.
            target_equivalence_verified = all(
                row.get("target_equivalence_verified") is True
                for row in group_rows
            )
            graph_hashes = sorted(
                {
                    str(row["coexpression_graph_sha256"])
                    for row in group_rows
                    if row.get("coexpression_graph_sha256")
                }
            )
            minimum_seed_gate_passed = len(seeds) >= 3
            duplicate_seed_gate_passed = not duplicate_seeds
            comparison_blockers: list[str] = []
            if not target_equivalence_verified:
                comparison_blockers.append(
                    "real-data target equivalence has not been verified"
                )
            if not minimum_seed_gate_passed:
                comparison_blockers.append(
                    "at least three distinct random seeds are required"
                )
            if not duplicate_seed_gate_passed:
                comparison_blockers.append(
                    "duplicate-seed artifacts detected; resolve duplicates before comparison"
                )
            if len(graph_hashes) != 1:
                comparison_blockers.append(
                    "co-expression graph hash is missing or differs across runs"
                )
            if not all(
                row.get("comparison_eligible") is True for row in group_rows
            ):
                comparison_blockers.append(
                    "run manifests do not authorize a protocol-matched performance comparison"
                )
            report_row.update(
                {
                    "comparison_eligible": (
                        target_equivalence_verified
                        and minimum_seed_gate_passed
                        and duplicate_seed_gate_passed
                        and len(graph_hashes) == 1
                        and all(
                            row.get("comparison_eligible") is True
                            for row in group_rows
                        )
                    ),
                    "target_equivalence_verified": target_equivalence_verified,
                    "minimum_seed_gate_passed": minimum_seed_gate_passed,
                    "duplicate_seed_gate_passed": duplicate_seed_gate_passed,
                    "coexpression_graph_hashes": graph_hashes,
                    "coexpression_graph_consistent": len(graph_hashes) == 1,
                    "duplicate_seed_runs_ignored": len(duplicate_seeds),
                    "duplicate_seeds": sorted(set(duplicate_seeds)),
                    "duplicate_seed_artifacts_ignored": sorted(duplicate_seed_artifacts),
                    "comparison_blockers": comparison_blockers,
                }
            )
        report_rows.append(report_row)

    return sorted(
        report_rows,
        key=lambda row: (row["dataset_name"], row["train_protocol"], row["model_type"]),
    )


def build_multiseed_report_from_artifacts(
    artifact_root: str | Path,
    *,
    min_runs: int = 2,
) -> list[dict[str, Any]]:
    """Scan an artifact root and aggregate any repeated-seed runs it contains."""
    return build_multiseed_report(scan_artifact_comparison_rows(artifact_root), min_runs=min_runs)

def load_multiseed_report(path: str | Path) -> list[dict[str, Any]]:
    """Load a multi-seed report from disk, returning only dict rows."""
    report_path = Path(path)
    if not report_path.exists():
        return []
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def select_multiseed_group(
    report_rows: list[dict[str, Any]],
    *,
    dataset_name: str | None = None,
    train_protocol: str | None = None,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Return the first report row matching the provided filters."""
    for row in report_rows:
        if dataset_name is not None and str(row.get("dataset_name")) != dataset_name:
            continue
        if train_protocol is not None and str(row.get("train_protocol")) != train_protocol:
            continue
        if model_type is not None and str(row.get("model_type")) != model_type:
            continue
        return row
    return {}


def format_multiseed_report(report_rows: list[dict[str, Any]], *, artifact_root: str | Path) -> str:
    """Render a concise text report for CLI use."""
    lines = [
        "PerturbScope-GPT multi-seed report",
        f"Artifact root: {Path(artifact_root).resolve()}",
        "",
    ]
    if not report_rows:
        lines.append("No groups with repeated runs were found.")
        return "\n".join(lines)

    for row in report_rows:
        lines.extend(
            [
                f"{row['group_label']} | dataset={row['dataset_name']} | protocol={row['train_protocol']}",
                f"  runs={row['num_runs']} | seeds={row['seeds'] or 'unknown'}",
            ]
        )
        if row.get("model_type") == "gears":
            lines.append(
                "  diagnostic-only: "
                f"comparison_eligible={row.get('comparison_eligible', False)} | "
                f"target_equivalence_verified={row.get('target_equivalence_verified', False)} | "
                f"three_seed_gate={row.get('minimum_seed_gate_passed', False)}"
            )
            for blocker in row.get("comparison_blockers", []):
                lines.append(f"  blocker: {blocker}")
        for metric_name in METRIC_FIELDS:
            stats = row["metrics"].get(metric_name)
            if not stats:
                continue
            lines.append(
                "  "
                f"{metric_name}: mean={float(stats['mean']):.4f} "
                f"std={float(stats['std']):.4f} "
                f"min={float(stats['min']):.4f} "
                f"max={float(stats['max']):.4f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()
