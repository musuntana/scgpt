from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_anndata
from src.data.pairing import build_training_bundle, load_processed_bundle
from src.data.preprocess import prepare_adata, select_hvgs_from_training_partitions
from src.data.schema import enrich_adata_for_preset, resolve_schema
from src.integrations.gears_contract import build_gears_split_contract, validate_source_identity
from src.integrations.gears_export import build_gears_input_adata, validate_rebuilt_bundle
from src.utils.config import ensure_keys, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight an unseen-group bundle and optionally export an exactly "
            "reproduced, GEARS-schema AnnData input."
        )
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-h5ad",
        default=None,
        help=(
            "Raw source H5AD. Re-runs preprocessing and exports gears_input.h5ad only "
            "if the saved bundle is reproduced exactly."
        ),
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--perturbation-col", default="auto")
    parser.add_argument("--control-label", default=None)
    parser.add_argument("--batch-col", default="auto")
    parser.add_argument("--context-cols", nargs="*", default=["auto"])
    parser.add_argument("--hvg-top-genes", type=int, default=None)
    parser.add_argument("--max-cells-per-perturbation", type=int, default=None)
    return parser.parse_args()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rebuild_source_and_export(
    source_h5ad: str | Path,
    bundle: dict[str, Any],
    *,
    data_config_path: str | Path,
    perturbation_col: str,
    control_label: str | None,
    batch_col: str,
    context_cols: list[str],
    hvg_top_genes: int | None,
    max_cells_per_perturbation: int | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    data_config = load_yaml(data_config_path)
    ensure_keys(
        data_config,
        [
            "preprocess.hvg_top_genes",
            "pairing.target",
            "split.val_fraction",
            "split.test_fraction",
        ],
    )
    schema_config = data_config.get("schema", {})
    adata = load_anndata(source_h5ad)
    adata = enrich_adata_for_preset(
        adata=adata,
        preset=str(schema_config.get("preset", "")),
        control_label_candidates=schema_config.get(
            "control_label_candidates", ["control"]
        ),
        multi_gene_delimiters=schema_config.get(
            "multi_gene_delimiters", [";", "+", ",", "|", "_"]
        ),
    )
    schema = resolve_schema(
        adata=adata,
        dataset_config=data_config,
        cli_perturbation_col=perturbation_col,
        cli_control_label=control_label,
        cli_batch_col=batch_col,
        cli_context_cols=context_cols,
    )
    source_report = validate_source_identity(
        adata,
        bundle,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        context_col=schema.context_cols[0] if len(schema.context_cols) == 1 else None,
    )

    preprocess_config = dict(data_config["preprocess"])
    if hvg_top_genes is not None:
        preprocess_config["hvg_top_genes"] = int(hvg_top_genes)
    if max_cells_per_perturbation is not None:
        preprocess_config["max_cells_per_perturbation"] = int(max_cells_per_perturbation)

    random_seed = int(data_config["split"]["random_seed"])
    prepared = prepare_adata(
        adata=adata,
        preprocess_config=preprocess_config,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        random_seed=random_seed,
        select_hvg=False,
        filter_genes_before_split=False,
        normalize_expression=False,
    )
    prepared = select_hvgs_from_training_partitions(
        adata=prepared,
        preprocess_config=preprocess_config,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        val_fraction=float(data_config["split"]["val_fraction"]),
        test_fraction=float(data_config["split"]["test_fraction"]),
        random_seed=random_seed,
    )
    rebuilt_bundle = build_training_bundle(
        adata=prepared,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        batch_col=schema.batch_col,
        context_cols=schema.context_cols,
        val_fraction=float(data_config["split"]["val_fraction"]),
        test_fraction=float(data_config["split"]["test_fraction"]),
        random_seed=random_seed,
    )
    reproduction_report = validate_rebuilt_bundle(rebuilt_bundle, bundle)
    gears_adata, export_report = build_gears_input_adata(
        prepared,
        bundle,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        context_cols=schema.context_cols,
    )
    return gears_adata, {
        "source_validation": source_report,
        "source_preprocessing_reproduction": reproduction_report,
        "gears_input": export_report,
    }, {
        "perturbation_col": schema.perturbation_col,
        "control_label": schema.control_label,
        "batch_col": schema.batch_col,
        "context_cols": schema.context_cols,
    }


def prepare_gears_preflight(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    source_h5ad: str | Path | None = None,
    data_config_path: str | Path = "configs/data.yaml",
    perturbation_col: str = "auto",
    control_label: str | None = None,
    batch_col: str = "auto",
    context_cols: list[str] | None = None,
    hvg_top_genes: int | None = None,
    max_cells_per_perturbation: int | None = None,
) -> dict[str, Any]:
    source = Path(bundle_dir)
    output = Path(output_dir)
    bundle = load_processed_bundle(source)
    split, manifest = build_gears_split_contract(bundle)

    manifest["bundle_artifacts_sha256"] = {
        filename: file_sha256(source / filename)
        for filename in ("arrays.npz", "splits.npz", "metadata.json")
    }
    gears_adata = None
    if source_h5ad is not None:
        gears_adata, source_reports, resolved_schema = _rebuild_source_and_export(
            source_h5ad,
            bundle,
            data_config_path=data_config_path,
            perturbation_col=perturbation_col,
            control_label=control_label,
            batch_col=batch_col,
            context_cols=context_cols or ["auto"],
            hvg_top_genes=hvg_top_genes,
            max_cells_per_perturbation=max_cells_per_perturbation,
        )
        manifest.update(source_reports)
        manifest["resolved_source_schema"] = resolved_schema
        manifest["source_h5ad_sha256"] = file_sha256(source_h5ad)
        manifest["source_cell_identity_verified"] = True
        manifest["source_preprocessing_reproduction_verified"] = True
        manifest["gears_input_h5ad_exported"] = True
        manifest["blocking_checks"].append(
            "train-only GEARS metadata/evaluation adapter; do not use default all-condition DE preprocessing"
        )
    else:
        manifest["gears_input_h5ad_exported"] = False

    filenames = [
        "gears_custom_split.pkl",
        "gears_custom_split.json",
        "gears_preflight.json",
        "gears_input.h5ad",
    ]
    existing = [name for name in filenames if (output / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing preflight artifacts: "
            + ", ".join(existing)
            + ". Choose a new output directory."
        )
    output.mkdir(parents=True, exist_ok=True)

    if gears_adata is not None:
        input_path = output / "gears_input.h5ad"
        gears_adata.write_h5ad(input_path, compression="gzip")
        manifest["gears_input_h5ad_sha256"] = file_sha256(input_path)

    # This pickle is produced locally from validated strings only. GEARS' own
    # custom-split loader uses pickle; never load arbitrary downloaded pickles.
    with (output / "gears_custom_split.pkl").open("wb") as handle:
        pickle.dump(split, handle, protocol=4)
    for filename, payload in (
        ("gears_custom_split.json", split),
        ("gears_preflight.json", manifest),
    ):
        with (output / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_gears_preflight(
        args.bundle_dir,
        args.output_dir,
        source_h5ad=args.source_h5ad,
        data_config_path=args.data_config,
        perturbation_col=args.perturbation_col,
        control_label=args.control_label,
        batch_col=args.batch_col,
        context_cols=args.context_cols,
        hvg_top_genes=args.hvg_top_genes,
        max_cells_per_perturbation=args.max_cells_per_perturbation,
    )
    print(
        f"GEARS preflight exported to {args.output_dir}: "
        f"{manifest['perturbation_count']} perturbations; "
        f"status={manifest['status']}, comparison_eligible=False."
    )
    if manifest["gears_input_h5ad_exported"]:
        print(
            "Exact-source GEARS-schema input exported. No GEARS model was trained; "
            "default GEARS DE preprocessing remains blocked."
        )
    else:
        print("No source H5AD supplied; only the custom split and preflight manifest were exported.")


if __name__ == "__main__":
    main()
