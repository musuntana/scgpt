from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_anndata
from src.data.pairing import load_processed_bundle
from src.data.preprocess import prepare_adata
from src.data.schema import enrich_adata_for_preset, resolve_schema
from src.evaluation.deg import compute_deg_artifact, save_deg_artifact
from src.utils.config import ensure_keys, load_yaml
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local DEG artifact for app ranking.")
    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to the input .h5ad file. Defaults to dataset.raw_path in config.",
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Processed bundle directory used by the app and model artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store deg_artifact.csv and metadata.",
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to the data configuration YAML.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="Path to the training configuration YAML with DEG thresholds.",
    )
    parser.add_argument(
        "--perturbation-col",
        default="auto",
        help="obs column holding perturbation labels, or auto.",
    )
    parser.add_argument(
        "--control-label",
        default=None,
        help="Label used for control cells. Defaults to config value.",
    )
    parser.add_argument(
        "--batch-col",
        default="auto",
        help="obs column used for batch-aware control means, or auto/none.",
    )
    parser.add_argument(
        "--context-cols",
        nargs="*",
        default=["auto"],
        help="Additional obs columns used for schema resolution, or auto.",
    )
    parser.add_argument(
        "--hvg-top-genes",
        type=int,
        default=None,
        help="Optional override for HVG count. Defaults to bundle gene count.",
    )
    parser.add_argument(
        "--max-cells-per-perturbation",
        type=int,
        default=None,
        help="Optional override for local-first subsampling. Defaults to bundle max count.",
    )
    return parser.parse_args()


def _bundle_max_cells_per_perturbation(bundle: dict) -> int:
    counts = {}
    for perturbation_index in bundle["perturbation_index"]:
        counts[int(perturbation_index)] = counts.get(int(perturbation_index), 0) + 1
    return max(counts.values(), default=0)


def _align_adata_to_bundle_genes(adata, bundle_gene_names: list[str]):
    current_genes = set(adata.var_names.astype(str).tolist())
    missing = [gene for gene in bundle_gene_names if gene not in current_genes]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "Prepared AnnData is missing bundle genes. "
            f"First missing genes: {preview}"
        )
    return adata[:, bundle_gene_names].copy()


def main() -> None:
    args = parse_args()
    data_config = load_yaml(args.config)
    train_config = load_yaml(args.train_config)
    ensure_keys(
        data_config,
        [
            "dataset.raw_path",
            "dataset.name",
            "preprocess.hvg_top_genes",
            "split.random_seed",
        ],
    )
    ensure_keys(
        train_config,
        [
            "deg.method",
            "deg.adjusted_pvalue_threshold",
            "deg.abs_logfoldchange_threshold",
        ],
    )

    bundle = load_processed_bundle(args.bundle_dir)
    bundle_gene_names = bundle["metadata"]["gene_names"]
    bundle_perturbation_names = bundle["metadata"]["perturbation_names"]

    preprocess_config = dict(data_config["preprocess"])
    preprocess_config["hvg_top_genes"] = (
        int(args.hvg_top_genes)
        if args.hvg_top_genes is not None
        else int(len(bundle_gene_names))
    )
    preprocess_config["max_cells_per_perturbation"] = (
        int(args.max_cells_per_perturbation)
        if args.max_cells_per_perturbation is not None
        else int(_bundle_max_cells_per_perturbation(bundle))
    )

    input_path = args.input_path or data_config["dataset"]["raw_path"]
    random_seed = int(data_config["split"]["random_seed"])
    adata = load_anndata(input_path)

    schema_config = data_config.get("schema", {})
    adata = enrich_adata_for_preset(
        adata=adata,
        preset=str(schema_config.get("preset", "")),
        control_label_candidates=schema_config.get("control_label_candidates", ["control"]),
        multi_gene_delimiters=schema_config.get(
            "multi_gene_delimiters", [";", "+", ",", "|", "_"]
        ),
    )
    schema = resolve_schema(
        adata=adata,
        dataset_config=data_config,
        cli_perturbation_col=args.perturbation_col,
        cli_control_label=args.control_label,
        cli_batch_col=args.batch_col,
        cli_context_cols=args.context_cols,
    )
    LOGGER.info(
        "Resolved schema | perturbation_col=%s | control_label=%s | batch_col=%s | context_cols=%s",
        schema.perturbation_col,
        schema.control_label,
        schema.batch_col,
        schema.context_cols,
    )

    prepared = prepare_adata(
        adata=adata,
        preprocess_config=preprocess_config,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        random_seed=random_seed,
    )
    prepared = _align_adata_to_bundle_genes(prepared, bundle_gene_names)

    deg_config = train_config["deg"]
    deg_df = compute_deg_artifact(
        adata=prepared,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        perturbation_names=bundle_perturbation_names,
        method=str(deg_config["method"]),
        adjusted_pvalue_threshold=float(deg_config["adjusted_pvalue_threshold"]),
        abs_logfoldchange_threshold=float(deg_config["abs_logfoldchange_threshold"]),
    )

    metadata = {
        "dataset_name": data_config["dataset"]["name"],
        "raw_path": str(input_path),
        "bundle_dir": str(args.bundle_dir),
        "perturbation_col": schema.perturbation_col,
        "control_label": schema.control_label,
        "num_rows": int(len(deg_df)),
        "num_genes": int(len(bundle_gene_names)),
        "num_perturbations": int(len(bundle_perturbation_names)),
        "preprocess": preprocess_config,
        "deg": deg_config,
    }
    csv_path, metadata_path = save_deg_artifact(
        deg_df=deg_df,
        output_dir=args.output_dir,
        metadata=metadata,
    )
    LOGGER.info("Saved DEG artifact to %s", csv_path)
    LOGGER.info("Saved DEG metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
