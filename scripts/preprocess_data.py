from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_anndata
from src.data.pairing import build_training_bundle, save_processed_bundle
from src.data.preprocess import prepare_adata
from src.data.schema import enrich_adata_for_preset, resolve_schema
from src.utils.config import ensure_keys, load_yaml
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess perturbation data into a bundle.")
    parser.add_argument("--input-path", required=True, help="Path to the input .h5ad file.")
    parser.add_argument("--output-dir", required=True, help="Directory to store processed arrays.")
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to the data configuration YAML.",
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
        help="Additional obs columns used when matching control means, or auto.",
    )
    parser.add_argument(
        "--hvg-top-genes",
        type=int,
        default=None,
        help="Optional override for HVG count.",
    )
    parser.add_argument(
        "--max-cells-per-perturbation",
        type=int,
        default=None,
        help="Optional override for local-first subsampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_yaml(args.config)
    ensure_keys(
        data_config,
        [
            "preprocess.hvg_top_genes",
            "pairing.target",
            "split.val_fraction",
            "split.test_fraction",
        ],
    )

    preprocess_config = dict(data_config["preprocess"])
    if args.hvg_top_genes is not None:
        preprocess_config["hvg_top_genes"] = int(args.hvg_top_genes)
    if args.max_cells_per_perturbation is not None:
        preprocess_config["max_cells_per_perturbation"] = int(args.max_cells_per_perturbation)

    random_seed = int(data_config["split"]["random_seed"])
    adata = load_anndata(args.input_path)
    schema_config = data_config.get("schema", {})
    adata = enrich_adata_for_preset(
        adata=adata,
        preset=str(schema_config.get("preset", "")),
        control_label_candidates=schema_config.get("control_label_candidates", ["control"]),
        multi_gene_delimiters=schema_config.get("multi_gene_delimiters", [";", "+", ",", "|", "_"]),
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
    bundle = build_training_bundle(
        adata=prepared,
        perturbation_col=schema.perturbation_col,
        control_label=schema.control_label,
        batch_col=schema.batch_col,
        context_cols=schema.context_cols,
        val_fraction=float(data_config["split"]["val_fraction"]),
        test_fraction=float(data_config["split"]["test_fraction"]),
        random_seed=random_seed,
    )
    save_processed_bundle(bundle, Path(args.output_dir))
    LOGGER.info("Saved processed bundle to %s", args.output_dir)


if __name__ == "__main__":
    main()
