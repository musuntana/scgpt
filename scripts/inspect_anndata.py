from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_anndata, validate_h5ad_file, write_json
from src.data.schema import enrich_adata_for_preset, resolve_schema, summarize_anndata
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an AnnData file and resolve schema.")
    parser.add_argument("--input-path", required=True, help="Path to the input .h5ad file.")
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to the data configuration YAML.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the inspection summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_yaml(args.config)
    try:
        validate_h5ad_file(args.input_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    adata = load_anndata(args.input_path, backed="r")

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
        cli_perturbation_col="auto",
        cli_control_label=None,
        cli_batch_col="auto",
        cli_context_cols=["auto"],
    )
    summary = summarize_anndata(adata)
    summary["resolved_schema"] = {
        "perturbation_col": schema.perturbation_col,
        "control_label": schema.control_label,
        "batch_col": schema.batch_col,
        "context_cols": schema.context_cols,
    }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json:
        write_json(args.output_json, summary)


if __name__ == "__main__":
    main()
