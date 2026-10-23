from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_anndata
from src.data.pairing import load_processed_bundle
from src.data.schema import enrich_adata_for_preset, resolve_schema
from src.integrations.gears_contract import (
    build_gears_split_contract,
    validate_source_identity,
)
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight unseen-group bundle and export official GEARS custom split."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-h5ad",
        default=None,
        help="Optional raw/source H5AD for cell, label and gene identity checks only.",
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_gears_preflight(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    source_h5ad: str | Path | None = None,
    data_config_path: str | Path = "configs/data.yaml",
) -> dict[str, Any]:
    source = Path(bundle_dir)
    output = Path(output_dir)
    bundle = load_processed_bundle(source)
    split, manifest = build_gears_split_contract(bundle)

    manifest["bundle_artifacts_sha256"] = {
        filename: file_sha256(source / filename)
        for filename in ("arrays.npz", "splits.npz", "metadata.json")
    }
    if source_h5ad is not None:
        adata = load_anndata(source_h5ad, backed="r")
        try:
            data_config = load_yaml(data_config_path)
            schema_config = data_config.get("schema", {})
            adata = enrich_adata_for_preset(
                adata,
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
                cli_perturbation_col=None,
                cli_control_label=None,
                cli_batch_col=None,
                cli_context_cols=None,
            )
            source_report = validate_source_identity(
                adata, bundle,
                perturbation_col=schema.perturbation_col,
                control_label=schema.control_label,
            )
            manifest["source_validation"] = source_report
            manifest["source_cell_identity_verified"] = True
            manifest["source_h5ad_sha256"] = file_sha256(Path(source_h5ad))
        finally:
            # Backed AnnData keeps a file handle open until explicitly closed.
            if getattr(adata, "isbacked", False):
                adata.file.close()

    artifacts = {
        "gears_custom_split.pkl": split,
        "gears_custom_split.json": split,
        "gears_preflight.json": manifest,
    }
    existing = [name for name in artifacts if (output / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing preflight artifacts: "
            + ", ".join(existing)
            + ". Choose a new output directory."
        )
    output.mkdir(parents=True, exist_ok=True)
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
    )
    print(
        f"GEARS preflight exported to {args.output_dir}: "
        f"{manifest['perturbation_count']} perturbations; "
        f"status={manifest['status']}, comparison_eligible=False."
    )
    print("No GEARS model was trained. See gears_preflight.json for remaining gates.")


if __name__ == "__main__":
    main()
