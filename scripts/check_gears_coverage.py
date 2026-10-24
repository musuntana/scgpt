from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pairing import load_processed_bundle
from src.integrations.gears_contract import build_gears_split_contract
from src.integrations.gears_coverage import build_go_graph_coverage_report
from src.integrations.gears_runtime import (
    attach_train_only_loss_metadata,
    validate_gears_gene_order,
)


EXPECTED_BUNDLE_ARTIFACTS = ("arrays.npz", "splits.npz", "metadata.json")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check official GEARS GO-annotation node coverage without building cell graphs, "
            "initializing a model, training, or generating metrics."
        )
    )
    parser.add_argument("--preflight-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--gears-data-dir",
        required=True,
        type=Path,
        help="Isolated directory for official GEARS GO-annotation cache.",
    )
    return parser.parse_args()


def validate_inputs(
    preflight_dir: Path, bundle_dir: Path, input_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = preflight_dir / "gears_preflight.json"
    split_json_path = preflight_dir / "gears_custom_split.json"
    for path in (manifest_path, split_json_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required preflight artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREFLIGHT_ONLY":
        raise ValueError("Expected a PREFLIGHT_ONLY manifest.")
    if (
        manifest.get("source_preprocessing_reproduction_verified") is not True
        or manifest.get("source_cell_identity_verified") is not True
        or manifest.get("gears_input_h5ad_exported") is not True
        or manifest.get("gears_input_h5ad_sha256") != file_sha256(input_path)
    ):
        raise ValueError("Source replay/export flags or input H5AD SHA-256 do not match.")

    expected_hash_names = set(EXPECTED_BUNDLE_ARTIFACTS)
    hashes = manifest.get("bundle_artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_hash_names:
        raise ValueError("Preflight must contain the exact processed-bundle hash set.")
    for name in EXPECTED_BUNDLE_ARTIFACTS:
        digest = hashes[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError(f"Invalid SHA-256 digest for {name}.")
        path = bundle_dir / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Processed bundle is missing or changed: {name}")

    bundle = load_processed_bundle(bundle_dir)
    expected_split, _ = build_gears_split_contract(bundle)
    split = json.loads(split_json_path.read_text(encoding="utf-8"))
    if split != expected_split:
        raise ValueError("GEARS split JSON differs from the hashed processed bundle.")
    if set(split) != {"train", "val", "test"} or any(not split[k] for k in split):
        raise ValueError("GEARS split must contain non-empty train/val/test conditions.")
    return manifest, bundle


def run_coverage_check(
    *,
    preflight_dir: Path,
    bundle_dir: Path,
    output_dir: Path,
    gears_data_dir: Path,
) -> dict[str, Any]:
    input_path = preflight_dir / "gears_input.h5ad"
    report_path = output_dir / "gears_go_graph_coverage.json"
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    if gears_data_dir.resolve() == output_dir.resolve():
        raise ValueError("GEARS annotation cache must be separate from report output.")

    manifest, bundle = validate_inputs(preflight_dir, bundle_dir, input_path)
    bundle_metadata = bundle["metadata"]
    adata = ad.read_h5ad(input_path)
    if "gene_name" not in adata.var or "condition" not in adata.obs:
        raise ValueError("GEARS H5AD must include var['gene_name'] and obs['condition'].")
    validate_gears_gene_order(
        bundle_gene_ids=list(bundle_metadata["gene_names"]),
        input_gene_ids=adata.var_names.astype(str).tolist(),
        input_gene_symbols=adata.var["gene_name"].astype(str).tolist(),
    )
    split = json.loads((preflight_dir / "gears_custom_split.json").read_text(encoding="utf-8"))
    for partition in ("train", "val", "test"):
        observed = set(
            adata.obs.loc[
                adata.obs["scgpt_split"].astype(str) == partition, "condition"
            ].astype(str)
        )
        if observed != set(split[partition]):
            raise ValueError(f"Input H5AD {partition} conditions differ from split JSON.")

    # Validate the same no-DE, train-only metadata contract used by the runtime.
    attach_train_only_loss_metadata(adata)

    try:
        from gears import PertData
    except ImportError as exc:
        raise RuntimeError(
            "Official cell-gears is not installed. Run this command inside the isolated "
            "GEARS/PyG environment; core uv.lock is intentionally unchanged."
        ) from exc

    gears_data_dir.mkdir(parents=True, exist_ok=True)
    pert_data = PertData(data_path=str(gears_data_dir), default_pert_graph=False)
    pert_data.adata = adata
    pert_data.set_pert_genes()
    annotation_path = gears_data_dir / "gene2go_all.pkl"
    if not annotation_path.is_file():
        raise FileNotFoundError(
            "Official GEARS annotation cache gene2go_all.pkl was not created."
        )

    report = build_go_graph_coverage_report(
        required_perturbations=list(bundle_metadata["perturbation_names"]),
        available_perturbations=[str(name) for name in pert_data.pert_names],
        input_h5ad_sha256=file_sha256(input_path),
        preflight_manifest_sha256=file_sha256(preflight_dir / "gears_preflight.json"),
        bundle_artifacts_sha256=manifest["bundle_artifacts_sha256"],
        go_annotation_sha256=file_sha256(annotation_path),
    )
    report["repository_commit"] = None
    report["official_gears_annotation_source"] = (
        "cell-gears PertData.gene2go / set_pert_genes"
    )
    report["report_path"] = str(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    report = run_coverage_check(
        preflight_dir=args.preflight_dir,
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        gears_data_dir=args.gears_data_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["go_graph_node_coverage"]["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
