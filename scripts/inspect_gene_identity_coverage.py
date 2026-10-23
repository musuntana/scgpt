from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gene_identity import build_gene_identity_coverage_report
from src.data.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect exact perturbation-to-gene mapping coverage for a processed bundle."
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Directory containing metadata.json from a processed bundle.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the coverage report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    metadata_path = bundle_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Processed bundle metadata is missing: {metadata_path}")

    metadata = read_json(metadata_path)
    gene_names = metadata.get("gene_names")
    perturbation_names = metadata.get("perturbation_names")
    if not isinstance(gene_names, list) or not isinstance(perturbation_names, list):
        raise ValueError(
            "metadata.json must contain list fields 'gene_names' and 'perturbation_names'."
        )

    report = build_gene_identity_coverage_report(gene_names, perturbation_names)
    if args.output_json is not None:
        write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
