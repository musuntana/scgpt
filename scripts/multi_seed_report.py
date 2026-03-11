from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.multiseed import build_multiseed_report_from_artifacts, format_multiseed_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated runs into mean/std multi-seed summaries.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="Artifact root containing per-run subdirectories.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="Minimum number of runs required before a group is included.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a human-readable report.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the report to disk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_rows = build_multiseed_report_from_artifacts(
        args.artifact_root,
        min_runs=args.min_runs,
    )
    if args.json:
        payload = json.dumps(report_rows, indent=2)
    else:
        payload = format_multiseed_report(report_rows, artifact_root=args.artifact_root)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(payload, encoding="utf-8")

    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
