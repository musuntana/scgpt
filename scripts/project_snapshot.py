from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.project_snapshot import (
    build_project_snapshot,
    format_project_snapshot,
    write_project_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an interview-friendly project snapshot for PerturbScope-GPT.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Path to the project root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the snapshot as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the snapshot JSON to disk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = build_project_snapshot(args.project_root)
    if args.output_path is not None:
        write_project_snapshot(snapshot, args.output_path)
    if args.json:
        print(json.dumps(snapshot, indent=2))
    else:
        print(format_project_snapshot(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
