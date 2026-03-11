from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.project_health import collect_project_health, format_health_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local project/demo readiness for PerturbScope-GPT.",
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
        help="Print the report as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = collect_project_health(args.project_root)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_health_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
