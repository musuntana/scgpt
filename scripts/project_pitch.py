from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.interview_script import (
    SUPPORTED_PITCH_TRACKS,
    build_interview_script,
    format_interview_script,
    write_interview_script_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an interview/demo speaking script for PerturbScope-GPT.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Path to the project root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--track",
        choices=list(SUPPORTED_PITCH_TRACKS),
        default="both",
        help="Role-specific pitch track to export. Defaults to both.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the script as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the rendered text or JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = build_interview_script(args.project_root, track=args.track)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.json:
            args.output_path.write_text(json.dumps(script, indent=2), encoding="utf-8")
        else:
            write_interview_script_text(script, args.output_path)
    if args.json:
        print(json.dumps(script, indent=2))
    else:
        print(format_interview_script(script))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
