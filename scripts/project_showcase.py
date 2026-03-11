from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.utils.project_health import collect_project_health
from src.utils.project_snapshot import build_project_snapshot, write_project_snapshot
from src.utils.showcase import build_showcase_plan, format_showcase_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and print an interview-ready live demo flow.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Path to the project root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/project_snapshot.json"),
        help="Where to write the snapshot JSON.",
    )
    parser.add_argument(
        "--force-refresh-synthetic",
        action="store_true",
        help="Regenerate the synthetic showcase even if it already exists.",
    )
    parser.add_argument(
        "--launch-app",
        action="store_true",
        help="Launch the Streamlit app after preparing the showcase.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    health = collect_project_health(project_root)
    plan = build_showcase_plan(
        project_root,
        health["modes"],
        launch_app=args.launch_app,
        force_refresh_synthetic=args.force_refresh_synthetic,
        snapshot_output_path=args.output_path,
    )

    actions_taken = {
        "generated_synthetic_showcase": False,
        "snapshot_written": False,
        "launch_app": False,
    }

    if plan["prepare_synthetic_showcase"]:
        subprocess.run(
            [str(project_root / "scripts/run_generate_synthetic_showcase.sh")],
            check=True,
        )
        actions_taken["generated_synthetic_showcase"] = True

    snapshot = build_project_snapshot(project_root)
    write_project_snapshot(snapshot, plan["snapshot_output_path"])
    actions_taken["snapshot_written"] = True

    print(format_showcase_report(snapshot, plan, actions_taken))

    if plan["launch_app"]:
        actions_taken["launch_app"] = True
        subprocess.run([str(project_root / "scripts/run_app.sh")], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
