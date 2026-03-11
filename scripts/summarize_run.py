from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.experiment import build_run_summary, write_run_summary
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a structured local summary artifact.")
    parser.add_argument("--bundle-dir", required=True, help="Directory with processed bundle files.")
    parser.add_argument("--output-dir", required=True, help="Training artifact directory.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to a saved model checkpoint.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to write the summary JSON. Defaults to <output-dir>/run_summary.json.",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["transformer", "mlp", "xgboost"],
        help="Model family used for the run.",
    )
    parser.add_argument(
        "--split-prefix",
        default="seen",
        choices=["seen", "unseen"],
        help="Training split protocol for the run.",
    )
    parser.add_argument(
        "--data-config", default="configs/data.yaml", help="Path to data configuration YAML."
    )
    parser.add_argument(
        "--model-config", default="configs/model.yaml", help="Path to model configuration YAML."
    )
    parser.add_argument(
        "--train-config", default="configs/train.yaml", help="Path to training configuration YAML."
    )
    parser.add_argument(
        "--history-path",
        default=None,
        help="Path to the saved history JSON. Defaults to <output-dir>/history.json when present.",
    )
    parser.add_argument(
        "--seen-metrics-path",
        default=None,
        help="Optional JSON path for seen_test metrics.",
    )
    parser.add_argument(
        "--unseen-metrics-path",
        default=None,
        help="Optional JSON path for unseen_test metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the recorded training seed in run_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    history_path = args.history_path or output_dir / "history.json"
    output_path = args.output_path or output_dir / "run_summary.json"

    summary = build_run_summary(
        bundle_dir=args.bundle_dir,
        checkpoint_path=args.checkpoint_path,
        output_dir=output_dir,
        model_type=args.model_type,
        split_prefix=args.split_prefix,
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        history_path=history_path,
        seen_metrics_path=args.seen_metrics_path,
        unseen_metrics_path=args.unseen_metrics_path,
        seed=args.seed,
    )
    destination = write_run_summary(summary, output_path)
    LOGGER.info("Wrote run summary to %s", destination)


if __name__ == "__main__":
    main()
