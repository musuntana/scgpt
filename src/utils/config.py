from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {file_path}, got {type(data)!r}")
    return data


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project_config(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load the default data, model, and training configuration files."""
    root = Path(config_dir)
    config = {
        "data": load_yaml(root / "data.yaml"),
        "model": load_yaml(root / "model.yaml"),
        "train": load_yaml(root / "train.yaml"),
    }
    return config


def get_nested(config: dict[str, Any], dotted_key: str, default: Any | None = None) -> Any:
    """Read a nested config value using dot notation."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def ensure_keys(config: dict[str, Any], required_keys: list[str]) -> None:
    """Validate that required dot-path keys are present."""
    missing = [key for key in required_keys if get_nested(config, key) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required configuration keys: {missing_str}")

