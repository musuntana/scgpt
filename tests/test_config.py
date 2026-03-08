from __future__ import annotations

from src.utils.config import ensure_keys, load_project_config


def test_load_project_config_reads_expected_sections():
    config = load_project_config("configs")
    assert set(config) == {"data", "model", "train"}


def test_ensure_keys_accepts_existing_paths():
    config = load_project_config("configs")
    ensure_keys(
        config,
        [
            "data.preprocess.hvg_top_genes",
            "model.transformer.d_model",
            "train.train.batch_size",
        ],
    )

