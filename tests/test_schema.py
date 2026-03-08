from __future__ import annotations

import pandas as pd

from src.data.schema import infer_control_label, normalize_perturbation_label


def test_normalize_perturbation_label_maps_control_and_multi_gene():
    assert (
        normalize_perturbation_label(
            "control",
            control_labels={"control", "ctrl"},
            multi_gene_delimiters=[";", "+", ",", "|", "_"],
        )
        == "control"
    )
    assert (
        normalize_perturbation_label(
            "AHR_KLF1",
            control_labels={"control"},
            multi_gene_delimiters=[";", "+", ",", "|", "_"],
        )
        == "AHR;KLF1"
    )


def test_infer_control_label_uses_candidates():
    labels = pd.Series(["AHR", "control", "KLF1"])
    assert infer_control_label(labels, configured_label="auto", candidates=["control"]) == "control"
