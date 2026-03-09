from __future__ import annotations

import numpy as np
from anndata import AnnData

from src.evaluation.deg import compute_deg_artifact, load_deg_artifact, save_deg_artifact


def test_compute_deg_artifact_and_reload(tmp_path):
    matrix = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.1, 1.0, 1.0],
            [1.0, 0.9, 1.0],
            [8.0, 1.0, 1.0],
            [8.2, 1.1, 1.0],
            [7.8, 1.0, 0.9],
        ],
        dtype=np.float32,
    )
    adata = AnnData(matrix)
    adata.obs["perturbation"] = ["control", "control", "control", "JUN", "JUN", "JUN"]
    adata.var_names = ["g1", "g2", "g3"]

    deg_df = compute_deg_artifact(
        adata=adata,
        perturbation_col="perturbation",
        control_label="control",
        perturbation_names=["JUN"],
        adjusted_pvalue_threshold=1.0,
        abs_logfoldchange_threshold=0.0,
    )

    assert not deg_df.empty
    assert set(
        [
            "perturbation",
            "rank",
            "gene",
            "logfoldchange",
            "adjusted_p_value",
            "deg_significance",
        ]
    ).issubset(deg_df.columns)

    csv_path, metadata_path = save_deg_artifact(
        deg_df=deg_df,
        output_dir=tmp_path,
        metadata={"dataset_name": "test"},
    )
    assert csv_path.exists()
    assert metadata_path.exists()

    reloaded = load_deg_artifact(csv_path)
    assert len(reloaded) == len(deg_df)
