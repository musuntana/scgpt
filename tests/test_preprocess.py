from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import src.data.preprocess as preprocess
from src.data.pairing import create_split_indices
from src.data.preprocess import (
    extend_hvg_mask_with_target_genes,
    select_hvgs_from_training_partitions,
)


def _make_adata() -> ad.AnnData:
    rng = np.random.default_rng(7)
    perturbations = ["control"] * 8
    obs_names = [f"control-{index}" for index in range(8)]
    for perturbation in ("A", "B", "C", "D", "E"):
        perturbations.extend([perturbation] * 10)
        obs_names.extend(f"{perturbation}-{index}" for index in range(10))

    return ad.AnnData(
        X=rng.normal(size=(len(perturbations), 8)).astype(np.float32),
        obs=pd.DataFrame(
            {"perturbation": perturbations},
            index=pd.Index(obs_names, name="sample_id"),
        ),
        var=pd.DataFrame(index=[f"GENE_{index}" for index in range(8)]),
    )


def test_hvg_selection_excludes_heldout_perturbation_samples(monkeypatch):
    adata = _make_adata()
    fitted_sample_ids: list[str] = []

    def fake_filter_genes(fit_adata, min_cells: int) -> None:
        expressed_cells = np.count_nonzero(np.asarray(fit_adata.X), axis=0)
        fit_adata._inplace_subset_var(expressed_cells >= min_cells)

    def fake_highly_variable_genes(fit_adata, n_top_genes: int) -> None:
        fitted_sample_ids.extend(fit_adata.obs_names.astype(str).tolist())
        variances = np.var(np.asarray(fit_adata.X), axis=0)
        selected = np.zeros(fit_adata.n_vars, dtype=bool)
        selected[np.argsort(variances)[-n_top_genes:]] = True
        fit_adata.var["highly_variable"] = selected

    monkeypatch.setattr(
        preprocess,
        "_require_scanpy",
        lambda: SimpleNamespace(
            pp=SimpleNamespace(
                filter_genes=fake_filter_genes,
                normalize_total=lambda fit_adata, target_sum: None,
                log1p=lambda fit_adata: None,
                highly_variable_genes=fake_highly_variable_genes,
            )
        ),
    )

    val_fraction = 0.2
    test_fraction = 0.2
    random_seed = 42
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    control_positions = np.flatnonzero(labels == "control")
    perturbed_positions = np.flatnonzero(labels != "control")
    perturbation_names = sorted(np.unique(labels[perturbed_positions]).tolist())
    perturbation_to_index = {
        name: index for index, name in enumerate(perturbation_names)
    }
    perturbation_index = np.asarray(
        [perturbation_to_index[name] for name in labels[perturbed_positions]],
        dtype=np.int64,
    )
    splits = create_split_indices(
        perturbation_index=perturbation_index,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    shared_train = np.intersect1d(
        splits["seen_train"], splits["unseen_train"], assume_unique=True
    )
    expected_training_ids = {
        str(adata.obs_names[perturbed_positions[index]]) for index in shared_train
    }
    expected_control_ids = {str(adata.obs_names[index]) for index in control_positions}

    # This gene is expressed only in samples held out by at least one protocol.
    # A global min-cells filter would retain it; a training-only filter must drop it.
    adata.X[:, -1] = 0.0
    heldout_ids = set(adata.obs_names.astype(str)) - expected_training_ids - expected_control_ids
    for row_index, sample_id in enumerate(adata.obs_names.astype(str)):
        if sample_id in heldout_ids:
            adata.X[row_index, -1] = 5.0

    result = select_hvgs_from_training_partitions(
        adata=adata,
        preprocess_config={"hvg_top_genes": 3},
        perturbation_col="perturbation",
        control_label="control",
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    assert set(fitted_sample_ids) == expected_training_ids | expected_control_ids
    assert not (set(fitted_sample_ids) & (set(adata.obs_names) - expected_training_ids - expected_control_ids))
    assert result.n_obs == adata.n_obs
    assert result.n_vars == 3
    assert "GENE_7" in adata.var_names
    assert "GENE_7" not in result.var_names

    diagnostics = result.uns["scgpt_preprocessing"]
    assert diagnostics["hvg_selection_method"] == "shared_seen_unseen_training_intersection"
    assert diagnostics["hvg_fit_perturbed_sample_count"] == len(shared_train)
    assert diagnostics["hvg_fit_control_sample_count"] == len(control_positions)
    assert diagnostics["heldout_perturbation_samples_excluded_from_hvg_fit"] is True
    assert diagnostics["protocols_protected"] == ["seen", "unseen"]


def test_hvg_selection_rejects_non_positive_gene_count(monkeypatch):
    monkeypatch.setattr(
        preprocess,
        "_require_scanpy",
        lambda: SimpleNamespace(pp=SimpleNamespace()),
    )

    with pytest.raises(ValueError, match="hvg_top_genes must be positive"):
        select_hvgs_from_training_partitions(
            adata=_make_adata(),
            preprocess_config={"hvg_top_genes": 0},
            perturbation_col="perturbation",
            control_label="control",
            val_fraction=0.2,
            test_fraction=0.2,
            random_seed=42,
        )


def test_target_aware_hvg_mask_retains_known_target_gene_names():
    mask, diagnostics = extend_hvg_mask_with_target_genes(
        gene_names=["A", "B", "C", "D"],
        hvg_mask=np.array([False, False, True, False]),
        perturbation_names=["A", "B"],
        max_panel_genes=3,
    )

    assert mask.tolist() == [True, True, True, False]
    assert diagnostics["known_target_gene_count"] == 2
    assert diagnostics["target_genes_added_to_hvgs"] == 2
    assert diagnostics["feature_panel_gene_count"] == 3
    assert diagnostics["heldout_perturbation_expression_used_for_target_retention"] is False


@pytest.mark.parametrize(
    ("names", "targets", "budget", "message"),
    [
        (["A", "B"], ["A", "C"], 3, "Unmapped targets"),
        (["A", "A", "B"], ["A"], 3, "ambiguous targets"),
        (["A", "B", "C"], ["A", "B"], 2, "exceeds hvg_upper_bound"),
        (["A", "B", "B"], ["A"], 3, "ambiguous duplicate gene names"),
    ],
)
def test_target_aware_hvg_mask_fails_closed_for_unmappable_or_oversized_panel(
    names, targets, budget, message
):
    selected = np.zeros(len(names), dtype=bool)
    selected[-2:] = True
    with pytest.raises(ValueError, match=message):
        extend_hvg_mask_with_target_genes(
            gene_names=names,
            hvg_mask=selected,
            perturbation_names=targets,
            max_panel_genes=budget,
        )


def test_target_aware_training_hvg_panel_preserves_hvg_semantics_and_target_coverage(
    monkeypatch,
):
    adata = _make_adata()
    adata.var_names = ["A", "B", "C", "D", "E", "GENE_5", "GENE_6", "GENE_7"]

    def fake_filter_genes(fit_adata, min_cells):
        expressed = np.count_nonzero(np.asarray(fit_adata.X), axis=0)
        fit_adata._inplace_subset_var(expressed >= min_cells)

    def fake_hvgs(fit_adata, n_top_genes):
        selected = np.zeros(fit_adata.n_vars, dtype=bool)
        selected[-n_top_genes:] = True
        fit_adata.var["highly_variable"] = selected

    monkeypatch.setattr(
        preprocess,
        "_require_scanpy",
        lambda: SimpleNamespace(
            pp=SimpleNamespace(
                filter_genes=fake_filter_genes,
                normalize_total=lambda _adata, target_sum: None,
                log1p=lambda _adata: None,
                highly_variable_genes=fake_hvgs,
            )
        ),
    )

    result = select_hvgs_from_training_partitions(
        adata=adata,
        preprocess_config={
            "hvg_top_genes": 2,
            "hvg_upper_bound": 7,
            "retain_perturbation_target_genes": True,
        },
        perturbation_col="perturbation",
        control_label="control",
        val_fraction=0.2,
        test_fraction=0.2,
        random_seed=42,
    )

    assert result.n_vars == 7
    assert set("ABCDE").issubset(set(result.var_names))
    assert int(result.var["highly_variable"].sum()) == 2
    assert bool(result.var["scgpt_feature_selected"].all())
    diagnostics = result.uns["scgpt_preprocessing"]
    assert diagnostics["schema_version"] == 2
    assert diagnostics["hvg_selected_gene_count"] == 2
    assert diagnostics["target_genes_added_to_hvgs"] == 5
    assert diagnostics["target_genes_retained"] == 5
    assert diagnostics["feature_panel_gene_count"] == 7


def test_target_aware_preprocessing_blocks_gene_removed_by_training_filter(monkeypatch):
    adata = _make_adata()
    adata.var_names = ["A", "B", "C", "D", "E", "GENE_5", "GENE_6", "GENE_7"]
    adata.X[:, 0] = 0.0

    def fake_filter_genes(fit_adata, min_cells):
        expressed = np.count_nonzero(np.asarray(fit_adata.X), axis=0)
        fit_adata._inplace_subset_var(expressed >= min_cells)

    monkeypatch.setattr(
        preprocess,
        "_require_scanpy",
        lambda: SimpleNamespace(
            pp=SimpleNamespace(
                filter_genes=fake_filter_genes,
                normalize_total=lambda _adata, target_sum: None,
                log1p=lambda _adata: None,
                highly_variable_genes=lambda fit_adata, n_top_genes: fit_adata.var.__setitem__(
                    "highly_variable",
                    np.asarray(
                        [False] * (fit_adata.n_vars - n_top_genes)
                        + [True] * n_top_genes,
                        dtype=bool,
                    ),
                ),
            )
        ),
    )

    with pytest.raises(ValueError, match="Unmapped targets"):
        select_hvgs_from_training_partitions(
            adata=adata,
            preprocess_config={
                "hvg_top_genes": 2,
                "retain_perturbation_target_genes": True,
            },
            perturbation_col="perturbation",
            control_label="control",
            val_fraction=0.2,
            test_fraction=0.2,
            random_seed=42,
        )
