from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import read_json
from src.data.pairing import load_processed_bundle
from src.evaluation.deg import DEG_ARTIFACT_FILENAME, DEG_METADATA_FILENAME, load_deg_artifact
from src.evaluation.inference import (
    build_gene_comparison_frame,
    build_perturbation_batch,
    load_torch_model_for_bundle,
    predict_delta_for_batch,
    summarize_perturbation_fit,
)
from src.evaluation.metrics import topk_overlap
from src.ranking.target_ranking import build_target_ranking
from src.utils.config import load_yaml

DEFAULT_BUNDLE_DIR = "data/processed/norman2019_demo_bundle"
DEFAULT_ARTIFACT_DIR = "artifacts/transformer_seen_norman2019_demo"
DEFAULT_MODEL_CONFIG = "configs/model.yaml"
DEFAULT_TRAIN_CONFIG = "configs/train.yaml"


@st.cache_data(show_spinner=False)
def load_bundle_cached(bundle_dir: str) -> dict:
    return load_processed_bundle(bundle_dir)


@st.cache_data(show_spinner=False)
def load_optional_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return read_json(file_path)


@st.cache_data(show_spinner=False)
def load_optional_deg_artifact(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return load_deg_artifact(file_path)


@st.cache_resource(show_spinner=False)
def load_model_cached(
    bundle_dir: str,
    checkpoint_path: str,
    model_config_path: str,
    model_type: str,
):
    bundle = load_processed_bundle(bundle_dir)
    return load_torch_model_for_bundle(
        bundle=bundle,
        checkpoint_path=checkpoint_path,
        model_config_path=model_config_path,
        model_type=model_type,
    )


def plot_prediction_scatter(comparison_df: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        comparison_df["observed_delta"],
        comparison_df["predicted_delta"],
        alpha=0.55,
        s=12,
    )
    axis.axline((0, 0), slope=1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Observed mean delta")
    axis.set_ylabel("Predicted delta")
    axis.set_title("Predicted vs Observed Delta")
    return figure


def top_gene_frame(comparison_df: pd.DataFrame, column: str, ascending: bool, top_n: int) -> pd.DataFrame:
    return (
        comparison_df.sort_values(column, ascending=ascending)
        .loc[:, ["gene", "predicted_delta", "observed_delta", "residual"]]
        .head(top_n)
        .reset_index(drop=True)
    )


st.set_page_config(page_title="PerturbScope-GPT", layout="wide")
st.title("PerturbScope-GPT")
st.caption("Local-first single-cell perturbation response demo")

bundle_dir = Path(
    st.sidebar.text_input(
        "Processed bundle directory",
        value=DEFAULT_BUNDLE_DIR,
    )
)
artifact_dir = Path(
    st.sidebar.text_input(
        "Artifact directory",
        value=DEFAULT_ARTIFACT_DIR,
    )
)
model_type = st.sidebar.selectbox("Torch model type", options=["transformer", "mlp"], index=0)
model_config_path = Path(
    st.sidebar.text_input(
        "Model config path",
        value=DEFAULT_MODEL_CONFIG,
    )
)
train_config_path = Path(
    st.sidebar.text_input(
        "Train config path",
        value=DEFAULT_TRAIN_CONFIG,
    )
)

checkpoint_path = artifact_dir / "best_model.pt"
run_summary_path = artifact_dir / "run_summary.json"
deg_artifact_path = artifact_dir / DEG_ARTIFACT_FILENAME
deg_metadata_path = artifact_dir / DEG_METADATA_FILENAME

if not bundle_dir.exists():
    st.info("Processed bundle is missing. Run `./scripts/run_norman2019_demo.sh` first.")
    st.stop()

if not checkpoint_path.exists():
    st.info(
        "Checkpoint is missing. Train a torch model first, for example "
        "`./scripts/run_train_transformer.sh --bundle-dir data/processed/norman2019_demo_bundle "
        "--output-dir artifacts/transformer_seen_norman2019_demo`."
    )
    st.stop()

bundle = load_bundle_cached(str(bundle_dir))
run_summary = load_optional_json(str(run_summary_path))
deg_metadata = load_optional_json(str(deg_metadata_path))
deg_artifact = load_optional_deg_artifact(str(deg_artifact_path))
train_config = load_yaml(train_config_path)
model = load_model_cached(
    bundle_dir=str(bundle_dir),
    checkpoint_path=str(checkpoint_path),
    model_config_path=str(model_config_path),
    model_type=model_type,
)

metadata = bundle["metadata"]
selected_perturbation = st.sidebar.selectbox(
    "Perturbation gene",
    options=metadata["perturbation_names"],
)
top_n = int(st.sidebar.slider("Top genes to display", min_value=10, max_value=30, value=15))

st.subheader("Artifact Summary")
left, center, right = st.columns(3)
left.metric("Samples", len(bundle["perturbation_index"]))
center.metric("Genes", len(metadata["gene_names"]))
right.metric("Perturbations", len(metadata["perturbation_names"]))

if run_summary:
    dataset = run_summary.get("dataset", {})
    validation = run_summary.get("validation", {})
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Cell Context", dataset.get("cell_context", "unknown"))
    metrics_col2.metric("Best Val Epoch", validation.get("best_epoch", "n/a"))
    best_validation = validation.get("best_validation", {})
    metrics_col3.metric(
        "Best Val Pearson",
        f"{best_validation.get('pearson_per_perturbation', 0.0):.4f}",
    )

deg_col1, deg_col2, deg_col3 = st.columns(3)
deg_col1.metric("DEG Artifact", "loaded" if not deg_artifact.empty else "missing")
deg_col2.metric(
    "DEG Rows",
    int(len(deg_artifact)) if not deg_artifact.empty else 0,
)
deg_col3.metric(
    "Ranking Mode",
    "predicted + DEG" if not deg_artifact.empty else "prediction-only",
)

batch = build_perturbation_batch(bundle, selected_perturbation)
predicted_delta = predict_delta_for_batch(model, batch)
comparison_df = build_gene_comparison_frame(
    gene_names=metadata["gene_names"],
    predicted_delta=predicted_delta,
    observed_delta=batch.observed_delta_mean,
)
fit_metrics = summarize_perturbation_fit(
    predicted_delta=predicted_delta,
    observed_delta=batch.observed_delta_mean,
)
selected_deg_df = pd.DataFrame()
if not deg_artifact.empty:
    selected_deg_df = deg_artifact[deg_artifact["perturbation"] == selected_perturbation].copy()

ranking_config = train_config.get("ranking", {})
if selected_deg_df.empty:
    abs_predicted_delta_weight = 1.0
    deg_significance_weight = 0.0
else:
    abs_predicted_delta_weight = float(ranking_config.get("abs_predicted_delta_weight", 0.5))
    deg_significance_weight = float(ranking_config.get("deg_significance_weight", 0.5))

ranking_df = build_target_ranking(
    gene_names=metadata["gene_names"],
    predicted_delta=predicted_delta,
    deg_df=selected_deg_df,
    abs_predicted_delta_weight=abs_predicted_delta_weight,
    deg_significance_weight=deg_significance_weight,
)

# Compute top-k overlap when DEG artifact is available
topk_k = 20
topk_overlap_value = None
if not selected_deg_df.empty:
    predicted_top_genes = (
        comparison_df.sort_values("abs_predicted_delta", ascending=False)["gene"]
        .tolist()[:topk_k]
    )
    true_top_genes = selected_deg_df["gene"].tolist()[:topk_k]
    if true_top_genes:
        topk_overlap_value = topk_overlap(predicted_top_genes, true_top_genes, topk_k)

st.subheader(f"Inference: {selected_perturbation}")
if topk_overlap_value is not None:
    metric1, metric2, metric3, metric4 = st.columns(4)
else:
    metric1, metric2, metric3 = st.columns(3)
    metric4 = None
metric1.metric("Matched Samples", batch.sample_count)
metric2.metric("Aggregated Pearson", f"{fit_metrics['pearson']:.4f}")
metric3.metric("Aggregated MSE", f"{fit_metrics['mse']:.4f}")
if metric4 is not None and topk_overlap_value is not None:
    metric4.metric(f"Top-{topk_k} DEG Overlap", f"{topk_overlap_value:.4f}")

st.caption(
    "Reference control is the mean matched control profile already stored in the processed bundle "
    "for the selected perturbation. Observed delta is the mean target delta across bundle samples "
    "for this perturbation."
)
if selected_deg_df.empty:
    st.warning(
        "No DEG rows were found for this perturbation. Ranking is currently prediction-only. "
        "Generate a DEG artifact with `./scripts/run_generate_deg_artifact.sh` and place it in the artifact directory."
    )
else:
    deg_caption = (
        f"Loaded {len(selected_deg_df)} DEG rows for this perturbation"
    )
    if deg_metadata:
        deg_config = deg_metadata.get("deg", {})
        deg_caption += (
            f" using method={deg_config.get('method', 'unknown')}, "
            f"adj_p<{deg_config.get('adjusted_pvalue_threshold', 'n/a')}, "
            f"|logFC|>{deg_config.get('abs_logfoldchange_threshold', 'n/a')}"
        )
    st.caption(deg_caption)

plot_col, ranking_col = st.columns([3, 2])
with plot_col:
    scatter_figure = plot_prediction_scatter(comparison_df)
    st.pyplot(scatter_figure, use_container_width=True)
    plt.close(scatter_figure)

with ranking_col:
    st.write("Top Target Ranking")
    st.dataframe(
        ranking_df.loc[
            :,
            ["rank", "gene", "predicted_delta", "deg_significance", "importance_score"],
        ].head(top_n),
        use_container_width=True,
        height=460,
    )

up_col, down_col = st.columns(2)
with up_col:
    st.write("Top Predicted Up Genes")
    st.dataframe(
        top_gene_frame(comparison_df, column="predicted_delta", ascending=False, top_n=top_n),
        use_container_width=True,
        height=380,
    )

with down_col:
    st.write("Top Predicted Down Genes")
    st.dataframe(
        top_gene_frame(comparison_df, column="predicted_delta", ascending=True, top_n=top_n),
        use_container_width=True,
        height=380,
    )

if not selected_deg_df.empty:
    st.subheader("True DEG Artifact")
    st.dataframe(
        selected_deg_df.loc[
            :,
            [
                "rank",
                "gene",
                "logfoldchange",
                "adjusted_p_value",
                "deg_significance",
                "score",
            ],
        ].head(50),
        use_container_width=True,
        height=420,
    )

st.subheader("Gene-Level Comparison")
st.dataframe(
    comparison_df.loc[
        :,
        [
            "gene",
            "predicted_delta",
            "observed_delta",
            "residual",
            "abs_predicted_delta",
            "abs_residual",
        ],
    ],
    use_container_width=True,
    height=520,
)
