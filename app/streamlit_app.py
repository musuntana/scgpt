from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import read_json
from src.data.pairing import load_processed_bundle
from src.evaluation.deg import DEG_ARTIFACT_FILENAME, DEG_METADATA_FILENAME, load_deg_artifact
from src.evaluation.error_analysis import (
    build_failure_mode_count_frame,
    build_worst_conditions_frame,
    select_perturbation_diagnostics,
)
from src.evaluation.inference import (
    build_gene_comparison_frame,
    build_perturbation_batch,
    load_torch_model_for_bundle,
    predict_delta_for_batch,
    summarize_perturbation_fit,
)
from src.evaluation.metrics import topk_overlap
from src.ranking.target_ranking import build_target_ranking
from src.utils.comparison import (
    plot_grouped_metric_bars,
    scan_artifact_comparison_rows,
    shorten_model_label,
)
from src.utils.config import load_yaml
from src.utils.multiseed import load_multiseed_report, select_multiseed_group

REAL_BUNDLE_DIR = "data/processed/norman2019_demo_bundle"
REAL_ARTIFACT_DIR = "artifacts/transformer_seen_norman2019_demo"
SYNTHETIC_BUNDLE_DIR = "data/processed/synthetic_demo_bundle"
SYNTHETIC_ARTIFACT_DIR = "artifacts/transformer_seen_synthetic_demo"
REAL_MODEL_CONFIG = "configs/model.yaml"
REAL_TRAIN_CONFIG = "configs/train.yaml"
SYNTHETIC_MODEL_CONFIG = "configs/model_synthetic_demo.yaml"
SYNTHETIC_TRAIN_CONFIG = "configs/train_synthetic_demo.yaml"


def resolve_default_demo_paths() -> tuple[str, str, str, str]:
    if Path(REAL_BUNDLE_DIR).exists() or Path(REAL_ARTIFACT_DIR).exists():
        return REAL_BUNDLE_DIR, REAL_ARTIFACT_DIR, REAL_MODEL_CONFIG, REAL_TRAIN_CONFIG
    if Path(SYNTHETIC_BUNDLE_DIR).exists() or Path(SYNTHETIC_ARTIFACT_DIR).exists():
        return (
            SYNTHETIC_BUNDLE_DIR,
            SYNTHETIC_ARTIFACT_DIR,
            SYNTHETIC_MODEL_CONFIG,
            SYNTHETIC_TRAIN_CONFIG,
        )
    return REAL_BUNDLE_DIR, REAL_ARTIFACT_DIR, REAL_MODEL_CONFIG, REAL_TRAIN_CONFIG


(
    DEFAULT_BUNDLE_DIR,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAIN_CONFIG,
) = resolve_default_demo_paths()


def infer_demo_mode(
    bundle_dir: str | Path,
    artifact_dir: str | Path,
    run_summary: dict | None = None,
) -> str:
    dataset = (run_summary or {}).get("dataset", {})
    candidates = [
        str(bundle_dir).lower(),
        str(artifact_dir).lower(),
        str(dataset.get("name", "")).lower(),
        str(dataset.get("source", "")).lower(),
        str(dataset.get("raw_path", "")).lower(),
    ]
    if any("synthetic" in candidate for candidate in candidates):
        return "synthetic"
    return "real"


def filter_rows_for_demo_mode(rows: list[dict], demo_mode: str) -> list[dict]:
    suffix = "_synthetic_demo" if demo_mode == "synthetic" else "_norman2019_demo"
    filtered = [row for row in rows if str(row["model"]).endswith(suffix)]
    return filtered or rows


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
def load_optional_list(path: str) -> list:
    """Load a JSON file that is expected to be a list (e.g. history.json)."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    import json
    with open(file_path) as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []

@st.cache_data(show_spinner=False)
def load_optional_multiseed_report(path: str) -> list[dict]:
    return load_multiseed_report(path)

@st.cache_data(show_spinner=False)
def load_optional_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)



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


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} ± {float(std):.4f}"


def display_error_summary(split_label: str, error_summary: dict, *, top_n: int = 4) -> None:
    st.write(split_label)
    if not error_summary:
        st.caption(f"No saved {split_label.lower()} error summary found.")
        return

    failure_mode_df = build_failure_mode_count_frame(error_summary)
    worst_pearson_df = build_worst_conditions_frame(
        error_summary,
        rank_by="worst_by_pearson",
        top_n=top_n,
    )
    worst_mse_df = build_worst_conditions_frame(
        error_summary,
        rank_by="worst_by_mse",
        top_n=top_n,
    )

    st.caption(
        f"{int(error_summary.get('num_perturbations', 0))} perturbations in saved "
        f"{split_label.lower()} diagnostics."
    )
    if not failure_mode_df.empty:
        st.write("Failure mode counts")
        st.dataframe(failure_mode_df, use_container_width=True, height=180)

    if not worst_pearson_df.empty:
        pearson_columns = [
            column
            for column in [
                "perturbation",
                "pearson",
                "sample_count",
                "failure_mode",
                "top_residual_genes",
            ]
            if column in worst_pearson_df.columns
        ]
        st.write("Worst Pearson")
        st.dataframe(
            worst_pearson_df.loc[:, pearson_columns].style.format(
                {"pearson": "{:.4f}"},
                na_rep="—",
            ),
            use_container_width=True,
            height=210,
        )

    if not worst_mse_df.empty:
        mse_columns = [
            column
            for column in [
                "perturbation",
                "mse",
                "sample_count",
                "failure_mode",
                "top_residual_genes",
            ]
            if column in worst_mse_df.columns
        ]
        st.write("Worst MSE")
        st.dataframe(
            worst_mse_df.loc[:, mse_columns].style.format(
                {"mse": "{:.4f}"},
                na_rep="—",
            ),
            use_container_width=True,
            height=210,
        )


def display_selected_split_diagnostics(
    split_label: str,
    perturbation_name: str,
    diagnostics: dict,
) -> None:
    st.write(split_label)
    if not diagnostics:
        st.caption(
            f"`{perturbation_name}` is not present in the saved {split_label.lower()} artifact."
        )
        return

    summary_df = pd.DataFrame(
        [
            {
                "Samples": diagnostics.get("sample_count"),
                "Pearson": diagnostics.get("pearson"),
                "MSE": diagnostics.get("mse"),
                "Failure Mode": diagnostics.get("failure_mode", "n/a"),
                "Error/Signal": diagnostics.get("error_to_signal_ratio"),
            }
        ]
    )
    st.dataframe(
        summary_df.style.format(
            {
                "Samples": "{:.0f}",
                "Pearson": "{:.4f}",
                "MSE": "{:.4f}",
                "Error/Signal": "{:.4f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        height=80,
    )
    top_residual_genes = diagnostics.get("top_residual_genes")
    if top_residual_genes:
        st.caption(f"Top residual genes: {top_residual_genes}")


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
        "--output-dir artifacts/transformer_seen_norman2019_demo`, "
        "or generate an offline demo with `./scripts/run_generate_synthetic_demo.sh`."
    )
    st.stop()

bundle = load_bundle_cached(str(bundle_dir))
run_summary = load_optional_json(str(run_summary_path))
deg_metadata = load_optional_json(str(deg_metadata_path))
deg_artifact = load_optional_deg_artifact(str(deg_artifact_path))
train_config = load_yaml(train_config_path)
history = load_optional_list(str(artifact_dir / "history.json"))
demo_mode = infer_demo_mode(bundle_dir, artifact_dir, run_summary)
multiseed_report = load_optional_multiseed_report(str(artifact_dir.parent / "multi_seed_report.json"))
multiseed_dataset_name = "synthetic_demo" if demo_mode == "synthetic" else "scperturb_norman2019"
transformer_multiseed = select_multiseed_group(
    multiseed_report,
    dataset_name=multiseed_dataset_name,
    train_protocol="seen",
    model_type="transformer",
)
seen_error_summary = load_optional_json(str(artifact_dir / "seen_test_error_summary.json"))
unseen_error_summary = load_optional_json(str(artifact_dir / "unseen_test_error_summary.json"))
seen_error_table = load_optional_csv(str(artifact_dir / "seen_test_per_perturbation.csv"))
unseen_error_table = load_optional_csv(str(artifact_dir / "unseen_test_per_perturbation.csv"))
model = load_model_cached(
    bundle_dir=str(bundle_dir),
    checkpoint_path=str(checkpoint_path),
    model_config_path=str(model_config_path),
    model_type=model_type,
)

if demo_mode == "synthetic":
    st.warning(
        "Currently showing the offline synthetic showcase. Use it to validate the pipeline and UI, "
        "not to make biological claims about real perturbation performance."
    )
else:
    st.info("Currently showing real Norman2019 demo artifacts.")

metadata = bundle["metadata"]
selected_perturbation = st.sidebar.selectbox(
    "Perturbation gene",
    options=metadata["perturbation_names"],
)
top_n = int(st.sidebar.slider("Top genes to display", min_value=10, max_value=30, value=15))

inference_tab, comparison_tab, history_tab = st.tabs(
    ["🔬 Inference", "📊 Model Comparison", "📈 Training History"]
)

with inference_tab:
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
 selected_seen_diagnostics = select_perturbation_diagnostics(
     seen_error_table,
     perturbation_name=selected_perturbation,
 )
 selected_unseen_diagnostics = select_perturbation_diagnostics(
     unseen_error_table,
     perturbation_name=selected_perturbation,
 )
 if selected_seen_diagnostics or selected_unseen_diagnostics:
     st.subheader("Stored Split Diagnostics")
     diag_left, diag_right = st.columns(2)
     with diag_left:
         display_selected_split_diagnostics(
             "Seen split",
             selected_perturbation,
             selected_seen_diagnostics,
         )
     with diag_right:
         display_selected_split_diagnostics(
             "Unseen split",
             selected_perturbation,
             selected_unseen_diagnostics,
         )
 if selected_deg_df.empty:
     st.warning(
         "No DEG rows were found for this perturbation. Ranking is currently prediction-only. "
         "Generate a DEG artifact with `./scripts/run_generate_deg_artifact.sh` and place it in the artifact directory."
     )
 else:
     deg_caption = f"Loaded {len(selected_deg_df)} DEG rows for this perturbation"
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

# ── Model Comparison tab ─────────────────────────────────────────────────────
with comparison_tab:
 st.subheader("Model Comparison — test-set metrics")
 if transformer_multiseed:
     st.markdown("#### Transformer Multi-Seed Stability")
     ms_col1, ms_col2, ms_col3, ms_col4 = st.columns(4)
     ms_col1.metric("Runs", int(transformer_multiseed.get("num_runs", 0)))
     ms_col2.metric(
         "Seen Pearson",
         format_mean_std(
             transformer_multiseed.get("seen_pearson_mean"),
             transformer_multiseed.get("seen_pearson_std"),
         ),
     )
     ms_col3.metric(
         "Unseen Pearson",
         format_mean_std(
             transformer_multiseed.get("unseen_pearson_mean"),
             transformer_multiseed.get("unseen_pearson_std"),
         ),
     )
     ms_col4.metric(
         "Unseen Top-100 DEG",
         format_mean_std(
             transformer_multiseed.get("unseen_top100_deg_mean"),
             transformer_multiseed.get("unseen_top100_deg_std"),
         ),
     )
     stability_prefix = (
         "Synthetic showcase stability only. "
         if demo_mode == "synthetic"
         else "Real Norman2019 Transformer stability across repeated seeds. "
     )
     artifact_labels = transformer_multiseed.get("artifact_labels") or []
     seeds = transformer_multiseed.get("seeds") or []
     stability_details: list[str] = []
     if artifact_labels:
         stability_details.append("Artifacts: " + ", ".join(str(label) for label in artifact_labels))
     if seeds:
         stability_details.append("Seeds: " + ", ".join(str(seed) for seed in seeds))
     st.caption(stability_prefix + (" ".join(stability_details) if stability_details else ""))
 if seen_error_summary or unseen_error_summary:
     st.markdown("#### Error-Analysis Highlights")
     error_caption_prefix = (
         "Synthetic showcase diagnostics only. "
         if demo_mode == "synthetic"
         else "Real Norman2019 diagnostics from saved split-level artifacts. "
     )
     st.caption(
         error_caption_prefix
         + "Failure modes are heuristic and intended for qualitative debugging only."
     )
     error_left, error_right = st.columns(2)
     with error_left:
         display_error_summary("Seen split", seen_error_summary)
     with error_right:
         display_error_summary("Unseen split", unseen_error_summary)
 artifact_root = str(artifact_dir.parent)
 rows = filter_rows_for_demo_mode(scan_artifact_comparison_rows(artifact_root), demo_mode)
 if not rows:
     st.warning(f"No run_summary.json files found under `{artifact_root}`.")
 else:
     df_cmp = pd.DataFrame(rows).set_index("model")

     # Numeric formatting for display
     display_cols = {
         "seen_pearson": "Seen Pearson",
         "seen_mse": "Seen MSE (per-pert)",
         "unseen_pearson": "Unseen Pearson",
         "unseen_mse": "Unseen MSE (per-pert)",
         "seen_top20_deg": "Seen Top-20 DEG",
         "seen_top100_deg": "Seen Top-100 DEG",
         "unseen_top20_deg": "Unseen Top-20 DEG",
         "unseen_top100_deg": "Unseen Top-100 DEG",
     }
     df_display = (
         df_cmp[[c for c in display_cols if c in df_cmp.columns]]
         .rename(columns=display_cols)
     )
     st.dataframe(df_display.style.format("{:.4f}", na_rep="—"), use_container_width=True)

     # Bar charts — Pearson and MSE
     pearson_rows = [
         row
         for row in rows
         if row.get("seen_pearson") is not None and row.get("unseen_pearson") is not None
     ]
     mse_rows = [
         row
         for row in rows
         if row.get("seen_mse") is not None and row.get("unseen_mse") is not None
     ]
     if pearson_rows or mse_rows:
         fig_cmp, (ax_p, ax_m) = plt.subplots(1, 2, figsize=(12, 4))
         if pearson_rows:
             pearson_labels = [shorten_model_label(row["model"]) for row in pearson_rows]
             plot_grouped_metric_bars(
                 ax_p,
                 pearson_labels,
                 [row["seen_pearson"] for row in pearson_rows],
                 [row["unseen_pearson"] for row in pearson_rows],
                 ylabel="Pearson (per-perturbation)",
                 title="Pearson Correlation",
                 annotate=len(pearson_labels) <= 6,
                 value_format="{:.3f}",
             )
         else:
             ax_p.set_title("Pearson Correlation")
             ax_p.text(0.5, 0.5, "Pearson metrics unavailable", ha="center", va="center")
             ax_p.set_axis_off()

         if mse_rows:
             mse_labels = [shorten_model_label(row["model"]) for row in mse_rows]
             plot_grouped_metric_bars(
                 ax_m,
                 mse_labels,
                 [row["seen_mse"] for row in mse_rows],
                 [row["unseen_mse"] for row in mse_rows],
                 ylabel="MSE (per-perturbation)",
                 title="MSE per Perturbation",
                 annotate=len(mse_labels) <= 6,
                 value_format="{:.4f}",
             )
         else:
             ax_m.set_title("MSE per Perturbation")
             ax_m.text(0.5, 0.5, "MSE metrics unavailable", ha="center", va="center")
             ax_m.set_axis_off()

         fig_cmp.tight_layout()
         st.pyplot(fig_cmp, use_container_width=True)
         plt.close(fig_cmp)

     caption_prefix = (
         "Synthetic showcase only. "
         if demo_mode == "synthetic"
         else "Real Norman2019 results only. "
     )
     st.caption(
         caption_prefix
         + "Seen split: stratified within each perturbation condition. "
         + "Unseen split: held-out perturbation genes not seen during training. "
         + "Metrics are per-perturbation (averaged over all genes for each condition)."
     )

# ── Training History tab ─────────────────────────────────────────────────────
with history_tab:
 st.subheader("Training History")
 if not history:
     st.info("No `history.json` found in the selected artifact directory.")
 else:
     hist_df = pd.DataFrame(history)
     fig_hist, (ax_loss, ax_pearson) = plt.subplots(1, 2, figsize=(12, 4))

     ax_loss.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", markersize=3,
                  label="train loss", color="steelblue")
     if "overall_mse" in hist_df.columns:
         ax_loss.plot(hist_df["epoch"], hist_df["overall_mse"], marker="s", markersize=3,
                      label="val MSE", color="coral", linestyle="--")
     ax_loss.set_xlabel("Epoch")
     ax_loss.set_ylabel("Loss / MSE")
     ax_loss.set_title("Train Loss & Val MSE")
     ax_loss.legend(fontsize=8)

     ax_pearson.plot(hist_df["epoch"], hist_df["pearson_per_perturbation"],
                     marker="o", markersize=3, color="seagreen")
     ax_pearson.set_xlabel("Epoch")
     ax_pearson.set_ylabel("Pearson (per-perturbation)")
     ax_pearson.set_title("Validation Pearson per Perturbation")

     fig_hist.tight_layout()
     st.pyplot(fig_hist, use_container_width=True)
     plt.close(fig_hist)

     st.dataframe(
         hist_df.style.format("{:.5f}", subset=[c for c in hist_df.columns if c != "epoch"])
                      .format("{:.0f}", subset=["epoch"]),
         use_container_width=True,
         height=340,
     )
     if run_summary:
         val = run_summary.get("validation", {})
         best_ep = val.get("best_epoch")
         if best_ep:
             st.caption(f"Best checkpoint saved at epoch {best_ep}.")
