from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pairing import load_processed_bundle


st.set_page_config(page_title="PerturbScope-GPT", layout="wide")
st.title("PerturbScope-GPT")
st.caption("Local-first single-cell perturbation response demo")

bundle_dir = Path(
    st.sidebar.text_input(
        "Processed bundle directory",
        value="data/processed/demo_bundle",
    )
)

if not bundle_dir.exists():
    st.info("No processed bundle found yet. Run `scripts/preprocess_data.py` first.")
    st.stop()

bundle = load_processed_bundle(bundle_dir)
metadata = bundle["metadata"]

st.subheader("Bundle Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Samples", len(bundle["perturbation_index"]))
col2.metric("Genes", len(metadata["gene_names"]))
col3.metric("Perturbations", len(metadata["perturbation_names"]))

st.subheader("Available Perturbations")
st.dataframe(metadata["perturbation_names"], use_container_width=True)

st.subheader("Next Step")
st.write(
    "This UI is ready to consume trained artifacts. The next implementation step is to "
    "load a saved model checkpoint, run inference for a selected perturbation, and show "
    "predicted delta expression plus DEG-based target rankings."
)
