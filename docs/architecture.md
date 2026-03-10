# PerturbScope-GPT — Architecture

## Overview

PerturbScope-GPT is a local-first machine learning pipeline for predicting
single-cell perturbation responses. Given the control expression profile of a
cell and a perturbation gene label, the model predicts the **delta expression**
(perturbed minus control, per gene) across all highly variable genes.

```
raw .h5ad  →  preprocessing  →  bundle  →  training  →  artifacts  →  app
```

---

## Data Flow

### 1. Raw data

Input: `NormanWeissman2019_filtered.h5ad` (K562, ~111,000 cells before
filtering, single-gene and multi-gene conditions).

### 2. Preprocessing (`src/data/`)

| Step | Implementation |
|---|---|
| Filter cells and genes | min 200 genes/cell, 3 cells/gene |
| Normalize | library-size normalize to 10,000 counts/cell |
| Log-transform | `log1p` |
| HVG selection | top 512 highly variable genes (scanpy) |
| Sparse → dense | after HVG slicing, for training efficiency |
| Control mean | batch-aware (or global fallback) per perturbation condition |
| Delta expression | `perturbed_cell - control_mean` per gene |
| Split | seen (stratified within condition) / unseen (held-out conditions) |

Output: `data/processed/<bundle_name>/` — a `.npz` archive of control
expression, delta expression, perturbation indices, and associated metadata
(gene list, perturbation map, split indices).

### 3. Model training (`src/training/`)

Three model classes share the same training contract:

| Model | File |
|---|---|
| Transformer | `src/models/transformer.py` |
| MLP baseline | `src/models/mlp.py` |
| XGBoost baseline | `src/models/xgboost_baseline.py` |

Checkpoints and JSON metrics land in `artifacts/<run_name>/`.

### 4. Evaluation (`src/evaluation/`)

- Pearson correlation (per-perturbation mean)
- MSE (per-perturbation mean)
- Top-k DEG overlap: predicted top-k genes vs. Wilcoxon DEG results from scanpy

### 5. Streamlit app (`app/`)

Loads a saved checkpoint, runs aggregated inference for a selected perturbation,
and displays:
- predicted vs. observed delta expression scatter
- top predicted genes
- target ranking combining predicted delta with DEG significance
- top-k DEG overlap chart

---

## Transformer Model

**File:** `src/models/transformer.py` — `TransformerPerturbationModel`

### Architecture

```
Input: control_expression  [B, G]   (G = 512 HVGs)
       perturbation_index  [B]      (integer label)

Gene token construction (per gene g):
  gene_token[g] = gene_embedding[g]          # learnable gene ID embedding
               + value_encoder(expr[g])       # linear projection of scalar value
               + perturbation_embedding[p]    # perturbation conditioning (additive)

Encoder:
  2× TransformerEncoderLayer
     d_model=128, n_heads=4, ffn_dim=256
     activation=GELU, pre-norm (norm_first=True)

Output head (per gene):
  LayerNorm → Linear(128 → 1) → squeeze

Output: predicted delta_expression  [B, G]
```

### Design choices

- **No positional encoding** — gene order has no inherent meaning in scRNA-seq.
- **Additive perturbation injection** — the perturbation embedding is broadcast
  to all gene tokens; this gives every gene equal access to the perturbation
  identity without a dedicated `[CLS]` token.
- **Pre-norm Transformer** — `norm_first=True` improves gradient flow for
  shallow (2-layer) models.
- **Output target = delta expression** — simplifies loss (MSE on delta) and
  makes evaluation directly comparable across perturbations.

### Default hyperparameters (`configs/model.yaml`)

| Parameter | Value |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `ffn_dim` | 256 |
| `dropout` | 0.1 |
| `hvg_count` | 512 |

---

## Baseline Models

### MLP (`src/models/mlp.py`)

A 3-layer fully connected network. Input: concatenation of control expression
and a one-hot perturbation vector. Output: predicted delta expression.

### XGBoost (`src/models/xgboost_baseline.py`)

Gradient-boosted trees. Each gene's delta expression is predicted as a separate
regression target using the same concatenated input representation. Metrics are
written at train time.

---

## Repository Layout

```
src/
  data/         loading, preprocessing, pairing, PyTorch datasets
  models/       TransformerPerturbationModel, MLP, XGBoost wrapper
  training/     loss functions, training loop, checkpoint utilities
  evaluation/   Pearson/MSE metrics, DEG overlap, analysis helpers
  ranking/      target prioritization (predicted delta + DEG significance)
  utils/        config loader, logger, seed control, comparison utilities

scripts/        thin CLI entrypoints — call src/ only, no business logic
configs/        YAML runtime configuration (data, model, training)
app/            Streamlit UI — inference and visualization only
notebooks/      EDA and model comparison Jupyter notebooks
tests/          unit and integration tests (pytest, synthetic fixtures)
docs/           architecture notes, dataset notes, figures
```

---

## Evaluation Standards

| Metric | Granularity |
|---|---|
| Pearson correlation | per-perturbation mean |
| MSE | per-perturbation mean |
| Top-k DEG overlap | k = 20, 50, 100 |

Seen (in-distribution) and unseen (held-out conditions) splits are always
reported separately to measure generalization.

---

## Configuration

All runtime knobs are controlled through YAML files under `configs/`:

| File | Controls |
|---|---|
| `configs/data.yaml` | dataset paths, filtering, normalization, HVG count, pairing, split protocol |
| `configs/model.yaml` | architecture dimensions, HVG count, memory guardrails |
| `configs/train.yaml` | learning rate, batch size, epochs, seed, evaluation thresholds |
