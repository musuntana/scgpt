# PerturbScope-GPT

PerturbScope-GPT is a local-first AI4Bio project for single-cell perturbation response prediction and target prioritization.

## Project Goal

Build a job-ready MVP that can:
- preprocess public Perturb-seq data into a reproducible training set
- predict perturbation-induced expression changes with a Transformer
- compare against MLP and XGBoost baselines
- produce DEG-based target rankings
- expose inference results in a Streamlit demo

## Local-First MVP Scope

- dataset: `scPerturb / Norman2019`
- sample scope: single-gene perturbations only
- target: `delta expression`
- primary metric: `per-perturbation Pearson`
- default HVG count: `512`

## Environment

This repository now uses `uv` as the default environment and dependency manager.

- Python version: `3.11`
- local virtual environment: `.venv`
- local uv cache: `.uv-cache`
- dependency source of truth: [`pyproject.toml`](/Users/musun/Desktop/scgpt/pyproject.toml)
- locked dependency snapshot: [`uv.lock`](/Users/musun/Desktop/scgpt/uv.lock)

### Bootstrap

Create the local development environment:

```bash
./scripts/bootstrap_env.sh
```

This will:
- create `.venv` in the current repository
- install Python `3.11` if `uv` needs to fetch it
- sync runtime and dev dependencies with `uv`

### Activate

```bash
source .venv/bin/activate
```

You can also avoid manual activation and run everything through `uv run` or the shell wrappers in `scripts/`.

## Main Commands

Run tests:

```bash
./scripts/run_tests.sh
```

Start the Streamlit app:

```bash
./scripts/run_app.sh
```

The app defaults to the real local demo artifacts:
- bundle: `data/processed/norman2019_demo_bundle`
- artifact dir: `artifacts/transformer_seen_norman2019_demo`
- checkpoint: `<artifact dir>/best_model.pt`

Current app behavior:
- load a saved torch checkpoint
- select a perturbation gene from the processed bundle
- run aggregated inference for that perturbation
- if `deg_artifact.csv` exists in the artifact directory, combine predicted delta with real DEG significance
- show predicted vs observed delta, top predicted genes, true DEG rows, target ranking, and top-k DEG overlap

Preprocess a dataset bundle:

```bash
./scripts/run_preprocess_demo.sh \
  --input-path data/raw/your_dataset.h5ad \
  --output-dir data/processed/demo_bundle
```

Download the default `Norman2019` dataset:

```bash
./scripts/download_norman2019.sh
```

Inspect AnnData schema and auto-resolved columns:

```bash
./scripts/run_inspect_anndata.sh \
  --input-path data/raw/NormanWeissman2019_filtered.h5ad \
  --output-json data/interim/norman2019_schema.json
```

Run the full Norman2019 local demo flow:

```bash
./scripts/run_norman2019_demo.sh
```

Train the Transformer:

```bash
./scripts/run_train_transformer.sh \
  --bundle-dir data/processed/demo_bundle \
  --output-dir artifacts/transformer_seen
```

Train baselines:

```bash
./scripts/run_train_baselines.sh \
  --bundle-dir data/processed/demo_bundle \
  --output-dir artifacts/baselines \
  --baseline mlp
```

The baseline output directory stores split-specific metrics:
- `mlp_seen_test_metrics.json` and `mlp_unseen_test_metrics.json`
- `xgboost_seen_test_metrics.json` and `xgboost_unseen_test_metrics.json`
- `xgboost_model.joblib` and `xgboost_run_summary.json` for the tree baseline

Generate a real DEG artifact for app ranking:

```bash
./scripts/run_generate_deg_artifact.sh \
  --input-path data/raw/NormanWeissman2019_filtered.h5ad \
  --bundle-dir data/processed/norman2019_demo_bundle \
  --output-dir artifacts/transformer_seen_norman2019_demo
```

Evaluate a saved model:

```bash
./scripts/run_evaluate.sh \
  --bundle-dir data/processed/demo_bundle \
  --checkpoint-path artifacts/transformer_seen/best_model.pt \
  --model-type transformer \
  --output-path artifacts/transformer_seen/test_metrics.json \
  --deg-artifact-path artifacts/transformer_seen/deg_artifact.csv
```

When `--deg-artifact-path` is provided, the evaluation also computes top-k DEG overlap metrics.

Write a structured local run summary:

```bash
./scripts/run_summarize_run.sh \
  --bundle-dir data/processed/demo_bundle \
  --output-dir artifacts/transformer_seen \
  --checkpoint-path artifacts/transformer_seen/best_model.pt \
  --model-type transformer \
  --split-prefix seen \
  --seen-metrics-path artifacts/transformer_seen/seen_test_metrics.json \
  --unseen-metrics-path artifacts/transformer_seen/unseen_test_metrics.json
```

## Results

This repository already includes one complete local run on the real `Norman2019` demo bundle:
- bundle: `10500` samples, `256` genes, `105` perturbations
- primary metric: `pearson_per_perturbation`
- artifact summaries:
  - [`artifacts/transformer_seen_norman2019_demo/run_summary.json`](/Users/musun/Desktop/scgpt/artifacts/transformer_seen_norman2019_demo/run_summary.json)
  - [`artifacts/mlp_seen_norman2019_demo/run_summary.json`](/Users/musun/Desktop/scgpt/artifacts/mlp_seen_norman2019_demo/run_summary.json)
  - [`artifacts/xgboost_seen_norman2019_demo/xgboost_run_summary.json`](/Users/musun/Desktop/scgpt/artifacts/xgboost_seen_norman2019_demo/xgboost_run_summary.json)

### Model Comparison

![Norman2019 demo bundle model comparison](docs/assets/model_comparison_seen_norman2019_demo.png)

| Model | Best Val Pearson | Seen Test Pearson | Seen Test MSE | Unseen Test Pearson | Unseen Test MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 0.6430 | 0.6523 | 0.0518 | 0.8652 | 0.0503 |
| MLP | 0.6366 | 0.6420 | 0.0520 | 0.8706 | 0.0503 |
| XGBoost | n/a | 0.6285 | 0.0526 | 0.8213 | 0.0494 |

Interpretation:
- Transformer is the strongest model on the main `seen_test` metric.
- MLP is close to Transformer and serves as a strong low-complexity baseline.
- XGBoost is competitive on MSE but trails on the main `per-perturbation Pearson` metric.
- `unseen_test` is easier than `seen_test` on this demo bundle, so this should be reported carefully.

### Streamlit Preview

![Transformer inference preview](docs/assets/transformer_inference_preview.png)

The preview above is generated from the same checkpoint and bundle that the Streamlit app loads by default.
When `artifacts/transformer_seen_norman2019_demo/deg_artifact.csv` exists, the ranking panel uses
`predicted_delta + DEG significance` instead of prediction-only scoring.
Launch the interactive app locally with:

```bash
./scripts/run_app.sh
```

Regenerate the README result assets with:

```bash
./scripts/run_generate_results_assets.sh
```

## Repository Documents

- [`PROJECT_PLAN.md`](/Users/musun/Desktop/scgpt/PROJECT_PLAN.md): development plan and architecture decisions
- [`AGENTS.md`](/Users/musun/Desktop/scgpt/AGENTS.md): implementation constraints and anti-drift guardrails
- [`pyproject.toml`](/Users/musun/Desktop/scgpt/pyproject.toml): uv project definition and dependency source of truth
- [`configs/data.yaml`](/Users/musun/Desktop/scgpt/configs/data.yaml): data and preprocessing defaults
- [`configs/model.yaml`](/Users/musun/Desktop/scgpt/configs/model.yaml): model and memory defaults
- [`configs/train.yaml`](/Users/musun/Desktop/scgpt/configs/train.yaml): training, evaluation, and ranking defaults

## Recommended Workflow

1. bootstrap the environment with `./scripts/bootstrap_env.sh`
2. place a `.h5ad` file under `data/raw/`
3. optionally inspect schema resolution with `./scripts/run_inspect_anndata.sh`
4. preprocess it into a bundle under `data/processed/`
5. train a minimal Transformer on the seen split
6. evaluate seen and unseen metrics
7. write `run_summary.json` for the completed experiment
8. inspect ranking outputs or start Streamlit for demo artifacts

## Notes

- `requirements.txt` is kept as a compatibility fallback; update `pyproject.toml` first and refresh `uv.lock` with `uv sync`.
- the shell wrappers export `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, and `PYTHONPATH` so the project can be run consistently from the repo root
- if you change runtime dependencies, resync with `./scripts/bootstrap_env.sh`
- each meaningful local training run should keep `history.json`, `best_model.pt`, evaluation JSONs, and `run_summary.json` in the same artifact directory
