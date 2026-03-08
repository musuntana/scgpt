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

Evaluate a saved model:

```bash
./scripts/run_evaluate.sh \
  --bundle-dir data/processed/demo_bundle \
  --checkpoint-path artifacts/transformer_seen/best_model.pt \
  --model-type transformer \
  --output-path artifacts/transformer_seen/test_metrics.json
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
6. evaluate metrics and inspect ranking outputs
7. start Streamlit to inspect the bundle or future inference artifacts

## Notes

- `requirements.txt` is kept as a compatibility fallback; update `pyproject.toml` first and refresh `uv.lock` with `uv sync`.
- the shell wrappers export `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, and `PYTHONPATH` so the project can be run consistently from the repo root
- if you change runtime dependencies, resync with `./scripts/bootstrap_env.sh`
