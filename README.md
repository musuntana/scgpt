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

## Repository Documents

- `PROJECT_PLAN.md`: development plan and architecture decisions
- `AGENTS.md`: implementation constraints and anti-drift guardrails
- `configs/data.yaml`: data and preprocessing defaults
- `configs/model.yaml`: model and memory defaults
- `configs/train.yaml`: training, evaluation, and ranking defaults

## Recommended Environment

- Python `3.10` or `3.11`
- local machine first; GPU optional, CPU subset runs acceptable

## Next Build Steps

1. scaffold `src/`, `scripts/`, `tests/`, `data/`, and `app/`
2. implement preprocessing and pairing
3. export torch-ready datasets
4. train a minimal Transformer on the seen split
5. add evaluation, ranking, and the demo app
