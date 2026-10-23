# Changelog

All notable changes to PerturbScope-GPT are documented here.

---

## [Unreleased]

- Training scripts now evaluate only the test split matching the selected training protocol; the generic evaluator rejects checkpoint/split protocol mismatches.
- Run summaries and model-comparison assets no longer mix seen/unseen test metrics from a single checkpoint; comparison figures use one matching protocol.
- Moved CLI gene filtering and HVG selection to the intersection of seen/unseen training perturbation samples plus the shared control reference; persisted feature-selection provenance in processed bundles and run summaries.
- Added regression tests proving that perturbation samples held out by either split protocol do not participate in feature selection.
- Added validation for split fractions, explicit fallback logging for infeasible stratification, and edge-case handling for one-sided validation/test splits.
- Added sample-coverage, split-disjointness, and unseen-group integrity regression tests.
- Documented limitations in the current unseen-perturbation evaluation protocol and qualified historical README metrics.
- Added backward-compatible experiment provenance to run summaries: Git state, runtime versions, split/seed details, and SHA-256 fingerprints for data, bundle, configs, and checkpoints.
- Documented how to compare runs and interpret missing provenance inputs.
- Fixed the seen-split stratification guard and added deterministic class-balance regression tests.
- Reduced CI duplication by validating code-relevant pull requests and supporting manual workflow dispatch; docs-only changes no longer trigger CI.

---

## [Reproducibility & analysis] — 2026-03

**Result robustness and failure-analysis pass**

- Added optional `--seed` overrides to training and run-summary scripts so repeated experiments can be recorded without editing YAML
- Added `scripts/multi_seed_report.py` and `scripts/run_multi_seed_report.sh` to aggregate repeated runs into mean/std summaries
- Extended `scripts/evaluate_model.py` with perturbation-level CSV exports and `*_error_summary.json` outputs for worst-case conditions
- Added heuristic perturbation failure modes (low signal, magnitude mismatch, high residual) to support qualitative debugging
- Updated `scripts/run_full_evaluation.sh` to emit per-split perturbation-level error artifacts automatically
- Tightened the real-vs-synthetic boundary in the Streamlit app and README so offline showcase artifacts are not confused with real biological results

---
## [Polish] — 2026-03

**Engineering & demo quality pass**

- Added GitHub Actions CI workflow (tests + ruff + mypy on every push)
- Added `Makefile` with targets: `test`, `lint`, `typecheck`, `demo`, `eval`, `clean`
- Added `.pre-commit-config.yaml` with ruff, ruff-format, and file sanity hooks
- Added `mypy` type checking; `mypy src/` reports zero issues across 27 source files
- Created `docs/architecture.md` — Transformer architecture, data flow, component map
- Revamped `README.md` hero section: key results upfront, 3-command quick-start
- Extracted `src/utils/comparison.py` from Streamlit app; added 8 new tests (30 total)
- Added two Jupyter notebooks: EDA (`01_data_exploration`) and model comparison (`02_model_comparison`)
- Added `scripts/run_full_evaluation.sh` for one-command seen+unseen evaluation across all models
- Streamlit app restructured into three tabs: Inference, Model Comparison, Training History

---

## [M4] — Streamlit demo

**Milestone: project is fully demo-ready**

- Built interactive Streamlit app: perturbation selection, inference, target ranking, DEG display
- Added synthetic demo generation scripts for offline showcasing without dataset download
- Added `scripts/run_generate_synthetic_showcase.sh` producing all three model artifacts and figures
- Committed synthetic showcase figures to `docs/assets/` for README and offline walkthroughs
- Added committed Norman2019 real-data result figures (`model_comparison_seen_norman2019_demo.png`)

---

## [M3] — Evaluation & ranking

**Milestone: full evaluation pipeline with seen/unseen split**

- Implemented per-perturbation Pearson, MSE, Top-k DEG overlap
- Added unseen perturbation split (held-out conditions not seen during training)
- Implemented `src/evaluation/deg.py`: scanpy Wilcoxon DEG computation and artifact export
- Implemented `src/ranking/target_ranking.py`: importance score = 0.5 × |predicted delta| + 0.5 × DEG significance
- Added `scripts/run_generate_deg_artifact.sh` and `scripts/run_evaluate.sh`
- Added `scripts/run_summarize_run.sh` for structured `run_summary.json` artifact
- Real Norman2019 results: Transformer unseen Pearson = 0.824, top-100 DEG overlap = 0.976

---

## [M2] — Model training

**Milestone: Transformer and baselines trained on Norman2019**

- Implemented `TransformerPerturbationModel`: gene token = gene embedding + value projection + perturbation embedding
- Implemented MLP baseline (3-layer FC network)
- Implemented XGBoost baseline (per-gene regression)
- Training loop with best-checkpoint saving, validation metrics, `history.json`
- All three models trained on Norman2019 demo bundle (10,500 × 512 × 105 conditions)
- Added `scripts/run_train_transformer.sh` and `scripts/run_train_baselines.sh`

---

## [M1] — Data pipeline

**Milestone: reproducible preprocessing and bundle export**

- Implemented full preprocessing pipeline: QC → normalize → log1p → HVG → sparse-to-dense
- Batch-aware control mean pairing; fallback to global control mean within cell context
- Delta expression target: `perturbed_cell - matched_control_mean`
- Seen/unseen split protocol with persistent split indices
- Bundle export as `.npz` with gene map, perturbation map, metadata
- AnnData schema auto-resolution for Norman2019 column names
- Added `scripts/download_norman2019.sh`, `scripts/run_preprocess_demo.sh`

---

## [M0] — Project initialization

**Milestone: reproducible local engineering skeleton**

- Project scaffolding: `src/`, `scripts/`, `configs/`, `tests/`, `app/`, `docs/`, `notebooks/`
- `uv`-managed environment with `pyproject.toml` as dependency source of truth
- YAML configuration files: `data.yaml`, `model.yaml`, `train.yaml`
- `AGENTS.md` anti-drift guardrails and architecture boundary rules
- `PROJECT_PLAN.md` scoping all milestones and key technical decisions
