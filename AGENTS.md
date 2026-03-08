# AGENTS.md

## Purpose
This repository builds `PerturbScope-GPT`, a local-first AI4Bio project for single-cell perturbation response prediction and target prioritization.

The repository exists to produce a job-ready, demo-ready MVP, not a large-scale paper reproduction platform.

## Source of Truth

The following files define project intent and must stay aligned:
- `PROJECT_PLAN.md`: scope, milestones, architecture decisions, deliverables
- `README.md`: setup, usage, and demo instructions
- `configs/*.yaml`: runtime configuration and local-safe defaults

If implementation needs broader scope, update `PROJECT_PLAN.md` first.

## Product Scope

The MVP includes only:
- one public dataset pipeline centered on `scPerturb / Norman2019`
- single-gene perturbation samples only
- one Transformer model for perturbation response prediction
- baseline comparisons with MLP and XGBoost
- evaluation with Pearson, MSE, and Top-k DEG overlap
- a Streamlit demo for inference and result visualization

Do not add these unless explicitly requested:
- multi-dataset training
- multi-gene combinatorial perturbation modeling
- distributed training
- large-scale pretraining
- workflow orchestration platforms
- cloud infrastructure
- database-backed services

## Local-First Principles

1. Keep the project runnable on a single developer machine.
2. Prefer a smaller, reproducible MVP over a larger but unstable system.
3. Restrict default model and data settings to local-safe values.
4. Expand data size or model size only after a successful baseline run.
5. Keep all scientific assumptions explicit in docs or configs.

## Architecture Rules

### Directory responsibilities
- `src/data`: data loading, validation, preprocessing, pairing, torch datasets
- `src/models`: model definitions only
- `src/training`: losses, training loops, checkpoint handling
- `src/evaluation`: metrics, DEG logic, analysis utilities
- `src/ranking`: target prioritization logic
- `src/utils`: config loading, logging helpers, seed control
- `app`: Streamlit UI only
- `scripts`: CLI entrypoints that orchestrate `src/`
- `tests`: unit and integration tests

### Boundary rules
- UI code must not contain training logic.
- Scripts must call code in `src/`; they must not duplicate business logic.
- Model files must not read raw files directly.
- Evaluation code must not silently mutate training artifacts.
- Ranking logic must document every score component and its meaning.
- `src/` code must not hard-code local data paths. Paths must come from config or CLI arguments.

## Data Standards

1. Use `AnnData` as the canonical in-memory structure for single-cell data.
2. Keep raw input matrices sparse through loading and preprocessing whenever feasible.
3. Convert to dense tensors only after HVG selection and only for the slices needed for training.
4. Preserve raw metadata whenever feasible.
5. Record preprocessing assumptions explicitly:
   - filtering thresholds
   - normalization method
   - log transform
   - HVG selection count
   - sparse-to-dense stage
   - pairing strategy
6. Never hard-code gene indices without saving mapping artifacts.
7. Raw data is immutable. Derived outputs go to `data/interim` or `data/processed`.
8. Large datasets and generated model artifacts must not be committed unless intentionally versioned.

## Dataset Scope Rules

1. Default dataset is `scPerturb / Norman2019`.
2. Default sample scope is single-gene perturbations only.
3. Exclude multi-gene combinations unless the plan is explicitly expanded.
4. Use a matched control mean strategy for MVP instead of per-cell pairing.
5. If batch metadata exists, use batch-aware control means. If not, fall back to a documented global control mean within the same cell context.

## Modeling Standards

1. Start with a minimal Transformer that is easy to train and inspect.
2. Default target is `delta expression`.
3. Default perturbation conditioning is additive injection into every gene token.
4. Default positional encoding is disabled for MVP.
5. Keep baseline implementations functional and comparable.
6. Use configuration files for:
   - HVG count
   - hidden dimensions
   - number of heads
   - depth
   - dropout
   - batch size
   - learning rate
   - seed
7. Fix random seeds for training and evaluation runs.
8. Save best checkpoints based on a declared validation metric.

## Memory Guardrails

1. Default `HVG` should stay in the `512-800` range.
2. Do not exceed `1000 HVG` without a documented memory estimate.
3. If local resources are tight, reduce in this order:
   - HVG count
   - batch size
   - model depth
4. Do not introduce linear-attention variants unless standard attention has already been shown to be impractical for the chosen local settings.

## Evaluation Standards

Minimum required metrics:
- Pearson correlation
- MSE
- Top-k DEG overlap

Evaluation rules:
- primary reporting granularity is `per-perturbation`
- secondary reporting can include `per-gene`
- `per-cell` analysis is optional, not the headline metric
- seen and unseen perturbation evaluations must be clearly separated
- avoid leakage across perturbation groups
- document any heuristic used in DEG or ranking logic
- do not present attention as causal evidence

## DEG Standards

Default DEG definition for MVP:
- `scanpy.tl.rank_genes_groups`
- method: `wilcoxon`
- adjusted p-value threshold: `< 0.05`
- absolute log fold change threshold: `> 0.25`

If DEG definitions change, update configs and docs together.

## Ranking Standards

1. Attention weights must not be used directly as ranking features in MVP.
2. Ranking must be based on documented score components only.
3. Default score should combine:
   - normalized absolute predicted delta
   - normalized DEG significance
4. Weight choices must be explicit in config or code comments.

## Streamlit App Rules

The app must:
- load a saved model or a clearly labeled demo artifact
- fail with actionable messages if artifacts are missing
- keep inference latency reasonable for local demo-sized inputs
- show labels and plot titles that are biologically interpretable

The app must not:
- retrain models on startup
- contain hidden file path assumptions
- silently fabricate scientific outputs

## Environment Rules

1. Prefer Python `3.10` or `3.11`.
2. If dependency compatibility conflicts appear, prefer the most stable environment over the newest version.
3. Record the chosen Python version in `README.md` and environment setup files.

## Coding Conventions

1. Prefer type hints on public functions.
2. Write short docstrings for non-trivial modules and functions.
3. Keep functions focused; do not mix loading, transformation, and reporting in one function.
4. Avoid global state except for logger setup and constants.
5. Use `pathlib` where practical.
6. Use ASCII by default unless existing files require Unicode.
7. Use `logger.py` or similarly explicit names; avoid names that shadow standard library modules.

## Config Standards

1. All runtime knobs must be configurable through YAML and/or CLI arguments.
2. Add a centralized config loader under `src/utils/config.py`.
3. Validate required config fields before starting training or preprocessing.
4. Do not scatter default values across multiple modules.

## Testing Requirements

At minimum, add or maintain tests for:
- dataset shape and field consistency
- config loading and validation
- metric calculations
- ranking output schema
- training entrypoints that can run on tiny synthetic fixtures

Prefer fast tests with small synthetic fixtures.
Do not rely on large external datasets for routine test execution.

## Experiment Tracking

For each meaningful training run, capture:
- dataset identifier
- preprocessing parameters
- split protocol
- model configuration
- seed
- metrics
- checkpoint path

If no experiment tracker is configured, write a structured local summary artifact.

## Documentation Requirements

Any substantial change must update relevant docs:
- new module -> update `README.md` or module docstring
- new config -> document expected fields
- changed preprocessing -> update `PROJECT_PLAN.md` if assumptions changed
- changed split logic -> update the evaluation section in docs
- changed ranking formula -> explain score definition and limitations

## Decision Framework

When multiple implementation options exist, choose in this order:
1. correctness
2. reproducibility
3. local runnability
4. simplicity
5. extensibility

Do not choose sophistication over clarity unless the added complexity is necessary.

## Anti-Drift Guardrails

Stop and reassess if a proposed change would:
- introduce a new major subsystem not listed in `PROJECT_PLAN.md`
- require non-local infrastructure for the main workflow
- expand from one dataset MVP to a general platform
- add multi-gene perturbation support before the single-gene path is stable
- replace a simple baseline before it has been benchmarked
- treat the project as a full paper reproduction without a demo path

If drift is necessary, first update:
1. scope in `PROJECT_PLAN.md`
2. acceptance criteria
3. affected architecture notes

## Definition of Done For Code Changes

A change is complete only when:
- code is placed in the correct module boundary
- configuration is explicit
- basic tests or sanity checks pass
- docs are updated when behavior changes
- no unnecessary scope expansion was introduced

## Agent Workflow

Before making major changes:
1. read `PROJECT_PLAN.md`
2. identify the current phase and target deliverable
3. confirm the work stays inside the local-first MVP
4. implement the smallest change that moves that deliverable forward

When editing:
1. prefer library code under `src/`
2. keep scripts thin
3. add tests alongside behavior changes
4. document assumptions near the code or config

Before finishing:
1. run relevant tests or sanity checks
2. summarize what changed
3. list unresolved assumptions or data dependencies
