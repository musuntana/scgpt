# Evaluation protocol audit (Norman2019)

**Status: protocol limitations identified; historical metrics are retained but the
zero-shot interpretation is not validated.** This document is a reproducibility
guardrail, not a claim that the existing numerical outputs were calculated
incorrectly.

## Current split behavior

- `src/data/pairing.py::_three_way_stratified_split` creates sample-level seen
  train/validation/test partitions. The same perturbation identity can appear
  in all three partitions.
- `src/data/pairing.py::_three_way_group_split` assigns each perturbation ID
  to one unseen partition. The current implementation now rejects invalid
  fractions and too few groups for the requested non-empty partitions.
- Split arrays are saved in `splits.npz`. Newly generated bundles also persist
  sample counts, perturbation-group counts, index integrity, stratification
  status, and fallback reasons in `metadata.json`; run summaries expose these
  under `split.audit`. Legacy bundles receive computed index/group integrity
  counts, but their original stratification fallback reason cannot be recovered.
  Regenerate the bundle before using the corrected split code.

## Two limitations that affect the meaning of "unseen"

### 1. The model uses categorical perturbation IDs

Both `src/models/transformer.py` and `src/models/mlp.py` create a learned
`nn.Embedding(num_perturbations, ...)` table.

- A seen-protocol checkpoint has training examples from every perturbation
  identity, so its performance on the separate unseen-group indices is not a
  held-out-identity evaluation.
- An unseen-protocol checkpoint holds out perturbation identities, but their
  embedding rows receive no training updates. This architecture does not
  provide a meaningful learned representation for those held-out IDs.

The training entrypoints now evaluate only the test split matching the selected
training protocol and write `training_protocol.json` beside the checkpoint.
The generic evaluation CLI checks that record (or a legacy run summary) and
rejects cross-protocol evaluation. Run summaries, comparison rows, project
snapshots, and multi-seed reports now retain only metrics matching the run's
training protocol. Historical files may still contain old cross-protocol
metrics; they are retained for traceability but are not valid evidence of
zero-shot generalization.

A future zero-shot experiment needs a representation shared across perturbation
identities (for example, explicit gene-level or other perturbation features),
or the project must narrow its claim to sample-level generalization for
perturbations represented during training. Architecture changes and new
biological performance claims require separate controlled experiments.

## 2. Gene filtering and HVG selection

The CLI preprocessing path runs cell-level QC and deterministic
per-perturbation subsampling before planning both split protocols. It then calls
`src/data/preprocess.py::select_hvgs_from_training_partitions`, which fits
the `min_cells_per_gene` filter on raw counts from the intersection of
`seen_train` and `unseen_train` perturbation samples plus all filtered control
cells. After applying the retained-gene mask to all cells, it normalizes each
cell independently and optionally applies log1p, then fits HVGs on the same
training/reference subset. The selected genes are applied to the full AnnData
object, and the method, fit counts, seed, normalization settings, and control
scope are recorded in `metadata.json` and run summaries.

Using the intersection is conservative: fewer perturbation cells fit the gene
selection than are available in either individual training partition. However,
it means no perturbation sample held out by either protocol can influence gene
filtering or HVG selection, while retaining a single bundle compatible with the
existing two-protocol evaluation flow.

Per-cell normalization and log transformation operate independently per cell
after the retained gene universe has been fixed from the training/reference
subset; they do not fit cross-sample normalization statistics. Cell QC and
capped subsampling occur before the split and are based on per-cell QC/counts
or perturbation labels rather than fitting a predictive model, but should be
reconsidered if the evaluation target changes to generalization across batches
or cell contexts.

**Existing processed bundles are not retroactively corrected.** Re-run
preprocessing to create new bundles before making new claims. Their hashes and
preprocessing diagnostics should be retained with each result. Direct calls to
`prepare_adata()` using its backward-compatible defaults still perform
pre-split gene filtering/HVG selection; strict evaluation pipelines should use
the new split-aware selector or the CLI.

## What can be said today

- The README's existing values are historical outputs from saved artifacts.
- They may be useful as pipeline smoke-test or within-identity sample-holdout
  results, depending on how each checkpoint was trained.
- They must not be presented as validated zero-shot generalization to unseen
  perturbation identities until the split, representation, and preprocessing
  boundaries are corrected and re-evaluated.
- Split disjointness tests prove index/group separation only; they do not prove
  biological generalization or absence of preprocessing leakage.

## Verification checklist

1. Validate fractions before splitting and reject empty/undersized requested
   partitions with an actionable error.
2. Test sample-index disjointness and full sample coverage for seen and unseen
   partitions.
3. Test that perturbation groups are pairwise disjoint across unseen partitions.
4. Record sample counts, group counts, split protocol, and seed with each run.
5. Re-run the real Norman2019 pipeline with the corrected protocol and preserve
   the old artifacts as historical results rather than silently overwriting them.


## Checkpoint-selection safeguards

The training loop selects validation checkpoints using the metric's declared
direction: both MSE metrics are minimized, while per-perturbation and per-gene
Pearson correlations are maximized. Unknown checkpoint metrics and invalid epoch
or patience settings are rejected before training. Empty training and validation
loaders fail closed rather than yielding a synthetic zero score or a spurious
best checkpoint; empty evaluation splits cannot report metrics. Non-finite
training loss, gradients, evaluation predictions, targets, or metrics stop the
run rather than writing an apparently successful result.

These are engineering correctness checks, **not** evidence that unseen
perturbation generalization has been validated on real Norman2019 data.


## Checkpoint and processed-bundle identity

New `training_protocol.json` artifacts (schema 2) now record the SHA-256 hashes
of the three input files `arrays.npz`, `splits.npz`, and `metadata.json`.
Both the evaluation CLI and the Streamlit model loader verify these exact bytes
before loading a checkpoint. A bundle with the same number of genes but a
different gene order, different perturbation labels, or a changed train/test
split is therefore refused rather than silently producing a score.

Historical schema-1 artifacts can still be loaded for backward compatibility,
but their bundle identity is **unverified**, and the evaluation CLI issues a
warning. Do not treat scores produced from unverified legacy artifacts as
validated cross-run comparisons. Changing to another processed bundle requires
retraining or retrieving its matching checkpoint.


The `run_summary.json` best-epoch audit now uses the same metric direction
as training checkpoint selection. For MSE it selects the smallest validation
value; for Pearson it selects the largest. Missing, unsupported, or non-finite
validation scores are rejected instead of silently reporting an incorrect best
epoch.


## Model-configuration identity and UI access

New Transformer runs record a schema-3 training protocol including the selected
model type and the SHA-256 of the exact model YAML used at training time. The
evaluation CLI and Streamlit inference loader reject model-type mismatches and
changed configuration files **before** loading the weights, even when dropout
or other settings can change without altering parameter shapes.

MLP/XGBoost training protocols now record their model type as well; their
constructors do not read Transformer YAML. Historical protocols without enough
model configuration provenance remain readable but are **unverified** and
must not support claims of protocol-matched performance. The Streamlit model
selector also exposes the existing `transformer_shared_gene_identity` model.


## Observed DEG data versus independent prediction

The local DEG artifact is derived from **measured perturbation responses**
and may contain held-out condition labels. It is valid for explicitly labeled
retrospective error/DEG comparison but is **not** an independent feature for
predicting unseen perturbations. The Streamlit target ranking defaults to
`prediction_only`, using only the model-predicted absolute delta. The optional
`observed_deg_diagnostic` mode requires an explicit user selection and shows
a warning that observed DEG significance influences its scores. The DEG
generator now records this evaluation-only evidence scope in its metadata.
Do not use diagnostic-mode ranks as zero-shot prediction or prospective
target-prioritization evidence.
