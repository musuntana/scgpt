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
